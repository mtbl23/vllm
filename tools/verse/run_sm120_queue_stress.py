#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path

from check_sm120_chat_contract import (
    exact_context_messages,
    load_key,
    validate_endpoint,
)
from run_sm120_churn import (
    Totals,
    WorkerProgress,
    fetch_metrics,
    require,
    validate_metrics_samples,
    validate_worker_progress,
    worker,
)
from sm120_evidence_identity import add_identity_arguments, validated_identity


def build_prompt_pool(
    endpoint: str,
    key: str,
    model: str,
    pool_size: int,
) -> list[list[dict[str, str]]]:
    targets = (5500, 5750, 6000)

    def build(index: int) -> list[dict[str, str]]:
        return exact_context_messages(
            endpoint,
            key,
            model,
            targets[index % len(targets)],
            marker=f"Synthetic queue stress prefix {index:03d}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        prompts = list(pool.map(build, range(pool_size)))
    require(
        len({json.dumps(item, sort_keys=True) for item in prompts}) == pool_size,
        "stress prompts are not distinct",
    )
    return prompts


def run_phase(
    *,
    name: str,
    endpoint: str,
    key: str,
    model: str,
    prompts: list[list[dict[str, str]]],
    clients: int,
    active_capacity: int,
    duration_seconds: int,
    metrics_interval_seconds: float,
    require_sustained_active: bool,
) -> dict[str, object]:
    before = fetch_metrics(endpoint)
    require(
        before["running"] == 0 and before["waiting"] == 0,
        f"{name} requires an idle scheduler",
    )

    stop = threading.Event()
    totals = Totals()
    progress = [WorkerProgress() for _ in range(clients)]
    lock = threading.Lock()
    errors: list[str] = []
    barrier = threading.Barrier(clients + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [
            pool.submit(
                worker,
                worker_id,
                stop,
                endpoint,
                key,
                model,
                prompts,
                totals,
                progress[worker_id],
                lock,
                errors,
                barrier,
            )
            for worker_id in range(clients)
        ]
        try:
            barrier.wait(timeout=120)
        except threading.BrokenBarrierError as exc:
            stop.set()
            raise ValueError(f"{name} coordinated start barrier broke") from exc

        started = time.monotonic()
        deadline = started + duration_seconds
        next_sample = started
        samples: list[dict[str, float]] = []
        while not stop.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_sample:
                samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        **fetch_metrics(endpoint),
                    }
                )
                next_sample += metrics_interval_seconds
            stop.wait(min(0.1, max(0.0, deadline - time.monotonic())))

        stop.set()
        for future in futures:
            future.result(timeout=300)

    load_elapsed = time.monotonic() - started
    drain_deadline = time.monotonic() + 180
    after = fetch_metrics(endpoint)
    while (
        after["running"] != 0 or after["waiting"] != 0
    ) and time.monotonic() < drain_deadline:
        time.sleep(0.5)
        after = fetch_metrics(endpoint)

    require(not errors, errors[0] if errors else f"{name} worker failed")
    require(totals.errors == 0, f"{name} recorded request errors")
    require(after["running"] == 0, f"{name} running requests did not drain")
    require(after["waiting"] == 0, f"{name} waiting requests did not drain")
    require(
        after["preemptions"] == before["preemptions"],
        f"{name} preempted requests",
    )
    progress_evidence = validate_worker_progress(progress, totals)
    observed_max_waiting = int(max(item["waiting"] for item in samples))
    if require_sustained_active:
        metrics_evidence = validate_metrics_samples(
            samples,
            concurrency=active_capacity,
            duration_seconds=duration_seconds,
            interval_seconds=metrics_interval_seconds,
        )
        require(
            int(metrics_evidence["observed_max_running"]) == active_capacity,
            f"{name} did not saturate all {active_capacity} active slots",
        )
    else:
        require(len(samples) >= 3, f"{name} has too few metrics samples")
        maximum_gap = max(
            samples[index]["elapsed_seconds"] - samples[index - 1]["elapsed_seconds"]
            for index in range(1, len(samples))
        )
        require(
            maximum_gap <= max(3 * metrics_interval_seconds, 2.0),
            f"{name} metrics were not continuous",
        )
        for index, sample in enumerate(samples):
            require(
                all(math.isfinite(float(value)) for value in sample.values()),
                f"{name} emitted a non-finite metric",
            )
            require(
                0 <= sample["running"] <= active_capacity,
                f"{name} running metric exceeded active capacity",
            )
            require(sample["waiting"] >= 0, f"{name} waiting metric is negative")
            require(0 <= sample["kv_usage"] <= 1, f"{name} KV metric is invalid")
            require(
                sample["preemptions"] == samples[0]["preemptions"],
                f"{name} preemption metric changed",
            )
            if index:
                require(
                    sample["generation_tokens"]
                    >= samples[index - 1]["generation_tokens"],
                    f"{name} generation counter decreased",
                )
        generation_delta = (
            samples[-1]["generation_tokens"] - samples[0]["generation_tokens"]
        )
        require(generation_delta > 0, f"{name} made no decode progress")
        metrics_evidence = {
            "metrics_samples": len(samples),
            "maximum_metrics_gap_seconds": maximum_gap,
            "observed_max_running": int(max(item["running"] for item in samples)),
            "observed_max_waiting": observed_max_waiting,
            "observed_max_kv_cache_usage": max(item["kv_usage"] for item in samples),
            "observed_generation_tokens_delta": generation_delta,
        }
    if clients > active_capacity:
        require(observed_max_waiting > 0, f"{name} did not exercise the waiting queue")

    generation_delta = float(metrics_evidence["observed_generation_tokens_delta"])
    return {
        "name": name,
        "clients": clients,
        "active_capacity": active_capacity,
        "requested_duration_seconds": duration_seconds,
        "load_elapsed_seconds": round(load_elapsed, 3),
        "aggregate_decode_tokens_per_second": round(generation_delta / load_elapsed, 3),
        "completed_requests": totals.completed,
        "cancelled_requests": totals.cancelled,
        "stream_chunks": totals.chunks,
        "request_errors": totals.errors,
        "observed_max_waiting": observed_max_waiting,
        "progress_evidence": progress_evidence,
        "worker_progress": [asdict(item) for item in progress],
        "metrics_evidence": metrics_evidence,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded Verse max-capacity and overflow-queue stress gate."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--phase-seconds", type=int, default=450)
    parser.add_argument("--active-capacity", type=int, default=38)
    parser.add_argument("--overflow-clients", type=int, default=76)
    parser.add_argument("--prompt-pool-size", type=int, default=96)
    parser.add_argument("--metrics-interval-seconds", type=float, default=1.0)
    add_identity_arguments(parser)
    args = parser.parse_args()
    identity = validated_identity(args)

    require(args.phase_seconds >= 60, "each phase must run for at least 60 seconds")
    require(args.active_capacity == 38, "this appliance gate requires 38 active slots")
    require(
        args.overflow_clients > args.active_capacity,
        "overflow clients must exceed active capacity",
    )
    require(
        args.prompt_pool_size >= args.overflow_clients,
        "prompt pool must cover every overflow client",
    )
    endpoint = validate_endpoint(args.endpoint)
    key = load_key(args.api_key_file)
    prompts = build_prompt_pool(endpoint, key, args.model, args.prompt_pool_size)

    overall_started = time.monotonic()
    phases = [
        run_phase(
            name="max-active",
            endpoint=endpoint,
            key=key,
            model=args.model,
            prompts=prompts,
            clients=args.active_capacity,
            active_capacity=args.active_capacity,
            duration_seconds=args.phase_seconds,
            metrics_interval_seconds=args.metrics_interval_seconds,
            require_sustained_active=True,
        ),
        run_phase(
            name="overflow-queue",
            endpoint=endpoint,
            key=key,
            model=args.model,
            prompts=prompts,
            clients=args.overflow_clients,
            active_capacity=args.active_capacity,
            duration_seconds=args.phase_seconds,
            metrics_interval_seconds=args.metrics_interval_seconds,
            require_sustained_active=False,
        ),
    ]
    print(
        json.dumps(
            {
                "status": "pass",
                **identity,
                "stress_seconds": args.phase_seconds * 2,
                "wall_seconds_including_setup_and_drain": round(
                    time.monotonic() - overall_started, 3
                ),
                "prompt_token_targets": [5500, 5750, 6000],
                "max_completion_tokens": 128,
                "phases": phases,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
