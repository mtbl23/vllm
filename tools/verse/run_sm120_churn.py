#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_sm120_chat_contract import (
    NoRedirectHandler,
    exact_context_messages,
    load_key,
    validate_endpoint,
)

OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirectHandler()
)
METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "generation_tokens": "vllm:generation_tokens_total",
}


@dataclass
class Totals:
    completed: int = 0
    cancelled: int = 0
    errors: int = 0
    chunks: int = 0


@dataclass
class WorkerProgress:
    requests: int = 0
    completed: int = 0
    cancelled: int = 0
    chunks: int = 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        metric = fields[0].split("{", 1)[0]
        for name, expected in METRICS.items():
            if metric == expected:
                value = float(fields[1])
                require(math.isfinite(value), f"{name} metric is not finite")
                values[name].append(value)
    parsed: dict[str, float] = {}
    for name, found in values.items():
        require(len(found) == 1, f"expected exactly one {name} metric series")
        parsed[name] = found[0]
    return parsed


def fetch_metrics(endpoint: str) -> dict[str, float]:
    request = urllib.request.Request(f"{endpoint}/metrics", method="GET")
    with OPENER.open(request, timeout=15) as response:
        return parse_metrics(response.read().decode())


def consume_stream(
    endpoint: str,
    key: str,
    payload: dict[str, Any],
    cancel_after_first_content: bool,
) -> tuple[bool, int]:
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    chunks = 0
    with OPENER.open(request, timeout=240) as response:
        require(response.status == 200, "churn request did not return HTTP 200")
        for raw in response:
            line = raw.decode(errors="strict").strip()
            if not line:
                continue
            require(line.startswith("data: "), "churn stream emitted non-SSE data")
            data = line[6:].strip()
            if data == "[DONE]":
                return False, chunks
            chunk = json.loads(data)
            require("error" not in chunk, "churn stream emitted an error")
            choices = chunk.get("choices")
            require(isinstance(choices, list) and choices, "churn chunk lacks choices")
            delta = choices[0].get("delta")
            require(isinstance(delta, dict), "churn chunk lacks a delta")
            chunks += 1
            if cancel_after_first_content and delta.get("content"):
                return True, chunks
    raise ValueError("churn stream ended without [DONE]")


def worker(
    worker_id: int,
    stop: threading.Event,
    endpoint: str,
    key: str,
    model: str,
    prompts: list[list[dict[str, str]]],
    totals: Totals,
    progress: WorkerProgress,
    lock: threading.Lock,
    errors: list[str],
    start_barrier: threading.Barrier,
) -> None:
    try:
        start_barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        with lock:
            totals.errors += 1
            errors.append(f"worker {worker_id}: coordinated start barrier broke")
        stop.set()
        return

    iteration = 0
    while not stop.is_set():
        messages = prompts[(worker_id + iteration) % len(prompts)]
        cancel = (worker_id * 31 + iteration) % 5 == 0
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.85,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.08,
            "repetition_penalty": 1.08,
            "frequency_penalty": 0.5,
            "top_k": 0,
            "max_tokens": 128,
            "stream": True,
            "stop": ["\nRules:", "\nCharacter:"],
            "seed": worker_id * 100000 + iteration + 1,
            "ignore_eos": True,
        }
        try:
            cancelled, chunks = consume_stream(
                endpoint, key, payload, cancel_after_first_content=cancel
            )
            with lock:
                totals.chunks += chunks
                progress.requests += 1
                progress.chunks += chunks
                if cancelled:
                    totals.cancelled += 1
                    progress.cancelled += 1
                else:
                    totals.completed += 1
                    progress.completed += 1
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            with lock:
                totals.errors += 1
                errors.append(f"worker {worker_id}: {type(exc).__name__}: {exc}")
            stop.set()
            return
        iteration += 1


def validate_worker_progress(
    progress: list[WorkerProgress], totals: Totals
) -> dict[str, int]:
    require(bool(progress), "churn has no worker progress records")
    for worker_id, item in enumerate(progress):
        require(item.requests > 0, f"worker {worker_id} made no request progress")
        require(item.chunks > 0, f"worker {worker_id} produced no stream chunks")
        require(
            item.completed + item.cancelled == item.requests,
            f"worker {worker_id} has inconsistent request progress",
        )
    require(
        sum(item.completed for item in progress) == totals.completed
        and sum(item.cancelled for item in progress) == totals.cancelled
        and sum(item.chunks for item in progress) == totals.chunks,
        "per-worker progress does not match aggregate totals",
    )
    return {
        "workers_with_progress": len(progress),
        "minimum_requests_per_worker": min(item.requests for item in progress),
        "minimum_chunks_per_worker": min(item.chunks for item in progress),
    }


def validate_metrics_samples(
    samples: list[dict[str, float]],
    *,
    concurrency: int,
    duration_seconds: float,
    interval_seconds: float,
) -> dict[str, int | float]:
    require(interval_seconds > 0, "metrics interval must be positive")
    require(len(samples) >= 3, "too few continuous metrics samples")
    require(
        samples[0]["elapsed_seconds"] <= max(2 * interval_seconds, 1.0),
        "metrics sampling did not begin with the load",
    )
    require(
        samples[-1]["elapsed_seconds"]
        >= duration_seconds - max(2 * interval_seconds, 1.0),
        "metrics sampling did not cover the load duration",
    )

    maximum_gap = 0.0
    full_running_samples = 0
    full_running_streak = 0
    best_full_running_streak = 0
    best_full_running_seconds = 0.0
    best_full_running_generation_delta = 0.0
    streak_started = 0.0
    streak_started_generation = 0.0
    for index, sample in enumerate(samples):
        elapsed = sample["elapsed_seconds"]
        require(elapsed >= 0, "metrics sample timestamp is negative")
        if index:
            previous = samples[index - 1]
            gap = elapsed - previous["elapsed_seconds"]
            require(gap > 0, "metrics sample timestamps are not increasing")
            maximum_gap = max(maximum_gap, gap)
            for counter in (
                "generation_tokens",
                "preemptions",
                "prefix_queries",
                "prefix_hits",
            ):
                require(
                    sample[counter] >= previous[counter],
                    f"{counter} metric decreased during load",
                )
        require(
            0 <= sample["running"] <= concurrency,
            "running-request metric is outside the fixed worker range",
        )
        require(sample["waiting"] >= 0, "waiting-request metric is negative")
        require(0 <= sample["kv_usage"] <= 1, "KV usage metric is outside [0, 1]")
        require(
            sample["preemptions"] == samples[0]["preemptions"],
            "preemption metric changed during load",
        )
        if sample["running"] == concurrency:
            full_running_samples += 1
            if full_running_streak == 0:
                streak_started = elapsed
                streak_started_generation = sample["generation_tokens"]
            full_running_streak += 1
            streak_seconds = elapsed - streak_started
            generation_delta = sample["generation_tokens"] - streak_started_generation
            if generation_delta > 0 and (
                streak_seconds,
                full_running_streak,
            ) > (
                best_full_running_seconds,
                best_full_running_streak,
            ):
                best_full_running_streak = full_running_streak
                best_full_running_seconds = streak_seconds
                best_full_running_generation_delta = generation_delta
        else:
            full_running_streak = 0

    require(
        maximum_gap <= max(3 * interval_seconds, 2.0),
        "metrics samples were not continuous during load",
    )
    require(
        best_full_running_streak >= 3 and best_full_running_seconds >= interval_seconds,
        "metrics did not prove sustained "
        f"{concurrency} running requests with decode progress",
    )
    full_running_sample_fraction = full_running_samples / len(samples)
    require(
        full_running_sample_fraction >= 0.20,
        "fewer than 20 percent of continuous churn samples proved all "
        f"{concurrency} requests running",
    )
    generation_values = [sample["generation_tokens"] for sample in samples]
    require(
        max(generation_values) > min(generation_values),
        "generation-token metric did not prove decode progress during load",
    )
    require(
        max(sample["kv_usage"] for sample in samples) > 0,
        "KV occupancy was never observed during load",
    )
    return {
        "metrics_samples": len(samples),
        "maximum_metrics_gap_seconds": maximum_gap,
        "full_running_streak_samples": best_full_running_streak,
        "full_running_streak_seconds": best_full_running_seconds,
        "full_running_generation_tokens_delta": (best_full_running_generation_delta),
        "full_running_samples": full_running_samples,
        "full_running_sample_fraction": full_running_sample_fraction,
        "observed_max_running": int(max(sample["running"] for sample in samples)),
        "observed_max_kv_cache_usage": max(sample["kv_usage"] for sample in samples),
        "observed_generation_tokens_delta": max(generation_values)
        - min(generation_values),
    }


def collect_metrics_during_load(
    endpoint: str,
    stop: threading.Event,
    *,
    started: float,
    duration_seconds: float,
    interval_seconds: float,
) -> list[dict[str, float]]:
    require(interval_seconds > 0, "metrics interval must be positive")
    deadline = started + duration_seconds
    next_sample = started
    samples: list[dict[str, float]] = []
    while True:
        now = time.monotonic()
        if now >= next_sample:
            sample = fetch_metrics(endpoint)
            samples.append({"elapsed_seconds": time.monotonic() - started, **sample})
            next_sample += interval_seconds
            now = time.monotonic()
        if now >= deadline or stop.is_set():
            break
        wait_seconds = min(deadline - now, max(0.0, next_sample - now))
        stop.wait(wait_seconds)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the output-free Verse SM120 cancellation and KV churn gate."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=38)
    parser.add_argument("--prompt-pool-size", type=int, default=64)
    parser.add_argument("--metrics-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    try:
        require(args.duration_seconds >= 60, "duration must be at least 60 seconds")
        require(args.concurrency == 38, "the fixed churn gate requires 38 workers")
        require(args.prompt_pool_size >= 38, "prompt pool must cover every worker")
        require(
            0 < args.metrics_interval_seconds <= 5,
            "metrics interval must be in (0, 5] seconds",
        )
        endpoint = validate_endpoint(args.endpoint)
        key = load_key(args.api_key_file)
        prompt_targets = (1000, 5500, 6000)
        prompts = [
            exact_context_messages(
                endpoint,
                key,
                args.model,
                prompt_targets[index % len(prompt_targets)],
                marker=f"Synthetic churn prefix {index:03d}",
            )
            for index in range(args.prompt_pool_size)
        ]
        before = fetch_metrics(endpoint)
        require(
            before["running"] == 0 and before["waiting"] == 0,
            "churn requires an idle server",
        )

        stop = threading.Event()
        totals = Totals()
        worker_progress = [WorkerProgress() for _ in range(args.concurrency)]
        lock = threading.Lock()
        errors: list[str] = []
        start_barrier = threading.Barrier(args.concurrency + 1)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(
                    worker,
                    worker_id,
                    stop,
                    endpoint,
                    key,
                    args.model,
                    prompts,
                    totals,
                    worker_progress[worker_id],
                    lock,
                    errors,
                    start_barrier,
                )
                for worker_id in range(args.concurrency)
            ]
            try:
                start_barrier.wait(timeout=120)
            except threading.BrokenBarrierError as exc:
                stop.set()
                raise ValueError("coordinated churn start barrier broke") from exc
            started = time.monotonic()
            metrics_samples = collect_metrics_during_load(
                endpoint,
                stop,
                started=started,
                duration_seconds=args.duration_seconds,
                interval_seconds=args.metrics_interval_seconds,
            )
            stop.set()
            for future in futures:
                future.result(timeout=300)

        drain_deadline = time.monotonic() + 180
        after = fetch_metrics(endpoint)
        while (
            after["running"] != 0 or after["waiting"] != 0
        ) and time.monotonic() < drain_deadline:
            time.sleep(1)
            after = fetch_metrics(endpoint)

        require(not errors, errors[0] if errors else "churn worker failed")
        require(totals.errors == 0, "churn recorded request errors")
        require(totals.completed >= args.concurrency, "too few completed requests")
        require(totals.cancelled > 0, "cancellation route was not exercised")
        progress_evidence = validate_worker_progress(worker_progress, totals)
        metrics_evidence = validate_metrics_samples(
            metrics_samples,
            concurrency=args.concurrency,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.metrics_interval_seconds,
        )
        require(after["running"] == 0, "running requests did not drain")
        require(after["waiting"] == 0, "waiting requests did not drain")
        require(
            after["preemptions"] == before["preemptions"],
            "the runtime preempted requests during churn",
        )
        require(
            after["prefix_hits"] > before["prefix_hits"],
            "prefix-cache reuse was not observed",
        )
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        concurrent.futures.TimeoutError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "status": "pass",
                "duration_seconds": round(time.monotonic() - started, 3),
                "concurrency": args.concurrency,
                "prompt_pool_size": args.prompt_pool_size,
                "completed_requests": totals.completed,
                "cancelled_requests": totals.cancelled,
                "stream_chunks": totals.chunks,
                "request_errors": totals.errors,
                "coordinated_start_workers": args.concurrency,
                "worker_progress": [
                    {
                        "worker_id": worker_id,
                        "requests": item.requests,
                        "completed": item.completed,
                        "cancelled": item.cancelled,
                        "chunks": item.chunks,
                    }
                    for worker_id, item in enumerate(worker_progress)
                ],
                "progress_evidence": progress_evidence,
                "metrics_evidence": metrics_evidence,
                "metrics_samples": metrics_samples,
                "metrics_before": before,
                "metrics_after": after,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
