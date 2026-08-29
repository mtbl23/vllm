#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from check_sm120_chat_contract import (
    NoRedirectHandler,
    exact_context_messages,
    load_key,
    validate_endpoint,
)
from run_sm120_churn import (
    Totals,
    WorkerProgress,
    fetch_metrics,
    require,
    worker,
)


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirectHandler()
)


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "cannot calculate a percentile without samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def measured_request(
    endpoint: str,
    key: str,
    model: str,
    messages: list[dict[str, str]],
    prompt_target: int,
) -> dict[str, float | int]:
    before = fetch_metrics(endpoint)
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
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "seed": 9000000 + prompt_target,
    }
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    first_content_at: float | None = None
    completion_tokens: int | None = None
    with OPENER.open(request, timeout=300) as response:
        require(response.status == 200, "measured request did not return HTTP 200")
        for raw in response:
            line = raw.decode(errors="strict").strip()
            if not line:
                continue
            require(line.startswith("data: "), "measured stream emitted non-SSE data")
            data = line[6:].strip()
            if data == "[DONE]":
                ended = time.monotonic()
                break
            chunk = json.loads(data)
            require("error" not in chunk, "measured stream emitted an error")
            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta")
                if (
                    first_content_at is None
                    and isinstance(delta, dict)
                    and delta.get("content")
                ):
                    first_content_at = time.monotonic()
        else:
            raise ValueError("measured stream ended without [DONE]")

    require(first_content_at is not None, "measured request emitted no content")
    require(completion_tokens is not None and completion_tokens > 0, "usage missing")
    ttft = first_content_at - started
    end_to_end = ended - started
    decode_seconds = max(0.000001, ended - first_content_at)
    return {
        "prompt_tokens": prompt_target,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "end_to_end_seconds": end_to_end,
        "decode_tokens_per_second": completion_tokens / decode_seconds,
        "running_at_arrival": int(before["running"]),
        "waiting_at_arrival": int(before["waiting"]),
    }


def summarize(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    ttft = [float(item["ttft_seconds"]) for item in samples]
    end_to_end = [float(item["end_to_end_seconds"]) for item in samples]
    decode = [float(item["decode_tokens_per_second"]) for item in samples]
    waiting = [int(item["waiting_at_arrival"]) for item in samples]
    running = [int(item["running_at_arrival"]) for item in samples]
    return {
        "samples": len(samples),
        "ttft_p50_seconds": round(percentile(ttft, 0.50), 3),
        "ttft_p95_seconds": round(percentile(ttft, 0.95), 3),
        "ttft_max_seconds": round(max(ttft), 3),
        "end_to_end_p50_seconds": round(percentile(end_to_end, 0.50), 3),
        "end_to_end_p95_seconds": round(percentile(end_to_end, 0.95), 3),
        "end_to_end_max_seconds": round(max(end_to_end), 3),
        "decode_p50_tokens_per_second": round(percentile(decode, 0.50), 3),
        "decode_p05_tokens_per_second": round(percentile(decode, 0.05), 3),
        "running_at_arrival_p50": round(percentile([float(v) for v in running], 0.50), 1),
        "waiting_at_arrival_p50": round(percentile([float(v) for v in waiting], 0.50), 1),
        "waiting_at_arrival_max": max(waiting),
    }


def run_mode(
    *,
    mode: str,
    endpoint: str,
    key: str,
    model: str,
    background_clients: int,
    samples_per_prompt: int,
    background_prompts: list[list[dict[str, str]]],
    probe_prompts: dict[int, list[list[dict[str, str]]]],
) -> dict[str, Any]:
    before = fetch_metrics(endpoint)
    require(before["running"] == 0 and before["waiting"] == 0, f"{mode} requires idle")
    stop = threading.Event()
    totals = Totals()
    progress = [WorkerProgress() for _ in range(background_clients)]
    lock = threading.Lock()
    errors: list[str] = []
    barrier = threading.Barrier(background_clients + 1)
    samples: list[dict[str, float | int]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=background_clients) as pool:
        futures = [
            pool.submit(
                worker,
                worker_id,
                stop,
                endpoint,
                key,
                model,
                background_prompts,
                totals,
                progress[worker_id],
                lock,
                errors,
                barrier,
            )
            for worker_id in range(background_clients)
        ]
        barrier.wait(timeout=120)

        readiness_deadline = time.monotonic() + 180
        ready = fetch_metrics(endpoint)
        minimum_pressure = min(background_clients, 30)
        while (
            ready["running"] + ready["waiting"] < minimum_pressure
            and time.monotonic() < readiness_deadline
        ):
            time.sleep(0.25)
            ready = fetch_metrics(endpoint)
        require(
            ready["running"] + ready["waiting"] >= minimum_pressure,
            f"{mode} failed to establish realistic background pressure",
        )

        targets = [target for target in (2000, 4000, 6000) for _ in range(samples_per_prompt)]
        random.Random(22 if mode == "saturated" else 23).shuffle(targets)
        prompt_indexes = {2000: 0, 4000: 0, 6000: 0}
        measured_inputs: list[tuple[int, list[dict[str, str]]]] = []
        for target in targets:
            prompt_index = prompt_indexes[target]
            prompt_indexes[target] += 1
            measured_inputs.append(
                (target, probe_prompts[target][prompt_index])
            )
        require(not errors, errors[0] if errors else f"{mode} background failed")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(measured_inputs)
        ) as measured_pool:
            samples = list(
                measured_pool.map(
                    lambda item: measured_request(
                        endpoint,
                        key,
                        model,
                        item[1],
                        item[0],
                    ),
                    measured_inputs,
                )
            )

        stop.set()
        for future in futures:
            future.result(timeout=300)

    drain_deadline = time.monotonic() + 180
    after = fetch_metrics(endpoint)
    while (
        after["running"] != 0 or after["waiting"] != 0
    ) and time.monotonic() < drain_deadline:
        time.sleep(0.5)
        after = fetch_metrics(endpoint)
    require(not errors, errors[0] if errors else f"{mode} background failed")
    require(after["running"] == 0 and after["waiting"] == 0, f"{mode} did not drain")
    require(after["preemptions"] == before["preemptions"], f"{mode} preempted requests")

    grouped = {
        str(target): summarize(
            [item for item in samples if int(item["prompt_tokens"]) == target]
        )
        for target in (2000, 4000, 6000)
    }
    return {
        "mode": mode,
        "background_clients": background_clients,
        "measured_user_clients": len(samples),
        "total_simultaneous_clients": background_clients + len(samples),
        "samples_per_prompt": samples_per_prompt,
        "request_errors": totals.errors,
        "preemptions_delta": after["preemptions"] - before["preemptions"],
        "by_prompt_tokens": grouped,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Verse user-visible latency under realistic GPU pressure."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    args = parser.parse_args()
    require(args.samples_per_prompt >= 5, "at least five samples per prompt are required")

    endpoint = validate_endpoint(args.endpoint)
    key = load_key(args.api_key_file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        background_prompts = list(
            pool.map(
                lambda index: exact_context_messages(
                    endpoint,
                    key,
                    args.model,
                    (5500, 5750, 6000)[index % 3],
                    marker=f"Synthetic latency background {index:03d}",
                ),
                range(96),
            )
        )
        probe_items = list(
            pool.map(
                lambda item: (
                    item[0],
                    exact_context_messages(
                        endpoint,
                        key,
                        args.model,
                        item[0],
                        marker=f"Synthetic measured user {item[0]} sample {item[1]:02d}",
                    ),
                ),
                [
                    (target, index)
                    for target in (2000, 4000, 6000)
                    for index in range(args.samples_per_prompt * 2)
                ],
            )
        )
    probe_prompts = {
        target: [messages for item_target, messages in probe_items if item_target == target]
        for target in (2000, 4000, 6000)
    }

    started = time.monotonic()
    modes = [
        run_mode(
            mode="saturated",
            endpoint=endpoint,
            key=key,
            model=args.model,
            background_clients=14,
            samples_per_prompt=args.samples_per_prompt,
            background_prompts=background_prompts,
            probe_prompts=probe_prompts,
        ),
        run_mode(
            mode="overloaded",
            endpoint=endpoint,
            key=key,
            model=args.model,
            background_clients=52,
            samples_per_prompt=args.samples_per_prompt,
            background_prompts=background_prompts,
            probe_prompts={
                target: prompts[args.samples_per_prompt :]
                for target, prompts in probe_prompts.items()
            },
        ),
    ]
    print(
        json.dumps(
            {
                "status": "pass",
                "wall_seconds_including_setup_and_drain": round(
                    time.monotonic() - started, 3
                ),
                "completion_tokens_requested": 128,
                "modes": modes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
