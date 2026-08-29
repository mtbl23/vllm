#!/usr/bin/env python3
"""Measure loaded cold-to-warm user latency for the Verse SM120 runtime.

The harness creates one synthetic chat per concurrent client. It first sends all
chats together to establish their KV prefixes, then immediately sends one delta
turn for every same chat together. Reports contain timing and aggregate runtime
pressure only - generated text is never printed or persisted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
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
from run_sm120_churn import fetch_metrics, require


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirectHandler()
)
PROMPT_TARGETS = (2000, 4000, 6000)


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "cannot calculate a percentile without samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def completion(
    endpoint: str,
    key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, float | int], str]:
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
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.monotonic()
    first_content_at: float | None = None
    completion_tokens: int | None = None
    fragments: list[str] = []
    with OPENER.open(request, timeout=300) as response:
        require(response.status == 200, "completion did not return HTTP 200")
        for raw in response:
            line = raw.decode(errors="strict").strip()
            if not line:
                continue
            require(line.startswith("data: "), "stream emitted non-SSE data")
            data = line[6:].strip()
            if data == "[DONE]":
                ended = time.monotonic()
                break
            chunk = json.loads(data)
            require("error" not in chunk, "stream emitted an error")
            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            text = delta.get("content") if isinstance(delta, dict) else None
            if text:
                if first_content_at is None:
                    first_content_at = time.monotonic()
                fragments.append(text)
        else:
            raise ValueError("stream ended without [DONE]")
    require(first_content_at is not None, "completion emitted no content")
    require(completion_tokens is not None and completion_tokens > 0, "usage missing")
    content = "".join(fragments)
    require(bool(content), "completion content was empty")
    decode_seconds = max(0.000001, ended - first_content_at)
    return (
        {
            "completion_tokens": completion_tokens,
            "ttft_seconds": first_content_at - started,
            "end_to_end_seconds": ended - started,
            "decode_tokens_per_second": completion_tokens / decode_seconds,
        },
        content,
    )


def pressure_sampler(endpoint: str, stop: threading.Event, samples: list[dict[str, float]]) -> None:
    while not stop.wait(0.1):
        samples.append(fetch_metrics(endpoint))


def run_phase(
    endpoint: str,
    key: str,
    model: str,
    work: list[tuple[int, list[dict[str, str]], int]],
    *,
    max_tokens: int,
) -> tuple[list[tuple[int, dict[str, float | int], str]], dict[str, float]]:
    stop = threading.Event()
    pressure: list[dict[str, float]] = []
    sampler = threading.Thread(
        target=pressure_sampler, args=(endpoint, stop, pressure), daemon=True
    )
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as pool:
            futures = [
                pool.submit(
                    completion,
                    endpoint,
                    key,
                    model,
                    messages,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                for _, messages, seed in work
            ]
            results = [
                (target, *future.result(timeout=360))
                for (target, _, _), future in zip(work, futures, strict=True)
            ]
    finally:
        stop.set()
        sampler.join(timeout=2)
    require(bool(pressure), "runtime pressure sampler captured no measurements")
    return results, {
        "max_running": max(float(item["running"]) for item in pressure),
        "max_waiting": max(float(item["waiting"]) for item in pressure),
        "samples": len(pressure),
    }


def summarize(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    ttft = [float(item["ttft_seconds"]) for item in samples]
    end_to_end = [float(item["end_to_end_seconds"]) for item in samples]
    decode = [float(item["decode_tokens_per_second"]) for item in samples]
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
    }


def grouped(results: list[tuple[int, dict[str, float | int], str]]) -> dict[str, Any]:
    return {
        str(target): summarize(
            [metrics for item_target, metrics, _ in results if item_target == target]
        )
        for target in PROMPT_TARGETS
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=38)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    require(args.clients >= 6, "at least six loaded clients are required")
    require(args.max_tokens > 0, "max tokens must be positive")

    endpoint = validate_endpoint(args.endpoint)
    key = load_key(args.api_key_file)
    before = fetch_metrics(endpoint)
    require(before["running"] == 0 and before["waiting"] == 0, "runtime must be idle")

    assignments = [PROMPT_TARGETS[index % len(PROMPT_TARGETS)] for index in range(args.clients)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        prompts = list(
            pool.map(
                lambda item: exact_context_messages(
                    endpoint,
                    key,
                    args.model,
                    item[1] - 200,
                    marker=f"Synthetic warm user {item[0]:03d}",
                ),
                enumerate(assignments),
            )
        )

    cold_work = [
        (target, messages, 22000000 + index)
        for index, (target, messages) in enumerate(zip(assignments, prompts, strict=True))
    ]
    cold, cold_pressure = run_phase(
        endpoint, key, args.model, cold_work, max_tokens=args.max_tokens
    )

    warm_work: list[tuple[int, list[dict[str, str]], int]] = []
    for index, ((target, _, content), original) in enumerate(zip(cold, prompts, strict=True)):
        warm_work.append(
            (
                target,
                [
                    *original,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "I hold the thread of the scene and continue with one short new beat.",
                    },
                ],
                23000000 + index,
            )
        )
    warm, warm_pressure = run_phase(
        endpoint, key, args.model, warm_work, max_tokens=args.max_tokens
    )

    drain_deadline = time.monotonic() + 120
    after = fetch_metrics(endpoint)
    while (after["running"] or after["waiting"]) and time.monotonic() < drain_deadline:
        time.sleep(0.25)
        after = fetch_metrics(endpoint)
    require(after["running"] == 0 and after["waiting"] == 0, "runtime did not drain")
    require(after["preemptions"] == before["preemptions"], "runtime preempted requests")

    print(
        json.dumps(
            {
                "status": "pass",
                "clients": args.clients,
                "completion_tokens_requested": args.max_tokens,
                "seed_prompt_tokens_below_warm_target": 200,
                "cold": {"pressure": cold_pressure, "by_prompt_tokens": grouped(cold)},
                "warm_delta": {"pressure": warm_pressure, "by_prompt_tokens": grouped(warm)},
                "preemptions_delta": after["preemptions"] - before["preemptions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
