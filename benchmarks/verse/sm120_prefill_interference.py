#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

"""Measure cold 6K prefill interference against ongoing SM120 B01 decode."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import secrets
import statistics
import threading
import time

from sm120_b01 import (
    MetricSample,
    build_prompt,
    completion,
    fetch_metrics,
    poll_metrics,
    validate_loopback_endpoint,
)


def wait_for_running(endpoint: str, minimum: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fetch_metrics(endpoint).running >= minimum:
            return
        time.sleep(0.05)
    raise RuntimeError(f"server did not reach {minimum} running requests")


def generated_rate(samples: list[MetricSample], start: float, end: float) -> float:
    window = [sample for sample in samples if start <= sample.timestamp <= end]
    if len(window) < 2:
        raise RuntimeError("metric window contains fewer than two samples")
    duration = window[-1].timestamp - window[0].timestamp
    if duration <= 0:
        raise RuntimeError("metric window duration is not positive")
    return (window[-1].generated - window[0].generated) / duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--decoders", type=int, required=True)
    parser.add_argument("--prefills", type=int, required=True)
    parser.add_argument("--decode-prompt-tokens", type=int, default=4500)
    parser.add_argument("--decode-output-tokens", type=int, default=1024)
    parser.add_argument("--prefill-prompt-tokens", type=int, default=6000)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decoders < 1 or args.prefills < 1:
        raise SystemExit("decoders and prefills must be positive")
    if args.decoders + args.prefills > 46:
        raise SystemExit("the interference harness caps total submitted requests at 46")
    endpoint = validate_loopback_endpoint(args.endpoint)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    before = fetch_metrics(endpoint)
    if before.running or before.waiting:
        raise SystemExit("interference benchmark requires an idle server")

    nonce = secrets.token_hex(32)
    total_prompts = args.decoders + args.prefills
    prompts: list[tuple[str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(total_prompts, 16)
    ) as pool:
        futures = []
        for index in range(args.decoders):
            futures.append(
                pool.submit(
                    build_prompt,
                    endpoint,
                    api_key,
                    args.model,
                    index,
                    args.decode_prompt_tokens,
                    nonce,
                )
            )
        for index in range(args.prefills):
            futures.append(
                pool.submit(
                    build_prompt,
                    endpoint,
                    api_key,
                    args.model,
                    1000 + index,
                    args.prefill_prompt_tokens,
                    nonce,
                )
            )
        prompts = [future.result() for future in futures]

    decode_prompts = [prompt for prompt, _ in prompts[: args.decoders]]
    prefill_prompts = [prompt for prompt, _ in prompts[args.decoders :]]
    prefill_counts = [count for _, count in prompts[args.decoders :]]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.decoders, 16)
    ) as pool:
        list(
            pool.map(
                lambda prompt: completion(endpoint, api_key, args.model, prompt, 1),
                decode_prompts,
            )
        )

    stop = threading.Event()
    samples: list[MetricSample] = []
    poller = threading.Thread(
        target=poll_metrics,
        args=(endpoint, stop, samples, args.metrics_interval),
        daemon=True,
    )
    poller.start()
    decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.decoders)
    prefill_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefills)
    try:
        decode_futures = [
            decode_pool.submit(
                completion,
                endpoint,
                api_key,
                args.model,
                prompt,
                args.decode_output_tokens,
            )
            for prompt in decode_prompts
        ]
        wait_for_running(endpoint, args.decoders, args.timeout)
        baseline_start = time.monotonic()
        time.sleep(args.baseline_seconds)
        all_decoders_unfinished_before = all(
            not future.done() for future in decode_futures
        )
        running_before_injection = fetch_metrics(endpoint).running
        if (
            not all_decoders_unfinished_before
            or running_before_injection < args.decoders
        ):
            raise RuntimeError("decoder cohort was not active at prefill injection")
        injection_start = time.monotonic()

        def cold_prefill(prompt: str) -> tuple[int, float]:
            started = time.monotonic()
            tokens = completion(endpoint, api_key, args.model, prompt, 1)
            return tokens, time.monotonic() - started

        prefill_futures = [
            prefill_pool.submit(cold_prefill, prompt) for prompt in prefill_prompts
        ]
        prefill_results = [
            future.result(timeout=args.timeout) for future in prefill_futures
        ]
        injection_end = time.monotonic()
        all_decoders_unfinished_after = all(
            not future.done() for future in decode_futures
        )
        decode_results = [
            future.result(timeout=args.timeout) for future in decode_futures
        ]
        drained = time.monotonic()
    finally:
        stop.set()
        poller.join(timeout=2)
        decode_pool.shutdown(wait=False, cancel_futures=True)
        prefill_pool.shutdown(wait=False, cancel_futures=True)

    baseline_rate = generated_rate(samples, baseline_start, injection_start)
    mixed_rate = generated_rate(samples, injection_start, injection_end)
    prefill_outputs = sum(tokens for tokens, _ in prefill_results)
    mixed_seconds = injection_end - injection_start
    decode_rate_during_prefill = max(
        0.0,
        (mixed_rate * mixed_seconds - prefill_outputs) / mixed_seconds,
    )
    latencies = [latency for _, latency in prefill_results]
    mixed_samples = [
        sample
        for sample in samples
        if injection_start <= sample.timestamp <= injection_end
    ]
    maximum_running_during_prefill = max(sample.running for sample in mixed_samples)
    overlap_proven = (
        all_decoders_unfinished_before
        and all_decoders_unfinished_after
        and maximum_running_during_prefill >= args.decoders + 1
    )
    after = fetch_metrics(endpoint)
    result = {
        "status": "pass",
        "decoders": args.decoders,
        "prefills": args.prefills,
        "submitted_requests": args.decoders + args.prefills,
        "decode_prompt_tokens": args.decode_prompt_tokens,
        "decode_output_tokens": args.decode_output_tokens,
        "prefill_prompt_tokens_min": min(prefill_counts),
        "prefill_prompt_tokens_max": max(prefill_counts),
        "baseline_decode_tok_s": baseline_rate,
        "mixed_generation_tok_s": mixed_rate,
        "decode_tok_s_during_prefill": decode_rate_during_prefill,
        "decode_retention_ratio": decode_rate_during_prefill / baseline_rate,
        "aggregate_prefill_tok_s": sum(prefill_counts) / mixed_seconds,
        "prefill_wall_seconds": mixed_seconds,
        "prefill_latency_seconds": latencies,
        "prefill_latency_median_seconds": statistics.median(latencies),
        "prefill_latency_max_seconds": max(latencies),
        "decode_tokens_returned": sum(decode_results),
        "all_decoders_unfinished_before_prefill": all_decoders_unfinished_before,
        "all_decoders_unfinished_after_prefill": all_decoders_unfinished_after,
        "running_before_prefill": running_before_injection,
        "maximum_running_during_prefill": maximum_running_during_prefill,
        "decode_prefill_overlap_proven": overlap_proven,
        "maximum_running": max(sample.running for sample in samples),
        "maximum_waiting": max(sample.waiting for sample in samples),
        "preemptions_delta": after.preemptions - before.preemptions,
        "drain_seconds_after_prefill": drained - injection_end,
        "server_idle_after": after.running == 0 and after.waiting == 0,
    }
    if (
        result["preemptions_delta"]
        or not result["server_idle_after"]
        or not overlap_proven
        or decode_rate_during_prefill <= 0
    ):
        result["status"] = "fail"
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
