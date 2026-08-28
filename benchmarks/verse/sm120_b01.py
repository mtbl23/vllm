#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

"""Verse SM120 B01 decode benchmark.

The no-prewarm acceptance pass measures prefixes that are disjoint from every
other release nonce. It does not claim a cold server process or globally empty
caches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRIC_RE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
RELEASE_NONCE_RE = re.compile(r"[0-9a-f]{64}")


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


ENVIRONMENT_PROXY_BLOCKER = urllib.request.ProxyHandler({})
NO_REDIRECT_OPENER = urllib.request.build_opener(
    ENVIRONMENT_PROXY_BLOCKER, NoRedirectHandler()
)


@dataclass(frozen=True)
class MetricSnapshot:
    generated: float
    running: float
    waiting: float
    preemptions: float


@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    generated: float
    running: float
    waiting: float
    preemptions: float = 0.0


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def request_text(url: str, timeout: float = 30) -> str:
    request = urllib.request.Request(url, method="GET")
    with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
        return response.read().decode()


def validate_loopback_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("endpoint must be an uncredentialed loopback HTTP origin")
    if parsed.port is None:
        raise ValueError("endpoint must include an explicit port")
    return endpoint.rstrip("/")


def load_api_key(path: Path | None, env_name: str) -> str:
    if path is None:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"{env_name} is required")
        return value
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("API key file must be an absolute regular non-symlink file")
    raw = path.read_bytes()
    if b"\r" in raw or b"\0" in raw:
        raise ValueError("API key file contains forbidden bytes")
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("API key file must contain exactly one non-empty line")
    return lines[0].decode()


def validate_release_nonce(value: str) -> str:
    if RELEASE_NONCE_RE.fullmatch(value) is None:
        raise ValueError("release nonce must be exactly 64 lowercase hex characters")
    return value


def tokenize(endpoint: str, api_key: str, model: str, prompt: str) -> int:
    response = request_json(
        "POST",
        f"{endpoint}/tokenize",
        api_key,
        {"model": model, "prompt": prompt, "add_special_tokens": True},
    )
    return int(response["count"])


def build_prompt(
    endpoint: str,
    api_key: str,
    model: str,
    stream: int,
    target: int,
    release_nonce: str,
) -> tuple[str, int]:
    release_nonce = validate_release_nonce(release_nonce)
    prefix = (
        f"Synthetic Verse release {release_nonce} {target:04d}-token "
        f"capacity stream {stream:03d}. "
    )
    unit = (
        f"neutral-{stream:03d} continuity detail remains stable while the "
        "synthetic benchmark advances one ordinary event at a time. "
    )
    corpus = prefix + unit * (target + 64)
    low = len(prefix)
    high = len(corpus)
    best = prefix
    best_count = tokenize(endpoint, api_key, model, best)

    while low <= high:
        middle = (low + high) // 2
        candidate = corpus[:middle]
        count = tokenize(endpoint, api_key, model, candidate)
        if count <= target:
            best = candidate
            best_count = count
            low = middle + 1
        else:
            high = middle - 1

    if target - best_count > 2:
        raise RuntimeError(
            f"could not construct stream {stream} near {target} tokens: {best_count}"
        )
    return best, best_count


def completion(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    output_tokens: int,
) -> int:
    response = request_json(
        "POST",
        f"{endpoint}/v1/completions",
        api_key,
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    return int(response["usage"]["completion_tokens"])


def parse_metrics(text: str) -> MetricSnapshot:
    series: dict[str, list[float]] = {
        "generated": [],
        "running": [],
        "waiting": [],
        "preemptions": [],
    }
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match is None:
            continue
        name, raw_value = match.groups()
        value = float(raw_value)
        normalized = name.replace(":", "_")
        if normalized.endswith("generation_tokens_total"):
            series["generated"].append(value)
        elif normalized.endswith("num_requests_running"):
            series["running"].append(value)
        elif normalized.endswith("num_requests_waiting"):
            series["waiting"].append(value)
        elif normalized.endswith("num_preemptions"):
            series["preemptions"].append(value)
    for name, values in series.items():
        if len(values) != 1:
            raise ValueError(
                f"expected exactly one {name} metric series, found {len(values)}"
            )
    return MetricSnapshot(
        generated=series["generated"][0],
        running=series["running"][0],
        waiting=series["waiting"][0],
        preemptions=series["preemptions"][0],
    )


def fetch_metrics(endpoint: str) -> MetricSnapshot:
    return parse_metrics(request_text(f"{endpoint}/metrics"))


def require_server_idle(endpoint: str) -> MetricSnapshot:
    metrics = fetch_metrics(endpoint)
    if metrics.running != 0 or metrics.waiting != 0:
        raise RuntimeError(
            "B01 requires an idle server before benchmark ownership: "
            f"running={metrics.running:g}, waiting={metrics.waiting:g}"
        )
    return metrics


def wait_for_idle(endpoint: str, timeout: float, interval: float) -> MetricSnapshot:
    deadline = time.monotonic() + timeout
    while True:
        metrics = fetch_metrics(endpoint)
        if metrics.running == 0 and metrics.waiting == 0:
            return metrics
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "server did not drain after B01: "
                f"running={metrics.running:g}, waiting={metrics.waiting:g}"
            )
        time.sleep(interval)


def require_no_preemption_increase(before: float, after: float) -> None:
    if after < before:
        raise RuntimeError(
            "vllm:num_preemptions reset during B01; benchmark evidence is invalid"
        )
    if after > before:
        raise RuntimeError(
            "vllm:num_preemptions increased during B01: "
            f"before={before:g}, after={after:g}"
        )


def poll_metrics(
    endpoint: str,
    stop: threading.Event,
    samples: list[MetricSample],
    interval: float,
) -> None:
    while not stop.is_set():
        try:
            metrics = fetch_metrics(endpoint)
            samples.append(
                MetricSample(
                    time.monotonic(),
                    metrics.generated,
                    metrics.running,
                    metrics.waiting,
                    metrics.preemptions,
                )
            )
        except (OSError, ValueError):
            pass
        stop.wait(interval)


def longest_full_decode_window(
    samples: list[MetricSample],
    concurrency: int,
    interval: float,
    minimum_duration: float,
    minimum_samples: int,
) -> tuple[float, float, int] | None:
    groups: list[list[MetricSample]] = []
    current: list[MetricSample] = []
    for sample in samples:
        valid = sample.running >= concurrency and sample.waiting == 0
        contiguous = (
            not current or sample.timestamp - current[-1].timestamp <= interval * 3
        )
        if valid and contiguous:
            current.append(sample)
        else:
            if len(current) >= minimum_samples:
                groups.append(current)
            current = [sample] if valid else []
    if len(current) >= minimum_samples:
        groups.append(current)
    if not groups:
        return None
    group = max(groups, key=lambda item: item[-1].timestamp - item[0].timestamp)
    duration = group[-1].timestamp - group[0].timestamp
    if duration < minimum_duration:
        return None
    throughput = (group[-1].generated - group[0].generated) / duration
    return duration, throughput, len(group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic, output-free Verse SM120 B01 decode benchmark."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=38)
    parser.add_argument("--prompt-tokens", type=int, default=5500)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup-concurrency", type=int, default=8)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help=(
            "skip this invocation's explicit prefix prewarm; acceptance uses this "
            "for its first disjoint-prefix run, not as a cold-process claim"
        ),
    )
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument("--minimum-steady-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-steady-samples", type=int, default=50)
    parser.add_argument("--minimum-aggregate", type=float, default=992.0)
    parser.add_argument("--minimum-wall-ratio", type=float, default=0.9)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--fork-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument(
        "--release-nonce",
        help="64-character release nonce; a fresh nonce is generated when omitted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.concurrency <= 38:
        raise SystemExit("concurrency must be between 1 and the profile cap of 38")
    try:
        api_key = load_api_key(args.api_key_file, args.api_key_env)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    endpoint = validate_loopback_endpoint(args.endpoint)
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", args.image_digest):
        raise SystemExit("image digest must be immutable")
    if not re.fullmatch(r"[0-9a-f]{40}", args.fork_commit):
        raise SystemExit("fork commit must be a 40-character SHA")
    try:
        supplied_nonce = args.release_nonce
        release_nonce = validate_release_nonce(
            supplied_nonce if supplied_nonce is not None else secrets.token_hex(32)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    identity = request_json("GET", f"{endpoint}/version", api_key)
    ownership_metrics = require_server_idle(endpoint)

    prompts: list[str] = []
    counts: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.concurrency, 16)
    ) as pool:
        futures = [
            pool.submit(
                build_prompt,
                endpoint,
                api_key,
                args.model,
                stream,
                args.prompt_tokens,
                release_nonce,
            )
            for stream in range(args.concurrency)
        ]
        for future in futures:
            prompt, count = future.result()
            prompts.append(prompt)
            counts.append(count)

    if not args.skip_warmup:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.warmup_concurrency
        ) as pool:
            list(
                pool.map(
                    lambda prompt: completion(endpoint, api_key, args.model, prompt, 1),
                    prompts,
                )
            )

    before_metrics = wait_for_idle(endpoint, args.idle_timeout, args.metrics_interval)

    stop = threading.Event()
    samples: list[MetricSample] = []
    poller = threading.Thread(
        target=poll_metrics,
        args=(endpoint, stop, samples, args.metrics_interval),
        daemon=True,
    )
    poller.start()
    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            completion_tokens = list(
                pool.map(
                    lambda prompt: completion(
                        endpoint, api_key, args.model, prompt, args.output_tokens
                    ),
                    prompts,
                )
            )
        elapsed = time.monotonic() - started
        after_metrics = wait_for_idle(
            endpoint, args.idle_timeout, args.metrics_interval
        )
    finally:
        stop.set()
        poller.join(timeout=2)

    steady = longest_full_decode_window(
        samples,
        args.concurrency,
        args.metrics_interval,
        args.minimum_steady_seconds,
        args.minimum_steady_samples,
    )
    steady_duration, steady_throughput, steady_samples = steady or (0.0, 0.0, 0)
    completed_requests = len(completion_tokens)
    generated = sum(completion_tokens)
    expected_generated = args.concurrency * args.output_tokens
    exact_output_lengths = all(
        token_count == args.output_tokens for token_count in completion_tokens
    )
    max_running = max((sample.running for sample in samples), default=0.0)
    wall_throughput = generated / elapsed
    minimum_wall = args.minimum_aggregate * args.minimum_wall_ratio
    preemption_error = None
    try:
        require_no_preemption_increase(
            ownership_metrics.preemptions, after_metrics.preemptions
        )
    except RuntimeError as exc:
        preemption_error = str(exc)
    passed = (
        completed_requests == args.concurrency
        and generated == expected_generated
        and exact_output_lengths
        and steady is not None
        and steady_throughput >= args.minimum_aggregate
        and wall_throughput >= minimum_wall
        and preemption_error is None
    )
    result = {
        "status": "pass" if passed else "fail",
        "endpoint": endpoint,
        "server_version": identity.get("version"),
        "image_digest": args.image_digest,
        "fork_commit": args.fork_commit,
        "model_revision": args.model_revision,
        "gpu_name": args.gpu_name,
        "release_nonce": release_nonce,
        "concurrency": args.concurrency,
        "requested_prompt_tokens": args.prompt_tokens,
        "prompt_tokens_min": min(counts),
        "prompt_tokens_max": max(counts),
        "output_tokens_per_request": args.output_tokens,
        "completed_request_count": completed_requests,
        "expected_request_count": args.concurrency,
        "completion_tokens_by_request": completion_tokens,
        "all_requests_returned_exact_output_tokens": exact_output_lengths,
        "prewarmed": not args.skip_warmup,
        "prefix_preparation": (
            "no_explicit_prewarm_this_run"
            if args.skip_warmup
            else "explicitly_prewarmed_this_run"
        ),
        "generated_tokens": generated,
        "expected_generated_tokens": expected_generated,
        "max_running_requests_observed": max_running,
        "server_idle_before_ownership": (
            ownership_metrics.running == 0 and ownership_metrics.waiting == 0
        ),
        "server_idle_before_measurement": (
            before_metrics.running == 0 and before_metrics.waiting == 0
        ),
        "server_idle_after_measurement": (
            after_metrics.running == 0 and after_metrics.waiting == 0
        ),
        "preemptions_before": ownership_metrics.preemptions,
        "preemptions_before_measurement": before_metrics.preemptions,
        "preemptions_after": after_metrics.preemptions,
        "preemptions_delta": (
            after_metrics.preemptions - ownership_metrics.preemptions
        ),
        "preemptions_measurement_delta": (
            after_metrics.preemptions - before_metrics.preemptions
        ),
        "preemption_error": preemption_error,
        "wall_seconds": round(elapsed, 6),
        "wall_aggregate_tokens_per_second": round(wall_throughput, 3),
        "minimum_wall_aggregate_tokens_per_second": round(minimum_wall, 3),
        "steady_window_seconds": round(steady_duration, 6),
        "steady_window_samples": steady_samples,
        "steady_aggregate_tokens_per_second": round(steady_throughput, 3),
        "minimum_aggregate_tokens_per_second": args.minimum_aggregate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
