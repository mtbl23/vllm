#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import secrets
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from sm120_evidence_identity import (
    add_identity_arguments,
    validated_optional_identity,
)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def build_isolated_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirectHandler()
    )


OPENER = build_isolated_opener()

CAPACITY_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "kv_usage": "vllm:kv_cache_usage_perc",
}
CAPACITY_PROMPT_TOKENS = 4096
CAPACITY_COMPLETION_TOKENS = 2048
CAPACITY_BLOCK_BYTES = 294_912
CAPACITY_RESERVED_BLOCKS = 1
CAPACITY_TOTAL_KV_BYTES = 5_704_253_440
SEMANTIC_CANARY_SEEDS = (1103, 2207, 3301)
SEMANTIC_COMMON_WORDS = frozenset(
    {
        "a",
        "against",
        "and",
        "as",
        "at",
        "back",
        "but",
        "came",
        "come",
        "do",
        "for",
        "from",
        "had",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "lantern",
        "me",
        "my",
        "not",
        "of",
        "on",
        "rain",
        "returned",
        "she",
        "still",
        "that",
        "the",
        "there",
        "they",
        "this",
        "through",
        "to",
        "tower",
        "was",
        "we",
        "were",
        "what",
        "when",
        "with",
        "you",
        "your",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed.port is not None
        and parsed.path in ("", "/")
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment,
        "endpoint must be an uncredentialed loopback HTTP origin",
    )
    return endpoint.rstrip("/")


def load_key(path: Path) -> str:
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "API key file must be an absolute regular non-symlink file",
    )
    raw = path.read_bytes()
    require(b"\r" not in raw and b"\0" not in raw, "API key has forbidden bytes")
    lines = raw.splitlines()
    require(
        len(lines) == 1 and bool(lines[0]),
        "API key file must contain exactly one non-empty line",
    )
    return lines[0].decode()


def post(
    endpoint: str,
    path: str,
    key: str,
    payload: dict[str, Any],
    timeout: float = 180,
) -> tuple[int, Any]:
    request = urllib.request.Request(
        f"{endpoint}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            detail: Any = json.loads(body)
        except json.JSONDecodeError:
            detail = body[:1000]
        return exc.code, detail


def token_count(endpoint: str, key: str, model: str, messages: list[dict]) -> int:
    status, payload = post(
        endpoint,
        "/tokenize",
        key,
        {
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
        },
    )
    require(status == 200, f"/tokenize returned HTTP {status}")
    require(isinstance(payload, dict), "/tokenize did not return an object")
    return int(payload["count"])


def parse_sse(
    lines: Iterable[bytes],
    minimum_content_chunks: int = 1,
    expected_completion_tokens: int | None = None,
    expected_finish_reason: str | None = None,
    on_first_completion_token: Callable[[], None] | None = None,
    on_completion_progress: Callable[[int], None] | None = None,
) -> dict[str, int | bool | str]:
    require(minimum_content_chunks >= 1, "minimum content chunks must be positive")
    require(
        (expected_completion_tokens is None) == (expected_finish_reason is None),
        "completion-token and finish-reason expectations must be provided together",
    )
    if expected_completion_tokens is not None:
        require(
            expected_completion_tokens > 0,
            "expected completion tokens must be positive",
        )
    chunks = 0
    content_chunks = 0
    content_characters = 0
    completion_logprob_tokens = 0
    usage_completion_tokens: int | None = None
    finish_reasons: list[str] = []
    notified_first_token = False
    saw_done = False
    for raw in lines:
        line = raw.decode(errors="strict").strip()
        if not line:
            continue
        require(line.startswith("data: "), "stream contained a non-SSE data line")
        data = line[6:].strip()
        if data == "[DONE]":
            saw_done = True
            continue
        require(not saw_done, "stream emitted data after [DONE]")
        payload = json.loads(data)
        require(isinstance(payload, dict), "stream chunk is not an object")
        require("error" not in payload, "stream emitted an error object")
        usage = payload.get("usage")
        if usage is not None:
            require(isinstance(usage, dict), "stream usage is not an object")
            require(
                usage_completion_tokens is None,
                "stream emitted usage more than once",
            )
            value = usage.get("completion_tokens")
            require(
                isinstance(value, int) and not isinstance(value, bool),
                "stream usage lacks an integer completion token count",
            )
            usage_completion_tokens = value
        choices = payload.get("choices")
        require(isinstance(choices, list), "stream chunk lacks choices")
        if not choices:
            require(usage is not None, "stream chunk has neither choices nor usage")
            chunks += 1
            continue
        require(len(choices) == 1, "stream chunk has the wrong choice count")
        choice = choices[0]
        require(isinstance(choice, dict), "stream choice is not an object")
        delta = choice.get("delta")
        require(isinstance(delta, dict), "stream chunk lacks a delta")
        content = delta.get("content")
        if content is not None:
            require(isinstance(content, str), "stream content is not text")
            content_characters += len(content)
            if content:
                content_chunks += 1
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            require(isinstance(finish_reason, str), "stream finish reason is not text")
            finish_reasons.append(finish_reason)
        logprobs = choice.get("logprobs")
        if logprobs is not None:
            require(isinstance(logprobs, dict), "stream logprobs is not an object")
            token_items = logprobs.get("content")
            require(isinstance(token_items, list), "stream logprobs lack content")
            for item in token_items:
                require(isinstance(item, dict), "stream token logprob is not an object")
                require(isinstance(item.get("token"), str), "stream token is not text")
                value = item.get("logprob")
                require(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value)),
                    "stream contains a non-finite token logprob",
                )
            completion_logprob_tokens += len(token_items)
            if token_items and on_completion_progress is not None:
                on_completion_progress(completion_logprob_tokens)
            if token_items and not notified_first_token:
                notified_first_token = True
                if on_first_completion_token is not None:
                    on_first_completion_token()
        chunks += 1
    require(saw_done, "stream did not terminate with [DONE]")
    require(chunks > 0, "stream emitted no completion chunks")
    require(
        content_chunks >= minimum_content_chunks,
        f"stream emitted fewer than {minimum_content_chunks} content-bearing chunks",
    )
    result: dict[str, int | bool | str] = {
        "chunks": chunks,
        "content_chunks": content_chunks,
        "content_characters": content_characters,
        "saw_done": saw_done,
    }
    if expected_completion_tokens is not None:
        require(
            usage_completion_tokens == expected_completion_tokens,
            "stream usage has the wrong completion token count",
        )
        require(
            completion_logprob_tokens == expected_completion_tokens,
            "stream logprobs have the wrong completion token count",
        )
        require(
            finish_reasons == [expected_finish_reason],
            f"stream finish reasons were {finish_reasons!r}, expected "
            f"[{expected_finish_reason!r}]",
        )
        result.update(
            {
                "completion_tokens": completion_logprob_tokens,
                "usage_completion_tokens": usage_completion_tokens,
                "finish_reason": finish_reasons[0],
            }
        )
    return result


def stream_chat(
    endpoint: str,
    key: str,
    payload: dict[str, Any],
    timeout: float = 180,
    minimum_content_chunks: int = 1,
    expected_completion_tokens: int | None = None,
    expected_finish_reason: str | None = None,
    on_first_completion_token: Callable[[], None] | None = None,
    on_completion_progress: Callable[[int], None] | None = None,
) -> dict[str, int | bool | str]:
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            require(response.status == 200, "chat stream did not return HTTP 200")
            content_type = response.headers.get_content_type()
            require(content_type == "text/event-stream", "wrong stream content type")
            return parse_sse(
                response,
                minimum_content_chunks=minimum_content_chunks,
                expected_completion_tokens=expected_completion_tokens,
                expected_finish_reason=expected_finish_reason,
                on_first_completion_token=on_first_completion_token,
                on_completion_progress=on_completion_progress,
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise ValueError(f"chat stream returned HTTP {exc.code}: {detail}") from exc


def completion_fingerprint(payload: Any, expected_tokens: int) -> dict[str, Any]:
    require(isinstance(payload, dict), "completion response is not an object")
    choices = payload.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "wrong choice count")
    choice = choices[0]
    message = choice.get("message")
    require(isinstance(message, dict), "completion choice has no message")
    content = message.get("content")
    require(isinstance(content, str) and content, "completion has no text")
    logprobs = choice.get("logprobs")
    require(isinstance(logprobs, dict), "completion has no logprobs")
    token_items = logprobs.get("content")
    require(isinstance(token_items, list), "completion has no token logprobs")
    require(len(token_items) == expected_tokens, "wrong number of token logprobs")
    tokens: list[str] = []
    values: list[float] = []
    for item in token_items:
        require(isinstance(item, dict), "token logprob is not an object")
        token = item.get("token")
        value = item.get("logprob")
        require(isinstance(token, str), "completion token is not text")
        require(
            isinstance(value, int | float) and math.isfinite(float(value)),
            "completion contains a non-finite logprob",
        )
        tokens.append(token)
        values.append(float(value))
    usage = payload.get("usage")
    require(isinstance(usage, dict), "completion has no usage object")
    require(
        int(usage.get("completion_tokens", -1)) == expected_tokens,
        "completion usage has the wrong token count",
    )
    return {
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "tokens_sha256": hashlib.sha256("\0".join(tokens).encode()).hexdigest(),
        "completion_tokens": expected_tokens,
        "content_characters": len(content),
        "minimum_logprob": min(values),
        "maximum_logprob": max(values),
    }


def greedy_decode_evidence(
    endpoint: str,
    key: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
        "seed": 123456,
    }
    first_token_runs: list[dict[str, Any]] = []
    for _ in range(2):
        status, payload = post(
            endpoint,
            "/v1/chat/completions",
            key,
            {**request_payload, "max_tokens": 1},
        )
        require(status == 200, f"first-token canary returned HTTP {status}")
        first_token_runs.append(completion_fingerprint(payload, 1))
    decode_runs: list[dict[str, Any]] = []
    for _ in range(2):
        status, payload = post(
            endpoint,
            "/v1/chat/completions",
            key,
            {**request_payload, "max_tokens": 16},
        )
        require(status == 200, f"greedy decode canary returned HTTP {status}")
        decode_runs.append(completion_fingerprint(payload, 16))
    return {
        "scope": "finite_liveness_only",
        "deterministic_equality_required": False,
        "finite_first_token_runs": first_token_runs,
        "finite_decode_runs": decode_runs,
    }


def completion_text(payload: Any) -> str:
    require(isinstance(payload, dict), "semantic canary response is not an object")
    choices = payload.get("choices")
    require(
        isinstance(choices, list) and len(choices) == 1,
        "semantic canary response has the wrong choice count",
    )
    choice = choices[0]
    require(isinstance(choice, dict), "semantic canary choice is not an object")
    message = choice.get("message")
    require(isinstance(message, dict), "semantic canary choice has no message")
    content = message.get("content")
    require(
        isinstance(content, str) and content,
        "semantic canary completion has no text",
    )
    return content


def semantic_text_evidence(content: str, seed: int) -> dict[str, Any]:
    character_count = len(content)
    require(character_count >= 100, "semantic canary completion is implausibly short")
    printable_count = sum(
        character.isprintable() or character in "\n\r\t" for character in content
    )
    ascii_count = sum(ord(character) < 128 for character in content)
    alpha_count = sum(character.isalpha() for character in content)
    replacement_count = content.count("\ufffd")
    non_latin_letter_count = sum(
        character.isalpha()
        and ord(character) >= 128
        and "LATIN" not in unicodedata.name(character, "")
        for character in content
    )
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", content.lower())
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    common_word_count = sum(word in SEMANTIC_COMMON_WORDS for word in words)
    printable_fraction = printable_count / character_count
    ascii_fraction = ascii_count / character_count
    alphabetic_fraction = alpha_count / character_count
    common_word_fraction = common_word_count / max(1, len(words))
    unique_bigram_fraction = len(set(bigrams)) / max(1, len(bigrams))
    unique_trigram_fraction = len(set(trigrams)) / max(1, len(trigrams))
    sentence_mark_count = sum(content.count(mark) for mark in ".?!")

    require(
        printable_fraction >= 0.995,
        "semantic canary contains non-printable corruption",
    )
    require(replacement_count == 0, "semantic canary contains replacement characters")
    require(
        ascii_fraction >= 0.97,
        "semantic canary is not predominantly ASCII English",
    )
    require(
        non_latin_letter_count == 0,
        "semantic canary contains non-Latin script corruption",
    )
    require(len(words) >= 24, "semantic canary has too few English words")
    require(
        len(set(words)) >= 14,
        "semantic canary has implausibly low lexical diversity",
    )
    require(
        alphabetic_fraction >= 0.45,
        "semantic canary has too little alphabetic text",
    )
    require(
        common_word_fraction >= 0.15,
        "semantic canary lacks ordinary English structure",
    )
    require(
        unique_bigram_fraction >= 0.55,
        "semantic canary contains repetitive English word salad",
    )
    require(
        unique_trigram_fraction >= 0.65,
        "semantic canary contains repetitive phrase corruption",
    )
    require(
        sentence_mark_count >= 2,
        "semantic canary lacks sentence structure",
    )
    return {
        "seed": seed,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "character_count": character_count,
        "ascii_word_count": len(words),
        "unique_ascii_word_count": len(set(words)),
        "printable_fraction": printable_fraction,
        "ascii_fraction": ascii_fraction,
        "alphabetic_fraction": alphabetic_fraction,
        "common_word_fraction": common_word_fraction,
        "unique_bigram_fraction": unique_bigram_fraction,
        "unique_trigram_fraction": unique_trigram_fraction,
        "sentence_mark_count": sentence_mark_count,
        "replacement_character_count": replacement_count,
        "non_latin_letter_count": non_latin_letter_count,
    }


def semantic_rp_evidence(endpoint: str, key: str, model: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Kaelen, a guarded but kind fantasy ranger. Stay in "
                "character and reply only in natural English. Continue the "
                "roleplay with actions and dialogue, and address the user as you."
            ),
        },
        {
            "role": "user",
            "content": (
                "*I lower my lantern as rain drums against the ruined "
                "watchtower.* You returned. I thought the pass had taken you."
            ),
        },
    ]
    runs: list[dict[str, Any]] = []
    for seed in SEMANTIC_CANARY_SEEDS:
        status, payload = post(
            endpoint,
            "/v1/chat/completions",
            key,
            {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 64,
                "ignore_eos": True,
                "stream": False,
                "seed": seed,
            },
        )
        require(status == 200, f"semantic RP canary returned HTTP {status}")
        runs.append(semantic_text_evidence(completion_text(payload), seed))
    return {
        "scope": "safe_rp_semantic_integrity",
        "raw_output_retained": False,
        "runs": runs,
    }


def parse_metrics(text: str, metric_names: Mapping[str, str]) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    expected_to_name = {metric: name for name, metric in metric_names.items()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        name = expected_to_name.get(fields[0].split("{", 1)[0])
        if name is None:
            continue
        value = float(fields[1])
        require(math.isfinite(value), f"{name} metric is not finite")
        values[name].append(value)
    parsed: dict[str, float] = {}
    for name, found in values.items():
        require(
            len(found) == 1,
            f"expected exactly one {name} metric series, found {len(found)}",
        )
        parsed[name] = found[0]
    return parsed


def metrics_snapshot(
    endpoint: str, metric_names: Mapping[str, str] = CAPACITY_METRICS
) -> dict[str, float]:
    request = urllib.request.Request(f"{endpoint}/metrics", method="GET")
    with OPENER.open(request, timeout=30) as response:
        text = response.read().decode()
    return parse_metrics(text, metric_names)


def single_metric(endpoint: str, metric_name: str) -> float:
    return metrics_snapshot(endpoint, {"value": metric_name})["value"]


def build_messages(
    repetitions: int, marker: str = "Synthetic continuity detail"
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Continue a synthetic roleplay scene. Write only the character "
                "and world, never the user. Keep continuity exact. " + marker
            ),
        },
        {
            "role": "user",
            "content": f"{marker}. " + ("x " * repetitions),
        },
    ]


def exact_context_messages(
    endpoint: str,
    key: str,
    model: str,
    target: int,
    marker: str = "Synthetic continuity detail",
) -> list[dict[str, str]]:
    low = 0
    high = target * 3
    best: tuple[int, list[dict[str, str]]] | None = None
    while low <= high:
        middle = (low + high) // 2
        messages = build_messages(middle, marker)
        count = token_count(endpoint, key, model, messages)
        if count == target:
            return messages
        if count < target:
            best = (middle, messages)
            low = middle + 1
        else:
            high = middle - 1
    if best is not None:
        start = max(0, best[0] - 8)
        for repetitions in range(start, best[0] + 64):
            messages = build_messages(repetitions, marker)
            if token_count(endpoint, key, model, messages) == target:
                return messages
    raise ValueError(f"could not construct an exact {target}-token chat prompt")


def concurrent_boundary_acceptance(
    endpoint: str,
    key: str,
    model: str,
    concurrency: int = 38,
    metrics_poll_interval: float = 0.02,
) -> dict[str, Any]:
    prompt_tokens = CAPACITY_PROMPT_TOKENS
    completion_tokens = CAPACITY_COMPLETION_TOKENS
    configured_blocks = CAPACITY_TOTAL_KV_BYTES // CAPACITY_BLOCK_BYTES
    usable_blocks = configured_blocks - CAPACITY_RESERVED_BLOCKS
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        messages = list(
            pool.map(
                lambda index: exact_context_messages(
                    endpoint,
                    key,
                    model,
                    target=prompt_tokens,
                    marker=f"Synthetic distinct capacity detail {index:03d}",
                ),
                range(concurrency),
            )
        )
    require(
        len({json.dumps(item, sort_keys=True) for item in messages}) == concurrency,
        "capacity prompts are not distinct",
    )
    before = metrics_snapshot(endpoint)
    require(
        before["running"] == 0 and before["waiting"] == 0,
        "capacity check requires an idle scheduler",
    )
    require(
        before["preemptions"] >= 0 and before["kv_usage"] == 0,
        "capacity check requires zero allocated KV blocks before the cohort",
    )
    start_barrier = threading.Barrier(concurrency + 1)
    first_token_events = [threading.Event() for _ in range(concurrency)]
    completion_progress = [0 for _ in range(concurrency)]
    completion_progress_lock = threading.Lock()
    cache_salts = [secrets.token_hex(32) for _ in range(concurrency)]

    def _record_progress(index: int, tokens: int) -> None:
        with completion_progress_lock:
            require(
                completion_progress[index] <= tokens <= completion_tokens,
                "capacity completion progress moved backward or out of range",
            )
            completion_progress[index] = tokens

    def _run(index: int, item: list[dict[str, str]]) -> dict[str, int | bool | str]:
        try:
            start_barrier.wait(timeout=60)
        except threading.BrokenBarrierError as exc:
            raise ValueError("capacity request barrier broke") from exc
        return stream_chat(
            endpoint,
            key,
            {
                "model": model,
                "messages": item,
                "cache_salt": cache_salts[index],
                "temperature": 0,
                "max_tokens": completion_tokens,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "logprobs": True,
            },
            timeout=600,
            expected_completion_tokens=completion_tokens,
            expected_finish_reason="length",
            on_first_completion_token=first_token_events[index].set,
            on_completion_progress=lambda tokens: _record_progress(index, tokens),
        )

    observed_max_running = 0.0
    observed_max_kv_usage = 0.0
    running_metric_samples = 0
    simultaneous_decode_sample: dict[str, float] | None = None
    simultaneous_6143_sample: dict[str, float] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run, index, item) for index, item in enumerate(messages)
        ]
        try:
            start_barrier.wait(timeout=60)
        except threading.BrokenBarrierError as exc:
            raise ValueError("capacity request barrier broke") from exc
        while True:
            sample = metrics_snapshot(endpoint)
            running = sample["running"]
            require(
                0 <= running <= concurrency and sample["waiting"] >= 0,
                "capacity scheduler metrics are outside the request range",
            )
            require(
                0 <= sample["kv_usage"] <= 1,
                "capacity KV occupancy metric is outside [0, 1]",
            )
            require(
                sample["preemptions"] == before["preemptions"],
                "capacity preemption metric changed during the batch",
            )
            observed_max_running = max(observed_max_running, running)
            observed_max_kv_usage = max(observed_max_kv_usage, sample["kv_usage"])
            running_metric_samples += 1
            decoded_streams = sum(event.is_set() for event in first_token_events)
            unfinished_streams = sum(not future.done() for future in futures)
            with completion_progress_lock:
                minimum_completion_progress = min(completion_progress)
            if (
                running == concurrency
                and decoded_streams == concurrency
                and unfinished_streams == concurrency
            ):
                require(
                    sample["waiting"] == 0,
                    "capacity batch queued requests during simultaneous decode",
                )
                require(
                    sample["kv_usage"] > 0,
                    "KV occupancy was not positive during simultaneous decode",
                )
                simultaneous_decode_sample = sample
            if (
                running == concurrency
                and unfinished_streams == concurrency
                and minimum_completion_progress >= completion_tokens - 1
            ):
                require(
                    sample["waiting"] == 0,
                    "capacity batch queued requests near the 6144-token boundary",
                )
                require(
                    sample["kv_usage"] > 0,
                    "KV occupancy was not positive near the 6144-token boundary",
                )
                simultaneous_6143_sample = sample
            if all(future.done() for future in futures):
                break
            time.sleep(metrics_poll_interval)
        streams = [future.result() for future in futures]
    after = metrics_snapshot(endpoint)
    require(
        observed_max_running == concurrency,
        "capacity batch did not prove exactly "
        f"{concurrency} running requests; observed {observed_max_running:g}",
    )
    require(
        simultaneous_decode_sample is not None,
        "capacity batch did not prove all streams decoding while 38 were running",
    )
    require(
        simultaneous_6143_sample is not None,
        "capacity batch did not prove all 38 streams resident at 6143 context tokens",
    )
    require(
        after["preemptions"] == before["preemptions"],
        f"the exact {concurrency}-slot capacity batch triggered KV preemption",
    )
    require(
        after["running"] == 0 and after["waiting"] == 0,
        "capacity batch did not drain the scheduler",
    )
    require(
        all(
            int(stream["completion_tokens"]) == completion_tokens for stream in streams
        ),
        "capacity batch contains a stream with the wrong completion length",
    )
    require(
        all(
            int(stream["usage_completion_tokens"]) == completion_tokens
            for stream in streams
        ),
        "capacity batch contains a stream with the wrong usage completion length",
    )
    require(
        all(stream["finish_reason"] == "length" for stream in streams),
        "capacity batch contains an unexpected finish reason",
    )
    completion_evidence = [
        {
            "stream_index": index,
            "completion_tokens": int(stream["completion_tokens"]),
            "usage_completion_tokens": int(stream["usage_completion_tokens"]),
            "finish_reason": str(stream["finish_reason"]),
        }
        for index, stream in enumerate(streams)
    ]
    return {
        "concurrency": concurrency,
        "prompt_tokens_per_request": prompt_tokens,
        "max_tokens_per_request": completion_tokens,
        "context_tokens_per_request": prompt_tokens + completion_tokens,
        "total_stream_chunks": sum(int(stream["chunks"]) for stream in streams),
        "verified_exact_completion_streams": len(completion_evidence),
        "observed_completion_tokens_total": sum(
            item["completion_tokens"] for item in completion_evidence
        ),
        "observed_length_finish_streams": sum(
            item["finish_reason"] == "length" for item in completion_evidence
        ),
        "stream_completion_evidence": completion_evidence,
        "observed_max_running": observed_max_running,
        "simultaneous_decoding_streams": concurrency,
        "running_metric_samples": running_metric_samples,
        "kv_cache_usage_at_simultaneous_decode": simultaneous_decode_sample["kv_usage"],
        "simultaneous_resident_context_tokens_per_request": prompt_tokens
        + completion_tokens
        - 1,
        "kv_cache_usage_at_simultaneous_6143": simultaneous_6143_sample["kv_usage"],
        "observed_max_kv_cache_usage": observed_max_kv_usage,
        "kv_cache_usage_after_drain": after["kv_usage"],
        "kv_cache_block_bytes": CAPACITY_BLOCK_BYTES,
        "configured_kv_cache_blocks": configured_blocks,
        "reserved_kv_cache_blocks": CAPACITY_RESERVED_BLOCKS,
        "usable_kv_cache_blocks": usable_blocks,
        "configured_kv_cache_bytes": CAPACITY_TOTAL_KV_BYTES,
        "concurrent_6144_completion_proven": True,
        "concurrent_6143_residency_proven": True,
        "preemptions_before": before["preemptions"],
        "preemptions_after": after["preemptions"],
        "scheduler_running_before": before["running"],
        "scheduler_waiting_before": before["waiting"],
        "scheduler_running_after": after["running"],
        "scheduler_waiting_after": after["waiting"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the exact Verse streaming chat and 6144-token contract."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="verse-free")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help="Run only the ordinary streaming and deterministic startup canaries.",
    )
    add_identity_arguments(parser, required=False)
    args = parser.parse_args()

    try:
        endpoint = validate_endpoint(args.endpoint)
        key = load_key(args.api_key_file)
        ordinary_messages = build_messages(12)
        ordinary_count = token_count(endpoint, key, args.model, ordinary_messages)
        stream = stream_chat(
            endpoint,
            key,
            {
                "model": args.model,
                "messages": ordinary_messages,
                "temperature": 0.85,
                "top_p": 0.95,
                "min_p": 0.0,
                "repeat_penalty": 1.08,
                "repetition_penalty": 1.08,
                "frequency_penalty": 0.5,
                "top_k": 0,
                "max_tokens": 16,
                "stream": True,
                "stop": ["\nRules:", "\nCharacter:"],
                "seed": 123456,
            },
            minimum_content_chunks=2,
        )
        greedy_decode = greedy_decode_evidence(
            endpoint, key, args.model, ordinary_messages
        )
        semantic_rp = semantic_rp_evidence(endpoint, key, args.model)
        identity = validated_optional_identity(args)
        result: dict[str, Any] = {
            "status": "pass",
            "scope": "startup" if args.startup_only else "complete",
            "model": args.model,
            "ordinary_prompt_tokens": ordinary_count,
            "ordinary_stream": stream,
            "greedy_decode_evidence": greedy_decode,
            "semantic_rp_evidence": semantic_rp,
        }
        if identity is not None:
            result.update(identity)
        if not args.startup_only:
            accepted_messages = exact_context_messages(
                endpoint, key, args.model, target=6143
            )
            accepted_count = token_count(endpoint, key, args.model, accepted_messages)
            accepted = stream_chat(
                endpoint,
                key,
                {
                    "model": args.model,
                    "messages": accepted_messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "ignore_eos": True,
                    "stream": True,
                },
            )

            rejected_messages = exact_context_messages(
                endpoint, key, args.model, target=6144
            )
            rejected_count = token_count(endpoint, key, args.model, rejected_messages)
            rejected_status, _ = post(
                endpoint,
                "/v1/chat/completions",
                key,
                {
                    "model": args.model,
                    "messages": rejected_messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "ignore_eos": True,
                    "stream": False,
                },
            )
            require(
                rejected_status in {400, 413, 422},
                f"6145-token request returned HTTP {rejected_status}, "
                "expected rejection",
            )
            capacity = concurrent_boundary_acceptance(endpoint, key, args.model)
            result.update(
                {
                    "boundary_accepted_prompt_tokens": accepted_count,
                    "boundary_accepted_stream": accepted,
                    "boundary_rejected_prompt_tokens": rejected_count,
                    "boundary_rejected_http_status": rejected_status,
                    "exact_boundary_capacity": capacity,
                }
            )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
