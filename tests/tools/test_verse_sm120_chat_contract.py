# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2] / "tools" / "verse" / ("check_sm120_chat_contract.py")
)
SPEC = importlib.util.spec_from_file_location("verse_sm120_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_sse_accepts_text_and_done():
    result = MODULE.parse_sse(
        [
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b"data: [DONE]\n",
        ]
    )

    assert result == {
        "chunks": 2,
        "content_chunks": 2,
        "content_characters": 11,
        "saw_done": True,
    }


def test_parse_sse_rejects_missing_done():
    try:
        MODULE.parse_sse([b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'])
    except ValueError as exc:
        assert "[DONE]" in str(exc)
    else:
        raise AssertionError("unterminated SSE was accepted")


def test_parse_sse_rejects_buffered_content_when_two_chunks_required():
    lines = [
        b'data: {"choices":[{"delta":{"content":"hello world"}}]}\n',
        b"data: [DONE]\n",
    ]

    assert MODULE.parse_sse(lines)["content_chunks"] == 1
    with pytest.raises(ValueError, match="fewer than 2 content-bearing chunks"):
        MODULE.parse_sse(lines, minimum_content_chunks=2)


def _capacity_proof_lines(
    *,
    usage_tokens: int = 2,
    finish_reason: str = "length",
    logprob_tokens: int = 2,
):
    lines = []
    for index in range(logprob_tokens):
        payload = {
            "choices": [
                {
                    "delta": {"content": chr(ord("a") + index)},
                    "finish_reason": None,
                    "logprobs": {
                        "content": [{"token": str(index), "logprob": -0.25 - index}]
                    },
                }
            ]
        }
        lines.append(f"data: {json.dumps(payload)}\n".encode())
    finish = {
        "choices": [
            {
                "delta": {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ]
    }
    usage = {"choices": [], "usage": {"completion_tokens": usage_tokens}}
    lines.extend(
        [
            f"data: {json.dumps(finish)}\n".encode(),
            f"data: {json.dumps(usage)}\n".encode(),
            b"data: [DONE]\n",
        ]
    )
    return lines


def test_parse_sse_proves_exact_completion_length_and_finish_reason():
    first_token_notifications = 0

    def notify_first_token():
        nonlocal first_token_notifications
        first_token_notifications += 1

    result = MODULE.parse_sse(
        _capacity_proof_lines(),
        expected_completion_tokens=2,
        expected_finish_reason="length",
        on_first_completion_token=notify_first_token,
    )

    assert result["completion_tokens"] == 2
    assert result["usage_completion_tokens"] == 2
    assert result["finish_reason"] == "length"
    assert first_token_notifications == 1


@pytest.mark.parametrize(
    ("usage_tokens", "finish_reason", "logprob_tokens", "message"),
    [
        (1, "length", 2, "usage has the wrong completion token count"),
        (2, "stop", 2, "stream finish reasons"),
        (2, "length", 1, "logprobs have the wrong completion token count"),
    ],
)
def test_parse_sse_rejects_incomplete_capacity_proof(
    usage_tokens, finish_reason, logprob_tokens, message
):
    with pytest.raises(ValueError, match=message):
        MODULE.parse_sse(
            _capacity_proof_lines(
                usage_tokens=usage_tokens,
                finish_reason=finish_reason,
                logprob_tokens=logprob_tokens,
            ),
            expected_completion_tokens=2,
            expected_finish_reason="length",
        )


def test_opener_ignores_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: (_ for _ in ()).throw(
            AssertionError("environment proxy discovery was used")
        ),
    )

    opener = MODULE.build_isolated_opener()

    assert isinstance(opener, urllib.request.OpenerDirector)


def test_endpoint_must_be_loopback():
    assert MODULE.validate_endpoint("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000"
    )

    try:
        MODULE.validate_endpoint("https://example.com")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("remote endpoint was accepted")


def test_single_metric_requires_one_series(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return io.BytesIO(b"vllm:num_preemptions_total 7\n").read()

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(MODULE, "OPENER", Opener())
    assert (
        MODULE.single_metric("http://127.0.0.1:8000", "vllm:num_preemptions_total") == 7
    )


def test_capacity_metrics_are_parsed_as_one_atomic_snapshot():
    metrics = """
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 0
vllm:num_preemptions_total{engine="0"} 4
vllm:kv_cache_usage_perc{engine="0"} 0.8125
"""

    assert MODULE.parse_metrics(metrics, MODULE.CAPACITY_METRICS) == {
        "running": 38,
        "waiting": 0,
        "preemptions": 4,
        "kv_usage": 0.8125,
    }


def test_capacity_metrics_fail_closed_when_kv_occupancy_is_missing():
    metrics = """
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 0
vllm:num_preemptions_total{engine="0"} 0
"""

    with pytest.raises(ValueError, match="exactly one kv_usage metric series"):
        MODULE.parse_metrics(metrics, MODULE.CAPACITY_METRICS)


def test_completion_fingerprint_rejects_non_finite_logprob():
    payload = {
        "choices": [
            {
                "message": {"content": "x"},
                "logprobs": {"content": [{"token": "x", "logprob": float("nan")}]},
            }
        ],
        "usage": {"completion_tokens": 1},
    }
    try:
        MODULE.completion_fingerprint(payload, 1)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("a NaN model logprob was accepted")


def test_capacity_probe_overlaps_all_requests_and_observes_running(monkeypatch):
    concurrency = 4
    active = 0
    maximum_active = 0
    full_metric_samples = 0
    lock = threading.Lock()
    release = threading.Event()
    calls = []

    def exact_messages(*args, marker, **kwargs):
        return [{"role": "user", "content": marker}]

    def stream_chat(*args, **kwargs):
        nonlocal active, maximum_active
        kwargs["on_first_completion_token"]()
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            calls.append((args[2], kwargs))
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return {
            "chunks": 64,
            "content_chunks": 64,
            "content_characters": 64,
            "saw_done": True,
            "completion_tokens": 64,
            "usage_completion_tokens": 64,
            "finish_reason": "length",
        }

    def snapshot(_endpoint):
        nonlocal full_metric_samples
        with lock:
            running = active
        if running == concurrency:
            full_metric_samples += 1
            if full_metric_samples >= 2:
                release.set()
        return {
            "running": running,
            "waiting": 0,
            "preemptions": 0,
            "kv_usage": 0.75 if running else 0,
        }

    monkeypatch.setattr(MODULE, "exact_context_messages", exact_messages)
    monkeypatch.setattr(MODULE, "stream_chat", stream_chat)
    monkeypatch.setattr(MODULE, "metrics_snapshot", snapshot)

    result = MODULE.concurrent_boundary_acceptance(
        "http://127.0.0.1:8000",
        "key",
        "model",
        concurrency=concurrency,
        metrics_poll_interval=0,
    )

    assert maximum_active == concurrency
    assert result["observed_max_running"] == concurrency
    assert result["simultaneous_decoding_streams"] == concurrency
    assert result["kv_cache_usage_at_simultaneous_decode"] == 0.75
    assert result["verified_exact_completion_streams"] == concurrency
    assert result["observed_completion_tokens_total"] == concurrency * 64
    assert result["observed_length_finish_streams"] == concurrency
    assert result["stream_completion_evidence"] == [
        {
            "stream_index": index,
            "completion_tokens": 64,
            "usage_completion_tokens": 64,
            "finish_reason": "length",
        }
        for index in range(concurrency)
    ]
    assert result["prompt_tokens_per_request"] == 6080
    assert result["max_tokens_per_request"] == 64
    payloads = [payload for payload, _kwargs in calls]
    assert len({payload["messages"][0]["content"] for payload in payloads}) == 4
    assert all(payload["max_tokens"] == 64 for payload in payloads)
    assert all(payload["ignore_eos"] is True for payload in payloads)
    assert all(payload["logprobs"] is True for payload in payloads)
    assert all(
        payload["stream_options"] == {"include_usage": True} for payload in payloads
    )
    assert all(
        kwargs["expected_completion_tokens"] == 64
        and kwargs["expected_finish_reason"] == "length"
        for _payload, kwargs in calls
    )


def test_capacity_probe_rejects_preemption(monkeypatch):
    concurrency = 2
    active = 0
    saw_full_batch = False
    full_metric_samples = 0
    lock = threading.Lock()
    release = threading.Event()

    monkeypatch.setattr(
        MODULE,
        "exact_context_messages",
        lambda *args, marker, **kwargs: [{"role": "user", "content": marker}],
    )

    def stream_chat(*args, **kwargs):
        nonlocal active
        kwargs["on_first_completion_token"]()
        with lock:
            active += 1
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return {
            "chunks": 64,
            "content_chunks": 64,
            "content_characters": 64,
            "saw_done": True,
            "completion_tokens": 64,
            "usage_completion_tokens": 64,
            "finish_reason": "length",
        }

    def snapshot(_endpoint):
        nonlocal full_metric_samples, saw_full_batch
        with lock:
            running = active
        if running == concurrency:
            full_metric_samples += 1
            if full_metric_samples >= 2:
                saw_full_batch = True
                release.set()
        return {
            "running": running,
            "waiting": 0,
            "preemptions": 4 if saw_full_batch and running == 0 else 3,
            "kv_usage": 0.5 if running else 0,
        }

    monkeypatch.setattr(MODULE, "stream_chat", stream_chat)
    monkeypatch.setattr(MODULE, "metrics_snapshot", snapshot)

    with pytest.raises(ValueError, match="preemption metric changed"):
        MODULE.concurrent_boundary_acceptance(
            "http://127.0.0.1:8000",
            "key",
            "model",
            concurrency=concurrency,
            metrics_poll_interval=0,
        )


def test_capacity_probe_rejects_running_only_without_decode_evidence(monkeypatch):
    concurrency = 2
    active = 0
    lock = threading.Lock()
    release = threading.Event()

    monkeypatch.setattr(
        MODULE,
        "exact_context_messages",
        lambda *args, marker, **kwargs: [{"role": "user", "content": marker}],
    )

    def stream_chat(*args, **kwargs):
        nonlocal active
        with lock:
            active += 1
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return {
            "chunks": 64,
            "content_chunks": 64,
            "content_characters": 64,
            "saw_done": True,
            "completion_tokens": 64,
            "usage_completion_tokens": 64,
            "finish_reason": "length",
        }

    def snapshot(_endpoint):
        with lock:
            running = active
        if running == concurrency:
            release.set()
        return {
            "running": running,
            "waiting": 0,
            "preemptions": 0,
            "kv_usage": 0.5 if running else 0,
        }

    monkeypatch.setattr(MODULE, "stream_chat", stream_chat)
    monkeypatch.setattr(MODULE, "metrics_snapshot", snapshot)

    with pytest.raises(ValueError, match="did not prove all streams decoding"):
        MODULE.concurrent_boundary_acceptance(
            "http://127.0.0.1:8000",
            "key",
            "model",
            concurrency=concurrency,
            metrics_poll_interval=0,
        )


def test_startup_only_avoids_boundary_capacity(monkeypatch, tmp_path, capsys):
    key = tmp_path / "key"
    key.write_text("secret\n")
    monkeypatch.setattr(MODULE, "token_count", lambda *args, **kwargs: 42)
    stream_calls = []

    def stream_chat(*args, **kwargs):
        stream_calls.append(kwargs)
        return {
            "chunks": 1,
            "content_chunks": 2,
            "content_characters": 4,
            "saw_done": True,
        }

    monkeypatch.setattr(MODULE, "stream_chat", stream_chat)
    monkeypatch.setattr(
        MODULE,
        "deterministic_completion_check",
        lambda *args, **kwargs: {"content_sha256": "a", "tokens_sha256": "b"},
    )
    monkeypatch.setattr(
        MODULE,
        "exact_context_messages",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("boundary path ran during startup-only check")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--api-key-file",
            str(key),
            "--startup-only",
        ],
    )

    assert MODULE.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["scope"] == "startup"
    assert "exact_boundary_capacity" not in result
    assert stream_calls == [{"minimum_content_chunks": 2}]
