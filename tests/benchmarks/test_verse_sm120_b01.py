# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "verse" / "sm120_b01.py"
SPEC = importlib.util.spec_from_file_location("verse_sm120_b01", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_metrics_accepts_vllm_colon_names():
    metrics = """
vllm:generation_tokens_total{engine="0"} 125
vllm:prompt_tokens_total{engine="0"} 250
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 0
vllm:num_preemptions_total{engine="0"} 4
"""

    assert MODULE.parse_metrics(metrics) == MODULE.MetricSnapshot(
        generated=125.0,
        prompted=250.0,
        running=38.0,
        waiting=0.0,
        preemptions=4.0,
    )


def test_http_opener_disables_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")

    assert MODULE.ENVIRONMENT_PROXY_BLOCKER.proxies == {}
    assert not any(
        isinstance(handler, MODULE.urllib.request.ProxyHandler)
        for handler in MODULE.NO_REDIRECT_OPENER.handlers
    )


def test_require_server_idle_rejects_dirty_server(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "fetch_metrics",
        lambda _endpoint: MODULE.MetricSnapshot(100.0, 50.0, 1.0, 2.0, 3.0),
    )

    with pytest.raises(RuntimeError, match=r"running=1, waiting=2"):
        MODULE.require_server_idle("http://127.0.0.1:8000")


def test_preemption_increase_is_rejected():
    with pytest.raises(RuntimeError, match="num_preemptions increased"):
        MODULE.require_no_preemption_increase(7.0, 8.0)


def test_wait_for_idle_drains_busy_server(monkeypatch):
    snapshots = iter(
        [
            MODULE.MetricSnapshot(100.0, 50.0, 2.0, 1.0, 3.0),
            MODULE.MetricSnapshot(110.0, 60.0, 1.0, 0.0, 3.0),
            MODULE.MetricSnapshot(120.0, 60.0, 0.0, 0.0, 3.0),
        ]
    )
    monkeypatch.setattr(MODULE, "fetch_metrics", lambda _endpoint: next(snapshots))

    assert MODULE.wait_for_idle("http://127.0.0.1:8000", 1.0, 0.0) == (
        MODULE.MetricSnapshot(120.0, 60.0, 0.0, 0.0, 3.0)
    )


def test_longest_full_decode_window_ignores_partial_batch():
    sample = MODULE.MetricSample
    samples = [
        sample(0.0, 100.0, 1000.0, 20.0, 0.0),
        sample(1.0, 200.0, 2000.0, 38.0, 0.0),
        sample(2.0, 1200.0, 2000.0, 38.0, 0.0),
        sample(3.0, 1300.0, 2000.0, 37.0, 1.0),
    ]

    assert MODULE.longest_full_decode_window(samples, 38, 0.5, 1.0, 2) == (
        1.0,
        1000.0,
        2,
        0.0,
    )


def test_short_transient_is_not_a_valid_steady_window():
    sample = MODULE.MetricSample
    samples = [
        sample(0.0, 100.0, 1000.0, 38.0, 0.0),
        sample(0.05, 150.0, 1000.0, 38.0, 0.0),
    ]

    assert MODULE.longest_full_decode_window(samples, 38, 0.05, 10.0, 50) is None


def test_decode_window_excludes_samples_while_prompt_counter_changes():
    sample = MODULE.MetricSample
    samples = [
        sample(0.0, 100.0, 1000.0, 38.0, 0.0),
        sample(1.0, 300.0, 2000.0, 38.0, 0.0),
        sample(2.0, 700.0, 2000.0, 38.0, 0.0),
        sample(3.0, 1100.0, 2000.0, 38.0, 0.0),
    ]

    assert MODULE.longest_full_decode_window(samples, 38, 0.5, 1.0, 2) == (
        2.0,
        400.0,
        3,
        0.0,
    )


def test_endpoint_must_be_loopback():
    assert MODULE.validate_loopback_endpoint("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000"
    )

    try:
        MODULE.validate_loopback_endpoint("https://example.com:443")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback endpoint was accepted")


def test_api_key_file_is_loaded_without_environment(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("opaque-secret\n")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    assert MODULE.load_api_key(key_file.resolve(), "VLLM_API_KEY") == "opaque-secret"


def test_api_key_file_rejects_multiple_lines(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("one\ntwo\n")

    try:
        MODULE.load_api_key(key_file.resolve(), "VLLM_API_KEY")
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("multi-line API key was accepted")


def test_release_nonce_separates_otherwise_identical_prompts(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "tokenize",
        lambda _endpoint, _api_key, _model, prompt: len(prompt),
    )

    first, first_count = MODULE.build_prompt(
        "http://127.0.0.1:8000", "key", "model", 0, 512, "a" * 64
    )
    repeated, repeated_count = MODULE.build_prompt(
        "http://127.0.0.1:8000", "key", "model", 0, 512, "a" * 64
    )
    next_release, next_count = MODULE.build_prompt(
        "http://127.0.0.1:8000", "key", "model", 0, 512, "b" * 64
    )

    assert first == repeated
    assert first != next_release
    assert first_count == repeated_count == next_count == 512


def test_release_nonce_must_be_64_lowercase_hex_characters():
    with pytest.raises(ValueError, match="64 lowercase hex"):
        MODULE.validate_release_nonce("A" * 64)


def test_parse_metrics_rejects_multiple_engine_series():
    metrics = """
vllm:generation_tokens_total{engine="0"} 125
vllm:generation_tokens_total{engine="1"} 126
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 0
vllm:num_preemptions_total{engine="0"} 0
"""

    try:
        MODULE.parse_metrics(metrics)
    except ValueError as exc:
        assert "exactly one generated" in str(exc)
    else:
        raise AssertionError("multiple generation series were accepted")
