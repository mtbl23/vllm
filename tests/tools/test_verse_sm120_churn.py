# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

VERSE_TOOLS = Path(__file__).parents[2] / "tools" / "verse"
sys.path.insert(0, str(VERSE_TOOLS))
MODULE_PATH = VERSE_TOOLS / "run_sm120_churn.py"
SPEC = importlib.util.spec_from_file_location("verse_sm120_churn", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_metrics_accepts_exact_single_engine_series():
    text = """
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 2
vllm:num_preemptions_total{engine="0"} 0
vllm:prefix_cache_queries_total{engine="0"} 1000
vllm:prefix_cache_hits_total{engine="0"} 900
vllm:kv_cache_usage_perc{engine="0"} 0.7
vllm:generation_tokens_total{engine="0"} 12345
"""

    assert MODULE.parse_metrics(text) == {
        "running": 38,
        "waiting": 2,
        "preemptions": 0,
        "prefix_queries": 1000,
        "prefix_hits": 900,
        "kv_usage": 0.7,
        "generation_tokens": 12345,
    }


def test_parse_metrics_rejects_multiple_engine_series():
    text = """
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_running{engine="1"} 38
"""

    try:
        MODULE.parse_metrics(text)
    except ValueError as exc:
        assert "exactly one running" in str(exc)
    else:
        raise AssertionError("multiple metric series were accepted")


def test_parse_metrics_fails_closed_when_decode_progress_is_unobservable():
    text = """
vllm:num_requests_running{engine="0"} 38
vllm:num_requests_waiting{engine="0"} 0
vllm:num_preemptions_total{engine="0"} 0
vllm:prefix_cache_queries_total{engine="0"} 1000
vllm:prefix_cache_hits_total{engine="0"} 900
vllm:kv_cache_usage_perc{engine="0"} 0.7
"""

    with pytest.raises(ValueError, match="exactly one generation_tokens"):
        MODULE.parse_metrics(text)


def test_churn_client_disables_environment_proxies():
    assert not any(
        isinstance(handler, MODULE.urllib.request.ProxyHandler) and handler.proxies
        for handler in MODULE.OPENER.handlers
    )


class CoordinatedGate:
    def __init__(self, parties):
        self.parties = parties
        self.arrived = 0
        self.condition = threading.Condition()
        self.release = threading.Event()

    def wait(self, timeout):
        with self.condition:
            self.arrived += 1
            self.condition.notify_all()
        if not self.release.wait(timeout):
            raise threading.BrokenBarrierError

    def wait_for_arrivals(self):
        with self.condition:
            assert self.condition.wait_for(
                lambda: self.arrived == self.parties, timeout=2
            )


def test_workers_make_no_requests_before_coordinated_start(monkeypatch):
    concurrency = 2
    gate = CoordinatedGate(concurrency)
    consume_gate = threading.Barrier(concurrency)
    stop = threading.Event()
    totals = MODULE.Totals()
    progress = [MODULE.WorkerProgress() for _ in range(concurrency)]
    lock = threading.Lock()
    errors = []
    calls = []

    def consume_stream(*args, cancel_after_first_content):
        with lock:
            calls.append(cancel_after_first_content)
        consume_gate.wait(timeout=2)
        stop.set()
        return cancel_after_first_content, 1

    monkeypatch.setattr(MODULE, "consume_stream", consume_stream)
    threads = [
        threading.Thread(
            target=MODULE.worker,
            args=(
                worker_id,
                stop,
                "http://127.0.0.1:8000",
                "key",
                "model",
                [[{"role": "user", "content": "prompt"}]],
                totals,
                progress[worker_id],
                lock,
                errors,
                gate,
            ),
        )
        for worker_id in range(concurrency)
    ]
    for thread in threads:
        thread.start()

    gate.wait_for_arrivals()
    assert calls == []
    gate.release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert errors == []
    assert [item.requests for item in progress] == [1, 1]
    assert [item.chunks for item in progress] == [1, 1]
    assert MODULE.validate_worker_progress(progress, totals) == {
        "workers_with_progress": 2,
        "minimum_requests_per_worker": 1,
        "minimum_chunks_per_worker": 1,
    }


def test_worker_progress_rejects_an_idle_worker():
    progress = [
        MODULE.WorkerProgress(requests=1, completed=1, chunks=2),
        MODULE.WorkerProgress(),
    ]
    totals = MODULE.Totals(completed=1, chunks=2)

    with pytest.raises(ValueError, match="worker 1 made no request progress"):
        MODULE.validate_worker_progress(progress, totals)


def metric_sample(elapsed, running, generation_tokens, kv_usage=0.7):
    return {
        "elapsed_seconds": elapsed,
        "running": running,
        "waiting": 0,
        "preemptions": 0,
        "prefix_queries": 1000,
        "prefix_hits": 900,
        "kv_usage": kv_usage,
        "generation_tokens": generation_tokens,
    }


def test_metrics_are_sampled_continuously_for_the_load_duration(monkeypatch):
    clock = [100.0]
    fetch_calls = 0

    class Stop:
        @staticmethod
        def is_set():
            return False

        @staticmethod
        def wait(seconds):
            clock[0] += seconds

    def fetch_metrics(_endpoint):
        nonlocal fetch_calls
        fetch_calls += 1
        sample = metric_sample(
            0,
            running=38,
            generation_tokens=100 * fetch_calls,
        )
        del sample["elapsed_seconds"]
        return sample

    monkeypatch.setattr(MODULE.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(MODULE, "fetch_metrics", fetch_metrics)

    samples = MODULE.collect_metrics_during_load(
        "http://127.0.0.1:8000",
        Stop(),
        started=clock[0],
        duration_seconds=3,
        interval_seconds=1,
    )

    assert [sample["elapsed_seconds"] for sample in samples] == [0, 1, 2, 3]
    assert [sample["generation_tokens"] for sample in samples] == [100, 200, 300, 400]


def test_metrics_samples_prove_continuous_sustained_full_decode():
    samples = [
        metric_sample(0, 0, 100, kv_usage=0),
        metric_sample(1, 38, 200),
        metric_sample(2, 38, 300),
        metric_sample(3, 38, 400),
        metric_sample(4, 37, 500),
    ]

    evidence = MODULE.validate_metrics_samples(
        samples,
        concurrency=38,
        duration_seconds=4,
        interval_seconds=1,
    )

    assert evidence == {
        "metrics_samples": 5,
        "maximum_metrics_gap_seconds": 1,
        "full_running_streak_samples": 3,
        "full_running_streak_seconds": 2,
        "full_running_generation_tokens_delta": 200,
        "full_running_samples": 3,
        "full_running_sample_fraction": 0.6,
        "observed_max_running": 38,
        "observed_max_kv_cache_usage": 0.7,
        "observed_generation_tokens_delta": 400,
    }


def test_metrics_samples_reject_a_transient_full_running_spike():
    samples = [
        metric_sample(0, 0, 100, kv_usage=0),
        metric_sample(1, 38, 200),
        metric_sample(2, 38, 300),
        metric_sample(3, 37, 400),
        metric_sample(4, 37, 500),
    ]

    with pytest.raises(ValueError, match="did not prove sustained 38 running"):
        MODULE.validate_metrics_samples(
            samples,
            concurrency=38,
            duration_seconds=4,
            interval_seconds=1,
        )


def test_metrics_samples_reject_low_full_running_coverage():
    samples = [metric_sample(0, 0, 100, kv_usage=0)]
    for elapsed in range(1, 21):
        running = 38 if 4 <= elapsed <= 6 else 37
        samples.append(metric_sample(elapsed, running, 100 + 100 * elapsed))

    with pytest.raises(ValueError, match="fewer than 20 percent"):
        MODULE.validate_metrics_samples(
            samples,
            concurrency=38,
            duration_seconds=20,
            interval_seconds=1,
        )


def test_metrics_samples_reject_a_gap_during_load():
    samples = [
        metric_sample(0, 0, 100, kv_usage=0),
        metric_sample(1, 38, 200),
        metric_sample(2, 38, 300),
        metric_sample(6, 38, 600),
    ]

    with pytest.raises(ValueError, match="not continuous during load"):
        MODULE.validate_metrics_samples(
            samples,
            concurrency=38,
            duration_seconds=6,
            interval_seconds=1,
        )


def test_metrics_samples_reject_no_decode_progress():
    samples = [
        metric_sample(0, 0, 100, kv_usage=0),
        metric_sample(1, 38, 100),
        metric_sample(2, 38, 100),
        metric_sample(3, 38, 100),
        metric_sample(4, 0, 200, kv_usage=0),
    ]

    with pytest.raises(
        ValueError,
        match="did not prove sustained 38 running requests with decode progress",
    ):
        MODULE.validate_metrics_samples(
            samples,
            concurrency=38,
            duration_seconds=4,
            interval_seconds=1,
        )
