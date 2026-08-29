# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2] / "tools" / "verse" / ("evaluate_sm120_acceptance.py")
)
SPEC = importlib.util.spec_from_file_location("verse_sm120_acceptance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RELEASE_NONCE = "e" * 64


def report(
    target: int,
    steady: float,
    wall: float,
    prewarmed: bool = True,
    release_nonce: str = RELEASE_NONCE,
) -> dict:
    return {
        "status": "pass",
        "image_digest": f"registry/runtime@sha256:{'a' * 64}",
        "fork_commit": "b" * 40,
        "model_revision": "c" * 40,
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "server_version": "0.28.0",
        "release_nonce": release_nonce,
        "concurrency": 38,
        "requested_prompt_tokens": target,
        "prompt_tokens_min": target - 1,
        "prompt_tokens_max": target,
        "output_tokens_per_request": 512,
        "generated_tokens": 38 * 512,
        "expected_generated_tokens": 38 * 512,
        "completed_request_count": 38,
        "expected_request_count": 38,
        "completion_tokens_by_request": [512] * 38,
        "all_requests_returned_exact_output_tokens": True,
        "max_running_requests_observed": 38,
        "server_idle_before_ownership": True,
        "server_idle_before_measurement": True,
        "server_idle_after_measurement": True,
        "preemptions_delta": 0,
        "preemption_error": None,
        "steady_window_seconds": 12.0,
        "steady_window_samples": 200,
        "steady_window_prompt_tokens_delta": 0,
        "steady_aggregate_tokens_per_second": steady,
        "wall_aggregate_tokens_per_second": wall,
        "prewarmed": prewarmed,
        "prefix_preparation": (
            "explicitly_prewarmed_this_run"
            if prewarmed
            else "no_explicit_prewarm_this_run"
        ),
    }


def matrix(one_k: float = 1200, long: float = 1040) -> list[dict]:
    return [
        report(1000, one_k, one_k, False),
        report(1000, one_k, one_k),
        report(1000, one_k, one_k),
        report(5500, long, long, False),
        report(5500, long, long),
        report(5500, long, long),
    ]


def test_acceptance_passes_fixed_matrix():
    result = MODULE.evaluate(matrix())

    assert result["status"] == "pass"
    assert result["scenarios"]["1000"]["runs"] == 3
    assert result["release_nonce"] == RELEASE_NONCE
    assert result["production_weights"] == {"1000": 0.5, "5500": 0.5}
    assert sum(result["production_weights"].values()) == 1


def test_acceptance_rejects_scenario_regression():
    result = MODULE.evaluate(matrix(one_k=1000))

    assert result["status"] == "fail"
    assert result["scenarios"]["1000"]["status"] == "fail"


def test_acceptance_rejects_identity_drift():
    reports = matrix()
    reports[-1] = dict(reports[-1], fork_commit="d" * 40)

    try:
        MODULE.evaluate(reports)
    except ValueError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("identity drift was accepted")


def test_acceptance_rejects_missing_cold_run():
    reports = [
        dict(
            item,
            prewarmed=True,
            prefix_preparation="explicitly_prewarmed_this_run",
        )
        for item in matrix()
    ]

    try:
        MODULE.evaluate(reports)
    except ValueError as exc:
        assert "one disjoint-prefix" in str(exc)
    else:
        raise AssertionError("all-warm matrix was accepted")


def test_acceptance_applies_explicit_production_weights(monkeypatch):
    monkeypatch.setattr(MODULE, "PRODUCTION_WEIGHTS", {1000: 0.75, 5500: 0.25})

    result = MODULE.evaluate(matrix(one_k=1074.0, long=1102.05))

    assert result["weighted_improvement_over_legacy"] == 0.275
    assert result["status"] == "fail"


def test_acceptance_rejects_production_weights_that_do_not_sum_to_one(monkeypatch):
    monkeypatch.setattr(MODULE, "PRODUCTION_WEIGHTS", {1000: 0.6, 5500: 0.5})

    try:
        MODULE.evaluate(matrix())
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("invalid production weights were accepted")


def test_acceptance_rejects_release_nonce_drift():
    reports = matrix()
    reports[-1] = dict(reports[-1], release_nonce="f" * 64)

    try:
        MODULE.evaluate(reports)
    except ValueError as exc:
        assert "release nonce changed" in str(exc)
    else:
        raise AssertionError("mixed release nonces were accepted")


def test_acceptance_rejects_prefill_inside_decode_window():
    reports = matrix()
    reports[0] = dict(reports[0], steady_window_prompt_tokens_delta=1)

    try:
        MODULE.evaluate(reports)
    except ValueError as exc:
        assert "mixed prefill" in str(exc)
    else:
        raise AssertionError("prefill-contaminated decode window was accepted")
