#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SCENARIOS = {
    1000: {"minimum": 1074.0, "baseline": 895.0},
    5500: {"minimum": 992.0, "baseline": 734.7},
}
EXPECTED_RUNS = 3
EXPECTED_CONCURRENCY = 38
EXPECTED_OUTPUT_TOKENS = 512
MINIMUM_WEIGHTED_IMPROVEMENT = 0.30
PRODUCTION_WEIGHTS = {
    1000: 0.50,
    5500: 0.50,
}
RELEASE_NONCE_RE = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        require(isinstance(payload, dict), f"{path} does not contain an object")
        reports.append(payload)
    return reports


def evaluate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    require(
        set(PRODUCTION_WEIGHTS) == set(EXPECTED_SCENARIOS),
        "production weights do not cover the fixed B01 scenarios",
    )
    require(
        all(
            isinstance(weight, int | float)
            and math.isfinite(float(weight))
            and float(weight) > 0
            for weight in PRODUCTION_WEIGHTS.values()
        ),
        "production weights must be finite and positive",
    )
    require(
        math.isclose(
            math.fsum(float(weight) for weight in PRODUCTION_WEIGHTS.values()),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "production weights must sum to 1",
    )
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, str, str, str]] = set()
    release_nonces: set[str] = set()
    for report in reports:
        target = int(report.get("requested_prompt_tokens", -1))
        require(target in EXPECTED_SCENARIOS, f"unexpected prompt target {target}")
        require(report.get("status") == "pass", f"{target}-token run failed")
        require(
            int(report.get("concurrency", -1)) == EXPECTED_CONCURRENCY,
            f"{target}-token run used wrong concurrency",
        )
        require(
            int(report.get("output_tokens_per_request", -1)) == EXPECTED_OUTPUT_TOKENS,
            f"{target}-token run used wrong output length",
        )
        require(
            int(report.get("generated_tokens", -1))
            == int(report.get("expected_generated_tokens", -2)),
            f"{target}-token run did not complete every token",
        )
        require(
            int(report.get("completed_request_count", -1)) == EXPECTED_CONCURRENCY
            and int(report.get("expected_request_count", -1)) == EXPECTED_CONCURRENCY,
            f"{target}-token run did not complete every request",
        )
        per_request = report.get("completion_tokens_by_request")
        require(
            isinstance(per_request, list)
            and len(per_request) == EXPECTED_CONCURRENCY
            and all(int(value) == EXPECTED_OUTPUT_TOKENS for value in per_request),
            f"{target}-token run has incomplete per-request output",
        )
        require(
            report.get("all_requests_returned_exact_output_tokens") is True,
            f"{target}-token run did not prove exact output lengths",
        )
        require(
            float(report.get("max_running_requests_observed", 0))
            >= EXPECTED_CONCURRENCY,
            f"{target}-token run never observed full concurrency",
        )
        for key in (
            "server_idle_before_ownership",
            "server_idle_before_measurement",
            "server_idle_after_measurement",
        ):
            require(report.get(key) is True, f"{target}-token run failed {key}")
        require(
            float(report.get("preemptions_delta", math.inf)) == 0
            and report.get("preemption_error") is None,
            f"{target}-token run observed preemption",
        )
        require(
            target - 2 <= int(report.get("prompt_tokens_min", -1)) <= target,
            f"{target}-token run has a short minimum prompt",
        )
        require(
            target - 2 <= int(report.get("prompt_tokens_max", -1)) <= target,
            f"{target}-token run has a short maximum prompt",
        )
        require(
            float(report.get("steady_window_seconds", 0)) >= 10.0,
            f"{target}-token run lacks a 10-second steady window",
        )
        require(
            int(report.get("steady_window_samples", 0)) >= 50,
            f"{target}-token run lacks enough metric samples",
        )
        release_nonce = str(report.get("release_nonce", ""))
        require(
            RELEASE_NONCE_RE.fullmatch(release_nonce) is not None,
            f"{target}-token run has an invalid release nonce",
        )
        release_nonces.add(release_nonce)
        identities.add(
            (
                str(report.get("image_digest")),
                str(report.get("fork_commit")),
                str(report.get("model_revision")),
                str(report.get("gpu_name")),
                str(report.get("server_version")),
            )
        )
        grouped[target].append(report)

    require(len(identities) == 1, "benchmark identity changed between runs")
    require(len(release_nonces) == 1, "release nonce changed between benchmark runs")
    require(
        set(grouped) == set(EXPECTED_SCENARIOS),
        "both 1K and 5.5K scenarios are required",
    )

    scenario_results: dict[str, Any] = {}
    improvement_ratios: dict[int, float] = {}
    passed = True
    for target, expectation in EXPECTED_SCENARIOS.items():
        runs = grouped[target]
        require(
            len(runs) == EXPECTED_RUNS,
            f"{target}-token scenario requires exactly {EXPECTED_RUNS} runs",
        )
        warm_states = [bool(item.get("prewarmed")) for item in runs]
        prefix_states = [str(item.get("prefix_preparation")) for item in runs]
        require(
            warm_states.count(False) == 1 and warm_states.count(True) == 2,
            f"{target}-token scenario requires one disjoint-prefix and two "
            "explicitly prewarmed runs",
        )
        require(
            prefix_states.count("no_explicit_prewarm_this_run") == 1
            and prefix_states.count("explicitly_prewarmed_this_run") == 2,
            f"{target}-token scenario has the wrong prefix preparation matrix",
        )
        steady = [float(item["steady_aggregate_tokens_per_second"]) for item in runs]
        wall = [float(item["wall_aggregate_tokens_per_second"]) for item in runs]
        median_steady = statistics.median(steady)
        median_wall = statistics.median(wall)
        scenario_passed = (
            median_steady >= expectation["minimum"]
            and median_wall >= expectation["minimum"] * 0.9
        )
        passed &= scenario_passed
        improvement_ratios[target] = median_steady / expectation["baseline"] - 1.0
        scenario_results[str(target)] = {
            "status": "pass" if scenario_passed else "fail",
            "runs": len(runs),
            "disjoint_prefix_runs": warm_states.count(False),
            "explicitly_prewarmed_runs": warm_states.count(True),
            "steady_tokens_per_second": steady,
            "wall_tokens_per_second": wall,
            "median_steady_tokens_per_second": round(median_steady, 3),
            "median_wall_tokens_per_second": round(median_wall, 3),
            "minimum_steady_tokens_per_second": expectation["minimum"],
            "legacy_baseline_tokens_per_second": expectation["baseline"],
        }

    weighted_improvement = math.fsum(
        improvement_ratios[target] * float(PRODUCTION_WEIGHTS[target])
        for target in EXPECTED_SCENARIOS
    )
    passed &= weighted_improvement >= MINIMUM_WEIGHTED_IMPROVEMENT
    identity = next(iter(identities))
    return {
        "status": "pass" if passed else "fail",
        "release_nonce": next(iter(release_nonces)),
        "identity": {
            "image_digest": identity[0],
            "fork_commit": identity[1],
            "model_revision": identity[2],
            "gpu_name": identity[3],
            "server_version": identity[4],
        },
        "scenarios": scenario_results,
        "production_weights": {
            str(target): PRODUCTION_WEIGHTS[target] for target in EXPECTED_SCENARIOS
        },
        "weighted_improvement_over_legacy": round(weighted_improvement, 6),
        "minimum_weighted_improvement": MINIMUM_WEIGHTED_IMPROVEMENT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed Verse SM120 B01 acceptance matrix."
    )
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        result = evaluate(load_reports(args.reports))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
