# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).parents[2] / "benchmarks" / "verse"
B01_PATH = BENCHMARK_DIR / "sm120_b01.py"
B01_SPEC = importlib.util.spec_from_file_location("sm120_b01", B01_PATH)
assert B01_SPEC is not None and B01_SPEC.loader is not None
B01_MODULE = importlib.util.module_from_spec(B01_SPEC)
sys.modules[B01_SPEC.name] = B01_MODULE
B01_SPEC.loader.exec_module(B01_MODULE)

MODULE_PATH = BENCHMARK_DIR / "sm120_prefill_interference.py"
SPEC = importlib.util.spec_from_file_location(
    "verse_sm120_prefill_interference", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generated_rate_uses_exact_metric_window():
    sample = B01_MODULE.MetricSample
    samples = [
        sample(0.0, 100.0, 38.0, 0.0),
        sample(1.0, 200.0, 38.0, 0.0),
        sample(2.0, 500.0, 38.0, 0.0),
        sample(3.0, 900.0, 38.0, 0.0),
    ]

    assert MODULE.generated_rate(samples, 1.0, 3.0) == 350.0


def test_generated_rate_rejects_single_sample_window():
    sample = B01_MODULE.MetricSample
    samples = [sample(1.0, 200.0, 38.0, 0.0)]

    with pytest.raises(RuntimeError, match="fewer than two"):
        MODULE.generated_rate(samples, 0.0, 2.0)
