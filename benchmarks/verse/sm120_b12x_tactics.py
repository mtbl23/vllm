#!/usr/bin/env python3
"""Profile FlashInfer B12X NVFP4 tactics at an exact decode batch size.

FlashInfer's default FP4 autotuning maps small token counts to power-of-two
buckets. Verse's production decode batch is deliberately capped at 38, so this
tool times every valid tactic at the real M=38 shape instead of inferring that
the tactic chosen for the M=64 bucket is optimal.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

MODEL_SHAPES = (
    (1920, 30720),
    (1920, 8192),
    (1920, 9216),
    (2048, 3840),
    (4096, 3840),
    (7680, 3840),
)
LM_HEAD_SHAPE = (1920, 262144)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cold-cache-mib", type=int, default=96)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lm-head-only",
        action="store_true",
        help="Profile only Campaign 22's packed 3840x262144 output head.",
    )
    return parser.parse_args()


def benchmark_tactic(
    runner,
    inputs: list,
    tactic,
    cold_cache: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | list]:
    for _ in range(warmup):
        runner(inputs=inputs, tactic=tactic)
    torch.cuda.synchronize()

    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        cold_cache.zero_()
        start.record()
        runner(inputs=inputs, tactic=tactic)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    return {
        "tactic": tactic,
        "median_us": statistics.median(timings) * 1000.0,
        "mean_us": statistics.fmean(timings) * 1000.0,
        "minimum_us": min(timings) * 1000.0,
        "maximum_us": max(timings) * 1000.0,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.iterations <= 0 or args.warmup < 0:
        raise ValueError("batch size and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark is restricted to exact SM120")

    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm.gemm_base import (
        _b12x_gemm_fp4_runner,
        _MM_FP4_TUNING_CONFIG_128x4,
    )

    tuner = AutoTuner.get()
    tuner.load_configs(str(args.autotune_cache))
    runner = _b12x_gemm_fp4_runner(
        12,
        0,
        True,
        torch.bfloat16,
        True,
    )
    cold_cache = torch.empty(
        args.cold_cache_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    all_results: list[dict] = []
    model_shapes = (LM_HEAD_SHAPE,) if args.lm_head_only else MODEL_SHAPES
    for packed_k, n in model_shapes:
        real_k = packed_k * 2
        padded_m = ((args.batch_size + 127) // 128) * 128
        a = torch.randint(
            0,
            256,
            (args.batch_size, packed_k),
            dtype=torch.uint8,
            device="cuda",
        )
        # The public FlashInfer contract represents B as column-major (K, N),
        # so allocate its transposed storage contiguously before transposing.
        b = torch.randint(
            0,
            256,
            (n, packed_k),
            dtype=torch.uint8,
            device="cuda",
        ).T
        a_descale = torch.ones(
            (padded_m, real_k // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        b_descale = torch.ones(
            (n, real_k // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ).T
        alpha = torch.ones((), dtype=torch.float32, device="cuda")
        out = torch.empty((args.batch_size, n), dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
        inputs = [
            a,
            b,
            a_descale,
            b_descale,
            alpha,
            torch.bfloat16,
            out,
            16,
            True,
            workspace,
        ]

        _, cached_tactic = tuner.choose_one(
            "fp4_gemm",
            [runner],
            _MM_FP4_TUNING_CONFIG_128x4,
            inputs,
        )
        # The B12X runner's tactic validity is independent of the profile's
        # synthetic tensor values and shapes; it reads the real input tensors.
        tactics = runner.get_valid_tactics(inputs, None)
        shape_results = []
        for tactic in tactics:
            result = benchmark_tactic(
                runner,
                inputs,
                tactic,
                cold_cache,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            result["is_cached_tactic"] = tactic == cached_tactic
            shape_results.append(result)
        shape_results.sort(key=lambda item: item["median_us"])
        all_results.append(
            {
                "batch_size": args.batch_size,
                "packed_k": packed_k,
                "real_k": real_k,
                "n": n,
                "cached_tactic": cached_tactic,
                "results": shape_results,
            }
        )
        del a, b, a_descale, b_descale, out
        torch.cuda.empty_cache()

    document = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "cold_cache_mib": args.cold_cache_mib,
        "shapes": all_results,
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
