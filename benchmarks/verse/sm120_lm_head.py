#!/usr/bin/env python3
"""Research-only comparison of tied BF16 and quantized Gemma 4 LM heads.

This microbenchmark is not a release qualification. Campaign 22 retains the
original tied BF16 head because the measured quantized head changed top-1
outputs too often for a small end-to-end throughput gain.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=38)
    parser.add_argument("--hidden-size", type=int, default=3840)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def global_scale_inv(tensor: torch.Tensor) -> torch.Tensor:
    # FP8 E4M3 max (448) times FP4 E2M1 max (6), matching ModelOpt NVFP4.
    return torch.tensor(2688.0, device=tensor.device) / tensor.abs().max().float()


def time_cuda(call, warmup: int, iterations: int) -> tuple[torch.Tensor, dict]:
    result = None
    for _ in range(warmup):
        result = call()
    torch.cuda.synchronize()
    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        result = call()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    assert result is not None
    return result, {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
    }


def main() -> int:
    args = parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark requires exact SM120")
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = torch.randn(
        (args.batch_size, args.hidden_size), dtype=dtype, device=device
    )
    weight = torch.empty(
        (args.vocab_size, args.hidden_size), dtype=dtype, device=device
    )
    weight.normal_(mean=0.0, std=args.hidden_size**-0.5)

    weight_scale_inv = global_scale_inv(weight)
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(
        weight,
        weight_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    input_scale_inv = global_scale_inv(hidden)
    alpha = 1.0 / (input_scale_inv * weight_scale_inv)

    def bf16_head() -> torch.Tensor:
        return F.linear(hidden, weight)

    def nvfp4_head() -> torch.Tensor:
        hidden_fp4, hidden_blockscale = ops.scaled_fp4_quant(
            hidden,
            input_scale_inv,
            is_sf_swizzled_layout=True,
            backend="b12x",
        )
        return flashinfer_scaled_fp4_mm(
            hidden_fp4,
            weight_fp4,
            hidden_blockscale,
            weight_blockscale,
            alpha,
            dtype,
            backend="b12x",
        )

    bf16_output, bf16_timing = time_cuda(bf16_head, args.warmup, args.iterations)
    nvfp4_output, nvfp4_timing = time_cuda(nvfp4_head, args.warmup, args.iterations)
    bf16_f32 = bf16_output.float()
    nvfp4_f32 = nvfp4_output.float()
    relative_l2 = (
        torch.linalg.vector_norm(nvfp4_f32 - bf16_f32)
        / torch.linalg.vector_norm(bf16_f32)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        nvfp4_f32.flatten(), bf16_f32.flatten(), dim=0
    ).item()
    top1_agreement = (
        (nvfp4_output.argmax(dim=-1) == bf16_output.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )

    report = {
        "gpu": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "bf16": bf16_timing,
        "nvfp4": nvfp4_timing,
        "speedup": bf16_timing["median_ms"] / nvfp4_timing["median_ms"],
        "saved_ms": bf16_timing["median_ms"] - nvfp4_timing["median_ms"],
        "relative_l2": relative_l2,
        "cosine": cosine,
        "top1_agreement": top1_agreement,
        "weight_bytes": weight.numel() * weight.element_size(),
        "quantized_weight_bytes": (
            weight_fp4.numel() * weight_fp4.element_size()
            + weight_blockscale.numel() * weight_blockscale.element_size()
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
