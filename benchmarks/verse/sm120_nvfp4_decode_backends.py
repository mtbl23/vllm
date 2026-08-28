#!/usr/bin/env python3
"""Probe FlashInfer's SM120 NVFP4 paged-decode backends.

This is deliberately independent from the vLLM scheduler.  It answers the
first routing question cheaply: does a backend expose a kernel for the exact
Gemma 4 decode geometry, and what is its kernel-only latency on the same packed
KV pages?
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def _bench(
    fn: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> tuple[torch.Tensor, float]:
    out = fn()
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        out = fn()
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=6144)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument(
        "--fa2-vo-splits",
        type=int,
        default=1,
        help="Number of FA2 value/output head-dimension passes",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="Left attention window, -1 for full context",
    )
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--variants",
        default="fa2-bf16,xqa-bf16,trtllm-bf16,trtllm-fp8",
        help=("Comma-separated subset of fa2-bf16,xqa-bf16,trtllm-bf16,trtllm-fp8"),
    )
    args = parser.parse_args()

    if args.seq_len % args.page_size:
        raise ValueError("seq-len must be divisible by page-size")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")

    torch.manual_seed(17)
    device = torch.device("cuda")
    pages_per_seq = args.seq_len // args.page_size
    num_pages = args.batch_size * pages_per_seq
    packed_dim = args.head_dim // 2
    scale_dim = args.head_dim // 16

    # HND is vLLM's zero-copy physical layout for NVFP4 KV.
    # Match vLLM's block-contiguous physical layout exactly.  Each page owns
    # [K data | K scales | V data | V scales]; K/V views share the full-page
    # stride so cache block copies remain ordinary contiguous page copies.
    full_dim = packed_dim + scale_dim
    storage = torch.empty(
        (num_pages, 2 * args.kv_heads, args.page_size, full_dim),
        dtype=torch.uint8,
        device=device,
    )
    page_bytes = storage.stride(0)
    side_bytes = args.kv_heads * args.page_size * full_dim
    data_per_side = args.kv_heads * args.page_size * packed_dim
    data_shape = (num_pages, args.kv_heads, args.page_size, packed_dim)
    data_strides = (
        page_bytes,
        args.page_size * packed_dim,
        packed_dim,
        1,
    )
    scale_shape = (num_pages, args.kv_heads, args.page_size, scale_dim)
    scale_strides = (
        page_bytes,
        args.page_size * scale_dim,
        scale_dim,
        1,
    )
    k_cache = torch.as_strided(storage, data_shape, data_strides, storage_offset=0)
    v_cache = torch.as_strided(
        storage, data_shape, data_strides, storage_offset=side_bytes
    )
    k_cache.random_(0, 256)
    v_cache.random_(0, 256)
    k_scale = torch.as_strided(
        storage,
        scale_shape,
        scale_strides,
        storage_offset=data_per_side,
    ).view(torch.float8_e4m3fn)
    v_scale = torch.as_strided(
        storage,
        scale_shape,
        scale_strides,
        storage_offset=side_bytes + data_per_side,
    ).view(torch.float8_e4m3fn)
    k_scale.fill_(1.0)
    v_scale.fill_(1.0)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        args.batch_size, pages_per_seq
    )
    seq_lens = torch.full(
        (args.batch_size,), args.seq_len, dtype=torch.uint32, device=device
    )
    q_bf16 = torch.randn(
        (args.batch_size, args.q_heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    scale = 1.0 / math.sqrt(args.head_dim)

    fa2_run: Callable[[], torch.Tensor] | None = None
    if "fa2-bf16" in args.variants.split(","):
        vo_splits = args.fa2_vo_splits
        if vo_splits < 1 or args.head_dim % vo_splits:
            raise ValueError("fa2-vo-splits must divide head-dim")
        vo_dim = args.head_dim // vo_splits
        qo_indptr = torch.arange(args.batch_size + 1, dtype=torch.int32, device=device)
        paged_kv_indptr = (
            torch.arange(args.batch_size + 1, dtype=torch.int32, device=device)
            * pages_per_seq
        )
        paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_page_len = torch.full(
            (args.batch_size,), args.page_size, dtype=torch.int32, device=device
        )
        fa2 = BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="HND", backend="fa2"
        )
        fa2.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            last_page_len,
            num_qo_heads=args.q_heads,
            num_kv_heads=args.kv_heads,
            head_dim_qk=args.head_dim,
            head_dim_vo=vo_dim,
            page_size=args.page_size,
            causal=True,
            sm_scale=scale,
            window_left=args.window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
        fa2_out = torch.empty_like(q_bf16)
        fa2_chunks = [
            torch.empty(
                (args.batch_size, args.q_heads, vo_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(vo_splits)
        ]

        def fa2_run() -> torch.Tensor:
            for index, chunk in enumerate(fa2_chunks):
                data_start = index * (vo_dim // 2)
                scale_start = index * (vo_dim // 16)
                fa2.run(
                    q_bf16,
                    (
                        k_cache,
                        v_cache.narrow(-1, data_start, vo_dim // 2),
                    ),
                    q_scale=1.0,
                    k_scale=1.0,
                    v_scale=1.0,
                    out=chunk,
                    kv_cache_sf=(
                        k_scale,
                        v_scale.narrow(-1, scale_start, vo_dim // 16),
                    ),
                )
                fa2_out.narrow(-1, index * vo_dim, vo_dim).copy_(chunk)
            return fa2_out

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    outputs: dict[str, torch.Tensor] = {}
    for variant in variants:
        if variant == "fa2-bf16":
            assert fa2_run is not None
            run = fa2_run
        elif variant == "xqa-bf16":
            query = q_bf16
            backend = "xqa"
            out_dtype = torch.bfloat16
        elif variant == "trtllm-bf16":
            query = q_bf16
            backend = "trtllm-gen"
            out_dtype = torch.bfloat16
        elif variant == "trtllm-fp8":
            query = q_fp8
            backend = "trtllm-gen"
            out_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(f"unknown variant: {variant}")

        if variant != "fa2-bf16":

            def run(
                query: torch.Tensor = query,
                backend: str = backend,
                out_dtype: torch.dtype = out_dtype,
            ) -> torch.Tensor:
                return trtllm_batch_decode_with_kv_cache(
                    query=query,
                    kv_cache=(k_cache, v_cache),
                    kv_cache_sf=(k_scale, v_scale),
                    workspace_buffer=workspace,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    max_seq_len=args.seq_len,
                    bmm1_scale=scale,
                    bmm2_scale=1.0,
                    window_left=args.window_left,
                    out_dtype=out_dtype,
                    kv_layout="HND",
                    backend=backend,
                    q_len_per_req=1,
                )

        try:
            output, latency_ms = _bench(run, args.warmup, args.iterations)
        except Exception as exc:  # Report kernel-availability failures together.
            print(f"{variant}: ERROR: {type(exc).__name__}: {exc}")
            continue
        outputs[variant] = output.float()
        tokens_per_second = args.batch_size * 1000.0 / latency_ms
        print(f"{variant}: {latency_ms:.6f} ms, {tokens_per_second:.3f} decode tok/s")

    reference = outputs.get("fa2-bf16")
    if reference is not None:
        ref = reference.flatten()
        for name, output in outputs.items():
            if name == "fa2-bf16":
                continue
            candidate = output.flatten()
            cosine = torch.nn.functional.cosine_similarity(ref, candidate, dim=0).item()
            rel_l2 = ((candidate - ref).norm() / ref.norm()).item()
            print(f"{name} vs fa2-bf16: cosine={cosine:.8f}, rel_l2={rel_l2:.8f}")


if __name__ == "__main__":
    main()
