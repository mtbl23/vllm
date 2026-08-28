# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import unittest.mock
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    import flashinfer
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer is not supported for vLLM on ROCm.", allow_module_level=True
        )

import torch

NUM_HEADS = [(32, 8), (6, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 30.0]
SLIDING_WINDOWS = [None, 64]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_paged_kv_metadata(
    kv_lens: list[int],
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged-KV metadata tensors for fast_plan_decode tests.

    Returns:
        kv_indptr          – CPU int32, shape [num_seqs + 1]
        kv_indices         – CUDA int32, shape [total_blocks]
        kv_last_page_lens  – CPU int32, shape [num_seqs]
        block_tables       – CUDA int32, shape [num_seqs, max_blocks_per_seq]
    """
    num_seqs = len(kv_lens)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device="cuda"
    )

    indptr_list = [0]
    indices_list: list[int] = []
    last_lens_list: list[int] = []
    for i, seq_len in enumerate(kv_lens):
        n = (seq_len + block_size - 1) // block_size
        indices_list.extend(block_tables[i, :n].cpu().tolist())
        indptr_list.append(indptr_list[-1] + n)
        last_lens_list.append(seq_len % block_size or block_size)

    return (
        torch.tensor(indptr_list, dtype=torch.int32, device="cpu"),
        torch.tensor(indices_list, dtype=torch.int32, device="cuda"),
        torch.tensor(last_lens_list, dtype=torch.int32, device="cpu"),
        block_tables,
    )


def _dense_attention_reference(
    queries: list[torch.Tensor],
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    *,
    scale: float,
    sliding_window: int | None,
) -> torch.Tensor:
    """Compute attention from source tensors, independently of the KV cache."""
    outputs: list[torch.Tensor] = []
    for query, key, value in zip(queries, keys, values, strict=True):
        if query.shape[1] != key.shape[1]:
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        query_len = query.shape[0]
        kv_len = key.shape[0]
        query_positions = torch.arange(kv_len - query_len, kv_len, device=query.device)
        key_positions = torch.arange(kv_len, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if sliding_window is not None:
            allowed &= key_positions.unsqueeze(0) >= (
                query_positions.unsqueeze(1) - (sliding_window - 1)
            )

        scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
        scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hqk,khd->qhd", probabilities, value.float()).to(query.dtype)
        )
    return torch.cat(outputs, dim=0)


def _dequantize_nvfp4_sequences_from_cache(
    *,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: list[int],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    k_scale: float,
    v_scale: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reconstruct logical K/V sequences from the packed HND cache.

    This is intentionally independent of FlashInfer's attention reader. It
    decodes the bytes written by reshape_and_cache_flash and follows the
    scheduler's logical-to-physical block table, so the attention oracle
    measures kernel correctness rather than unavoidable NVFP4 quantization
    loss against the original BF16 tensors.
    """
    from tests.kernels.quantization.nvfp4_utils import (
        dequant_nvfp4_kv_cache,
    )
    from vllm.utils.torch_utils import nvfp4_split_data_scale

    k_side, v_side = kv_cache.split(num_kv_heads, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)
    k_dequantized = dequant_nvfp4_kv_cache(
        k_data,
        k_sf,
        k_scale,
        head_size,
        block_size,
        swizzle_sf=False,
    )
    v_dequantized = dequant_nvfp4_kv_cache(
        v_data,
        v_sf,
        v_scale,
        head_size,
        block_size,
        swizzle_sf=False,
    )

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for sequence_index, kv_len in enumerate(kv_lens):
        num_blocks = (kv_len + block_size - 1) // block_size
        physical_blocks = block_table[sequence_index, :num_blocks].long()
        keys.append(
            k_dequantized[physical_blocks]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_size)[:kv_len]
        )
        values.append(
            v_dequantized[physical_blocks]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_size)[:kv_len]
        )
    return keys, values


def _sm120_test_config(
    *,
    dtype: torch.dtype,
    num_q_heads: int,
) -> SimpleNamespace:
    from vllm.config import CUDAGraphMode

    model_config = SimpleNamespace(
        dtype=dtype,
        max_model_len=6144,
        get_num_attention_heads=lambda _parallel_config: num_q_heads,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="nvfp4"),
        attention_config=SimpleNamespace(
            disable_flashinfer_q_quantization=False,
            use_trtllm_attention=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512,
            max_num_seqs=38,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            dcp_kv_cache_interleave_size=1,
            dcp_comm_backend="allgather",
        ),
        use_v2_model_runner=False,
    )


def _make_common_attention_metadata(
    *,
    query_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    device: torch.device,
):
    """Build only the scheduler metadata needed by the production-path oracle."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    num_reqs = len(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()

    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    num_computed_tokens_cpu = torch.tensor(
        [kv_len - query_len for query_len, kv_len in zip(query_lens, kv_lens)],
        dtype=torch.int32,
    )

    max_blocks_per_request = (max(kv_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        num_reqs * max_blocks_per_request,
        dtype=torch.int32,
        device=device,
    ).view(num_reqs, max_blocks_per_request)
    # Scheduler-assigned physical pages are not monotonic in production after
    # churn. Reverse every row and rotate request ownership so the oracle will
    # fail if the kernel accidentally treats logical block positions as
    # physical page IDs. Physical pages remain unique across live requests.
    block_table_tensor = torch.roll(block_table_tensor.flip(1), shifts=1, dims=0)
    num_actual_tokens = sum(query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(kv_lens),
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(
            num_actual_tokens,
            dtype=torch.int64,
            device=device,
        ),
        causal=True,
    )


def _run_nvfp4_gemma4_production_path(
    *,
    query_lens: list[int],
    kv_lens: list[int],
    head_size: int,
    sliding_window: int | None,
) -> None:
    """Exercise MetadataBuilder -> cache update -> FlashInferImpl.forward."""
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend
    from vllm.v1.attention.backends.flashinfer import (
        FIDecode,
        FIPrefill,
        FlashInferImpl,
        FlashInferMetadataBuilder,
    )
    from vllm.v1.attention.backends.utils import (
        PerLayerParameters,
        get_kv_cache_layout,
        set_kv_cache_layout,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    assert len(query_lens) == len(kv_lens)
    assert all(query_len <= kv_len for query_len, kv_len in zip(query_lens, kv_lens))

    set_random_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 4
    scale = head_size**-0.5
    common = _make_common_attention_metadata(
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_size=page_size,
        device=device,
    )

    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for query_len, kv_len in zip(query_lens, kv_lens, strict=True):
        queries.append(
            torch.randn(
                query_len,
                num_q_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
        )
        keys.append(
            torch.randn(
                kv_len,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
        )
        values.append(0.25 * torch.randn_like(keys[-1]))

    query = torch.cat(queries, dim=0)
    key = torch.cat(
        [
            tensor[-query_len:]
            for tensor, query_len in zip(keys, query_lens, strict=True)
        ],
        dim=0,
    )
    value = torch.cat(
        [
            tensor[-query_len:]
            for tensor, query_len in zip(values, query_lens, strict=True)
        ],
        dim=0,
    )
    k_scale = torch.stack([tensor.abs().amax() for tensor in keys]).amax() / 448.0
    v_scale = torch.stack([tensor.abs().amax() for tensor in values]).amax() / 448.0
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        _k_scale=k_scale.float(),
        _v_scale=v_scale.float(),
        _q_scale_float=1.0,
        _k_scale_float=k_scale.float().item(),
        _v_scale_float=v_scale.float().item(),
        _o_scale_float=None,
    )

    config = _sm120_test_config(dtype=dtype, num_q_heads=num_q_heads)
    kv_cache_spec = FullAttentionSpec(
        block_size=page_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
        sliding_window=sliding_window,
    )
    layer_names = ["test_layer"]
    window_left = sliding_window - 1 if sliding_window is not None else -1

    def per_layer_parameters(*_args, **_kwargs):
        return {
            layer_names[0]: PerLayerParameters(
                window_left=window_left,
                logits_soft_cap=0.0,
                sm_scale=scale,
            )
        }

    def slots_for(sequence_index: int, start: int, count: int) -> torch.Tensor:
        offsets = torch.arange(start, start + count, device=device)
        blocks = common.block_table_tensor[sequence_index, offsets // page_size].long()
        return blocks * page_size + offsets % page_size

    query_cursor = 0
    for sequence_index, (query_len, kv_len) in enumerate(
        zip(query_lens, kv_lens, strict=True)
    ):
        common.slot_mapping[query_cursor : query_cursor + query_len] = slots_for(
            sequence_index, kv_len - query_len, query_len
        )
        query_cursor += query_len

    num_blocks = int(common.block_table_tensor.max().item()) + 1
    kv_cache = torch.zeros(
        num_blocks,
        2 * num_kv_heads,
        page_size,
        nvfp4_kv_cache_full_dim(head_size),
        dtype=torch.uint8,
        device=device,
    )

    set_kv_cache_layout("HND")
    get_kv_cache_layout.cache_clear()
    try:
        with (
            unittest.mock.patch.dict(
                os.environ,
                {
                    "VLLM_VERSE_RUNTIME_STRICT": "1",
                    "VLLM_NVFP4_KV_VOSPLIT": "1",
                },
            ),
            set_current_vllm_config(config),
            unittest.mock.patch.object(
                flashinfer_backend,
                "can_use_trtllm_attention",
                return_value=True,
            ) as can_use_trtllm_mock,
            unittest.mock.patch.object(
                flashinfer_backend,
                "use_trtllm_attention",
                return_value=False,
            ),
            unittest.mock.patch.object(
                flashinfer_backend,
                "get_num_attention_heads_from_layers",
                return_value=num_q_heads,
            ),
            unittest.mock.patch.object(
                flashinfer_backend,
                "get_per_layer_parameters",
                side_effect=per_layer_parameters,
            ),
        ):
            builder = FlashInferMetadataBuilder(
                kv_cache_spec,
                layer_names,
                config,
                device,
            )
            implementation = FlashInferImpl(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=sliding_window,
                kv_cache_dtype="nvfp4",
            )

            for sequence_index, (query_len, key_full, value_full) in enumerate(
                zip(query_lens, keys, values, strict=True)
            ):
                context_len = key_full.shape[0] - query_len
                if context_len:
                    implementation.do_kv_cache_update(
                        layer,
                        key_full[:context_len],
                        value_full[:context_len],
                        kv_cache,
                        slots_for(sequence_index, 0, context_len),
                    )

            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common,
            )
            implementation.do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                metadata.slot_mapping,
            )
            output = implementation.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                metadata,
                output=torch.empty_like(query),
            )

            assert builder.use_fa2_nvfp4_kv
            assert builder.disable_split_kv
            # Exact SM120 normally advertises dedicated XQA. The strict packed
            # NVFP4 path must observe that availability and then deliberately
            # clear it in favor of the native FA2 reader before execution.
            assert can_use_trtllm_mock.call_count >= 1
            assert builder.flashinfer_trtllm_api_decode_kernel is None
            assert not builder.use_trtllm_decode_attention
            assert not builder.use_dedicated_xqa
            if head_size > 256:
                assert builder.vo_split == 2
                assert builder.reorder_batch_threshold == 0
                assert metadata.num_decodes == 0
                assert metadata.num_prefills == len(query_lens)
                assert isinstance(metadata.prefill, FIPrefill)
            else:
                assert builder.vo_split == 1
                assert metadata.num_decodes == len(query_lens)
                assert metadata.num_prefills == 0
                assert isinstance(metadata.decode, FIDecode)

        dequantized_keys, dequantized_values = _dequantize_nvfp4_sequences_from_cache(
            kv_cache=kv_cache,
            block_table=common.block_table_tensor,
            kv_lens=kv_lens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=page_size,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
        )
        expected = _dense_attention_reference(
            queries,
            dequantized_keys,
            dequantized_values,
            scale=scale,
            sliding_window=sliding_window,
        )
        assert torch.isfinite(output).all()
        relative_error = torch.linalg.vector_norm(
            output.float() - expected.float()
        ) / torch.linalg.vector_norm(expected.float())
        cosine_similarity = torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
        output_norm_ratio = torch.linalg.vector_norm(output.float()) / (
            torch.linalg.vector_norm(expected.float())
        )
        projected_gain = torch.sum(output.float() * expected.float()) / torch.sum(
            expected.float().square()
        )
        assert relative_error.item() < 0.01
        assert cosine_similarity.item() > 0.999
        assert 0.99 < output_norm_ratio.item() < 1.01
        assert 0.99 < projected_gain.item() < 1.01
    finally:
        set_kv_cache_layout(None)
        get_kv_cache_layout.cache_clear()


@pytest.mark.parametrize(
    ("query_lens", "kv_lens"),
    [
        ([1, 5], [33, 47]),
        ([1, 5], [1000, 5500]),
        ([1], [6144]),
        ([512], [1024]),
    ],
)
@torch.inference_mode()
def test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference(
    query_lens: list[int], kv_lens: list[int]
) -> None:
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        pytest.skip("requires exact SM120")

    _run_nvfp4_gemma4_production_path(
        query_lens=query_lens,
        kv_lens=kv_lens,
        head_size=512,
        sliding_window=None,
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    ("query_lens", "kv_lens"),
    [
        ([1, 1, 1], [1000, 5500, 6144]),
        ([16, 64], [1000, 5500]),
    ],
)
def test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference(
    query_lens: list[int], kv_lens: list[int]
) -> None:
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        pytest.skip("requires exact SM120")

    _run_nvfp4_gemma4_production_path(
        query_lens=query_lens,
        kv_lens=kv_lens,
        head_size=256,
        sliding_window=1024,
    )


def _make_cg_decode_wrapper(
    num_seqs: int,
    kv_indices_buffer: torch.Tensor,
    workspace_buffer: torch.Tensor,
    use_tensor_cores: bool = True,
) -> "flashinfer.BatchDecodeWithPagedKVCacheWrapper":
    """Create a cudagraph-enabled BatchDecodeWithPagedKVCacheWrapper.

    *kv_indices_buffer* is shared with the caller so that fast_plan_decode
    can avoid the device-to-device index copy on subsequent (cudagraph) calls.
    """
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices_buffer=kv_indices_buffer,
        paged_kv_last_page_len_buffer=torch.zeros(
            num_seqs, dtype=torch.int32, device="cuda"
        ),
        use_tensor_cores=use_tensor_cores,
    )


def test_fast_decode_plan_importable() -> None:
    """fast_decode_plan must be importable from flashinfer.decode.

    This is a forward-compatibility smoke test: if FlashInfer reorganises its
    public API the import will fail before any other test does.
    """
    from flashinfer.decode import fast_decode_plan  # noqa: F401

    assert callable(fast_decode_plan)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_warmup_uses_full_plan(dtype: torch.dtype) -> None:
    """On the first call fast_plan_decode must route through self.plan() and
    flip vllm_first_call to False on the wrapper object."""
    from unittest.mock import patch

    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)

    kv_lens = [128, 64]
    block_size = 16
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices.clone(), workspace)

    assert getattr(wrapper, "vllm_first_call", True) is True

    with patch.object(wrapper, "plan", wraps=wrapper.plan) as mock_plan:
        fast_plan_decode(
            wrapper,
            indptr_cpu=kv_indptr,
            indices=kv_indices,
            last_page_len_cpu=kv_last_page_lens,
            num_qo_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            page_size=block_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        mock_plan.assert_called_once()

    assert wrapper.vllm_first_call is False, (
        "vllm_first_call should be False after the first fast_plan_decode call"
    )


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_matches_full_plan(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """fast_plan_decode's cudagraph path (delegating to FlashInfer's
    fast_decode_plan) must produce attention output numerically identical to
    a standard plan() call.

    Both the warmup call (self.plan) and the subsequent fast call
    (fast_decode_plan) are verified against the same reference.
    """
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = num_heads

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    # Reference output via the standard plan()
    workspace_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    ref_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_ref, "NHD", use_tensor_cores=True
    )
    ref_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref_output = ref_wrapper.run(query, key_value_cache)

    # CUDAGraph wrapper exercised through fast_plan_decode
    kv_indices_buf = kv_indices.clone()
    workspace_cg = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    cg_wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices_buf, workspace_cg)

    plan_kwargs: dict = dict(
        indptr_cpu=kv_indptr,
        indices=kv_indices_buf,
        last_page_len_cpu=kv_last_page_lens,
        num_qo_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # First call – warmup path (routes through self.plan)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    warmup_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(warmup_output, ref_output, atol=1e-2, rtol=1e-2)

    # Second call – fast path (routes through fast_decode_plan from FlashInfer)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    fast_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(fast_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_decode_with_paged_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, key_value_cache)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_prefill_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(
        query,
        key_value_cache,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 132), (5, 18)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
def test_flashinfer_prefill_with_paged_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    pytest.skip("TODO: fix the accuracy issue")
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    kv_cache_fp8 = torch.cat([key_cache / k_scale, value_cache / v_scale], dim=1).to(
        kv_cache_dtype
    )

    assert kv_cache_fp8.shape == key_value_cache.shape
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache.squeeze(1),
        value_cache=value_cache.squeeze(1),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    del query
    del block_tables
    # verify prefill fp8
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.skip(reason="TODO: fix the accuracy issue")
@torch.inference_mode
def test_flashinfer_decode_with_paged_fp8_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    # test doesn't work for num_heads = (16,16)
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    use_tensor_cores = True
    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    key_cache_fp8 = (key_cache / k_scale).to(kv_cache_dtype)
    value_cache_fp8 = (value_cache / v_scale).to(kv_cache_dtype)
    assert key_cache_fp8.shape[1] == 1 and value_cache_fp8.shape[1] == 1
    kv_cache_fp8 = torch.cat([key_cache_fp8, value_cache_fp8], dim=1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )
    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    # Temporary fix: Increasing the tolerance. Seems like a flashinfer issue
    (
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )
