# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


def test_triton_metadata_uses_cache_group_kv_geometry():
    """Hybrid groups must not inherit model-wide KV geometry."""
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel_config: 32,
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 512,
        rswa_window=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=object(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=40),
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=256,
        dtype=torch.uint8,
    )

    with patch(
        "vllm.v1.attention.backends.triton_attn.get_num_attention_heads_from_layers",
        return_value=16,
    ):
        builder = TritonAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["model.layers.0.self_attn"],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.num_heads_q == 16
    assert builder.num_heads_kv == 8
    assert builder.headdim == 256
    assert builder.seq_threshold_3D == 16
    assert builder.softmax_segm_output.shape[-1] == 256
