# SPDX-License-Identifier: Apache-2.0
"""Self-contained native NVFP4 KV-store oracle for the Verse appliance."""

import random

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    create_kv_caches_with_random_flash,
    nvfp4_split_data_scale,
    set_random_seed,
)


def _assert_nvfp4_roundtrip(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    require_finite = (
        torch.isfinite(actual_flat).all() and torch.isfinite(expected_flat).all()
    )
    assert require_finite, "NVFP4 roundtrip produced non-finite values"

    cosine = torch.nn.functional.cosine_similarity(
        actual_flat.unsqueeze(0), expected_flat.unsqueeze(0)
    ).item()
    relative_l2 = (
        torch.linalg.vector_norm(actual_flat - expected_flat)
        / torch.linalg.vector_norm(expected_flat).clamp_min(1e-12)
    ).item()
    norm_ratio = (
        torch.linalg.vector_norm(actual_flat)
        / torch.linalg.vector_norm(expected_flat).clamp_min(1e-12)
    ).item()
    nonzero_fraction = torch.count_nonzero(actual_flat).item() / actual_flat.numel()

    assert cosine >= 0.99, f"NVFP4 roundtrip cosine {cosine:.6f} < 0.99"
    assert relative_l2 <= 0.15, f"NVFP4 relative L2 {relative_l2:.6f} > 0.15"
    assert 0.85 <= norm_ratio <= 1.15, (
        f"NVFP4 norm ratio {norm_ratio:.6f} is outside [0.85, 1.15]"
    )
    assert nonzero_fraction >= 0.75, (
        f"NVFP4 nonzero fraction {nonzero_fraction:.6f} < 0.75"
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    ("num_heads", "head_size"),
    (
        pytest.param(8, 64, id="shape-regression"),
        pytest.param(4, 512, id="gemma4-runtime"),
    ),
)
@torch.inference_mode()
def test_verse_sm120_nvfp4_physical_hnd_roundtrip(
    device: str,
    num_heads: int,
    head_size: int,
) -> None:
    if not current_platform.has_device_capability(120):
        pytest.skip("Verse NVFP4 KV oracle requires SM120.")

    num_tokens = 42
    block_size = 16
    num_blocks = 64
    dtype = torch.bfloat16

    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens),
        dtype=torch.long,
        device=device,
    )
    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype, device=device)
    _, key, value = qkv.unbind(dim=1)

    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        "nvfp4",
        dtype,
        device=device,
        cache_layout="HND",
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    key_cache_physical = key_cache.permute(0, 2, 1, 3)
    value_cache_physical = value_cache.permute(0, 2, 1, 3)

    nvfp4_key_data, key_scale_cache = nvfp4_split_data_scale(key_cache)
    nvfp4_value_data, value_scale_cache = nvfp4_split_data_scale(value_cache)
    k_scale = (key.abs().amax() / 448.0).to(torch.float32)
    v_scale = (value.abs().amax() / 448.0).to(torch.float32)

    ops.reshape_and_cache_flash(
        key,
        value,
        key_cache_physical,
        value_cache_physical,
        slot_mapping,
        "nvfp4",
        k_scale,
        v_scale,
    )

    def dequantize(
        data_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        global_scale: torch.Tensor,
        swizzle_sf: bool,
    ) -> torch.Tensor:
        result_hnd = dequant_nvfp4_kv_cache(
            data_cache.permute(0, 2, 1, 3),
            scale_cache.permute(0, 2, 1, 3),
            global_scale,
            head_size,
            block_size,
            swizzle_sf=swizzle_sf,
        )
        return result_hnd.permute(0, 2, 1, 3)

    result_key = dequantize(nvfp4_key_data, key_scale_cache, k_scale.item(), False)
    result_value = dequantize(
        nvfp4_value_data,
        value_scale_cache,
        v_scale.item(),
        current_platform.get_device_capability().major < 12,
    )
    result_key = result_key.reshape(num_slots, num_heads, head_size)[slot_mapping]
    result_value = result_value.reshape(num_slots, num_heads, head_size)[slot_mapping]

    _assert_nvfp4_roundtrip(result_key, key)
    _assert_nvfp4_roundtrip(result_value, value)
