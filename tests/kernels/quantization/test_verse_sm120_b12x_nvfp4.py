# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch
from nvfp4_utils import (
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
    get_nvfp4_global_scale,
)

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferB12xNvFp4LinearKernel,
)
from vllm.utils.flashinfer import has_flashinfer_b12x_gemm
from vllm.utils.torch_utils import set_random_seed

PROFILE_TOKEN_COUNTS = (
    pytest.param(1, id="decode"),
    pytest.param(38, id="max-seqs"),
    pytest.param(256, id="max-batched-tokens"),
)


@pytest.mark.parametrize("num_tokens", PROFILE_TOKEN_COUNTS)
@torch.inference_mode()
def test_flashinfer_b12x_nvfp4_linear_matches_reference(num_tokens: int) -> None:
    assert torch.cuda.is_available(), "the release oracle requires CUDA"
    assert torch.cuda.get_device_capability(0) == (12, 0), (
        "the release oracle requires an exact SM120 GPU"
    )
    assert importlib.util.find_spec("b12x") is None, (
        "the standalone b12x package must remain uninstalled"
    )
    assert has_flashinfer_b12x_gemm(), (
        "the pinned FlashInfer build lacks its native B12X GEMM"
    )
    supported, reason = FlashInferB12xNvFp4LinearKernel.is_supported()
    assert supported, reason

    set_random_seed(20260828)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    input_size = 512
    output_size = 384
    block_size = 16

    inputs = torch.randn(
        (num_tokens, input_size), dtype=dtype, device=device
    ).contiguous()
    weights = torch.randn(
        (output_size, input_size), dtype=dtype, device=device
    ).contiguous()
    input_global_scale_inv = get_nvfp4_global_scale(inputs)
    weight_global_scale_inv = get_nvfp4_global_scale(weights)

    input_fp4, input_scale_swizzled = ops.scaled_fp4_quant(
        inputs,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    weight_fp4, weight_scale_swizzled = ops.scaled_fp4_quant(
        weights,
        weight_global_scale_inv,
        is_sf_swizzled_layout=True,
    )
    weight_scale_linear = convert_swizzled_to_linear(
        weight_scale_swizzled,
        output_size,
        input_size,
        block_size,
    ).contiguous()

    dequantized_inputs = dequantize_nvfp4_to_dtype(
        input_fp4,
        input_scale_swizzled,
        input_global_scale_inv,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_128x4_layout=True,
    )
    dequantized_weights = dequantize_nvfp4_to_dtype(
        weight_fp4,
        weight_scale_swizzled,
        weight_global_scale_inv,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_128x4_layout=True,
    )
    expected = torch.matmul(dequantized_inputs, dequantized_weights.t())

    layer = torch.nn.Module()
    layer.output_size_per_partition = output_size
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight_fp4, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(weight_scale_linear, requires_grad=False),
    )
    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(input_global_scale_inv, requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(
            1.0 / (input_global_scale_inv * weight_global_scale_inv),
            requires_grad=False,
        ),
    )

    kernel = FlashInferB12xNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    actual = kernel.apply_weights(layer, inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(expected).all().item()
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    relative_error = (
        torch.linalg.vector_norm(actual_f32 - expected_f32)
        / torch.linalg.vector_norm(expected_f32)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    ).item()
    assert relative_error < 0.025, f"relative_error={relative_error:.6f}"
    assert cosine > 0.999, f"cosine={cosine:.6f}"


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    (
        pytest.param(3840, 4096, id="gemma-q-proj"),
        pytest.param(3840, 15360, id="gemma-gate-proj"),
        pytest.param(15360, 3840, id="gemma-down-proj"),
    ),
)
@torch.inference_mode()
def test_flashinfer_b12x_nvfp4_gemma_shapes_match_reference(
    input_size: int,
    output_size: int,
) -> None:
    """Cover the real dense layer shapes used by Verse's Gemma 4 model."""
    assert torch.cuda.is_available(), "the release oracle requires CUDA"
    assert torch.cuda.get_device_capability(0) == (12, 0), (
        "the release oracle requires an exact SM120 GPU"
    )
    assert has_flashinfer_b12x_gemm(), (
        "the pinned FlashInfer build lacks its native B12X GEMM"
    )

    set_random_seed(20260828)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    inputs = torch.randn((1, input_size), dtype=dtype, device=device).contiguous()
    weights = torch.randn(
        (output_size, input_size), dtype=dtype, device=device
    ).contiguous()
    input_global_scale_inv = get_nvfp4_global_scale(inputs)
    weight_global_scale_inv = get_nvfp4_global_scale(weights)

    input_fp4, input_scale_swizzled = ops.scaled_fp4_quant(
        inputs,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    weight_fp4, weight_scale_swizzled = ops.scaled_fp4_quant(
        weights,
        weight_global_scale_inv,
        is_sf_swizzled_layout=True,
    )
    weight_scale_linear = convert_swizzled_to_linear(
        weight_scale_swizzled,
        output_size,
        input_size,
        16,
    ).contiguous()

    dequantized_inputs = dequantize_nvfp4_to_dtype(
        input_fp4,
        input_scale_swizzled,
        input_global_scale_inv,
        dtype=dtype,
        device=device,
        block_size=16,
        is_sf_128x4_layout=True,
    )
    dequantized_weights = dequantize_nvfp4_to_dtype(
        weight_fp4,
        weight_scale_swizzled,
        weight_global_scale_inv,
        dtype=dtype,
        device=device,
        block_size=16,
        is_sf_128x4_layout=True,
    )
    expected = torch.matmul(dequantized_inputs, dequantized_weights.t())

    layer = torch.nn.Module()
    layer.output_size_per_partition = output_size
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight_fp4, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(weight_scale_linear, requires_grad=False),
    )
    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(input_global_scale_inv, requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(
            1.0 / (input_global_scale_inv * weight_global_scale_inv),
            requires_grad=False,
        ),
    )

    kernel = FlashInferB12xNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    actual = kernel.apply_weights(layer, inputs)

    relative_error = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float())
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    print(
        f"shape={input_size}x{output_size} "
        f"relative_error={relative_error:.6f} cosine={cosine:.6f}"
    )
    assert relative_error < 0.025, f"relative_error={relative_error:.6f}"
    assert cosine > 0.999, f"cosine={cosine:.6f}"
