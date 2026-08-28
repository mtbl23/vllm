# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (no-GPU) routing check for Gemma 4 NVFP4 KV -> FLASHINFER.

On exact SM120 the FlashInfer FA2 asymmetric paged
kernel serves Gemma 4 heterogeneous-head full-attention layers
(head_dim_qk=512, head_dim_vo=256) directly, so Gemma4Config must route
NVFP4-KV configs to FLASHINFER instead of the TRITON_ATTN fallback.

Runs under a mocked platform/capability; no CUDA required. Mocking
pattern adapted from the campaign triton-retirement selection test.
"""

from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

ALL_KNOBS = (
    "VLLM_KV_CACHE_LAYOUT",
    "VLLM_NVFP4_KV_VOSPLIT",
    "VLLM_VERSE_RUNTIME_STRICT",
    "VLLM_VERSE_NVFP4_XQA_DECODE",
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
    "VLLM_BATCH_INVARIANT",
)

CC12_0 = DeviceCapability(12, 0)
CC12_1 = DeviceCapability(12, 1)
CC9_0 = DeviceCapability(9, 0)


@pytest.fixture(autouse=True)
def _clear_knobs(monkeypatch):
    for name in ALL_KNOBS:
        monkeypatch.delenv(name, raising=False)
    yield


class _FakeArchConfig:
    """Minimal stand-in for ``model_config.model_arch_config``.

    Gemma4Config reads per-layer ``head_size`` by index and the total layer
    count, so that is all this needs to expose.
    """

    def __init__(self, head_sizes):
        self._head_sizes = head_sizes
        self.total_num_hidden_layers = len(head_sizes)

    def __getitem__(self, index):
        return SimpleNamespace(head_size=self._head_sizes[index])


_LAYER_CYCLE = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]
_LAYER_TYPES = _LAYER_CYCLE * 8


def _mock_vllm_config(
    *,
    backend=None,
    backend_per_kind=None,
    cache_dtype="nvfp4",
    kv_cache_dtype_skip_layers=None,
    head_dim=256,
    global_head_dim=512,
    quantization="modelopt_fp4",
    layer_types=None,
    language_model_only=True,
    max_model_len=6144,
    max_num_seqs=38,
    max_num_batched_tokens=256,
    enforce_eager=False,
    linear_backend="flashinfer_b12x",
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    cudagraph_capture_sizes=None,
    max_cudagraph_capture_size=38,
    async_scheduling=False,
    speculative=False,
    tensor_parallel_size=1,
    kv_offloading_size=None,
    sliding_window=1024,
    disable_sliding_window=False,
):
    layer_types = list(layer_types or _LAYER_TYPES)
    head_sizes = [
        global_head_dim if lt == "full_attention" else head_dim for lt in layer_types
    ]
    capture_sizes = (
        [1, 8, 16, 24, 32, 38]
        if cudagraph_capture_sizes is None
        else cudagraph_capture_sizes
    )
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=backend,
            backend_per_kind=backend_per_kind or {},
            flash_attn_version=None,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            kv_cache_dtype_skip_layers=kv_cache_dtype_skip_layers or [],
            kv_offloading_size=kv_offloading_size,
            block_size=16,
            kv_cache_memory_bytes=5_704_253_440,
            gpu_memory_utilization=0.94,
            enable_prefix_caching=True,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=async_scheduling,
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=SimpleNamespace() if speculative else None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
        ),
        kernel_config=SimpleNamespace(linear_backend=linear_backend),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            model_arch_config=_FakeArchConfig(head_sizes),
            hf_text_config=SimpleNamespace(
                layer_types=layer_types,
                sliding_window=sliding_window,
            ),
            quantization=quantization,
            multimodal_config=SimpleNamespace(language_model_only=language_model_only),
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            disable_sliding_window=disable_sliding_window,
        ),
    )


class _FakePlatform:
    """Delegates everything to the real current_platform except the
    compute-capability accessors, which are pinned to ``capability``."""

    def __init__(self, capability):
        self._cap = capability
        from vllm.platforms import current_platform as real

        self._real = real

    def is_cuda(self):
        return True

    def get_device_capability(self, device_id=0):
        return self._cap

    def is_device_capability_family(self, cap, device_id=0):
        return (self._cap.to_int() // 10) == (cap // 10)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def fake_cc(monkeypatch):
    def _set(capability):
        import vllm.platforms as platforms_mod
        import vllm.v1.attention.backends.fa_utils as fa_utils_mod

        fake = _FakePlatform(capability)
        monkeypatch.setattr(platforms_mod, "current_platform", fake, raising=False)
        monkeypatch.setattr(fa_utils_mod, "is_fa_version_supported", lambda v: False)
        return fake

    return _set


def _gemma4_route(vllm_config):
    from vllm.model_executor.models.config import Gemma4Config

    Gemma4Config.verify_and_update_config(vllm_config)
    return vllm_config.attention_config.backend


def _enable_strict_verse(monkeypatch):
    monkeypatch.setenv("VLLM_VERSE_RUNTIME_STRICT", "1")
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
    monkeypatch.setenv("VLLM_VERSE_NVFP4_XQA_DECODE", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(64 * 1024 * 1024))


@pytest.mark.parametrize("capability", [CC12_0, CC12_1])
def test_nvfp4_cc12_does_not_enable_verse_route_by_default(fake_cc, capability):
    fake_cc(capability)
    cfg = _mock_vllm_config()
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_nvfp4_cc12_opt_in_without_strict_profile_does_not_route(fake_cc, monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
    fake_cc(CC12_0)
    cfg = _mock_vllm_config()
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_nvfp4_hopper_does_not_route(fake_cc):
    fake_cc(CC9_0)
    cfg = _mock_vllm_config()
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_nvfp4_explicit_user_backend_wins(fake_cc):
    fake_cc(CC12_0)
    cfg = _mock_vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_bf16_kv_keeps_triton_fallback(fake_cc):
    fake_cc(CC12_0)
    cfg = _mock_vllm_config(cache_dtype="auto")
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_strict_verse_runtime_routes_supported_tuple(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    fake_cc(CC12_0)
    assert _gemma4_route(_mock_vllm_config()) == AttentionBackendEnum.FLASHINFER


@pytest.mark.parametrize(
    "capability,cache_dtype,backend,head_dim,global_head_dim,error",
    [
        (CC9_0, "nvfp4", None, 256, 512, "exact compute capability SM120"),
        (CC12_1, "nvfp4", None, 256, 512, "exact compute capability SM120"),
        (CC12_0, "auto", None, 256, 512, "kv-cache-dtype nvfp4"),
        (
            CC12_0,
            "nvfp4",
            AttentionBackendEnum.TRITON_ATTN,
            256,
            512,
            "FLASHINFER attention backend",
        ),
        (CC12_0, "nvfp4", None, 128, 512, "48-layer pattern"),
        (CC12_0, "nvfp4", None, 256, 256, "48-layer pattern"),
    ],
)
def test_strict_verse_runtime_rejects_unsupported_tuple(
    fake_cc,
    monkeypatch,
    capability,
    cache_dtype,
    backend,
    head_dim,
    global_head_dim,
    error,
):
    _enable_strict_verse(monkeypatch)
    fake_cc(capability)
    cfg = _mock_vllm_config(
        backend=backend,
        cache_dtype=cache_dtype,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
    )
    with pytest.raises(ValueError, match=error):
        _gemma4_route(cfg)


def test_strict_verse_runtime_requires_vo_split(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "0")
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match="VLLM_NVFP4_KV_VOSPLIT=1"):
        _gemma4_route(_mock_vllm_config())


@pytest.mark.parametrize(
    "sliding_window,disable_sliding_window,error",
    [
        (None, False, "sliding_window=1024"),
        (2048, False, "sliding_window=1024"),
        (1024, True, "remain enabled"),
    ],
)
def test_strict_verse_runtime_pins_sliding_attention_mask(
    fake_cc,
    monkeypatch,
    sliding_window,
    disable_sliding_window,
    error,
):
    _enable_strict_verse(monkeypatch)
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match=error):
        _gemma4_route(
            _mock_vllm_config(
                sliding_window=sliding_window,
                disable_sliding_window=disable_sliding_window,
            )
        )


def test_strict_verse_runtime_requires_nvfp4_weights(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        _gemma4_route(_mock_vllm_config(quantization="compressed-tensors"))


def test_strict_verse_runtime_requires_text_only_loading(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match="text-only loading"):
        _gemma4_route(_mock_vllm_config(language_model_only=False))


def test_strict_verse_runtime_requires_hnd_layout(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match="VLLM_KV_CACHE_LAYOUT=HND"):
        _gemma4_route(_mock_vllm_config())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"max_model_len": 8192}, "max_model_len=6144"),
        ({"max_num_seqs": 40}, "max_num_seqs=38"),
        ({"max_num_batched_tokens": 512}, "max_num_batched_tokens=256"),
        ({"async_scheduling": True}, "synchronous B01"),
        ({"speculative": True}, "speculative decoding"),
        ({"tensor_parallel_size": 2}, "TP1/PP1/DP1/DCP1"),
        ({"kv_offloading_size": 8.0}, "KV offload"),
        ({"enforce_eager": True}, "graph-enabled"),
        ({"linear_backend": "auto"}, "linear-backend flashinfer_b12x"),
        ({"cudagraph_mode": CUDAGraphMode.NONE}, "FULL_DECODE_ONLY"),
        ({"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 38]}, "capture sizes"),
        ({"max_cudagraph_capture_size": 64}, "max capture size 38"),
        (
            {"kv_cache_dtype_skip_layers": ["sliding_attention"]},
            "dtype skip layers",
        ),
        (
            {"backend_per_kind": {"sliding_window_attention": "TRITON_ATTN"}},
            "per-kind attention backend",
        ),
        ({"cache_dtype": "nvfp4_4over6"}, "kv-cache-dtype nvfp4"),
        ({"layer_types": _LAYER_TYPES[:-1]}, "48-layer pattern"),
    ],
)
def test_strict_verse_runtime_requires_fixed_serving_tuple(
    fake_cc, monkeypatch, overrides, error
):
    _enable_strict_verse(monkeypatch)
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match=error):
        _gemma4_route(_mock_vllm_config(**overrides))


def test_strict_verse_runtime_rejects_batch_invariant_override(fake_cc, monkeypatch):
    _enable_strict_verse(monkeypatch)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    fake_cc(CC12_0)
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
        _gemma4_route(_mock_vllm_config())
