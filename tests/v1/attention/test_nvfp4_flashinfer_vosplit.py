# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the consumer Blackwell NVFP4 VO split."""

from types import SimpleNamespace

import pytest

try:
    import vllm.v1.attention.backends.flashinfer as flashinfer_backend
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferBackend,
        FlashInferMetadataBuilder,
        _vo_split_factor,
    )

    HAS_FLASHINFER = True
except Exception:
    HAS_FLASHINFER = False


pytestmark = pytest.mark.skipif(
    not HAS_FLASHINFER, reason="FlashInfer attention backend not importable"
)


@pytest.mark.parametrize(
    "head_size,is_nvfp4,expected",
    [
        (128, True, 1),
        (256, True, 1),
        (256, False, 1),
        (512, True, 2),
        (512, False, 2),
    ],
)
def test_vo_split_factor(head_size, is_nvfp4, expected, monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
    assert _vo_split_factor(head_size, is_nvfp4) == expected


def test_vo_split_factor_nvfp4_fails_closed_without_knob(monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "0")
    with pytest.raises(ValueError, match="two-pass VO split"):
        _vo_split_factor(512, True)


class _ConsumerBlackwellPlatform:
    def get_device_capability(self, device_id=0):
        return SimpleNamespace(major=12, minor=0)

    def is_device_capability(self, capability, device_id=0):
        return capability == 120

    def is_device_capability_family(self, capability, device_id=0):
        return capability == 120


def test_nvfp4_consumer_blackwell_requires_hnd_layout(monkeypatch):
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="nvfp4"))
    monkeypatch.setattr(
        flashinfer_backend, "current_platform", _ConsumerBlackwellPlatform()
    )
    monkeypatch.setattr(
        flashinfer_backend, "get_current_vllm_config_or_none", lambda: config
    )
    assert FlashInferBackend.get_required_kv_cache_layout() == "HND"


def test_non_nvfp4_consumer_blackwell_does_not_force_hnd(monkeypatch):
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
    monkeypatch.setattr(
        flashinfer_backend, "current_platform", _ConsumerBlackwellPlatform()
    )
    monkeypatch.setattr(
        flashinfer_backend, "get_current_vllm_config_or_none", lambda: config
    )
    assert FlashInferBackend.get_required_kv_cache_layout() is None


def test_nvfp4_consumer_blackwell_cudagraph_fails_closed_to_single_token(
    monkeypatch,
):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="nvfp4"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    monkeypatch.setattr(
        flashinfer_backend, "current_platform", _ConsumerBlackwellPlatform()
    )
    assert (
        FlashInferMetadataBuilder.get_cudagraph_support(config, None)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
