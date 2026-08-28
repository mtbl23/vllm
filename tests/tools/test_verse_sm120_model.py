# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools" / "verse"))

from prepare_sm120_model import (  # noqa: E402
    FORBIDDEN_HUGGING_FACE_ENVIRONMENT,
    OFFICIAL_HUGGING_FACE_ENDPOINT,
    download_model_snapshot,
    full_checksum_verification,
    load_token_file,
    materialize_model_directory,
    sha256_file,
    validate_ready_marker,
    verify_materialized_model_directory,
    verify_model_directory,
    write_ready_marker,
)
from validate_sm120_profile import EXPECTED_PROFILE  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_model(cache_dir: Path) -> tuple[Path, Path, dict]:
    snapshot_dir = cache_dir / "models--verse--campaign" / "snapshots" / ("a" * 40)
    model_dir = snapshot_dir / EXPECTED_PROFILE["VERSE_MODEL_SUBDIR"]
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "tie_word_embeddings": True,
        "quantization_config": {
            "quant_algo": "NVFP4",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "weights": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float",
                    },
                    "input_activations": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float",
                    },
                }
            },
        },
    }
    config_data = (json.dumps(config, sort_keys=True) + "\n").encode()
    payload_data = b"synthetic weights"
    (model_dir / "config.json").write_bytes(config_data)
    (model_dir / "model.safetensors").write_bytes(payload_data)
    manifest = {
        "kind": "campaign22_final_matched_modelopt_nvfp4_w4a4",
        "candidate": {
            "files": {
                "config.json": {
                    "bytes": len(config_data),
                    "sha256": _sha(config_data),
                },
                "model.safetensors": {
                    "bytes": len(payload_data),
                    "sha256": _sha(payload_data),
                },
            }
        },
    }
    (model_dir / "final_w4a4_build_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    return snapshot_dir, model_dir, manifest


def test_model_snapshot_full_verification_and_ready_marker(tmp_path: Path):
    snapshot_dir, model_dir, manifest = make_model(tmp_path)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}

    verify_model_directory(model_dir, tmp_path, manifest_sha)
    full_checksum_verification(model_dir, manifest)
    materialized, verification = materialize_model_directory(
        model_dir, tmp_path, profile
    )
    write_ready_marker(tmp_path, snapshot_dir, materialized, profile, verification)

    ready = validate_ready_marker(tmp_path, profile)
    assert ready["model_directory"] == str(materialized.resolve())
    assert ready["verification"]["files"] == 2
    assert all(not path.is_symlink() for path in materialized.rglob("*"))


def test_ready_marker_rejects_same_size_weight_mutation(tmp_path: Path):
    snapshot_dir, model_dir, manifest = make_model(tmp_path)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}
    verify_model_directory(model_dir, tmp_path, manifest_sha)
    full_checksum_verification(model_dir, manifest)
    materialized, verification = materialize_model_directory(
        model_dir, tmp_path, profile
    )
    write_ready_marker(tmp_path, snapshot_dir, materialized, profile, verification)

    weights = materialized / "model.safetensors"
    weights.chmod(0o644)
    weights.write_bytes(b"x" * weights.stat().st_size)
    weights.chmod(0o444)

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_ready_marker(tmp_path, profile)


def test_materialization_resolves_hugging_face_style_blob_symlink(tmp_path: Path):
    _, model_dir, manifest = make_model(tmp_path)
    weights = model_dir / "model.safetensors"
    blob = tmp_path / "blobs" / "model-blob"
    blob.parent.mkdir()
    weights.replace(blob)
    weights.symlink_to(blob)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}

    verify_model_directory(model_dir, tmp_path, manifest_sha)
    full_checksum_verification(model_dir, manifest)
    materialized, _ = materialize_model_directory(model_dir, tmp_path, profile)

    materialized_weights = materialized / "model.safetensors"
    assert materialized_weights.is_file()
    assert not materialized_weights.is_symlink()
    assert materialized_weights.read_bytes() == b"synthetic weights"


def test_materialized_model_rejects_unlisted_file(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}
    materialized, _ = materialize_model_directory(model_dir, tmp_path, profile)
    materialized.chmod(0o755)
    (materialized / "unlisted.json").write_text("{}\n")
    materialized.chmod(0o555)

    with pytest.raises(ValueError, match="file set does not match"):
        verify_materialized_model_directory(materialized, tmp_path, manifest_sha)


def test_materialized_model_rejects_untrusted_ancestor(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}
    materialized, _ = materialize_model_directory(model_dir, tmp_path, profile)
    parent = materialized.parent
    original_mode = parent.stat().st_mode & 0o777
    parent.chmod(0o775)
    try:
        with pytest.raises(ValueError, match="group/world writable"):
            verify_materialized_model_directory(
                materialized,
                tmp_path,
                manifest_sha,
                required_owner=os.getuid(),
            )
    finally:
        parent.chmod(original_mode)


def test_mounted_model_accepts_model_directory_as_trust_root(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")
    profile = {**EXPECTED_PROFILE, "VERSE_MODEL_MANIFEST_SHA256": manifest_sha}
    materialized, expected = materialize_model_directory(model_dir, tmp_path, profile)

    actual = verify_materialized_model_directory(
        materialized,
        materialized,
        manifest_sha,
        required_owner=os.getuid(),
    )

    assert actual == expected


def test_model_snapshot_rejects_w4a16_config(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["config_groups"]["group_0"]["input_activations"][
        "num_bits"
    ] = 16
    config_path.write_text(json.dumps(config) + "\n")
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")

    with pytest.raises(ValueError, match="W4A4"):
        verify_model_directory(model_dir, tmp_path, manifest_sha)


def test_model_snapshot_rejects_quantized_output_head(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"] = []
    config_path.write_text(json.dumps(config) + "\n")
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")

    with pytest.raises(ValueError, match="exclude lm_head"):
        verify_model_directory(model_dir, tmp_path, manifest_sha)


def test_model_snapshot_rejects_file_outside_cache(tmp_path: Path):
    _, model_dir, _ = make_model(tmp_path / "cache")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"synthetic weights")
    model_path = model_dir / "model.safetensors"
    model_path.unlink()
    model_path.symlink_to(outside)
    manifest_sha = sha256_file(model_dir / "final_w4a4_build_manifest.json")

    with pytest.raises(ValueError, match="escapes the cache"):
        verify_model_directory(model_dir, tmp_path / "cache", manifest_sha)


def test_hugging_face_token_must_be_owner_only_and_outside_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    token = tmp_path / "token"
    token.write_text("hf_test\n")
    token.chmod(0o600)

    assert load_token_file(token, cache) == "hf_test"

    token.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        load_token_file(token, cache)

    nested = cache / "token"
    nested.write_text("hf_nested\n")
    nested.chmod(0o600)
    with pytest.raises(ValueError, match="outside the model cache"):
        load_token_file(nested, cache)


@pytest.mark.parametrize(
    "environment_name",
    [
        "HF_ENDPOINT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "Https_Proxy",
    ],
)
def test_model_download_rejects_endpoint_and_proxy_environment_before_token_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_name: str
):
    expected_environment = {
        "HF_ENDPOINT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
    assert set(FORBIDDEN_HUGGING_FACE_ENVIRONMENT) == expected_environment
    for name in list(os.environ):
        if name.upper() in expected_environment:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment_name, "")
    called = False

    def snapshot_download(**_: object) -> str:
        nonlocal called
        called = True
        return str(tmp_path / "unexpected")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    with pytest.raises(ValueError, match=environment_name):
        download_model_snapshot(EXPECTED_PROFILE, tmp_path, "hf_secret")

    assert called is False


def test_model_download_pins_official_hugging_face_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    forbidden = {name.upper() for name in FORBIDDEN_HUGGING_FACE_ENVIRONMENT}
    for name in list(os.environ):
        if name.upper() in forbidden:
            monkeypatch.delenv(name, raising=False)
    snapshot = tmp_path / "snapshot"
    received: dict[str, object] = {}

    def snapshot_download(**kwargs: object) -> str:
        received.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    result = download_model_snapshot(EXPECTED_PROFILE, tmp_path, "hf_secret")

    assert result == snapshot
    assert received["endpoint"] == OFFICIAL_HUGGING_FACE_ENDPOINT
    assert received["endpoint"] == "https://huggingface.co"
    assert received["token"] == "hf_secret"
