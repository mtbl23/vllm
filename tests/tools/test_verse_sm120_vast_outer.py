# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "tools" / "verse"
sys.path.insert(0, str(TOOLS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _load(
    "sm120_release_fixtures", Path(__file__).with_name("test_verse_sm120_release.py")
)
outer = _load(
    "finalize_sm120_vast_outer_release",
    TOOLS / "finalize_sm120_vast_outer_release.py",
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload))


def _live() -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "pid": 4242,
        "uid": 2000,
        "gid": 0,
        "executable_sha256": "3" * 64,
        "command_sha256": "4" * 64,
        "process_start_ticks": 12345,
        "process_started_at_unix": 1_787_000_000.0,
        "boot_id": fixtures.BOOT_ID,
        "machine_id_sha256": fixtures.MACHINE_ID_SHA256,
        "gpu": fixtures.GPU,
        "fork_commit": fixtures.FORK_COMMIT,
        "wheel_version": f"0.28.0+verse.{fixtures.FORK_COMMIT[:12]}",
        "model_revision": fixtures.MODEL_REVISION,
    }


def _provider() -> dict:
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "instance": {
            "id": 123456,
            "machine_id": 98765,
            "image_uuid": fixtures.IMAGE_DIGEST,
            "gpu_name": "RTX 5070 Ti",
            "num_gpus": 1,
            "actual_status": "running",
            "intended_status": "running",
            "cur_state": "running",
            "ssh_host": "ssh.example.test",
            "ssh_port": 22022,
            "start_date": 1_787_000_000.0,
        },
    }


def _runtime_id(provider: dict, live: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "provider": outer.provider_identity(provider, fixtures.IMAGE_DIGEST),
                "live": outer.live_identity(live, fixtures.FORK_COMMIT),
                "ssh": "SHA256:test-fingerprint",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _release_tree(tmp_path: Path) -> Path:
    core = fixtures.release_tree(tmp_path / "core")
    release = tmp_path / "outer"
    release.mkdir()
    for path in (core / "short").iterdir():
        if (
            path.name.startswith(("b01-", "prefill-"))
            or path.name == "chat-contract.json"
        ):
            shutil.copyfile(path, release / path.name)
    for name in (
        "queue-stress.json",
        "user-latency.json",
        "warm-latency.json",
        "churn.json",
        "post-churn-chat-contract.json",
        "image-receipt.json",
    ):
        shutil.copyfile(core / name, release / name)

    live = _live()
    provider = _provider()
    runtime_id = _runtime_id(provider, live)
    for name in (
        "chat-contract.json",
        "queue-stress.json",
        "user-latency.json",
        "warm-latency.json",
        "churn.json",
        "post-churn-chat-contract.json",
    ):
        path = release / name
        payload = json.loads(path.read_text())
        payload["container_id"] = runtime_id
        _write(path, payload)

    cuda_log = fixtures.cuda_log().replace(
        f"VERSE_CUDA_GPU_IDENTITY={fixtures.GPU_IDENTITY}",
        f"VERSE_CUDA_GPU_IDENTITY={fixtures.GPU_IDENTITY}, 12.0",
    )
    cuda_log += "SM120_HOSTED_CUDA_GATES_PASSED\n"
    _write(release / "cuda-oracle.log", cuda_log)
    _write(release / "image-verification.json", fixtures.cuda_image_verification())
    _write(release / "provider-before.json", provider)
    _write(release / "provider-after.json", provider)
    _write(release / "live-process-before.json", live)
    _write(release / "live-process-after.json", live)
    _write(release / "ssh-host-fingerprint.txt", "SHA256:test-fingerprint\n")
    _write(release / "ssh-known-hosts", "[ssh.example.test]:22022 ssh-ed25519 test\n")
    _write(release / "image.txt", fixtures.IMAGE_DIGEST + "\n")
    _write(release / "commit.txt", fixtures.FORK_COMMIT + "\n")
    _write(release / "attestation-verification.json", fixtures.image_attestation())
    _write(
        release / "qualification-tools.json",
        {
            "schema_version": 1,
            "sha256": {
                relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in outer.EXPECTED_QUALIFICATION_TOOLS
            },
        },
    )
    _write(
        release / "auth-proof.json",
        {
            "status": "pass",
            "unauthenticated": {
                "/invocations": 401,
                "/tokenize": 401,
                "/docs": 401,
                "/openapi.json": 401,
                "/v1/models": 401,
            },
            "liveness": {"/health": 200, "/metrics": 200},
        },
    )
    _write(
        release / "server.log",
        "\n".join(
            (
                "Strict Verse Gemma 4 runtime validated",
                "Using the FlashInfer FA2 paged wrapper",
                "head_dim=512, page_size=64, speculative_tokens=0, xqa=True",
            )
        ),
    )
    return release


def test_outer_finalizer_accepts_one_immutable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(outer, "IMAGE_RE", re.compile(r".+@sha256:[0-9a-f]{64}"))
    result = outer.finalize(_release_tree(tmp_path))

    assert result["status"] == "pass"
    assert result["scope"] == "disposable_image_qualification"
    assert result["b01"]["status"] == "pass"
    assert result["image_attestation"]["source_commit"] == fixtures.FORK_COMMIT


def test_outer_finalizer_rejects_live_process_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(outer, "IMAGE_RE", re.compile(r".+@sha256:[0-9a-f]{64}"))
    release = _release_tree(tmp_path)
    path = release / "live-process-after.json"
    payload = json.loads(path.read_text())
    payload["pid"] += 1
    _write(path, payload)

    with pytest.raises(ValueError, match="server process, host, or GPU changed"):
        outer.finalize(release)


def test_outer_finalizer_rejects_untracked_qualification_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(outer, "IMAGE_RE", re.compile(r".+@sha256:[0-9a-f]{64}"))
    release = _release_tree(tmp_path)
    path = release / "qualification-tools.json"
    payload = json.loads(path.read_text())
    payload["sha256"]["tools/verse/run_sm120_churn.py"] = "0" * 64
    _write(path, payload)

    with pytest.raises(ValueError, match="tools differ from the release source"):
        outer.finalize(release)


def test_outer_finalizer_rejects_unexpected_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(outer, "IMAGE_RE", re.compile(r".+@sha256:[0-9a-f]{64}"))
    release = _release_tree(tmp_path)
    _write(release / "raw-output.json", {"content": "must not be archived"})

    with pytest.raises(ValueError, match="artifact inventory drifted"):
        outer.finalize(release)
