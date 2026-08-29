# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
VALIDATOR = ROOT / "tools" / "verse" / "validate_sm120_container.py"
RUNNER = ROOT / "tools" / "verse" / "run_sm120_server.sh"
CHECKER = ROOT / "tools" / "verse" / "check_sm120_server.sh"
IMAGE_VERIFIER = ROOT / "tools" / "verse" / "verify_sm120_image.py"
IMAGE_RECEIPT = ROOT / "tools" / "verse" / "sm120_image_receipt.py"
WHEEL_IDENTITY = ROOT / "tools" / "verse" / "build_sm120_wheel_identity.py"
COMMIT = "2" * 40
CONTAINER_ID = "c" * 64
IMAGE = f"registry/runtime@sha256:{'a' * 64}"
REVISION = "e2c6cd9c3302e91c032a378a607009c82ba16fac"
GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
IMAGE_RECEIPT_SHA256 = "d" * 64
MANIFEST_SHA256 = hashlib.sha256(
    (ROOT / "requirements/verse-sm120-flashinfer.lock").read_bytes()
).hexdigest()


@pytest.fixture(scope="module")
def image_verifier():
    spec = importlib.util.spec_from_file_location(
        "verse_sm120_image_verifier", IMAGE_VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def image_receipt():
    spec = importlib.util.spec_from_file_location(
        "verse_sm120_image_receipt", IMAGE_RECEIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wheel_identity():
    spec = importlib.util.spec_from_file_location(
        "verse_sm120_wheel_identity", WHEEL_IDENTITY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_vllm_wheel_requirements(image_verifier) -> list[str]:
    flashinfer = (
        "flashinfer-python @ " + image_verifier.EXPECTED_FLASHINFER_REQUIREMENT_URL
    )
    return [
        flashinfer,
        "nvidia-cutlass-dsl[cu13]==4.7.0",
        "nvidia-cudnn-frontend==1.27.0",
    ]


def base_container(tmp_path: Path) -> tuple[dict, list[str]]:
    runtime = tmp_path / "runtime"
    model = tmp_path / "model"
    secret = tmp_path / "secret"
    for path in (runtime, model):
        path.mkdir()
    model_directory = model / "verified-model"
    model_directory.mkdir()
    secret.write_text("secret\n")
    secret.chmod(0o600)
    (runtime / ".verse-sm120-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": IMAGE,
                "fork_commit": COMMIT,
                "profile": "sm120-gemma4-nvfp4-v4",
                "gpu_device": "0",
                "gpu_uuid": GPU_UUID,
                "image_receipt_sha256": IMAGE_RECEIPT_SHA256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    model_path = "/models/model"
    command = [
        model_path,
        "--served-model-name",
        "verse-free",
        "--quantization",
        "modelopt_fp4",
        "--dtype",
        "bfloat16",
        "--linear-backend",
        "flashinfer_b12x",
        "--max-model-len",
        "6144",
        "--block-size",
        "16",
        "--max-num-seqs",
        "38",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.94",
        "--kv-cache-memory-bytes",
        "5704253440",
        "--kv-cache-dtype",
        "nvfp4",
        "--attention-backend",
        "FLASHINFER",
        "--enable-prefix-caching",
        "--no-disable-hybrid-kv-cache-manager",
        "--no-async-scheduling",
        "--language-model-only",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--generation-config",
        "vllm",
        "--compilation-config",
        (
            '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
            '"cudagraph_capture_sizes":[1,8,16,24,32,38],'
            '"max_cudagraph_capture_size":38}'
        ),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    container = {
        "Id": CONTAINER_ID,
        "Image": "sha256:image-id",
        "RestartCount": 0,
        "State": {
            "Running": True,
            "OOMKilled": False,
            "Error": "",
            "StartedAt": "2026-08-28T00:00:00Z",
        },
        "Config": {
            "Image": IMAGE,
            "Labels": {
                "ai.vllm.build.commit": COMMIT,
                "ai.verse.vllm.wheel.version": f"0.28.0+verse.{COMMIT[:12]}",
                "ai.verse.runtime.profile": "sm120-gemma4-nvfp4-v4",
                "ai.verse.gpu.ordinal": "0",
                "ai.verse.gpu.uuid": GPU_UUID,
                "ai.verse.flashinfer.release": "0.6.18.dev20260819",
                "ai.verse.flashinfer.manifest.sha256": MANIFEST_SHA256,
                "ai.verse.base.build": (
                    "pytorch/manylinux2_28-builder:cuda13.0@sha256:"
                    "7710cbc19d7ee951134e2e827f8ec89237c993095eb2581dd5e74f58e4e278c7"
                ),
                "ai.verse.base.runtime": (
                    "nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:"
                    "97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4"
                ),
            },
            "Entrypoint": ["/usr/local/bin/verse-sm120-entrypoint.sh"],
            "User": "2000:0",
            "Cmd": command,
            "Env": [
                "VLLM_API_KEY_FILE=/run/secrets/vllm_api_key",
                f"VERSE_VLLM_WHEEL_VERSION=0.28.0+verse.{COMMIT[:12]}",
                "UV_OVERRIDE=/etc/uv-overrides-verse-sm120.txt",
                "VLLM_VERSE_RUNTIME_STRICT=1",
                "VLLM_NVFP4_KV_VOSPLIT=1",
                "VLLM_VERSE_NVFP4_XQA_DECODE=1",
                "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864",
                "VLLM_KV_CACHE_LAYOUT=HND",
                "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0",
                "VLLM_USE_FLASHINFER_SAMPLER=0",
                "VLLM_MAX_N_SEQUENCES=1",
                "VLLM_MAX_COMPLETION_PROMPTS=1",
                "VLLM_MAX_STOP_STRINGS=4",
                "FLASHINFER_WORKSPACE_BASE=/cache/flashinfer",
                "CUDA_CACHE_PATH=/cache/cuda",
                "TORCH_HOME=/cache/torch",
                "TORCH_EXTENSIONS_DIR=/cache/torch-extensions",
                "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
                "HF_HOME=/cache/huggingface",
                "VLLM_CACHE_ROOT=/cache/vllm",
                "TRITON_CACHE_DIR=/cache/triton",
                "XDG_CACHE_HOME=/cache/xdg",
                "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer-autotune",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            ],
        },
        "HostConfig": {
            "Privileged": False,
            "CapAdd": [],
            "CapDrop": ["ALL"],
            "IpcMode": "private",
            "PidMode": "",
            "NetworkMode": "default",
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges"],
            "RestartPolicy": {"Name": "unless-stopped"},
            "Tmpfs": {"/tmp": "rw,nosuid,size=4g"},
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "DeviceIDs": [GPU_UUID],
                    "Capabilities": [["gpu"]],
                }
            ],
        },
        "Mounts": [
            {
                "Destination": "/cache",
                "Source": str(runtime),
                "Type": "bind",
                "RW": True,
            },
            {
                "Destination": "/models/model",
                "Source": str(model_directory),
                "Type": "bind",
                "RW": False,
            },
            {
                "Destination": "/run/secrets/vllm_api_key",
                "Source": str(secret),
                "Type": "bind",
                "RW": False,
            },
        ],
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}
        },
    }
    args = [
        "--container-id",
        CONTAINER_ID,
        "--image",
        IMAGE,
        "--expected-commit",
        COMMIT,
        "--served-model",
        "verse-free",
        "--gpu-device",
        "0",
        "--gpu-uuid",
        GPU_UUID,
        "--runtime-cache",
        str(runtime),
        "--model-cache",
        str(model),
        "--model-directory",
        str(model_directory),
        "--api-key-file",
        str(secret),
        "--image-receipt-sha256",
        IMAGE_RECEIPT_SHA256,
    ]
    return container, args


def run_validator(container: dict, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        input=json.dumps([container]),
        capture_output=True,
        text=True,
    )


def run_gpu_guards(
    tmp_path: Path, identity: str, compute_processes: str
) -> subprocess.CompletedProcess[str]:
    source = RUNNER.read_text()
    start = source.index("verify_gpu_identity()")
    end = source.index("\nfor broad_path", start)
    guards = source[start:end]
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *--query-gpu=index,uuid* ]]; then
  printf '%s\n' "$FAKE_GPU_IDENTITY"
  exit 0
fi
if [[ "$*" == *--query-compute-apps=pid* ]]; then
  printf '%s' "$FAKE_GPU_COMPUTE_PROCESSES"
  exit 0
fi
exit 2
"""
    )
    nvidia_smi.chmod(0o755)
    expected_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    script = f"""set -euo pipefail
GPU_DEVICE=0
GPU_UUID={expected_uuid}
{guards}
verify_gpu_identity
require_idle_gpu
"""
    return subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_GPU_IDENTITY": identity,
            "FAKE_GPU_COMPUTE_PROCESSES": compute_processes,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_container_accepts_exact_contract(tmp_path: Path):
    container, args = base_container(tmp_path)
    result = run_validator(container, args)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["host_port"] == 8001


def test_container_rejects_mismatched_expected_container_id(tmp_path: Path):
    container, args = base_container(tmp_path)
    args[args.index("--container-id") + 1] = "d" * 64

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "container ID does not match" in result.stderr


def test_container_rejects_second_port_binding(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["NetworkSettings"]["Ports"]["8000/tcp"].append(
        {"HostIp": "0.0.0.0", "HostPort": "8002"}
    )

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "exactly one port binding" in result.stderr


def test_container_rejects_api_key_in_config(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Env"].append("VLLM_API_KEY=leaked")

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "API key is exposed" in result.stderr


@pytest.mark.parametrize(
    "environment_entry",
    (
        "FLASHINFER_CUBIN_DIR=/tmp/cubins",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "VLLM_BATCH_INVARIANT=0",
        "VLLM_BATCH_INVARIANT=1",
        "VLLM_DISABLED_KERNELS=FlashInferB12xNvFp4LinearKernel",
        "VLLM_SERVER_DEV_MODE=1",
        "VLLM_TEST_FORCE_FP8_MARLIN=1",
    ),
)
def test_container_rejects_backend_changing_environment(
    tmp_path: Path, environment_entry: str
):
    container, args = base_container(tmp_path)
    container["Config"]["Env"].append(environment_entry)

    result = run_validator(container, args)

    name = environment_entry.split("=", 1)[0]
    assert result.returncode != 0
    assert f"forbidden runtime environment: {name}" in result.stderr


def test_container_rejects_duplicate_environment_name(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Env"].append("VLLM_VERSE_RUNTIME_STRICT=0")

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "duplicate runtime environment: VLLM_VERSE_RUNTIME_STRICT" in result.stderr


def test_container_accepts_unrelated_inherited_environment(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Env"].append("NVIDIA_VISIBLE_DEVICES=all")

    result = run_validator(container, args)

    assert result.returncode == 0, result.stderr


def test_image_verifier_accepts_exact_unconditional_wheel_requirements(
    image_verifier,
):
    requirements = exact_vllm_wheel_requirements(image_verifier)

    verified = image_verifier.verify_vllm_wheel_requirements(
        SimpleNamespace(requires=requirements)
    )

    assert set(verified) == {
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cudnn-frontend",
    }


def test_image_verifier_records_exact_native_and_wheel_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_verifier
):
    vllm_root = tmp_path / "site-packages" / "vllm"
    vllm_root.mkdir(parents=True)
    native = vllm_root / "_C_stable_libtorch.abi3.so"
    native.write_bytes(b"exact-native-extension")
    wheel_manifest = tmp_path / "vllm-wheel.json"
    wheel_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wheel": {"filename": "vllm-test.whl", "sha256": "a" * 64},
                "native_extension": {
                    "member": "vllm/_C_stable_libtorch.abi3.so",
                    "bytes": native.stat().st_size,
                    "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                },
            }
        )
    )
    monkeypatch.setattr(
        image_verifier.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(native)),
    )

    identity = image_verifier.verify_vllm_binary_identity(vllm_root, wheel_manifest)

    assert identity["native_extension"] == {
        "path": str(native),
        "wheel_member": "vllm/_C_stable_libtorch.abi3.so",
        "bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    assert identity["wheel_artifact"] == {
        "filename": "vllm-test.whl",
        "sha256": "a" * 64,
        "manifest_sha256": hashlib.sha256(wheel_manifest.read_bytes()).hexdigest(),
    }


def test_wheel_identity_binds_native_extension_bytes(tmp_path: Path, wheel_identity):
    wheel = tmp_path / "vllm-test.whl"
    native = b"exact-native-extension"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vllm/_C_stable_libtorch.abi3.so", native)

    identity = wheel_identity.build_identity(wheel)

    assert identity["wheel"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert identity["native_extension"] == {
        "member": "vllm/_C_stable_libtorch.abi3.so",
        "bytes": len(native),
        "sha256": hashlib.sha256(native).hexdigest(),
    }


def test_image_verifier_rejects_native_unrelated_to_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_verifier
):
    vllm_root = tmp_path / "site-packages" / "vllm"
    vllm_root.mkdir(parents=True)
    native = vllm_root / "_C_stable_libtorch.abi3.so"
    native.write_bytes(b"substituted-native-extension")
    manifest = tmp_path / "vllm-wheel.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wheel": {"filename": "vllm-test.whl", "sha256": "a" * 64},
                "native_extension": {
                    "member": "vllm/_C_stable_libtorch.abi3.so",
                    "bytes": len(b"expected-native-extension"),
                    "sha256": hashlib.sha256(b"expected-native-extension").hexdigest(),
                },
            }
        )
    )
    monkeypatch.setattr(
        image_verifier.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(native)),
    )

    with pytest.raises(SystemExit, match="does not match the declared wheel"):
        image_verifier.verify_vllm_binary_identity(vllm_root, manifest)


def test_external_image_receipt_binds_digest_and_runtime_binary(
    tmp_path: Path, image_receipt
):
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "status": "valid",
                "vllm_binary_identity": {
                    "wheel_artifact": {
                        "filename": "vllm-test.whl",
                        "sha256": "1" * 64,
                        "manifest_sha256": "2" * 64,
                    },
                    "native_extension": {
                        "wheel_member": "vllm/_C_stable_libtorch.abi3.so",
                        "sha256": "3" * 64,
                    },
                },
            }
        )
    )
    output = tmp_path / "receipt.json"
    args = SimpleNamespace(
        output=output.resolve(),
        verification=verification.resolve(),
        image=IMAGE,
        fork_commit=COMMIT,
        runtime_profile="sm120-gemma4-nvfp4-v4",
        source_archive_sha256="4" * 64,
        vllm_wheel_version=f"0.28.0+verse.{COMMIT[:12]}",
    )
    image_receipt.create_receipt(args)
    args.receipt = output.resolve()

    image_receipt.verify_receipt(args)
    payload = json.loads(output.read_text())
    payload["binary_identity"]["native_extension_sha256"] = "5" * 64
    output.write_text(json.dumps(payload))
    output.chmod(0o600)

    with pytest.raises(ValueError, match="runtime binary identity differs"):
        image_receipt.verify_receipt(args)


@pytest.mark.parametrize(
    "wheel_entry",
    (
        "../vllm-test.whl",
        "dist/subdir/vllm-test.whl",
        "dist/not-a-wheel.txt",
    ),
)
def test_image_verifier_rejects_noncanonical_wheel_identity_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_verifier,
    wheel_entry: str,
):
    vllm_root = tmp_path / "site-packages" / "vllm"
    vllm_root.mkdir(parents=True)
    native = vllm_root / "_C_stable_libtorch.abi3.so"
    native.write_bytes(b"exact-native-extension")
    wheel_manifest = tmp_path / "vllm-wheel.json"
    wheel_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "wheel": {"filename": wheel_entry, "sha256": "a" * 64},
                "native_extension": {
                    "member": "vllm/_C_stable_libtorch.abi3.so",
                    "bytes": native.stat().st_size,
                    "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                },
            }
        )
    )
    monkeypatch.setattr(
        image_verifier.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(native)),
    )

    with pytest.raises(SystemExit, match="wheel identity manifest is malformed"):
        image_verifier.verify_vllm_binary_identity(vllm_root, wheel_manifest)


@pytest.mark.parametrize(
    "requirement_index",
    (0, 1, 2),
    ids=("flashinfer", "cutlass", "cudnn"),
)
def test_image_verifier_rejects_markers_on_exact_wheel_requirements(
    image_verifier, requirement_index: int
):
    requirements = exact_vllm_wheel_requirements(image_verifier)
    separator = " ; " if " @ " in requirements[requirement_index] else "; "
    requirements[requirement_index] += f"{separator}python_version < '0'"

    with pytest.raises(SystemExit, match="wheel requirement must be unconditional"):
        image_verifier.verify_vllm_wheel_requirements(
            SimpleNamespace(requires=requirements)
        )


@pytest.mark.parametrize(
    "environment_name",
    (
        "FLASHINFER_CUBIN_DIR",
        "FLASHINFER_DISABLE_VERSION_CHECK",
        "VLLM_BATCH_INVARIANT",
        "VLLM_DISABLED_KERNELS",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_TEST_FORCE_FP8_MARLIN",
    ),
)
def test_image_verifier_rejects_backend_changing_environment(
    image_verifier, environment_name: str
):
    message = f"forbidden runtime environment: {environment_name}"
    with pytest.raises(SystemExit, match=message):
        image_verifier.verify_runtime_environment({environment_name: ""})


def test_image_verifier_accepts_unrelated_environment(image_verifier):
    image_verifier.verify_runtime_environment({"NVIDIA_VISIBLE_DEVICES": "all"})


def test_container_rejects_vllm_wheel_version_drift(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Labels"]["ai.verse.vllm.wheel.version"] = (
        "0.28.0+verse.ffffffffffff"
    )

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "wrong vLLM wheel version label" in result.stderr


def test_container_rejects_command_drift(tmp_path: Path):
    container, args = base_container(tmp_path)
    index = container["Config"]["Cmd"].index("38")
    container["Config"]["Cmd"][index] = "40"

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "command does not exactly match" in result.stderr


def test_container_rejects_restart(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["RestartCount"] = 1

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "has restarted" in result.stderr


def test_container_rejects_shared_host_ipc(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["HostConfig"]["IpcMode"] = "host"

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "host IPC" in result.stderr


def test_container_rejects_root_runtime_user(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["User"] = ""

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "UID 2000" in result.stderr


def test_container_rejects_reused_runtime_cache_identity(tmp_path: Path):
    container, args = base_container(tmp_path)
    runtime = Path(args[args.index("--runtime-cache") + 1])
    marker = runtime / ".verse-sm120-runtime.json"
    payload = json.loads(marker.read_text())
    payload["fork_commit"] = "3" * 40
    marker.write_text(json.dumps(payload) + "\n")

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "runtime cache identity" in result.stderr


def test_container_rejects_gpu_uuid_marker_drift(tmp_path: Path):
    container, args = base_container(tmp_path)
    runtime = Path(args[args.index("--runtime-cache") + 1])
    marker = runtime / ".verse-sm120-runtime.json"
    payload = json.loads(marker.read_text())
    payload["gpu_uuid"] = "GPU-11111111-2222-3333-4444-555555555555"
    marker.write_text(json.dumps(payload) + "\n")

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "runtime cache identity" in result.stderr


def test_container_rejects_gpu_identity_label_drift(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Labels"]["ai.verse.gpu.uuid"] = (
        "GPU-11111111-2222-3333-4444-555555555555"
    )

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "wrong GPU UUID label" in result.stderr


def test_container_rejects_gpu_ordinal_label_drift(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["Config"]["Labels"]["ai.verse.gpu.ordinal"] = "1"

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "wrong GPU ordinal label" in result.stderr


def test_container_rejects_gpu_device_request_by_ordinal(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["0"]

    result = run_validator(container, args)

    assert result.returncode != 0
    assert "wrong GPU UUID selected" in result.stderr


def test_container_accepts_initial_no_restart_policy(tmp_path: Path):
    container, args = base_container(tmp_path)
    container["HostConfig"]["RestartPolicy"]["Name"] = "no"
    args.extend(["--restart-policy", "no"])

    result = run_validator(container, args)

    assert result.returncode == 0, result.stderr


def test_launcher_binds_post_create_lifecycle_to_captured_container_id():
    source = RUNNER.read_text()
    lifecycle = source[source.index("CONTAINER_ID=$(docker create") :]

    expected_commands = {
        "start": 'docker start "$CONTAINER_ID"',
        "update": 'docker update --restart unless-stopped "$CONTAINER_ID"',
    }
    for command, invocation in expected_commands.items():
        assert invocation in lifecycle
        assert f'docker {command} "$CONTAINER"' not in lifecycle
    cleanup = source[source.index("container_name_is_owned_by_id()") :]
    for command in ("inspect", "logs", "rm"):
        assert f"docker {command}" in cleanup
        assert '"$CONTAINER_ID"' in cleanup
        assert f'docker {command} "$CONTAINER"' not in cleanup
    assert "{{.Id}} {{.Name}}" in cleanup
    assert '[[ $identity == "$CONTAINER_ID /$CONTAINER" ]]' in cleanup
    assert cleanup.index("if container_name_is_owned_by_id") < cleanup.index(
        'docker rm --force "$CONTAINER_ID"'
    )
    assert lifecycle.count('VERSE_VLLM_CONTAINER_ID="$CONTAINER_ID"') == 2
    assert 'VERSE_VLLM_CONTAINER="$CONTAINER_ID"' not in lifecycle


def test_launcher_requires_exact_idle_gpu_identity_before_create():
    source = RUNNER.read_text()
    create = source.index("CONTAINER_ID=$(docker create")
    pre_create = source[:create]

    assert "require_env VERSE_VLLM_GPU_UUID" in pre_create
    assert "VERSE_VLLM_GPU_UUID must be an exact full GPU UUID" in pre_create
    assert "--query-gpu=index,uuid" in pre_create
    assert "--query-compute-apps=pid" in pre_create
    assert pre_create.rindex("verify_gpu_identity") < create
    assert pre_create.rindex("require_idle_gpu") < create
    assert ".verse-sm120-runtime.json" in pre_create
    assert ".verse-sm120-runtime-gpu.json" not in pre_create
    assert '"gpu_device": sys.argv[4]' in pre_create
    assert '"gpu_uuid": sys.argv[5]' in pre_create
    assert '--label ai.verse.gpu.ordinal="$GPU_DEVICE"' in source[create:]
    assert '--label ai.verse.gpu.uuid="$GPU_UUID"' in source[create:]
    assert '--gpus "device=$GPU_UUID"' in source[create:]


def test_gpu_guard_accepts_matching_idle_device(tmp_path: Path):
    result = run_gpu_guards(
        tmp_path,
        "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "",
    )

    assert result.returncode == 0, result.stderr


def test_gpu_guard_rejects_ordinal_uuid_mismatch(tmp_path: Path):
    result = run_gpu_guards(
        tmp_path,
        "0, GPU-11111111-2222-3333-4444-555555555555",
        "",
    )

    assert result.returncode != 0
    assert "does not resolve to VERSE_VLLM_GPU_UUID" in result.stderr


def test_gpu_guard_rejects_preexisting_compute_process(tmp_path: Path):
    result = run_gpu_guards(
        tmp_path,
        "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "4242\n",
    )

    assert result.returncode != 0
    assert "pre-existing compute process" in result.stderr


def test_launcher_verifies_image_source_archive_provenance():
    source = RUNNER.read_text()

    assert "ai.verse.source.archive.sha256" in source
    assert "org.opencontainers.image.revision" in source
    assert 'git -C "$ROOT" -c tar.umask=0000 archive' in source
    assert 'IMAGE_SOURCE_ARCHIVE == "$EXPECTED_SOURCE_ARCHIVE"' in source


def test_live_checker_requires_and_verifies_exact_gpu_uuid():
    source = CHECKER.read_text()

    assert "require_env VERSE_VLLM_GPU_UUID" in source
    assert '--gpu-uuid "$GPU_UUID"' in source
    assert 'docker exec "$CONTAINER_ID" nvidia-smi' in source
    assert "--query-gpu=uuid" in source
    assert 'VISIBLE_GPU_UUID == "$GPU_UUID"' in source


def test_live_checker_requires_full_container_id_for_every_docker_operation():
    source = CHECKER.read_text()

    assert "require_env VERSE_VLLM_CONTAINER_ID" in source
    assert "VERSE_VLLM_CONTAINER_ID must be a full 64-character container ID" in source
    assert '--container-id "$CONTAINER_ID"' in source
    expected_commands = (
        'docker inspect "$CONTAINER_ID"',
        'docker exec "$CONTAINER_ID"',
        'docker logs --since "$STARTED_AT" "$CONTAINER_ID"',
    )
    assert all(command in source for command in expected_commands)
    assert '"$CONTAINER"' not in source
    assert "VERSE_VLLM_CONTAINER:-" not in source
