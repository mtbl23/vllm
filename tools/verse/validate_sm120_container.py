#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any

from validate_sm120_profile import EXPECTED_PROFILE

FLASHINFER_MANIFEST = Path(__file__).parents[2] / (
    "requirements/verse-sm120-flashinfer.lock"
)
BUILD_BASE_IMAGE = (
    "pytorch/manylinux2_28-builder:cuda13.0@sha256:"
    "7710cbc19d7ee951134e2e827f8ec89237c993095eb2581dd5e74f58e4e278c7"
)
FINAL_BASE_IMAGE = (
    "nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:"
    "97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4"
)
RUNTIME_IDENTITY_MARKER = ".verse-sm120-runtime.json"
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the exact Verse SM120 Docker container contract."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--runtime-cache", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument(
        "--restart-policy",
        choices=("no", "unless-stopped"),
        default="unless-stopped",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def expected_model_path() -> str:
    return "/models/model"


def expected_command(served_model: str) -> list[str]:
    profile = EXPECTED_PROFILE
    return [
        expected_model_path(),
        "--served-model-name",
        served_model,
        "--quantization",
        profile["VERSE_QUANTIZATION"],
        "--dtype",
        "bfloat16",
        "--linear-backend",
        profile["VERSE_LINEAR_BACKEND"],
        "--max-model-len",
        profile["VERSE_MAX_MODEL_LEN"],
        "--max-num-seqs",
        profile["VERSE_MAX_NUM_SEQS"],
        "--max-num-batched-tokens",
        profile["VERSE_MAX_NUM_BATCHED_TOKENS"],
        "--gpu-memory-utilization",
        profile["VERSE_GPU_MEMORY_UTILIZATION"],
        "--kv-cache-dtype",
        profile["VERSE_KV_CACHE_DTYPE"],
        "--attention-backend",
        profile["VERSE_ATTENTION_BACKEND"],
        "--enable-prefix-caching",
        "--no-disable-hybrid-kv-cache-manager",
        "--no-async-scheduling",
        "--language-model-only",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--generation-config",
        "vllm",
        "--enforce-eager",
        "--compilation-config",
        profile["VERSE_COMPILATION_CONFIG"],
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def validate_host_paths(args: argparse.Namespace) -> None:
    for name, path in (
        ("runtime cache", args.runtime_cache),
        ("model cache", args.model_cache),
    ):
        require(
            path.is_absolute() and path.is_dir(), f"{name} is not an absolute directory"
        )
        require(not path.is_symlink(), f"{name} is a symlink")
        require(path.resolve(strict=True) == path, f"{name} is not canonical")
    runtime = args.runtime_cache.resolve(strict=True)
    model = args.model_cache.resolve(strict=True)
    require(
        runtime != model
        and runtime not in model.parents
        and model not in runtime.parents,
        "runtime and model caches are not disjoint",
    )
    model_directory = args.model_directory
    require(
        model_directory.is_absolute()
        and model_directory.is_dir()
        and not model_directory.is_symlink(),
        "verified model directory is not an absolute regular directory",
    )
    resolved_model_directory = model_directory.resolve(strict=True)
    require(
        resolved_model_directory == model_directory,
        "verified model directory is not canonical",
    )
    require(
        model in resolved_model_directory.parents,
        "verified model directory is outside the model cache",
    )
    secret = args.api_key_file
    require(
        secret.is_absolute() and secret.is_file() and not secret.is_symlink(),
        "API key file is not an absolute regular non-symlink file",
    )
    resolved_secret = secret.resolve(strict=True)
    require(resolved_secret == secret, "API key file is not canonical")
    require(
        runtime not in resolved_secret.parents and model not in resolved_secret.parents,
        "API key file is inside a cache",
    )
    mode = stat.S_IMODE(resolved_secret.stat().st_mode)
    require(
        not mode & (stat.S_IRWXG | stat.S_IRWXO),
        "API key file is not owner-only",
    )


def validate_mounts(container: dict[str, Any], args: argparse.Namespace) -> None:
    mounts = {mount["Destination"]: mount for mount in container.get("Mounts", [])}
    expected = {
        "/cache": (args.runtime_cache.resolve(), True),
        "/models/model": (args.model_directory.resolve(), False),
        "/run/secrets/vllm_api_key": (args.api_key_file.resolve(), False),
    }
    require(set(mounts) == set(expected), "container mounts do not match the profile")
    for destination, (source, writable) in expected.items():
        mount = mounts[destination]
        require(mount.get("Type") == "bind", f"{destination} must be a bind mount")
        require(
            Path(mount.get("Source", "")).resolve() == source,
            f"{destination} source does not match the validated path",
        )
        require(bool(mount.get("RW")) is writable, f"{destination} has wrong mode")


def validate_gpu(container: dict[str, Any], gpu_uuid: str) -> None:
    requests = container["HostConfig"].get("DeviceRequests") or []
    require(len(requests) == 1, "container must have exactly one GPU request")
    request = requests[0]
    require(request.get("Driver") in ("", "nvidia"), "unexpected GPU driver")
    require(request.get("DeviceIDs") == [gpu_uuid], "wrong GPU UUID selected")
    capabilities = request.get("Capabilities") or []
    require(any("gpu" in group for group in capabilities), "GPU capability is absent")


def validate_runtime_identity(args: argparse.Namespace) -> None:
    marker_path = args.runtime_cache / RUNTIME_IDENTITY_MARKER
    require(marker_path.is_file(), "runtime cache identity marker is absent")
    require(not marker_path.is_symlink(), "runtime cache identity marker is a symlink")
    with marker_path.open(encoding="utf-8") as handle:
        marker = json.load(handle)
    require(
        marker
        == {
            "schema_version": 1,
            "image": args.image,
            "fork_commit": args.expected_commit,
            "profile": "sm120-gemma4-nvfp4-v1",
            "gpu_device": args.gpu_device,
            "gpu_uuid": args.gpu_uuid,
        },
        "runtime cache identity does not match the candidate",
    )


def validate_container(container: dict[str, Any], args: argparse.Namespace) -> dict:
    validate_host_paths(args)
    require(
        CONTAINER_ID_RE.fullmatch(args.container_id) is not None,
        "container ID is invalid",
    )
    require(container.get("Id") == args.container_id, "container ID does not match")
    require(args.gpu_device.isdigit(), "GPU ordinal is invalid")
    require(GPU_UUID_RE.fullmatch(args.gpu_uuid) is not None, "GPU UUID is invalid")
    require(
        args.served_model == EXPECTED_PROFILE["VERSE_SERVED_MODEL_NAME"],
        "served model name does not match the fixed Verse profile",
    )
    state = container["State"]
    require(state.get("Running") is True, "container is not running")
    require(state.get("OOMKilled") is False, "container was OOM-killed")
    require(not state.get("Error"), f"container state error: {state.get('Error')}")
    require(container.get("RestartCount") == 0, "container has restarted")

    config = container["Config"]
    labels = config.get("Labels") or {}
    require(config.get("Image") == args.image, "container uses the wrong image")
    require(
        labels.get("ai.vllm.build.commit") == args.expected_commit,
        "container has the wrong fork commit",
    )
    expected_wheel_version = f"0.28.0+verse.{args.expected_commit[:12]}"
    require(
        labels.get("ai.verse.vllm.wheel.version") == expected_wheel_version,
        "container has the wrong vLLM wheel version label",
    )
    require(
        labels.get("ai.verse.runtime.profile") == "sm120-gemma4-nvfp4-v1",
        "container has the wrong Verse runtime profile",
    )
    require(
        labels.get("ai.verse.gpu.ordinal") == args.gpu_device,
        "container has the wrong GPU ordinal label",
    )
    require(
        labels.get("ai.verse.gpu.uuid") == args.gpu_uuid,
        "container has the wrong GPU UUID label",
    )
    expected_manifest = hashlib.sha256(FLASHINFER_MANIFEST.read_bytes()).hexdigest()
    require(
        labels.get("ai.verse.flashinfer.release") == "0.6.18.dev20260819",
        "container has the wrong FlashInfer release label",
    )
    require(
        labels.get("ai.verse.flashinfer.manifest.sha256") == expected_manifest,
        "container has the wrong FlashInfer manifest label",
    )
    require(
        labels.get("ai.verse.base.build") == BUILD_BASE_IMAGE,
        "container has the wrong immutable build base",
    )
    require(
        labels.get("ai.verse.base.runtime") == FINAL_BASE_IMAGE,
        "container has the wrong immutable runtime base",
    )
    require(
        config.get("Entrypoint") == ["/usr/local/bin/verse-sm120-entrypoint.sh"],
        "container has the wrong entrypoint",
    )
    require(config.get("User") == "2000:0", "container is not running as UID 2000")
    require(
        config.get("Cmd") == expected_command(args.served_model),
        "container command does not exactly match the fixed profile",
    )

    environment = set(config.get("Env") or [])
    for required in (
        "VLLM_API_KEY_FILE=/run/secrets/vllm_api_key",
        f"VERSE_VLLM_WHEEL_VERSION={expected_wheel_version}",
        "VLLM_VERSE_RUNTIME_STRICT=1",
        "VLLM_NVFP4_KV_VOSPLIT=1",
        "VLLM_KV_CACHE_LAYOUT=HND",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
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
    ):
        require(required in environment, f"missing runtime environment: {required}")
    require(
        not any(item.startswith("VLLM_API_KEY=") for item in environment),
        "API key is exposed in Docker configuration",
    )

    host = container["HostConfig"]
    require(host.get("Privileged") is not True, "container is privileged")
    require(not (host.get("CapAdd") or []), "container adds Linux capabilities")
    require("ALL" in (host.get("CapDrop") or []), "Linux capabilities are not dropped")
    require(host.get("IpcMode") in ("", "private"), "container shares host IPC")
    require(
        host.get("PidMode") not in ("host",), "container shares the host PID namespace"
    )
    require(
        host.get("NetworkMode") not in ("host",),
        "container shares the host network namespace",
    )
    require(host.get("ReadonlyRootfs") is True, "root filesystem is writable")
    require(
        "no-new-privileges" in (host.get("SecurityOpt") or []),
        "no-new-privileges is absent",
    )
    require(
        host.get("RestartPolicy", {}).get("Name") == args.restart_policy,
        "unexpected restart policy",
    )
    require("/tmp" in (host.get("Tmpfs") or {}), "writable /tmp tmpfs is absent")
    validate_gpu(container, args.gpu_uuid)
    validate_mounts(container, args)
    validate_runtime_identity(args)

    bindings = container["NetworkSettings"]["Ports"].get("8000/tcp") or []
    require(len(bindings) == 1, "container must expose exactly one port binding")
    binding = bindings[0]
    require(binding.get("HostIp") == "127.0.0.1", "port is not loopback-only")
    host_port = binding.get("HostPort", "")
    require(host_port.isdigit() and 0 < int(host_port) < 65536, "invalid host port")

    return {
        "status": "valid",
        "container_id": args.container_id,
        "image_id": container["Image"],
        "host_port": int(host_port),
        "started_at": state["StartedAt"],
        "fork_commit": args.expected_commit,
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "gpu_device": args.gpu_device,
        "gpu_uuid": args.gpu_uuid,
    }


def main() -> int:
    args = parse_args()
    payload = json.load(sys.stdin)
    require(isinstance(payload, list) and len(payload) == 1, "expected one container")
    print(json.dumps(validate_container(payload[0], args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
