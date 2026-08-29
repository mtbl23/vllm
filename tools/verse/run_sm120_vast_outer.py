#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Start the immutable Campaign 22 image in a Vast SSH container.

Vast's SSH launch mode replaces the OCI entrypoint. This helper is copied into
the already-pulled immutable image, validates the exact runtime and model, then
drops permanently to the image's UID 2000 before executing the real Verse
entrypoint. It is qualification-only and never opens a public listener.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED_MODEL_REVISION = "e2c6cd9c3302e91c032a378a607009c82ba16fac"
EXPECTED_PROFILE = "sm120-gemma4-nvfp4-v4"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5070 Ti"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--runtime-cache", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--server-pid-file", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def canonical_new_or_existing_directory(path: Path, owner: int) -> Path:
    require(path.is_absolute(), f"{path} must be absolute")
    path.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"{path} must not be a symlink")
    resolved = path.resolve(strict=True)
    require(resolved == path, f"{path} must be canonical")
    os.chown(path, owner, 0)
    os.chmod(path, 0o750)
    return resolved


def validate_secret(path: Path, *, owner: int) -> bytes:
    require(path.is_absolute() and path.is_file(), f"{path} must be a file")
    require(not path.is_symlink(), f"{path} must not be a symlink")
    require(path.resolve(strict=True) == path, f"{path} must be canonical")
    mode = stat.S_IMODE(path.stat().st_mode)
    require(mode == 0o600, f"{path} must have mode 0600")
    require(path.stat().st_uid == owner, f"{path} has the wrong owner")
    raw = path.read_bytes()
    lines = raw.splitlines()
    require(
        len(lines) == 1 and bool(lines[0]) and b"\r" not in raw and b"\0" not in raw,
        f"{path} must contain one non-empty clean line",
    )
    return lines[0]


def erase_secret(path: Path) -> None:
    length = path.stat().st_size
    with path.open("r+b", buffering=0) as handle:
        handle.write(b"\0" * length)
        handle.flush()
        os.fsync(handle.fileno())
    path.unlink()


def gpu_identity() -> dict[str, object]:
    query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        .stdout.strip()
        .splitlines()
    )
    require(len(query) == 1, "qualification image must see exactly one GPU")
    fields = [field.strip() for field in query[0].split(",")]
    require(len(fields) == 5, "GPU identity is malformed")
    name, uuid, memory, driver, capability = fields
    require(name == EXPECTED_GPU_NAME, f"unexpected GPU model: {name}")
    require(capability == "12.0", f"unexpected compute capability: {capability}")
    require(re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", uuid) is not None, "bad GPU UUID")
    return {
        "name": name,
        "uuid": uuid,
        "memory_total_mib": int(memory),
        "driver_version": driver,
        "compute_capability": [12, 0],
    }


def prepare_model(args: argparse.Namespace) -> Path:
    validate_secret(args.hf_token_file, owner=0)
    result = subprocess.run(
        [
            sys.executable,
            "/opt/verse-tools/prepare_sm120_model.py",
            "--cache-dir",
            str(args.model_cache),
            "--token-file",
            str(args.hf_token_file),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    prepared = json.loads(result.stdout)
    require(prepared.get("status") == "ready", "model preparation did not pass")
    verified = subprocess.run(
        [
            sys.executable,
            "/opt/verse-tools/prepare_sm120_model.py",
            "--cache-dir",
            str(args.model_cache),
            "--verify-ready",
            "--require-root-owner",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(verified.stdout)
    require(payload.get("revision") == EXPECTED_MODEL_REVISION, "wrong model revision")
    materialized = Path(payload["model_directory"])
    require(
        materialized.is_dir() and materialized.resolve(strict=True) == materialized,
        "bad model path",
    )
    served = Path("/models/model")
    require(
        not served.exists() and not served.is_symlink(), "/models/model must be absent"
    )
    served.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    served.mkdir(mode=0o755)
    subprocess.run(
        ["cp", "-a", "-l", f"{materialized}/.", str(served)],
        check=True,
    )
    mounted = subprocess.run(
        [
            sys.executable,
            "/opt/verse-tools/prepare_sm120_model.py",
            "--model-directory",
            str(served),
            "--verify-mounted-model",
            "--require-root-owner",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    require(
        json.loads(mounted.stdout).get("status") == "valid",
        "served model failed verification",
    )
    return served


def clean_environment(
    api_key_file: Path, runtime_cache: Path, wheel_version: str
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": (
            "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        ),
        "HOME": "/home/vllm",
        "USER": "vllm",
        "LOGNAME": "vllm",
        "PYTHONUNBUFFERED": "1",
        "VLLM_API_KEY_FILE": str(api_key_file),
        "VERSE_VLLM_WHEEL_VERSION": wheel_version,
        "VLLM_VERSE_RUNTIME_STRICT": "1",
        "VLLM_NVFP4_KV_VOSPLIT": "1",
        "VLLM_VERSE_NVFP4_XQA_DECODE": "1",
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": "67108864",
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_MAX_N_SEQUENCES": "1",
        "VLLM_MAX_COMPLETION_PROMPTS": "1",
        "VLLM_MAX_STOP_STRINGS": "4",
        "FLASHINFER_WORKSPACE_BASE": str(runtime_cache / "flashinfer"),
        "CUDA_CACHE_PATH": str(runtime_cache / "cuda"),
        "TORCH_HOME": str(runtime_cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(runtime_cache / "torch-extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime_cache / "torchinductor"),
        "HF_HOME": str(runtime_cache / "huggingface"),
        "VLLM_CACHE_ROOT": str(runtime_cache / "vllm"),
        "TRITON_CACHE_DIR": str(runtime_cache / "triton"),
        "XDG_CACHE_HOME": str(runtime_cache / "xdg"),
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(
            runtime_cache / "flashinfer-autotune"
        ),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    return environment


def server_arguments(model: Path, port: int) -> list[str]:
    return [
        str(model),
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
        '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,8,16,24,32,38],"max_cudagraph_capture_size":38}',
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def wait_for_server(port: int, api_key: bytes, process_id: int) -> None:
    deadline = time.monotonic() + 900
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key.decode()}"},
    )
    while time.monotonic() < deadline:
        exited, status = os.waitpid(process_id, os.WNOHANG)
        require(exited == 0, f"server exited with wait status {status}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.load(response)
                require(response.status == 200, "model discovery failed")
                require(payload["data"][0]["id"] == "verse-free", "wrong model alias")
                unauthenticated = urllib.request.Request(
                    f"http://127.0.0.1:{port}/invocations", method="POST"
                )
                try:
                    urllib.request.urlopen(unauthenticated, timeout=5)
                except urllib.error.HTTPError as error:
                    require(
                        error.code == 401, "raw invocation auth did not fail closed"
                    )
                else:
                    raise RuntimeError(
                        "raw invocation endpoint bypassed authentication"
                    )
                return
        except (OSError, KeyError, ValueError, urllib.error.URLError):
            time.sleep(2)
    raise RuntimeError("server did not become ready within 15 minutes")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_live_process(
    evidence: Path,
    process_id: int,
    gpu: dict[str, object],
    expected_commit: str,
    wheel_version: str,
) -> None:
    status = {}
    for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    executable = Path(os.readlink(f"/proc/{process_id}/exe"))
    command = Path(f"/proc/{process_id}/cmdline").read_bytes().split(b"\0")
    process_stat = Path(f"/proc/{process_id}/stat").read_text().split()
    require(len(process_stat) > 21, "server process stat is malformed")
    start_ticks = int(process_stat[21])
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    require(status.get("Uid", "").split()[:1] == ["2000"], "server UID drifted")
    require(status.get("Gid", "").split()[:1] == ["0"], "server GID drifted")
    require(command and command[0], "server command line is empty")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    machine_id = Path("/etc/machine-id").read_text().strip()
    payload = {
        "schema_version": 1,
        "status": "pass",
        "pid": process_id,
        "uid": 2000,
        "gid": 0,
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "command_sha256": hashlib.sha256(b"\0".join(command)).hexdigest(),
        "process_start_ticks": start_ticks,
        "process_started_at_unix": boot_time + start_ticks / os.sysconf("SC_CLK_TCK"),
        "boot_id": boot_id,
        "machine_id_sha256": hashlib.sha256(machine_id.encode()).hexdigest(),
        "gpu": gpu,
        "fork_commit": expected_commit,
        "wheel_version": wheel_version,
        "model_revision": EXPECTED_MODEL_REVISION,
    }
    output = evidence / "live-process.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o640)


def main() -> int:
    args = parse_args()
    require(os.geteuid() == 0, "qualification bootstrap must start as root")
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is not None,
        "expected commit must be exactly 40 lowercase hex characters",
    )
    wheel_version = f"0.28.0+verse.{args.expected_commit[:12]}"
    require(0 < args.port < 65536, "invalid port")
    require(not args.server_pid_file.exists(), "server PID file already exists")
    require(not args.server_log.exists(), "server log already exists")

    runtime_cache = canonical_new_or_existing_directory(args.runtime_cache, 2000)
    require(not any(runtime_cache.iterdir()), "runtime cache must start empty")
    model_cache = canonical_new_or_existing_directory(args.model_cache, 0)
    evidence = canonical_new_or_existing_directory(args.evidence_dir, 0)
    require(runtime_cache != model_cache, "model and runtime caches must be disjoint")
    require(
        runtime_cache not in model_cache.parents, "nested cache paths are forbidden"
    )
    require(
        model_cache not in runtime_cache.parents, "nested cache paths are forbidden"
    )

    image_verification = subprocess.run(
        ["/usr/local/bin/verify-verse-sm120-image", "--require-gpu"],
        check=True,
        text=True,
        capture_output=True,
    )
    verification = json.loads(image_verification.stdout)
    require(verification.get("status") == "valid", "image verification failed")
    require(
        verification.get("vllm_wheel_version") == wheel_version,
        "wrong vLLM wheel version",
    )
    (evidence / "image-verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    gpu = gpu_identity()
    (evidence / "gpu-identity.json").write_text(
        json.dumps(gpu, indent=2, sort_keys=True) + "\n"
    )

    model = prepare_model(args)
    erase_secret(args.hf_token_file)
    validate_secret(args.api_key_file, owner=0)
    os.chown(args.api_key_file, 2000, 0)
    api_key = validate_secret(args.api_key_file, owner=2000)
    for name in (
        "flashinfer",
        "cuda",
        "torch",
        "torch-extensions",
        "torchinductor",
        "huggingface",
        "vllm",
        "triton",
        "xdg",
        "flashinfer-autotune",
    ):
        canonical_new_or_existing_directory(runtime_cache / name, 2000)

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    args.server_log.touch(mode=0o640, exist_ok=False)
    os.chown(args.server_log, 2000, 0)
    log_handle = args.server_log.open("ab", buffering=0)
    environment = clean_environment(args.api_key_file, runtime_cache, wheel_version)
    command = [
        "/usr/local/bin/verse-sm120-entrypoint.sh",
        *server_arguments(model, args.port),
    ]

    def demote() -> None:
        os.setgroups([])
        os.setgid(0)
        os.setuid(2000)
        os.chdir("/home/vllm")

    process_id = os.fork()
    if process_id == 0:
        try:
            os.setsid()
            null_input = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null_input, 0)
            os.dup2(log_handle.fileno(), 1)
            os.dup2(log_handle.fileno(), 2)
            os.close(null_input)
            demote()
            os.execve(command[0], command, environment)
        except OSError:
            os._exit(127)
    log_handle.close()
    args.server_pid_file.write_text(f"{process_id}\n")
    os.chmod(args.server_pid_file, 0o600)
    wait_for_server(args.port, api_key, process_id)
    record_live_process(evidence, process_id, gpu, args.expected_commit, wheel_version)
    print(
        json.dumps(
            {
                "status": "ready",
                "pid": process_id,
                "endpoint": f"http://127.0.0.1:{args.port}",
                "model_directory": str(model),
                "fork_commit": args.expected_commit,
                "profile": EXPECTED_PROFILE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
