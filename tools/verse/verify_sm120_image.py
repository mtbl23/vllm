#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from packaging.requirements import Requirement

EXPECTED_DISTRIBUTIONS = {
    "flashinfer-python": "0.6.18.dev20260819",
    "flashinfer-cubin": "0.6.18.dev20260819",
    "flashinfer-jit-cache": "0.6.18.dev20260819+cu130",
    "nvidia-cutlass-dsl": "4.7.0",
    "nvidia-cudnn-frontend": "1.27.0",
    "nvidia-nccl-cu13": "2.29.7",
}
EXPECTED_FLASHINFER_COMMIT = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
EXPECTED_FLASHINFER_REQUIREMENT_URL = (
    "https://github.com/flashinfer-ai/flashinfer/releases/download/"
    "nightly-v0.6.18-20260819/"
    "flashinfer_python-0.6.18.dev20260819-py3-none-any.whl"
    "#sha256=50ad966220b5160f17fcb9e064bdfbcda726ec779fb0c74fd3449b3c48c66600"
)
VLLM_WHEEL_VERSION_RE = re.compile(r"0\.28\.0\+verse\.[0-9a-f]{12}")
NATIVE_MEMBER_RE = re.compile(r"vllm/_C_stable_libtorch(?:\.[^.]+)*\.so")
VLLM_WHEEL_IDENTITY_MANIFEST = Path("/opt/verse/identity/vllm-wheel.json")
FORBIDDEN_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "FLASHINFER_CUBIN_DIR",
        "FLASHINFER_DISABLE_VERSION_CHECK",
        "VLLM_BATCH_INVARIANT",
        "VLLM_DISABLED_KERNELS",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_TEST_FORCE_FP8_MARLIN",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the immutable Verse SM120 runtime dependency tuple."
    )
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def normalized(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def installed_distributions() -> dict[str, list[importlib.metadata.Distribution]]:
    result: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            result.setdefault(normalized(name), []).append(distribution)
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_vllm_binary_identity(
    vllm_root: Path,
    wheel_manifest: Path = VLLM_WHEEL_IDENTITY_MANIFEST,
) -> dict[str, object]:
    spec = importlib.util.find_spec("vllm._C_stable_libtorch")
    require(spec is not None and spec.origin, "vLLM native extension is absent")
    native_path = Path(spec.origin).resolve()
    require(
        native_path.is_file() and native_path.is_relative_to(vllm_root),
        "vLLM native extension is loaded outside the installed distribution",
    )

    require(
        wheel_manifest.is_file() and not wheel_manifest.is_symlink(),
        "vLLM wheel identity manifest is absent",
    )
    try:
        manifest = json.loads(wheel_manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise SystemExit("vLLM wheel identity manifest is malformed") from error
    wheel = manifest.get("wheel") if isinstance(manifest, dict) else None
    native = manifest.get("native_extension") if isinstance(manifest, dict) else None
    require(
        isinstance(manifest, dict)
        and set(manifest) == {"schema_version", "wheel", "native_extension"}
        and manifest.get("schema_version") == 1
        and isinstance(wheel, dict)
        and set(wheel) == {"filename", "sha256"}
        and isinstance(wheel.get("filename"), str)
        and Path(wheel["filename"]).name == wheel["filename"]
        and wheel["filename"].endswith(".whl")
        and re.fullmatch(r"[0-9a-f]{64}", str(wheel.get("sha256", ""))) is not None
        and isinstance(native, dict)
        and set(native) == {"member", "bytes", "sha256"}
        and isinstance(native.get("member"), str)
        and NATIVE_MEMBER_RE.fullmatch(native["member"]) is not None
        and isinstance(native.get("bytes"), int)
        and native["bytes"] > 0
        and re.fullmatch(r"[0-9a-f]{64}", str(native.get("sha256", ""))) is not None,
        "vLLM wheel identity manifest is malformed",
    )
    native_sha256 = sha256_file(native_path)
    require(
        native_path.name == Path(native["member"]).name
        and native_path.stat().st_size == native["bytes"]
        and native_sha256 == native["sha256"],
        "loaded vLLM native extension does not match the declared wheel artifact",
    )
    return {
        "native_extension": {
            "path": str(native_path),
            "wheel_member": native["member"],
            "bytes": native_path.stat().st_size,
            "sha256": native_sha256,
        },
        "wheel_artifact": {
            "filename": wheel["filename"],
            "sha256": wheel["sha256"],
            "manifest_sha256": sha256_file(wheel_manifest),
        },
    }


def verify_runtime_environment(environment: Mapping[str, str]) -> None:
    for forbidden in sorted(FORBIDDEN_RUNTIME_ENVIRONMENT_NAMES):
        require(
            forbidden not in environment,
            f"forbidden runtime environment: {forbidden}",
        )


def verify_vllm_wheel_requirements(
    distribution: importlib.metadata.Distribution,
) -> dict[str, str]:
    parsed: dict[str, list[Requirement]] = {}
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        parsed.setdefault(normalized(requirement.name), []).append(requirement)

    def exactly_one(name: str) -> Requirement:
        matches = parsed.get(normalized(name), [])
        require(len(matches) == 1, f"expected exactly one {name} wheel requirement")
        requirement = matches[0]
        require(
            requirement.marker is None,
            f"{name} wheel requirement must be unconditional: {requirement}",
        )
        return requirement

    flashinfer = exactly_one("flashinfer-python")
    require(
        flashinfer.url == EXPECTED_FLASHINFER_REQUIREMENT_URL,
        f"wrong FlashInfer wheel requirement: {flashinfer}",
    )
    cutlass = exactly_one("nvidia-cutlass-dsl")
    require(
        str(cutlass.specifier) == "==4.7.0" and cutlass.extras == {"cu13"},
        f"wrong CUTLASS wheel requirement: {cutlass}",
    )
    cudnn = exactly_one("nvidia-cudnn-frontend")
    require(
        str(cudnn.specifier) == "==1.27.0" and not cudnn.extras,
        f"wrong cuDNN frontend wheel requirement: {cudnn}",
    )
    require(
        "quack-kernels" not in parsed,
        "the Verse wheel still requires quack-kernels",
    )
    return {
        "flashinfer-python": str(flashinfer),
        "nvidia-cutlass-dsl": str(cutlass),
        "nvidia-cudnn-frontend": str(cudnn),
    }


def main() -> int:
    args = parse_args()
    require(sys.version_info[:2] == (3, 12), "Verse image requires Python 3.12")
    require(platform.machine() == "x86_64", "Verse image requires x86_64")
    verify_runtime_environment(os.environ)

    distributions = installed_distributions()
    paths: dict[str, str] = {}
    vllm_wheel_version = os.environ.get("VERSE_VLLM_WHEEL_VERSION", "")
    require(
        VLLM_WHEEL_VERSION_RE.fullmatch(vllm_wheel_version) is not None,
        "invalid or missing VERSE_VLLM_WHEEL_VERSION",
    )
    vllm_matches = distributions.get("vllm", [])
    require(len(vllm_matches) == 1, "expected exactly one vllm distribution")
    require(
        vllm_matches[0].version == vllm_wheel_version,
        f"wrong vllm wheel: {vllm_matches[0].version}",
    )
    wheel_requirements = verify_vllm_wheel_requirements(vllm_matches[0])
    vllm_root = Path(vllm_matches[0].locate_file("")).resolve()
    paths["vllm"] = str(vllm_root)
    vllm_binary_identity = verify_vllm_binary_identity(vllm_root / "vllm")
    require(
        not distributions.get("deep-ep"),
        "DeepEP must not be installed in the single-GPU Verse appliance",
    )
    require(
        not distributions.get("b12x"),
        "standalone B12X must not be installed; the Verse appliance uses "
        "FlashInfer's CUTLASS-4.7-compatible SM120 B12X backend",
    )
    for name, version in EXPECTED_DISTRIBUTIONS.items():
        matches = distributions.get(normalized(name), [])
        require(len(matches) == 1, f"expected exactly one {name} distribution")
        distribution = matches[0]
        require(
            distribution.version == version,
            f"wrong {name}: {distribution.version}",
        )
        location = Path(distribution.locate_file("")).resolve()
        require(
            str(location).startswith(("/usr/local/", "/opt/venv/")),
            f"{name} is loaded from an unexpected path: {location}",
        )
        paths[name] = str(location)

    require(importlib.util.find_spec("flashinfer_cubin") is not None, "cubin missing")
    require(
        importlib.util.find_spec("flashinfer_jit_cache") is not None,
        "FlashInfer JIT cache missing",
    )

    import flashinfer
    import torch

    require(torch.__version__ == "2.13.0+cu130", f"wrong Torch: {torch.__version__}")
    require(torch.version.cuda == "13.0", f"wrong Torch CUDA: {torch.version.cuda}")
    commit = getattr(flashinfer, "__git_commit__", None)
    require(commit == EXPECTED_FLASHINFER_COMMIT, f"wrong FlashInfer commit: {commit}")

    check = subprocess.run(
        ["uv", "pip", "check", "--system"],
        capture_output=True,
        text=True,
    )
    require(
        check.returncode == 0,
        f"dependency check failed: {check.stdout}{check.stderr}",
    )

    gpu: dict[str, object] | None = None
    if args.require_gpu:
        require(torch.cuda.is_available(), "CUDA is unavailable")
        capability = torch.cuda.get_device_capability()
        require(capability == (12, 0), f"wrong GPU capability: {capability}")
        device_name = torch.cuda.get_device_name()
        require(
            device_name.endswith("RTX 5070 Ti"),
            f"wrong GPU model: {device_name}",
        )
        gpu = {
            "name": device_name,
            "capability": capability,
        }

    print(
        json.dumps(
            {
                "status": "valid",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "vllm_wheel_version": vllm_wheel_version,
                "vllm_wheel_requirements": wheel_requirements,
                "vllm_binary_identity": vllm_binary_identity,
                "flashinfer_commit": commit,
                "distributions": EXPECTED_DISTRIBUTIONS,
                "distribution_paths": paths,
                "gpu": gpu,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
