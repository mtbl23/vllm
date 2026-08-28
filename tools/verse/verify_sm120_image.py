#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
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
        return matches[0]

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
    for forbidden in ("FLASHINFER_DISABLE_VERSION_CHECK", "FLASHINFER_CUBIN_DIR"):
        require(
            not os.environ.get(forbidden),
            f"forbidden override is set: {forbidden}",
        )

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
    paths["vllm"] = str(Path(vllm_matches[0].locate_file("")).resolve())
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
