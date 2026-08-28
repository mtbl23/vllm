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

EXPECTED_DISTRIBUTIONS = {
    "flashinfer-python": "0.6.18.dev20260819",
    "flashinfer-cubin": "0.6.18.dev20260819",
    "flashinfer-jit-cache": "0.6.18.dev20260819+cu130",
    "nvidia-cutlass-dsl": "4.7.0",
    "nvidia-cudnn-frontend": "1.27.0",
    "nvidia-nccl-cu13": "2.29.7",
}
EXPECTED_FLASHINFER_COMMIT = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
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
    paths["vllm"] = str(Path(vllm_matches[0].locate_file("")).resolve())
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
