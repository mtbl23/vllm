#!/usr/bin/env python3
"""Download the immutable Campaign 22 NVFP4 model tree from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "marak12/verse-campaign22-gemma4-12b-compact"
DEFAULT_MODEL_SUBDIR = "quantized/w4a4/candidate-unified-checkpoint-010674-nvfp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-subdir", default=DEFAULT_MODEL_SUBDIR)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Hugging Face token file is empty")

    args.destination.mkdir(parents=True, exist_ok=True)
    os.chmod(args.destination, 0o700)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            allow_patterns=[f"{args.model_subdir}/*"],
            local_dir=args.destination,
        )
    finally:
        args.token_file.unlink(missing_ok=True)

    model_path = args.destination / args.model_subdir
    required = (
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"download is incomplete; missing: {missing}")
    print(model_path)


if __name__ == "__main__":
    main()
