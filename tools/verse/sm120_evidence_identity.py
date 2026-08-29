from __future__ import annotations

import argparse
import re
from typing import Any

IMAGE_DIGEST_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
SHA40_RE = re.compile(r"[0-9a-f]{40}")
NONCE_RE = re.compile(r"[0-9a-f]{64}")
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")


def add_identity_arguments(
    parser: argparse.ArgumentParser, *, required: bool = True
) -> None:
    parser.add_argument("--image-digest", required=required)
    parser.add_argument("--fork-commit", required=required)
    parser.add_argument("--model-revision", required=required)
    parser.add_argument("--gpu-name", required=required)
    parser.add_argument("--release-nonce", required=required)
    parser.add_argument("--container-id", required=required)


def validated_identity(args: argparse.Namespace) -> dict[str, Any]:
    if IMAGE_DIGEST_RE.fullmatch(args.image_digest) is None:
        raise ValueError("image digest must be immutable")
    if SHA40_RE.fullmatch(args.fork_commit) is None:
        raise ValueError("fork commit must be exactly 40 lowercase hex characters")
    if SHA40_RE.fullmatch(args.model_revision) is None:
        raise ValueError("model revision must be exactly 40 lowercase hex characters")
    if NONCE_RE.fullmatch(args.release_nonce) is None:
        raise ValueError("release nonce must be exactly 64 lowercase hex characters")
    if CONTAINER_ID_RE.fullmatch(args.container_id) is None:
        raise ValueError("container ID must be exactly 64 lowercase hex characters")
    gpu_name = str(args.gpu_name)
    if not gpu_name or len(gpu_name) > 512 or "\n" in gpu_name or "\r" in gpu_name:
        raise ValueError("GPU identity must be one non-empty line")
    return {
        "image_digest": args.image_digest,
        "fork_commit": args.fork_commit,
        "model_revision": args.model_revision,
        "gpu_name": gpu_name,
        "release_nonce": args.release_nonce,
        "container_id": args.container_id,
    }


def validated_optional_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    names = (
        "image_digest",
        "fork_commit",
        "model_revision",
        "gpu_name",
        "release_nonce",
        "container_id",
    )
    present = [getattr(args, name, None) is not None for name in names]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("qualification identity must be complete or entirely absent")
    return validated_identity(args)
