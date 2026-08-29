#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

IMAGE_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
HEX40_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
WHEEL_VERSION_RE = re.compile(r"0\.28\.0\+verse\.[0-9a-f]{12}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be an absolute regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def validate_receipt_file(path: Path) -> dict:
    payload = load_json(path)
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("image receipt path must be canonical")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise ValueError("image receipt must have exact mode 0600")
    if path.stat().st_uid != os.geteuid():
        raise ValueError("image receipt must be owned by the caller")
    return payload


def identity_from_verification(verification: dict) -> dict[str, str]:
    binary = verification.get("vllm_binary_identity")
    if not isinstance(binary, dict):
        raise ValueError("image verification lacks vLLM binary identity")
    wheel = binary.get("wheel_artifact")
    native = binary.get("native_extension")
    if not isinstance(wheel, dict) or not isinstance(native, dict):
        raise ValueError("image verification has malformed vLLM binary identity")
    result = {
        "wheel_filename": str(wheel.get("filename", "")),
        "wheel_sha256": str(wheel.get("sha256", "")),
        "wheel_manifest_sha256": str(wheel.get("manifest_sha256", "")),
        "native_extension_member": str(native.get("wheel_member", "")),
        "native_extension_sha256": str(native.get("sha256", "")),
    }
    if (
        Path(result["wheel_filename"]).name != result["wheel_filename"]
        or not result["wheel_filename"].endswith(".whl")
        or not result["native_extension_member"].startswith("vllm/")
        or not result["native_extension_member"].endswith(".so")
        or any(
            HEX64_RE.fullmatch(result[name]) is None
            for name in (
                "wheel_sha256",
                "wheel_manifest_sha256",
                "native_extension_sha256",
            )
        )
    ):
        raise ValueError("image verification has malformed binary hashes")
    return result


def validate_receipt_shape(receipt: dict) -> None:
    expected = {
        "schema_version",
        "status",
        "approved_at",
        "image_digest",
        "fork_commit",
        "runtime_profile",
        "source_archive_sha256",
        "vllm_wheel_version",
        "binary_identity",
    }
    if set(receipt) != expected or receipt.get("schema_version") != 1:
        raise ValueError("image receipt schema is invalid")
    if receipt.get("status") != "approved":
        raise ValueError("image receipt is not approved")
    if IMAGE_RE.fullmatch(str(receipt.get("image_digest", ""))) is None:
        raise ValueError("image receipt digest is invalid")
    if HEX40_RE.fullmatch(str(receipt.get("fork_commit", ""))) is None:
        raise ValueError("image receipt fork commit is invalid")
    if HEX64_RE.fullmatch(str(receipt.get("source_archive_sha256", ""))) is None:
        raise ValueError("image receipt source archive hash is invalid")
    if WHEEL_VERSION_RE.fullmatch(str(receipt.get("vllm_wheel_version", ""))) is None:
        raise ValueError("image receipt wheel version is invalid")
    binary = receipt.get("binary_identity")
    if (
        not isinstance(binary, dict)
        or identity_from_verification(
            {
                "vllm_binary_identity": {
                    "wheel_artifact": {
                        "filename": binary.get("wheel_filename"),
                        "sha256": binary.get("wheel_sha256"),
                        "manifest_sha256": binary.get("wheel_manifest_sha256"),
                    },
                    "native_extension": {
                        "wheel_member": binary.get("native_extension_member"),
                        "sha256": binary.get("native_extension_sha256"),
                    },
                }
            }
        )
        != binary
    ):
        raise ValueError("image receipt binary identity is invalid")


def create_receipt(args: argparse.Namespace) -> None:
    if args.output.exists() or not args.output.is_absolute():
        raise ValueError("receipt output must be a new absolute path")
    verification = load_json(args.verification)
    if verification.get("status") != "valid":
        raise ValueError("image verification did not pass")
    receipt = {
        "schema_version": 1,
        "status": "approved",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "image_digest": args.image,
        "fork_commit": args.fork_commit,
        "runtime_profile": args.runtime_profile,
        "source_archive_sha256": args.source_archive_sha256,
        "vllm_wheel_version": args.vllm_wheel_version,
        "binary_identity": identity_from_verification(verification),
    }
    validate_receipt_shape(receipt)
    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_receipt(args: argparse.Namespace) -> None:
    receipt = validate_receipt_file(args.receipt)
    validate_receipt_shape(receipt)
    expected = {
        "image_digest": args.image,
        "fork_commit": args.fork_commit,
        "runtime_profile": args.runtime_profile,
        "source_archive_sha256": args.source_archive_sha256,
        "vllm_wheel_version": args.vllm_wheel_version,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"image receipt {name} does not match the candidate")
    if args.verification is not None:
        verification = load_json(args.verification)
        if verification.get("status") != "valid":
            raise ValueError("image verification did not pass")
        if identity_from_verification(verification) != receipt["binary_identity"]:
            raise ValueError("runtime binary identity differs from the image receipt")
    print(
        json.dumps(
            {
                "status": "valid",
                "receipt_sha256": sha256_file(args.receipt),
                "image_digest": args.image,
                "fork_commit": args.fork_commit,
                "binary_identity": receipt["binary_identity"],
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in ("create", "verify"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--image", required=True)
        sub.add_argument("--fork-commit", required=True)
        sub.add_argument("--runtime-profile", required=True)
        sub.add_argument("--source-archive-sha256", required=True)
        sub.add_argument("--vllm-wheel-version", required=True)
        sub.add_argument("--verification", type=Path)
    create = subparsers.choices["create"]
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.choices["verify"]
    verify.add_argument("--receipt", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    for name, pattern in (
        ("image", IMAGE_RE),
        ("fork_commit", HEX40_RE),
        ("source_archive_sha256", HEX64_RE),
        ("vllm_wheel_version", WHEEL_VERSION_RE),
    ):
        if pattern.fullmatch(str(getattr(args, name))) is None:
            raise ValueError(f"{name} is invalid")
    if args.action == "create":
        if args.verification is None:
            raise ValueError("create requires --verification")
        create_receipt(args)
    else:
        verify_receipt(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
