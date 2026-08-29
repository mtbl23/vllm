#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

NATIVE_MEMBER_RE = re.compile(r"vllm/_C_stable_libtorch(?:\.[^.]+)*\.so")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_identity(wheel: Path) -> dict[str, object]:
    if not wheel.is_file() or wheel.is_symlink() or wheel.suffix != ".whl":
        raise ValueError("wheel must be one regular non-symlink .whl file")
    with zipfile.ZipFile(wheel) as archive:
        native_members = [
            name for name in archive.namelist() if NATIVE_MEMBER_RE.fullmatch(name)
        ]
        if len(native_members) != 1:
            raise ValueError("wheel must contain exactly one stable native extension")
        native_member = native_members[0]
        native_bytes = archive.read(native_member)
    return {
        "schema_version": 1,
        "wheel": {
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
        },
        "native_extension": {
            "member": native_member,
            "bytes": len(native_bytes),
            "sha256": hashlib.sha256(native_bytes).hexdigest(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_identity(args.wheel)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
