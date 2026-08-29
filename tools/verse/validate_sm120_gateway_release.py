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
from pathlib import Path

from validate_sm120_profile import EXPECTED_PROFILE

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_owned_file(path: Path) -> tuple[dict, str]:
    require(
        path.is_absolute() and path.is_file(),
        "release manifest must be an absolute file",
    )
    require(not path.is_symlink(), "release manifest must not be a symlink")
    require(
        path.resolve(strict=True) == path, "release manifest path must be canonical"
    )
    metadata = path.stat()
    require(
        metadata.st_uid == os.geteuid(), "release manifest must be owned by the caller"
    )
    require(
        not stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH),
        "release manifest must not be group/world writable",
    )
    parent = path.parent
    parent_metadata = parent.stat()
    require(
        parent.resolve(strict=True) == parent,
        "release manifest parent must be canonical",
    )
    require(
        parent_metadata.st_uid == os.geteuid(),
        "release manifest parent has the wrong owner",
    )
    require(
        not stat.S_IMODE(parent_metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH),
        "release manifest parent must not be group/world writable",
    )
    raw = path.read_bytes()
    payload = json.loads(raw)
    require(isinstance(payload, dict), "release manifest is not an object")
    return payload, hashlib.sha256(raw).hexdigest()


def validate(
    payload: dict,
    *,
    container_id: str,
    image_digest: str,
    fork_commit: str,
) -> dict[str, str]:
    require(payload.get("status") == "pass", "release manifest did not pass")
    require(
        payload.get("scope") == "pre_cutover_candidate_qualification",
        "release manifest has the wrong scope",
    )
    require(
        payload.get("profile") == EXPECTED_PROFILE["VERSE_RUNTIME_PROFILE"],
        "release manifest has the wrong profile",
    )
    require(
        payload.get("source_commit") == fork_commit,
        "release manifest has the wrong commit",
    )
    require(
        payload.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "release manifest has the wrong model revision",
    )
    nonce = str(payload.get("release_nonce", ""))
    require(HEX64.fullmatch(nonce) is not None, "release manifest has a bad nonce")
    container = payload.get("container")
    require(isinstance(container, dict), "release manifest lacks container identity")
    require(
        container.get("id") == container_id, "release manifest has the wrong container"
    )
    require(
        container.get("image_digest") == image_digest,
        "release manifest has the wrong image",
    )
    return {
        "candidate_container_id": container_id,
        "image_digest": image_digest,
        "fork_commit": fork_commit,
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "release_nonce": nonce,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--fork-commit", required=True)
    args = parser.parse_args()
    require(
        CONTAINER_ID.fullmatch(args.container_id) is not None, "invalid container ID"
    )
    require(
        IMAGE_DIGEST.fullmatch(args.image_digest) is not None, "invalid image digest"
    )
    require(HEX40.fullmatch(args.fork_commit) is not None, "invalid fork commit")
    payload, manifest_sha256 = load_owned_file(args.manifest)
    identity = validate(
        payload,
        container_id=args.container_id,
        image_digest=args.image_digest,
        fork_commit=args.fork_commit,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "release_manifest_sha256": manifest_sha256,
                **identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
