#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Bind an image-qualified SM120 release to one exact candidate container."""

from __future__ import annotations

import argparse
import json
import re
import secrets
from pathlib import Path
from typing import Any

from validate_sm120_gateway_release import load_owned_file
from validate_sm120_profile import EXPECTED_PROFILE

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE = re.compile(r".+@sha256:[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_qualification(
    payload: dict[str, Any], *, image: str, commit: str
) -> dict[str, Any]:
    require(payload.get("status") == "pass", "image qualification did not pass")
    require(
        payload.get("scope") == "disposable_image_qualification",
        "manifest is not an image qualification",
    )
    require(
        payload.get("profile") == EXPECTED_PROFILE["VERSE_RUNTIME_PROFILE"],
        "image qualification has the wrong profile",
    )
    require(payload.get("image_digest") == image, "image qualification drifted")
    require(payload.get("source_commit") == commit, "source commit drifted")
    require(
        payload.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "model revision drifted",
    )
    attestation = payload.get("image_attestation")
    require(isinstance(attestation, dict), "image qualification lacks attestation")
    image_repository, image_sha256 = image.rsplit("@sha256:", 1)
    expected = {
        "image_repository": image_repository,
        "image_sha256": image_sha256,
        "source_commit": commit,
        "signer_workflow": ".github/workflows/verse-sm120-image.yml",
        "source_ref": "refs/heads/verse/v0.28-sm120-nvfp4-fa2",
    }
    for name, value in expected.items():
        require(attestation.get(name) == value, f"attestation drifted: {name}")
    require(
        HEX64.fullmatch(str(attestation.get("verification_sha256", ""))) is not None,
        "attestation verification hash is invalid",
    )
    require(
        isinstance(payload.get("b01"), dict) and payload["b01"].get("status") == "pass",
        "image qualification lacks passing B01 evidence",
    )
    return attestation


def bind(
    qualification: dict[str, Any],
    qualification_sha256: str,
    candidate: dict[str, Any],
    candidate_sha256: str,
    *,
    container_id: str,
    image: str,
    commit: str,
) -> dict[str, Any]:
    attestation = validate_qualification(qualification, image=image, commit=commit)
    expected_candidate = {
        "status": "valid",
        "container_id": container_id,
        "image_digest": image,
        "fork_commit": commit,
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "gateway_host_port": 8080,
        "restart_policy": "no",
    }
    for name, value in expected_candidate.items():
        require(candidate.get(name) == value, f"candidate validation drifted: {name}")
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "pre_cutover_candidate_binding",
        "profile": EXPECTED_PROFILE["VERSE_RUNTIME_PROFILE"],
        "source_commit": commit,
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "release_nonce": secrets.token_hex(32),
        "qualification_manifest_sha256": qualification_sha256,
        "candidate_validation_sha256": candidate_sha256,
        "image_attestation": attestation,
        "container": {"id": container_id, "image_digest": image},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    require(HEX64.fullmatch(args.container_id) is not None, "invalid container ID")
    require(IMAGE.fullmatch(args.image) is not None, "invalid image digest")
    require(HEX40.fullmatch(args.expected_commit) is not None, "invalid commit")
    qualification, qualification_sha256 = load_owned_file(args.qualification_manifest)
    candidate, candidate_sha256 = load_owned_file(args.candidate_validation)
    print(
        json.dumps(
            bind(
                qualification,
                qualification_sha256,
                candidate,
                candidate_sha256,
                container_id=args.container_id,
                image=args.image,
                commit=args.expected_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
