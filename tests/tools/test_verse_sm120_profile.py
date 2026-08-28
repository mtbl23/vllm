# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
VALIDATOR = ROOT / "tools" / "verse" / "validate_sm120_profile.py"
PROFILE = ROOT / "tools" / "verse" / "sm120_profile.env"
COMMIT = "2" * 40


def run_validator(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--expected-commit",
            COMMIT,
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_profile_accepts_immutable_image():
    result = run_validator("--image", f"registry/runtime@sha256:{'a' * 64}")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"


def test_profile_rejects_mutable_image_by_default():
    result = run_validator("--image", "registry/runtime:latest")

    assert result.returncode != 0
    assert "pinned by sha256 digest" in result.stderr


def test_profile_rejects_configuration_drift(tmp_path: Path):
    drifted = tmp_path / "profile.env"
    drifted.write_text(
        PROFILE.read_text().replace("VERSE_MAX_NUM_SEQS=38", "VERSE_MAX_NUM_SEQS=40")
    )

    result = run_validator(
        "--profile",
        str(drifted),
        "--image",
        f"registry/runtime@sha256:{'a' * 64}",
    )

    assert result.returncode != 0
    assert "profile drifted" in result.stderr


def test_profile_rejects_unknown_key(tmp_path: Path):
    drifted = tmp_path / "profile.env"
    drifted.write_text(PROFILE.read_text() + "VERSE_UNREVIEWED_SETTING=1\n")

    result = run_validator(
        "--profile",
        str(drifted),
        "--image",
        f"registry/runtime@sha256:{'a' * 64}",
    )

    assert result.returncode != 0
    assert "profile drifted" in result.stderr


def test_profile_never_executes_shell(tmp_path: Path):
    marker = tmp_path / "executed"
    drifted = tmp_path / "profile.env"
    drifted.write_text(PROFILE.read_text() + f"touch {marker}\n")

    result = run_validator(
        "--profile",
        str(drifted),
        "--image",
        f"registry/runtime@sha256:{'a' * 64}",
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_profile_emits_shell_safe_assignments():
    result = run_validator(
        "--image",
        f"registry/runtime@sha256:{'a' * 64}",
        "--emit-shell",
    )

    assert result.returncode == 0, result.stderr
    assert "VERSE_MAX_MODEL_LEN=6144" in result.stdout
    assert "cudagraph_mode" in result.stdout
