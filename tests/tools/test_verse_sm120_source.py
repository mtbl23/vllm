# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "tools" / "verse" / "verify_sm120_source.sh"
BUILD = Path(__file__).parents[2] / "tools" / "verse" / "build_sm120_image.sh"
RUN_SERVER = Path(__file__).parents[2] / "tools" / "verse" / "run_sm120_server.sh"
SHA256_HELPER = Path(__file__).parents[2] / "tools" / "verse" / "sm120_sha256.bash"


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def make_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "repository"
    script = repository / "tools" / "verse" / SOURCE.name
    script.parent.mkdir(parents=True)
    shutil.copy2(SOURCE, script)
    run("git", "init", "-q", cwd=repository)
    run("git", "config", "user.email", "test@example.invalid", cwd=repository)
    run("git", "config", "user.name", "Verse test", cwd=repository)
    run("git", "add", ".", cwd=repository)
    committed = run("git", "commit", "-qm", "fixture", cwd=repository)
    assert committed.returncode == 0, committed.stderr
    commit = run("git", "rev-parse", "HEAD", cwd=repository).stdout.strip()
    return repository, script, commit


def invoke(script: Path, commit: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), commit],
        cwd=script.parents[2],
        env={**os.environ, "VERSE_VLLM_EXPECTED_COMMIT": commit},
        capture_output=True,
        text=True,
        check=False,
    )


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def make_build_repository(tmp_path: Path) -> tuple[Path, Path, Path, str, bytes]:
    repository = tmp_path / "build-repository"
    script = repository / "tools" / "verse" / BUILD.name
    script.parent.mkdir(parents=True)
    shutil.copy2(BUILD, script)
    shutil.copy2(SHA256_HELPER, script.parent / SHA256_HELPER.name)
    dockerfile = repository / "docker" / "Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM scratch\n")
    placeholder = repository / "tools" / "verse" / "archive-git-context" / ".keep"
    placeholder.parent.mkdir()
    placeholder.write_text("deterministic archive-build Git mount\n")
    lock = repository / "requirements" / "verse-sm120-flashinfer.lock"
    lock.parent.mkdir()
    lock.write_text("committed dependency lock\n")
    tracked = repository / "tracked.txt"
    tracked.write_text("committed source\n")
    run("git", "init", "-q", cwd=repository)
    run("git", "config", "user.email", "test@example.invalid", cwd=repository)
    run("git", "config", "user.name", "Verse test", cwd=repository)
    run("git", "add", ".", cwd=repository)
    committed = run("git", "commit", "-qm", "build fixture", cwd=repository)
    assert committed.returncode == 0, committed.stderr
    commit = run("git", "rev-parse", "HEAD", cwd=repository).stdout.strip()
    archive = subprocess.run(
        ["git", "-c", "tar.umask=0000", "archive", "--format=tar", commit],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    return repository, script, tracked, commit, archive


def install_fake_build_commands(
    tmp_path: Path,
    tracked: Path,
    commit: str,
    source_sha256: str,
    inspect_source_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, str]]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    real_git = shutil.which("git")
    assert real_git is not None
    write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == status && ${2:-} == --short ]]; then
  "$REAL_GIT" "$@"
  printf 'mutated source\n' >"$MUTATE_PATH"
  exit 0
fi
exec "$REAL_GIT" "$@"
""",
    )
    write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == buildx && ${2:-} == build ]]; then
  printf '%s\n' "$@" >"$FAKE_DOCKER_ARGS"
  tee "$FAKE_DOCKER_CONTEXT" >/dev/null
  exit 0
fi
if [[ ${1:-} == image && ${2:-} == inspect ]]; then
  case ${4:-} in
    '{{.Id}}') printf 'sha256:image-id\n' ;;
    *ai.vllm.build.commit*) printf '%s\n' "$FAKE_COMMIT" ;;
    *org.opencontainers.image.revision*) printf '%s\n' "$FAKE_COMMIT" ;;
    *ai.verse.source.archive.sha256*) printf '%s\n' "$FAKE_INSPECT_SOURCE_SHA256" ;;
    *ai.verse.vllm.wheel.version*) printf '0.28.0+verse.%s\n' "${FAKE_COMMIT:0:12}" ;;
    *) exit 2 ;;
  esac
  exit 0
fi
[[ ${1:-} == run ]]
""",
    )
    docker_args = tmp_path / "docker-args.txt"
    docker_context = tmp_path / "docker-context.tar"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_GIT": real_git,
        "MUTATE_PATH": str(tracked),
        "FAKE_DOCKER_ARGS": str(docker_args),
        "FAKE_DOCKER_CONTEXT": str(docker_context),
        "FAKE_COMMIT": commit,
        "FAKE_INSPECT_SOURCE_SHA256": inspect_source_sha256 or source_sha256,
    }
    return docker_args, docker_context, environment


def invoke_build(
    script: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script)],
        cwd=script.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def invoke_sha256_helper(
    fake_bin: Path, target: Path, hash_log: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; verse_sha256 "$2"',
            "bash",
            str(SHA256_HELPER),
            str(target),
        ],
        env={"PATH": str(fake_bin), "HASH_LOG": str(hash_log)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_source_gate_accepts_exact_clean_commit(tmp_path: Path):
    _, script, commit = make_repository(tmp_path)

    result = invoke(script, commit)

    assert result.returncode == 0, result.stderr
    assert f"commit={commit}" in result.stdout


def test_source_gate_rejects_dirty_or_wrong_commit(tmp_path: Path):
    repository, script, commit = make_repository(tmp_path)
    dirty = repository / "dirty.txt"
    dirty.write_text("uncommitted\n")

    result = invoke(script, commit)

    assert result.returncode != 0
    assert "clean source tree" in result.stderr

    dirty.unlink()
    wrong = invoke(script, "f" * 40)
    assert wrong.returncode != 0
    assert "does not match candidate" in wrong.stderr


def test_image_build_streams_exact_commit_archive_and_binds_provenance(
    tmp_path: Path,
):
    repository, script, tracked, commit, expected_archive = make_build_repository(
        tmp_path
    )
    source_sha256 = hashlib.sha256(expected_archive).hexdigest()
    docker_args, docker_context, environment = install_fake_build_commands(
        tmp_path, tracked, commit, source_sha256
    )

    result = invoke_build(script, environment)

    assert result.returncode == 0, result.stderr
    assert tracked.read_text() == "mutated source\n"
    actual_archive = docker_context.read_bytes()
    assert actual_archive == expected_archive
    with tarfile.open(fileobj=io.BytesIO(actual_archive), mode="r:") as archive:
        source = archive.extractfile("tracked.txt")
        assert source is not None
        assert source.read() == b"committed source\n"
        placeholder = archive.extractfile("tools/verse/archive-git-context/.keep")
        assert placeholder is not None
        assert b"deterministic archive-build Git mount" in placeholder.read()
        assert not any(member.name == ".git" for member in archive.getmembers())
    arguments = docker_args.read_text().splitlines()
    assert arguments[-1] == "-"
    assert f"ai.verse.source.archive.sha256={source_sha256}" in arguments
    assert f"org.opencontainers.image.revision={commit}" in arguments
    wheel_version = f"0.28.0+verse.{commit[:12]}"
    assert f"VLLM_VERSION_OVERRIDE={wheel_version}" in arguments
    assert "VLLM_VERSE_SM120_WHEEL=1" in arguments
    assert "GIT_REPO_CHECK=0" in arguments
    assert "GIT_REPO_MOUNT_SOURCE=tools/verse/archive-git-context" in arguments
    assert f"ai.verse.vllm.wheel.version={wheel_version}" in arguments
    assert f"vllm_wheel_version={wheel_version}" in result.stdout
    assert f"source_archive_sha256={source_sha256}" in result.stdout
    assert run("git", "status", "--short", cwd=repository).stdout


def test_image_build_rejects_mismatched_source_archive_label(tmp_path: Path):
    _, script, tracked, commit, expected_archive = make_build_repository(tmp_path)
    source_sha256 = hashlib.sha256(expected_archive).hexdigest()
    _, _, environment = install_fake_build_commands(
        tmp_path,
        tracked,
        commit,
        source_sha256,
        inspect_source_sha256="0" * 64,
    )

    result = invoke_build(script, environment)

    assert result.returncode != 0
    assert "source archive label does not match" in result.stderr


def test_sha256_helper_prefers_sha256sum(tmp_path: Path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    hash_log = tmp_path / "hash.log"
    target = tmp_path / "payload"
    target.write_text("payload\n")
    write_executable(
        fake_bin / "sha256sum",
        """#!/bin/bash
printf 'sha256sum:%s\n' "$*" >>"$HASH_LOG"
printf '%064d  %s\n' 0 "$1"
""",
    )
    write_executable(
        fake_bin / "shasum",
        """#!/bin/bash
printf 'shasum:%s\n' "$*" >>"$HASH_LOG"
printf '%064d  %s\n' 1 "${3:-}"
""",
    )

    result = invoke_sha256_helper(fake_bin, target, hash_log)

    assert result.returncode == 0, result.stderr
    assert hash_log.read_text() == f"sha256sum:{target}\n"


def test_sha256_helper_falls_back_to_shasum_a_256(tmp_path: Path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    hash_log = tmp_path / "hash.log"
    target = tmp_path / "payload"
    target.write_text("payload\n")
    write_executable(
        fake_bin / "shasum",
        """#!/bin/bash
printf 'shasum:%s\n' "$*" >>"$HASH_LOG"
printf '%064d  %s\n' 0 "$3"
""",
    )

    result = invoke_sha256_helper(fake_bin, target, hash_log)

    assert result.returncode == 0, result.stderr
    assert hash_log.read_text() == f"shasum:-a 256 {target}\n"


def test_sha256_helper_fails_closed_without_supported_tool(tmp_path: Path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    hash_log = tmp_path / "hash.log"
    target = tmp_path / "payload"
    target.write_text("payload\n")

    result = invoke_sha256_helper(fake_bin, target, hash_log)

    assert result.returncode == 127
    assert "sha256sum or shasum is required" in result.stderr


def test_build_and_run_scripts_use_portable_sha256_helper():
    for script in (BUILD, RUN_SERVER):
        text = script.read_text()
        assert 'source "$ROOT/tools/verse/sm120_sha256.bash"' in text
        assert "verse_sha256" in text
        assert "shasum -a 256" not in text


def test_dockerfile_preserves_upstream_git_builds_and_archive_builds():
    dockerfile = (SOURCE.parents[2] / "docker" / "Dockerfile").read_text()

    assert "ARG GIT_REPO_MOUNT_SOURCE=.git" in dockerfile
    assert dockerfile.count("source=${GIT_REPO_MOUNT_SOURCE},target=.git") == 2
    assert "source=.git,target=.git" not in dockerfile
    assert "ENV VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE}" in dockerfile
    assert "VLLM_VERSE_SM120_WHEEL=${VLLM_VERSE_SM120_WHEEL}" in dockerfile
    assert dockerfile.count("ARG VLLM_VERSE_SM120_WHEEL=0") == 2
    assert "ENV UV_OVERRIDE=/etc/uv-overrides-verse-sm120.txt" in dockerfile
    assert "requirements/verse-sm120-runtime.lock" in dockerfile
    assert "--no-deps --require-hashes" in dockerfile
    assert "        deep_ep" in dockerfile
    assert "        b12x" in dockerfile


def test_setup_metadata_has_an_explicit_verse_runtime_dependency_contract():
    setup = (SOURCE.parents[2] / "setup.py").read_text()
    requirements = (
        SOURCE.parents[2] / "requirements" / "verse-sm120-wheel.txt"
    ).read_text()

    assert 'os.getenv("VLLM_VERSE_SM120_WHEEL") == "1"' in setup
    assert 'build_vllm_flash_attn = os.getenv("VLLM_VERSE_SM120_WHEEL") != "1"' in setup
    for package in (
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cudnn-frontend",
    ):
        assert package in requirements
    assert "quack-kernels" not in requirements
    verifier = (
        SOURCE.parents[2] / "tools" / "verse" / "verify_sm120_image.py"
    ).read_text()
    assert "verify_vllm_wheel_requirements" in verifier
    assert 'not distributions.get("b12x")' in verifier
    assert "must not ship bundled vllm-flash-attn extensions" in verifier


def test_verse_wheel_omits_the_unused_bundled_flash_attention_matrix():
    root = SOURCE.parents[2]
    cmake = (root / "CMakeLists.txt").read_text()
    fa_utils = (
        root / "vllm" / "v1" / "attention" / "backends" / "fa_utils.py"
    ).read_text()

    assert 'if (NOT "$ENV{VLLM_VERSE_SM120_WHEEL}" STREQUAL "1")' in cmake
    assert "include(cmake/external_projects/vllm_flash_attn.cmake)" in cmake
    assert "_CUDA_FLASH_ATTN_AVAILABLE = False" in fa_utils
    assert "if not envs.VLLM_VERSE_RUNTIME_STRICT:" in fa_utils
    assert "CUDA FlashAttention is unavailable in this vLLM build" in fa_utils
    assert "return _CUDA_FLASH_ATTN_AVAILABLE" in fa_utils
