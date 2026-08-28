#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from validate_sm120_profile import EXPECTED_PROFILE, load_profile

READY_MARKER = ".verse-sm120-model-ready.json"
OFFICIAL_HUGGING_FACE_ENDPOINT = "https://huggingface.co"
FORBIDDEN_HUGGING_FACE_ENVIRONMENT = (
    "HF_ENDPOINT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_token_file(path: Path, cache_dir: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(
            "Hugging Face token file must be an absolute regular non-symlink file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("Hugging Face token path must be canonical")
    try:
        resolved.relative_to(cache_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Hugging Face token file must be outside the model cache")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("Hugging Face token file must be owner-only")
    raw = resolved.read_bytes()
    if len(raw) > 4096 or b"\r" in raw or b"\0" in raw:
        raise ValueError("Hugging Face token file has an invalid encoding")
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(
            "Hugging Face token file must contain exactly one non-empty line"
        )
    return lines[0].decode()


def validate_hugging_face_transport(
    environment: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environment is None else environment
    forbidden = {name.upper() for name in FORBIDDEN_HUGGING_FACE_ENVIRONMENT}
    configured = sorted(name for name in environment if name.upper() in forbidden)
    if configured:
        raise ValueError(
            "Hugging Face endpoint and proxy overrides are forbidden: "
            + ", ".join(configured)
        )


def download_model_snapshot(
    profile: dict[str, str], cache_dir: Path, token: str
) -> Path:
    validate_hugging_face_transport()

    import huggingface_hub

    return Path(
        huggingface_hub.snapshot_download(
            repo_id=profile["VERSE_MODEL_REPOSITORY"],
            revision=profile["VERSE_MODEL_REVISION"],
            allow_patterns=[f"{profile['VERSE_MODEL_SUBDIR']}/*"],
            cache_dir=cache_dir,
            token=token,
            endpoint=OFFICIAL_HUGGING_FACE_ENDPOINT,
        )
    )


def verify_model_directory(
    model_dir: Path, cache_dir: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    if not model_dir.is_dir() or not _inside(model_dir, cache_dir):
        raise ValueError("model snapshot must be a directory inside the model cache")

    config_path = model_dir / "config.json"
    manifest_path = model_dir / "final_w4a4_build_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError("model snapshot is missing config or build manifest")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("model build manifest checksum does not match the profile")

    config = _load_json(config_path)
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]:
        raise ValueError(
            "model architecture is not the validated Gemma 4 unified model"
        )
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_algo") != "NVFP4":
        raise ValueError("model weights are not ModelOpt NVFP4")
    group = (quant.get("config_groups") or {}).get("group_0") or {}
    weights = group.get("weights") or {}
    activations = group.get("input_activations") or {}
    expected_quant = {
        "dynamic": False,
        "group_size": 16,
        "num_bits": 4,
        "type": "float",
    }
    if weights != expected_quant or activations != expected_quant:
        raise ValueError("model is not the validated W4A4 NVFP4 tuple")
    text_config = config.get("text_config")
    if (
        config.get("tie_word_embeddings") is not True
        or not isinstance(text_config, dict)
        or text_config.get("tie_word_embeddings") is not True
    ):
        raise ValueError("model must retain the validated tied BF16 output head")
    ignored_modules = quant.get("ignore")
    if not isinstance(ignored_modules, list) or "lm_head" not in ignored_modules:
        raise ValueError("model must exclude lm_head from NVFP4 quantization")

    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "campaign22_final_matched_modelopt_nvfp4_w4a4":
        raise ValueError("model build manifest has the wrong campaign kind")
    candidate = manifest.get("candidate") or {}
    files = candidate.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise ValueError("model build manifest has no candidate file inventory")

    total_bytes = 0
    for relative_name, metadata in files.items():
        path = model_dir / relative_name
        if not path.is_file() or not _inside(path, cache_dir):
            raise ValueError(
                f"model file is missing or escapes the cache: {relative_name}"
            )
        expected_bytes = int(metadata["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"model file size mismatch: {relative_name}")
        total_bytes += expected_bytes

    return {
        "manifest_sha256": expected_manifest_sha256,
        "config_sha256": sha256_file(config_path),
        "files": len(files),
        "bytes": total_bytes,
    }


def full_checksum_verification(model_dir: Path, manifest: dict[str, Any]) -> None:
    files = (manifest.get("candidate") or {}).get("files") or {}
    for relative_name, metadata in files.items():
        if sha256_file(model_dir / relative_name) != metadata["sha256"]:
            raise ValueError(f"model file checksum mismatch: {relative_name}")


def verify_materialized_model_directory(
    model_dir: Path,
    cache_dir: Path,
    expected_manifest_sha256: str,
    required_owner: int | None = None,
) -> dict[str, Any]:
    verification = verify_model_directory(
        model_dir, cache_dir, expected_manifest_sha256
    )
    manifest = _load_json(model_dir / "final_w4a4_build_manifest.json")
    files = (manifest.get("candidate") or {}).get("files") or {}
    expected_files = set(files)
    expected_files.add("final_w4a4_build_manifest.json")
    actual_files = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError(
            "materialized model file set does not match the manifest: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in Path(name).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ValueError("materialized model directory set is not closed")
    paths = [model_dir / name for name in expected_files]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"materialized model contains a symlink: {path.name}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"materialized model file is writable: {path.name}")
        if required_owner is not None and path.stat().st_uid != required_owner:
            raise ValueError(
                f"materialized model file has the wrong owner: {path.name}"
            )
    directories = [model_dir]
    directories.extend(path for path in model_dir.rglob("*") if path.is_dir())
    for directory in directories:
        mode = stat.S_IMODE(directory.stat().st_mode)
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(
                f"materialized model directory is writable: {directory.name}"
            )
        if required_owner is not None and directory.stat().st_uid != required_owner:
            raise ValueError(
                f"materialized model directory has the wrong owner: {directory.name}"
            )
    if required_owner is not None:
        trusted_root = cache_dir.resolve(strict=True)
        if model_dir != trusted_root:
            current = model_dir.parent
            while True:
                metadata = current.stat()
                mode = stat.S_IMODE(metadata.st_mode)
                if metadata.st_uid != required_owner:
                    raise ValueError(
                        f"materialized model ancestor has the wrong owner: {current}"
                    )
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    raise ValueError(
                        "materialized model ancestor is group/world writable: "
                        f"{current}"
                    )
                if current == trusted_root:
                    break
                try:
                    current.relative_to(trusted_root)
                except ValueError as exc:
                    raise ValueError(
                        "materialized model ancestor escaped the model cache"
                    ) from exc
                current = current.parent
    full_checksum_verification(model_dir, manifest)
    return verification


def materialize_model_directory(
    source_dir: Path,
    cache_dir: Path,
    profile: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    root = cache_dir / "verse-sm120-materialized"
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    destination = root / (
        f"{profile['VERSE_MODEL_REVISION']}-"
        f"{profile['VERSE_MODEL_MANIFEST_SHA256'][:16]}"
    )
    if destination.exists():
        verification = verify_materialized_model_directory(
            destination,
            cache_dir,
            profile["VERSE_MODEL_MANIFEST_SHA256"],
        )
        return destination, verification

    manifest_path = source_dir / "final_w4a4_build_manifest.json"
    manifest = _load_json(manifest_path)
    files = (manifest.get("candidate") or {}).get("files") or {}
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    try:
        for relative_name in files:
            relative = Path(relative_name)
            if (
                relative.is_absolute()
                or relative == Path(".")
                or ".." in relative.parts
            ):
                raise ValueError(f"unsafe model manifest path: {relative_name}")
            source = source_dir / relative
            target = temporary / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=True)
            target.chmod(0o444)
        target_manifest = temporary / manifest_path.name
        if not target_manifest.exists():
            shutil.copyfile(manifest_path, target_manifest, follow_symlinks=True)
            target_manifest.chmod(0o444)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        verification = verify_materialized_model_directory(
            temporary,
            cache_dir,
            profile["VERSE_MODEL_MANIFEST_SHA256"],
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o755)
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o644)
            shutil.rmtree(temporary)
        raise
    return destination, verification


def write_ready_marker(
    cache_dir: Path,
    snapshot_dir: Path,
    model_dir: Path,
    profile: dict[str, str],
    verification: dict[str, Any],
) -> Path:
    relative_snapshot = snapshot_dir.resolve().relative_to(cache_dir.resolve())
    relative_model = model_dir.resolve().relative_to(cache_dir.resolve())
    marker = cache_dir / READY_MARKER
    temporary = cache_dir / f"{READY_MARKER}.tmp"
    payload = {
        "schema_version": 1,
        "profile_version": profile["VERSE_PROFILE_VERSION"],
        "repository": profile["VERSE_MODEL_REPOSITORY"],
        "revision": profile["VERSE_MODEL_REVISION"],
        "subdir": profile["VERSE_MODEL_SUBDIR"],
        "snapshot_relative_path": str(relative_snapshot),
        "model_relative_path": str(relative_model),
        "verification": verification,
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    temporary.replace(marker)
    return marker


def validate_ready_marker(
    cache_dir: Path,
    profile: dict[str, str],
    required_owner: int | None = None,
) -> dict[str, Any]:
    marker_path = cache_dir / READY_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("model ready marker must be a regular non-symlink file")
    if required_owner is not None:
        metadata = marker_path.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid != required_owner:
            raise ValueError("model ready marker has the wrong owner")
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("model ready marker is group/world writable")
    marker = _load_json(marker_path)
    expected = {
        "schema_version": 1,
        "profile_version": profile["VERSE_PROFILE_VERSION"],
        "repository": profile["VERSE_MODEL_REPOSITORY"],
        "revision": profile["VERSE_MODEL_REVISION"],
        "subdir": profile["VERSE_MODEL_SUBDIR"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"model ready marker mismatch: {key}")
    model_dir = cache_dir / str(marker.get("model_relative_path", ""))
    verification = verify_materialized_model_directory(
        model_dir,
        cache_dir,
        profile["VERSE_MODEL_MANIFEST_SHA256"],
        required_owner,
    )
    if marker.get("verification") != verification:
        raise ValueError("model ready marker verification data is stale")
    return {**marker, "model_directory": str(model_dir.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and verify the exact Verse SM120 model snapshot."
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--verify-mounted-model", action="store_true")
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("sm120_profile.env"),
    )
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--verify-ready", action="store_true")
    parser.add_argument("--require-root-owner", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_profile(args.profile)
    if profile != EXPECTED_PROFILE:
        raise SystemExit("refusing to prepare a model for a drifted runtime profile")
    if args.verify_mounted_model:
        if args.model_directory is None:
            raise SystemExit(
                "--model-directory is required with --verify-mounted-model"
            )
        model_dir = args.model_directory.resolve(strict=True)
        verification = verify_materialized_model_directory(
            model_dir,
            model_dir,
            profile["VERSE_MODEL_MANIFEST_SHA256"],
            0 if args.require_root_owner else None,
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "model_directory": str(model_dir),
                    **verification,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.cache_dir is None:
        raise SystemExit("--cache-dir is required")
    cache_dir = args.cache_dir.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_ready:
        print(
            json.dumps(
                validate_ready_marker(
                    cache_dir,
                    profile,
                    0 if args.require_root_owner else None,
                ),
                sort_keys=True,
            )
        )
        return 0

    if args.token_file is None:
        raise SystemExit("--token-file is required when downloading the model")
    if os.geteuid() != 0:
        raise SystemExit(
            "model preparation must run as root so the serving tree is immutable"
        )
    cache_metadata = cache_dir.stat()
    cache_mode = stat.S_IMODE(cache_metadata.st_mode)
    if cache_metadata.st_uid != 0 or cache_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise SystemExit("model cache must be root-owned and not group/world writable")
    try:
        validate_hugging_face_transport()
        token = load_token_file(args.token_file, cache_dir)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    snapshot_dir = download_model_snapshot(profile, cache_dir, token)
    model_dir = snapshot_dir / profile["VERSE_MODEL_SUBDIR"]
    verify_model_directory(model_dir, cache_dir, profile["VERSE_MODEL_MANIFEST_SHA256"])
    full_checksum_verification(
        model_dir, _load_json(model_dir / "final_w4a4_build_manifest.json")
    )
    materialized_dir, verification = materialize_model_directory(
        model_dir, cache_dir, profile
    )
    marker = write_ready_marker(
        cache_dir, snapshot_dir, materialized_dir, profile, verification
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "marker": str(marker),
                "model_directory": str(materialized_dir.resolve()),
                **verification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
