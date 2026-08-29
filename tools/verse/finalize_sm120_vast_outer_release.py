#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Finalize Campaign 22 evidence from one Vast outer OCI container."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_sm120_acceptance import evaluate as evaluate_b01
from finalize_sm120_release import (
    B01_EXPECTED_SCENARIOS,
    CUDA_TEST_ARTIFACT_RELATIVE_PATHS,
    EXPECTED_CUDA_TEST_COUNTS,
    EXPECTED_PROFILE_IDENTITY,
    ROOT,
    parse_gpu_csv,
    validate_b01_summary,
    validate_bound_identity,
    validate_chat_contract,
    validate_churn,
    validate_image_attestation,
    validate_image_receipt,
    validate_prefill_interference,
    validate_queue_stress,
    validate_user_latency,
    validate_warm_latency,
)
from validate_sm120_profile import EXPECTED_PROFILE

IMAGE_RE = re.compile(r"ghcr\.io/mtbl23/verse-vllm@sha256:[0-9a-f]{64}")
SHA40_RE = re.compile(r"[0-9a-f]{40}")
SHA64_RE = re.compile(r"[0-9a-f]{64}")
GPU_UUID_RE = re.compile(r"GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")
FATAL_LOG_RE = re.compile(
    r"fallback.*(triton|xqa)|using.*triton_attn|traceback|fatal|out of memory|"
    r"oom-killed|cuda graph.*(fail|hang)|illegal memory access",
    re.IGNORECASE,
)
EXPECTED_QUALIFICATION_TOOLS = (
    "benchmarks/verse/sm120_b01.py",
    "benchmarks/verse/sm120_prefill_interference.py",
    "tools/verse/check_sm120_chat_contract.py",
    "tools/verse/evaluate_sm120_acceptance.py",
    "tools/verse/finalize_sm120_vast_outer_release.py",
    "tools/verse/probe_sm120_outer_auth.py",
    "tools/verse/run_sm120_churn.py",
    "tools/verse/run_sm120_outer_cuda_gates.sh",
    "tools/verse/run_sm120_queue_stress.py",
    "tools/verse/run_sm120_user_latency.py",
    "tools/verse/run_sm120_vast_outer.py",
    "tools/verse/run_sm120_vast_release_gates.sh",
    "tools/verse/run_sm120_warm_latency.py",
    "tools/verse/sm120_evidence_identity.py",
    "tools/verse/snapshot_sm120_vast_outer.py",
    "tools/verse/verify_sm120_attestation.sh",
)
EXPECTED_ARTIFACTS = (
    "attestation-verification.json",
    "auth-proof.json",
    "b01-1000-run-1.json",
    "b01-1000-run-2.json",
    "b01-1000-run-3.json",
    "b01-5500-run-1.json",
    "b01-5500-run-2.json",
    "b01-5500-run-3.json",
    "b01-summary.json",
    "chat-contract.json",
    "churn.json",
    "commit.txt",
    "cuda-oracle.log",
    "image-receipt.json",
    "image-verification.json",
    "image.txt",
    "live-process-after.json",
    "live-process-before.json",
    "post-churn-chat-contract.json",
    "prefill-30x8.json",
    "prefill-37x1.json",
    "provider-after.json",
    "provider-before.json",
    "qualification-tools.json",
    "queue-stress.json",
    "server.log",
    "ssh-host-fingerprint.txt",
    "ssh-known-hosts",
    "user-latency.json",
    "warm-latency.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_object(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid evidence: {path}")
    payload = json.loads(path.read_bytes())
    require(isinstance(payload, dict), f"evidence is not an object: {path}")
    return payload


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_qualification_tools(payload: dict[str, Any]) -> dict[str, str]:
    require(payload.get("schema_version") == 1, "bad qualification tool schema")
    hashes = payload.get("sha256")
    require(isinstance(hashes, dict), "qualification tool hashes are absent")
    expected = {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in EXPECTED_QUALIFICATION_TOOLS
    }
    require(hashes == expected, "qualification tools differ from the release source")
    return expected


def provider_identity(payload: dict[str, Any], image: str) -> tuple[Any, ...]:
    captured = datetime.fromisoformat(str(payload.get("captured_at", "")))
    require(captured.tzinfo is not None, "provider capture time lacks a timezone")
    require(
        captured <= datetime.now(timezone.utc), "provider capture is from the future"
    )
    instance = payload.get("instance")
    require(isinstance(instance, dict), "provider evidence lacks an instance")
    require(
        int(instance.get("id", 0)) > 0
        and int(instance.get("machine_id", 0)) > 0
        and instance.get("actual_status") == "running"
        and instance.get("intended_status") == "running"
        and instance.get("cur_state") == "running",
        "provider instance is not one exact running allocation",
    )
    require(instance.get("image_uuid") == image, "provider launched another image")
    require(instance.get("gpu_name") == "RTX 5070 Ti", "provider used the wrong GPU")
    require(
        int(instance.get("num_gpus", 0)) == 1, "provider exposed the wrong GPU count"
    )
    require(
        isinstance(instance.get("ssh_host"), str)
        and 0 < int(instance.get("ssh_port", 0)) < 65536,
        "provider SSH identity is incomplete",
    )
    return (
        instance["id"],
        instance["machine_id"],
        instance["image_uuid"],
        instance["gpu_name"],
        instance["num_gpus"],
        instance["ssh_host"],
        instance["ssh_port"],
        instance.get("start_date"),
    )


def live_identity(payload: dict[str, Any], commit: str) -> tuple[Any, ...]:
    require(payload.get("status") == "pass", "live process snapshot did not pass")
    require(
        payload.get("fork_commit") == commit
        and payload.get("wheel_version") == f"0.28.0+verse.{commit[:12]}"
        and payload.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "live process uses another runtime identity",
    )
    require(payload.get("uid") == 2000 and payload.get("gid") == 0, "bad server UID")
    require(int(payload.get("pid", 0)) > 1, "bad server PID")
    for field in (
        "executable_sha256",
        "command_sha256",
        "machine_id_sha256",
    ):
        require(
            SHA64_RE.fullmatch(str(payload.get(field, ""))) is not None, f"bad {field}"
        )
    gpu = payload.get("gpu")
    require(isinstance(gpu, dict), "live process lacks GPU identity")
    require(
        gpu.get("name") == "NVIDIA GeForce RTX 5070 Ti"
        and gpu.get("compute_capability") == [12, 0]
        and GPU_UUID_RE.fullmatch(str(gpu.get("uuid", ""))) is not None,
        "live process has the wrong GPU",
    )
    return (
        payload["pid"],
        payload.get("process_start_ticks"),
        payload.get("process_started_at_unix"),
        payload["executable_sha256"],
        payload["command_sha256"],
        payload.get("boot_id"),
        payload["machine_id_sha256"],
        json.dumps(gpu, sort_keys=True),
    )


def validate_cuda_log(
    raw: bytes, image_verification: dict[str, Any], expected_gpu: dict[str, Any]
) -> None:
    text = raw.decode("utf-8")
    verification, _ = json.JSONDecoder().raw_decode(text.lstrip())
    require(verification == image_verification, "CUDA image verification drifted")
    for marker in (
        "VERSE_ROUTING_GATES_PASSED",
        "VERSE_GPU_ORACLE_PASSED",
        "VERSE_KV_STORE_ORACLE_PASSED",
        "VERSE_B12X_ORACLE_PASSED",
        "SM120_HOSTED_CUDA_GATES_PASSED",
    ):
        require(text.splitlines().count(marker) == 1, f"bad CUDA marker: {marker}")
    require(FATAL_LOG_RE.search(text) is None, "CUDA log contains a fatal or fallback")
    tests: dict[str, str] = {}
    counts = {name: 0 for name in EXPECTED_CUDA_TEST_COUNTS}
    for line in text.splitlines():
        if not line.startswith("VERSE_CUDA_TEST_RESULT=passed "):
            continue
        suite, node_id = line.removeprefix("VERSE_CUDA_TEST_RESULT=passed ").split(
            " ", 1
        )
        require(suite in counts and node_id not in tests, "bad CUDA test inventory")
        tests[node_id] = suite
        counts[suite] += 1
    require(counts == EXPECTED_CUDA_TEST_COUNTS, "CUDA test count drifted")
    hashes: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("VERSE_CUDA_TEST_ARTIFACT_SHA256="):
            continue
        value = line.removeprefix("VERSE_CUDA_TEST_ARTIFACT_SHA256=")
        match = re.fullmatch(r"([0-9a-f]{64})  (tests/.+\.py)", value)
        require(match is not None and match.group(2) not in hashes, "bad CUDA hash")
        hashes[match.group(2)] = match.group(1)
    expected_hashes = {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in CUDA_TEST_ARTIFACT_RELATIVE_PATHS
    }
    require(hashes == expected_hashes, "CUDA tests differ from the release source")
    gpu_lines = [
        line.removeprefix("VERSE_CUDA_GPU_IDENTITY=")
        for line in text.splitlines()
        if line.startswith("VERSE_CUDA_GPU_IDENTITY=")
    ]
    require(len(gpu_lines) == 1, "CUDA log has the wrong GPU inventory")
    fields = [field.strip() for field in next(csv.reader(gpu_lines))]
    require(len(fields) == 5 and fields[4] == "12.0", "CUDA GPU record is malformed")
    observed = {
        "name": fields[0],
        "uuid": fields[1],
        "memory_total_mib": int(fields[2]),
        "driver_version": fields[3],
        "compute_capability": [12, 0],
    }
    require(observed == expected_gpu, "CUDA and server GPU identities differ")


def finalize(release_dir: Path) -> dict[str, Any]:
    release_dir = release_dir.resolve(strict=True)
    actual = {path.name for path in release_dir.iterdir()}
    require(actual == set(EXPECTED_ARTIFACTS), "release artifact inventory drifted")
    for name in EXPECTED_ARTIFACTS:
        path = release_dir / name
        require(path.is_file() and not path.is_symlink(), f"invalid evidence: {path}")
    image = (release_dir / "image.txt").read_text().strip()
    commit = (release_dir / "commit.txt").read_text().strip()
    require(IMAGE_RE.fullmatch(image) is not None, "image digest is invalid")
    require(SHA40_RE.fullmatch(commit) is not None, "fork commit is invalid")
    before_provider = load_object(release_dir / "provider-before.json")
    after_provider = load_object(release_dir / "provider-after.json")
    require(
        provider_identity(before_provider, image)
        == provider_identity(after_provider, image),
        "provider allocation changed during qualification",
    )
    before_live = load_object(release_dir / "live-process-before.json")
    after_live = load_object(release_dir / "live-process-after.json")
    require(
        live_identity(before_live, commit) == live_identity(after_live, commit),
        "server process, host, or GPU changed during qualification",
    )
    ssh_fingerprint = (release_dir / "ssh-host-fingerprint.txt").read_text().strip()
    require(ssh_fingerprint.startswith("SHA256:"), "SSH host fingerprint is invalid")
    runtime_id = hashlib.sha256(
        json.dumps(
            {
                "provider": provider_identity(before_provider, image),
                "live": live_identity(before_live, commit),
                "ssh": ssh_fingerprint,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    gpu = before_live["gpu"]
    receipt = load_object(release_dir / "image-receipt.json")
    image_verification = load_object(release_dir / "image-verification.json")
    candidate = {
        "Id": runtime_id,
        "Config": {
            "Image": image,
            "Labels": {
                "ai.vllm.build.commit": commit,
                "ai.verse.runtime.profile": EXPECTED_PROFILE_IDENTITY,
                "ai.verse.source.archive.sha256": receipt.get("source_archive_sha256"),
                "ai.verse.vllm.wheel.version": f"0.28.0+verse.{commit[:12]}",
            },
        },
    }
    validate_image_receipt(
        receipt, container=candidate, image_verification=image_verification
    )
    validate_cuda_log(
        (release_dir / "cuda-oracle.log").read_bytes(), image_verification, gpu
    )

    reports = []
    for target in B01_EXPECTED_SCENARIOS:
        for run in range(1, 4):
            report = load_object(release_dir / f"b01-{target}-run-{run}.json")
            require(
                report.get("prewarmed") is (run != 1),
                "B01 prefix-preparation matrix drifted",
            )
            reports.append(report)
    b01 = evaluate_b01(reports)
    require(
        b01 == load_object(release_dir / "b01-summary.json"),
        "B01 summary differs from raw reports",
    )
    validate_b01_summary(b01, candidate)
    require(parse_gpu_csv(b01["identity"]["gpu_name"], "B01") == gpu, "B01 GPU drifted")
    nonce = b01["release_nonce"]
    for name, decoders, prefills in (
        ("prefill-37x1.json", 37, 1),
        ("prefill-30x8.json", 30, 8),
    ):
        validate_prefill_interference(
            load_object(release_dir / name),
            candidate,
            decoders=decoders,
            prefills=prefills,
            release_nonce=nonce,
            expected_gpu=gpu,
        )
    bound = (
        ("chat-contract.json", validate_chat_contract),
        ("queue-stress.json", validate_queue_stress),
        ("user-latency.json", validate_user_latency),
        ("warm-latency.json", validate_warm_latency),
        ("churn.json", validate_churn),
        ("post-churn-chat-contract.json", validate_chat_contract),
    )
    for name, validator in bound:
        payload = load_object(release_dir / name)
        validate_bound_identity(
            payload,
            container=candidate,
            release_nonce=nonce,
            expected_gpu=gpu,
            label=name,
        )
        validator(payload)
    auth = load_object(release_dir / "auth-proof.json")
    require(auth.get("status") == "pass", "auth proof did not pass")
    require(
        auth.get("unauthenticated")
        == {
            "/invocations": 401,
            "/tokenize": 401,
            "/docs": 401,
            "/openapi.json": 401,
            "/v1/models": 401,
        }
        and auth.get("liveness") == {"/health": 200, "/metrics": 200},
        "raw application routes did not fail closed",
    )
    logs = (release_dir / "server.log").read_text(errors="strict")
    for marker in (
        "Strict Verse Gemma 4 runtime validated",
        "Using the FlashInfer FA2 paged wrapper",
        "head_dim=512, page_size=64, speculative_tokens=0, xqa=True",
    ):
        require(marker in logs, f"server log lacks marker: {marker}")
    require(
        FATAL_LOG_RE.search(logs) is None, "server log contains a fatal or fallback"
    )
    attestation_payload = json.loads(
        (release_dir / "attestation-verification.json").read_bytes()
    )
    attestation = validate_image_attestation(attestation_payload, container=candidate)
    qualification_tools = validate_qualification_tools(
        load_object(release_dir / "qualification-tools.json")
    )
    artifacts = {
        name: sha256_bytes((release_dir / name).read_bytes())
        for name in EXPECTED_ARTIFACTS
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "disposable_image_qualification",
        "qualification_mode": "vast_outer_immutable_image",
        "profile": EXPECTED_PROFILE_IDENTITY,
        "release_nonce": nonce,
        "runtime_id": runtime_id,
        "image_digest": image,
        "source_commit": commit,
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "image_attestation": {
            **attestation,
            "verification_sha256": sha256_bytes(
                (release_dir / "attestation-verification.json").read_bytes()
            ),
        },
        "gpu": gpu,
        "provider_instance_id": provider_identity(before_provider, image)[0],
        "qualification_tools_sha256": qualification_tools,
        "b01": b01,
        "artifacts_sha256": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.release_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
