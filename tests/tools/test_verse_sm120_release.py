# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2] / "tools" / "verse" / "finalize_sm120_release.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("verse_sm120_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

IMAGE_DIGEST = f"registry/runtime@sha256:{'a' * 64}"
FORK_COMMIT = "b" * 40
MODEL_REVISION = "e2c6cd9c3302e91c032a378a607009c82ba16fac"
CONTAINER_ID = "c" * 64
GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU_IDENTITY = f"NVIDIA GeForce RTX 5070 Ti, {GPU_UUID}, 16303, 590.00"
RELEASE_NONCE = "e" * 64
MACHINE_ID_SHA256 = "1" * 64
BOOT_ID = "11111111-2222-3333-4444-555555555555"
DOCKER_ID = "DOCKER:HOST:1234"
HOST_IDENTITY_SHA256 = hashlib.sha256(
    f"{MACHINE_ID_SHA256}\n{DOCKER_ID}\n{BOOT_ID}\n".encode()
).hexdigest()
HOST = {
    "identity_sha256": HOST_IDENTITY_SHA256,
    "machine_id_sha256": MACHINE_ID_SHA256,
    "boot_id": BOOT_ID,
    "docker_id": DOCKER_ID,
    "docker_name": "disposable-sm120",
}
GPU = {
    "name": "NVIDIA GeForce RTX 5070 Ti",
    "uuid": GPU_UUID,
    "memory_total_mib": 16303,
    "driver_version": "590.00",
    "compute_capability": [12, 0],
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def container(started_at: str = "2026-08-28T00:00:00Z") -> list[dict]:
    return [
        {
            "Id": CONTAINER_ID,
            "Image": "sha256:image-id",
            "RestartCount": 0,
            "State": {
                "Running": True,
                "OOMKilled": False,
                "StartedAt": started_at,
            },
            "Config": {
                "Image": IMAGE_DIGEST,
                "Labels": {
                    "ai.vllm.build.commit": FORK_COMMIT,
                    "ai.verse.gpu.uuid": GPU_UUID,
                },
            },
            "HostConfig": {
                "DeviceRequests": [
                    {
                        "Driver": "nvidia",
                        "DeviceIDs": ["0"],
                        "Capabilities": [["gpu"]],
                    }
                ]
            },
        }
    ]


def stream(content_chunks: int = 2) -> dict:
    return {
        "chunks": content_chunks,
        "content_chunks": content_chunks,
        "content_characters": 32,
        "saw_done": True,
    }


def chat_contract() -> dict:
    return {
        "status": "pass",
        "scope": "complete",
        "model": "verse-free",
        "ordinary_prompt_tokens": 64,
        "ordinary_stream": stream(),
        "deterministic_completion": {
            "completion_tokens": 16,
            "content_sha256": "c" * 64,
            "tokens_sha256": "d" * 64,
            "content_characters": 48,
            "minimum_logprob": -2.0,
            "maximum_logprob": -0.1,
        },
        "boundary_accepted_prompt_tokens": 6143,
        "boundary_accepted_stream": stream(1),
        "boundary_rejected_prompt_tokens": 6144,
        "boundary_rejected_http_status": 400,
        "exact_boundary_capacity": {
            "concurrency": 38,
            "prompt_tokens_per_request": 6080,
            "max_tokens_per_request": 64,
            "context_tokens_per_request": 6144,
            "total_stream_chunks": 76,
            "observed_max_running": 38,
            "running_metric_samples": 10,
            "preemptions_before": 0,
            "preemptions_after": 0,
            "scheduler_running_before": 0,
            "scheduler_waiting_before": 0,
            "scheduler_running_after": 0,
            "scheduler_waiting_after": 0,
        },
    }


def b01_report(target: int, run: int) -> dict:
    throughput = 1200.0 if target == 1000 else 1040.0
    prewarmed = run != 1
    return {
        "status": "pass",
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "gpu_name": GPU_IDENTITY,
        "server_version": "0.28.0",
        "release_nonce": RELEASE_NONCE,
        "concurrency": 38,
        "requested_prompt_tokens": target,
        "prompt_tokens_min": target - 1,
        "prompt_tokens_max": target,
        "output_tokens_per_request": 512,
        "generated_tokens": 38 * 512,
        "expected_generated_tokens": 38 * 512,
        "completed_request_count": 38,
        "expected_request_count": 38,
        "completion_tokens_by_request": [512] * 38,
        "all_requests_returned_exact_output_tokens": True,
        "max_running_requests_observed": 38,
        "server_idle_before_ownership": True,
        "server_idle_before_measurement": True,
        "server_idle_after_measurement": True,
        "preemptions_delta": 0,
        "preemption_error": None,
        "steady_window_seconds": 12.0,
        "steady_window_samples": 200,
        "steady_aggregate_tokens_per_second": throughput,
        "wall_aggregate_tokens_per_second": throughput,
        "prewarmed": prewarmed,
        "prefix_preparation": (
            "explicitly_prewarmed_this_run"
            if prewarmed
            else "no_explicit_prewarm_this_run"
        ),
    }


def b01_reports() -> list[dict]:
    return [b01_report(target, run) for target in (1000, 5500) for run in range(1, 4)]


def b01_summary(reports: list[dict] | None = None) -> dict:
    return MODULE.evaluate_b01(reports if reports is not None else b01_reports())


def cuda_image_verification() -> dict:
    return {
        "status": "valid",
        "python": "3.12.11",
        "torch": "2.13.0+cu130",
        "torch_cuda": "13.0",
        "flashinfer_commit": MODULE.EXPECTED_FLASHINFER_COMMIT,
        "distributions": MODULE.EXPECTED_DISTRIBUTIONS,
        "distribution_paths": {
            name: "/usr/local/lib/python3.12/site-packages"
            for name in MODULE.EXPECTED_DISTRIBUTIONS
        },
        "gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "capability": [12, 0]},
    }


def cuda_tests() -> list[dict]:
    routing = [
        {
            "node_id": (
                "tests/v1/attention/"
                "test_gemma4_nvfp4_flashinfer_routing.py::"
                f"test_routing_{index}"
            ),
            "result": "passed",
            "suite": "routing",
        }
        for index in range(24)
    ]
    routing.extend(
        {
            "node_id": (
                "tests/v1/attention/test_nvfp4_flashinfer_vosplit.py::"
                f"test_vosplit_{index}"
            ),
            "result": "passed",
            "suite": "routing",
        }
        for index in range(23)
    )
    gpu = [
        {
            "node_id": MODULE.CUDA_GPU_TEST_PREFIXES[0] + f"[case-{index}]",
            "result": "passed",
            "suite": "gpu_oracle",
        }
        for index in range(4)
    ]
    gpu.extend(
        {
            "node_id": MODULE.CUDA_GPU_TEST_PREFIXES[1] + f"[case-{index}]",
            "result": "passed",
            "suite": "gpu_oracle",
        }
        for index in range(2)
    )
    return [*routing, *gpu]


def cuda_artifact_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((MODULE.ROOT / relative).read_bytes()).hexdigest()
        for relative in MODULE.CUDA_TEST_ARTIFACT_RELATIVE_PATHS
    }


def cuda_log() -> str:
    lines = [
        json.dumps(cuda_image_verification(), indent=2, sort_keys=True),
        f"VERSE_CUDA_GPU_IDENTITY={GPU_IDENTITY}",
    ]
    lines.extend(
        f"VERSE_CUDA_TEST_RESULT={test['result']} {test['suite']} {test['node_id']}"
        for test in cuda_tests()
    )
    lines.extend(
        f"VERSE_CUDA_TEST_ARTIFACT_SHA256={digest}  {relative}"
        for relative, digest in cuda_artifact_hashes().items()
    )
    lines.extend(MODULE.CUDA_MARKERS)
    return "\n".join(lines) + "\n"


def cuda_identity(log: str) -> dict:
    return {
        "schema_version": 3,
        "status": "pass",
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "gpu_selector": GPU_UUID,
        "gpu": GPU,
        "host": HOST,
        "tests": cuda_tests(),
        "test_artifacts_sha256": cuda_artifact_hashes(),
        "test_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "oracle_markers": list(MODULE.CUDA_MARKERS),
        "image_verification": cuda_image_verification(),
    }


def server_record() -> str:
    return (
        "status=healthy\n"
        "container=verse-vllm-sm120\n"
        "endpoint=http://127.0.0.1:8000\n"
        f"commit={FORK_COMMIT}\n"
        "profile=sm120-gemma4-nvfp4-v2\n"
    )


def churn() -> dict:
    return {
        "status": "pass",
        "duration_seconds": 7201,
        "concurrency": 38,
        "prompt_pool_size": 64,
        "completed_requests": 100,
        "cancelled_requests": 20,
        "stream_chunks": 1000,
        "request_errors": 0,
        "metrics_before": {
            "running": 0,
            "waiting": 0,
            "preemptions": 0,
            "prefix_hits": 100,
        },
        "metrics_after": {
            "running": 0,
            "waiting": 0,
            "preemptions": 0,
            "prefix_hits": 200,
        },
    }


def candidate_host() -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "container_id": CONTAINER_ID,
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "container_gpu_selector": "0",
        "gpu": GPU,
        "host": HOST,
    }


def release_tree(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    reports = b01_reports()
    for relative, report in zip(MODULE.B01_REPORT_RELATIVE_PATHS, reports, strict=True):
        write_json(release / relative, report)
    write_json(release / "short/b01-summary.json", b01_summary(reports))
    write_json(release / "short/chat-contract.json", chat_contract())
    write_json(release / "churn.json", churn())
    write_json(release / "post-churn-chat-contract.json", chat_contract())
    for relative in (
        "short/container-before.json",
        "short/container-after.json",
        "container-before-cuda.json",
        "container-after-cuda.json",
        "container-after-churn.json",
    ):
        write_json(release / relative, container())
    for relative in (
        "short/preflight.txt",
        "short/postflight.txt",
        "post-churn-server.txt",
    ):
        write_text(release / relative, server_record())
    log = cuda_log()
    write_text(release / "cuda-oracle.log", log)
    write_json(release / "cuda-oracle.json", cuda_identity(log))
    write_json(release / "candidate-host.json", candidate_host())
    return release


def test_release_finalizer_accepts_one_unchanged_container(tmp_path: Path):
    result = MODULE.finalize(release_tree(tmp_path))

    assert result["status"] == "pass"
    assert result["scope"] == "pre_cutover_candidate_qualification"
    assert result["release_nonce"] == RELEASE_NONCE
    assert result["container"]["id"] == CONTAINER_ID
    assert result["cuda_oracle"]["gpu"]["compute_capability"] == [12, 0]
    assert result["cuda_oracle"]["gpu"]["uuid"] == GPU_UUID
    assert result["disposable_host"]["identity_sha256"] == HOST_IDENTITY_SHA256
    assert set(result["artifacts_sha256"]) == set(
        MODULE.EXPECTED_ARTIFACT_RELATIVE_PATHS
    )


def test_release_finalizer_rejects_restart(tmp_path: Path):
    release = release_tree(tmp_path)
    changed = container("2026-08-28T01:00:00Z")
    write_json(release / "container-after-churn.json", changed)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("a restarted container was accepted")


def test_release_finalizer_rejects_placeholder_chat_evidence(tmp_path: Path):
    release = release_tree(tmp_path)
    write_json(release / "short/chat-contract.json", {"status": "pass"})

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "startup-only" in str(exc)
    else:
        raise AssertionError("status-only chat evidence was accepted")


def test_release_finalizer_rejects_capacity_without_real_overlap(tmp_path: Path):
    release = release_tree(tmp_path)
    payload = chat_contract()
    payload["exact_boundary_capacity"]["observed_max_running"] = 1
    write_json(release / "short/chat-contract.json", payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "observed_max_running" in str(exc)
    else:
        raise AssertionError("serialized-only capacity was accepted")


def test_release_finalizer_rejects_churn_preemption(tmp_path: Path):
    release = release_tree(tmp_path)
    payload = churn()
    payload["metrics_after"]["preemptions"] = 1
    write_json(release / "churn.json", payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "preempted" in str(exc)
    else:
        raise AssertionError("preempting churn was accepted")


def test_release_finalizer_rejects_b01_identity_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    reports = []
    for relative in MODULE.B01_REPORT_RELATIVE_PATHS:
        path = release / relative
        payload = json.loads(path.read_text())
        payload["fork_commit"] = "f" * 40
        write_json(path, payload)
        reports.append(payload)
    write_json(release / "short/b01-summary.json", b01_summary(reports))

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "source identity" in str(exc)
    else:
        raise AssertionError("B01 identity drift was accepted")


def test_release_finalizer_rejects_summary_not_derived_from_raw_reports(
    tmp_path: Path,
):
    release = release_tree(tmp_path)
    payload = b01_summary()
    payload["weighted_improvement_over_legacy"] = 0.99
    write_json(release / "short/b01-summary.json", payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "six raw benchmark reports" in str(exc)
    else:
        raise AssertionError("a summary detached from raw reports was accepted")


def test_release_finalizer_rejects_raw_report_nonce_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / MODULE.B01_REPORT_RELATIVE_PATHS[-1]
    payload = json.loads(path.read_text())
    payload["release_nonce"] = "f" * 64
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "release nonce changed" in str(exc)
    else:
        raise AssertionError("mixed raw-report release nonces were accepted")


def test_release_finalizer_requires_machine_readable_cuda_identity(tmp_path: Path):
    release = release_tree(tmp_path)
    (release / "cuda-oracle.json").unlink()

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "missing release artifacts" in str(exc)
        assert "cuda-oracle.json" in str(exc)
    else:
        raise AssertionError("release evidence without CUDA identity was accepted")


def test_release_finalizer_rejects_cuda_log_tampering(tmp_path: Path):
    release = release_tree(tmp_path)
    with (release / "cuda-oracle.log").open("a") as handle:
        handle.write("tampered\n")

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "log hash" in str(exc)
    else:
        raise AssertionError("tampered CUDA output was accepted")


def test_release_finalizer_rejects_cuda_image_identity_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "cuda-oracle.json"
    payload = json.loads(path.read_text())
    payload["image_digest"] = f"registry/runtime@sha256:{'d' * 64}"
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "CUDA image identity" in str(exc)
    else:
        raise AssertionError("CUDA evidence for another image was accepted")


def test_release_finalizer_rejects_missing_cuda_host_identity(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "cuda-oracle.json"
    payload = json.loads(path.read_text())
    del payload["host"]
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "CUDA identity schema" in str(exc)
    else:
        raise AssertionError("CUDA evidence without a host identity was accepted")


def test_release_finalizer_rejects_candidate_host_mismatch(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "candidate-host.json"
    payload = json.loads(path.read_text())
    payload["host"]["docker_id"] = "DOCKER:OTHER:1234"
    payload["host"]["identity_sha256"] = hashlib.sha256(
        (
            f"{payload['host']['machine_id_sha256']}\n"
            f"{payload['host']['docker_id']}\n{payload['host']['boot_id']}\n"
        ).encode()
    ).hexdigest()
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "host identities differ" in str(exc)
    else:
        raise AssertionError("evidence from another candidate host was accepted")


def test_release_finalizer_rejects_candidate_gpu_uuid_mismatch(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "candidate-host.json"
    payload = json.loads(path.read_text())
    payload["gpu"]["uuid"] = "GPU-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee"
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "GPU UUID differs" in str(exc)
    else:
        raise AssertionError("evidence from another candidate GPU was accepted")


def test_release_finalizer_rejects_container_gpu_selector_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "container-after-churn.json"
    payload = json.loads(path.read_text())
    payload[0]["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["1"]
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "GPU binding changed" in str(exc)
    else:
        raise AssertionError("container GPU selector drift was accepted")


def test_release_wrappers_anchor_exact_container_and_gpu_ids():
    release_runner = (
        MODULE.ROOT / "tools/verse/run_sm120_release_gates.sh"
    ).read_text()
    cuda_runner = (MODULE.ROOT / "tools/verse/run_sm120_cuda_gates.sh").read_text()

    assert "require_env VERSE_VLLM_CONTAINER_ID" in release_runner
    assert "${VERSE_VLLM_CONTAINER:-" not in release_runner
    assert 'docker inspect "$VERSE_VLLM_CONTAINER_ID"' in release_runner
    assert 'docker exec "$VERSE_VLLM_CONTAINER_ID"' in release_runner
    assert "require_env VERSE_VLLM_GPU_UUID" in cuda_runner
    assert '--gpus "device=$VERSE_VLLM_GPU_UUID"' in cuda_runner
