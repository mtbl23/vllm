# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
SOURCE_ARCHIVE_SHA256 = "8" * 64
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


def qualification_identity() -> dict:
    return {
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "gpu_name": GPU_IDENTITY,
        "release_nonce": RELEASE_NONCE,
        "container_id": CONTAINER_ID,
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
                    "ai.verse.runtime.profile": MODULE.EXPECTED_PROFILE_IDENTITY,
                    "ai.verse.source.archive.sha256": SOURCE_ARCHIVE_SHA256,
                    "ai.verse.vllm.wheel.version": f"0.28.0+verse.{FORK_COMMIT[:12]}",
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
        **qualification_identity(),
        "scope": "complete",
        "model": "verse-free",
        "ordinary_prompt_tokens": 64,
        "ordinary_stream": stream(),
        "greedy_decode_evidence": {
            "scope": "finite_liveness_only",
            "deterministic_equality_required": False,
            "finite_first_token_runs": [
                {
                    "completion_tokens": 1,
                    "content_sha256": character * 64,
                    "tokens_sha256": character * 64,
                    "content_characters": 3,
                    "minimum_logprob": -2.0,
                    "maximum_logprob": -0.1,
                }
                for character in ("c", "d")
            ],
            "finite_decode_runs": [
                {
                    "completion_tokens": 16,
                    "content_sha256": character * 64,
                    "tokens_sha256": character * 64,
                    "content_characters": 48,
                    "minimum_logprob": -2.0,
                    "maximum_logprob": -0.1,
                }
                for character in ("e", "f")
            ],
        },
        "semantic_rp_evidence": {
            "scope": "safe_rp_semantic_integrity",
            "raw_output_retained": False,
            "runs": [
                {
                    "seed": seed,
                    "content_sha256": character * 64,
                    "character_count": 320,
                    "ascii_word_count": 52,
                    "unique_ascii_word_count": 34,
                    "printable_fraction": 1.0,
                    "ascii_fraction": 1.0,
                    "alphabetic_fraction": 0.75,
                    "common_word_fraction": 0.35,
                    "replacement_character_count": 0,
                    "non_latin_letter_count": 0,
                }
                for seed, character in zip(
                    (1103, 2207, 3301), ("7", "8", "9"), strict=True
                )
            ],
        },
        "boundary_accepted_prompt_tokens": 6143,
        "boundary_accepted_stream": stream(1),
        "boundary_rejected_prompt_tokens": 6144,
        "boundary_rejected_http_status": 400,
        "exact_boundary_capacity": {
            "concurrency": 38,
            "prompt_tokens_per_request": 4096,
            "max_tokens_per_request": 2048,
            "context_tokens_per_request": 6144,
            "total_stream_chunks": 76,
            "verified_exact_completion_streams": 38,
            "observed_completion_tokens_total": 77824,
            "observed_length_finish_streams": 38,
            "stream_completion_evidence": [
                {
                    "stream_index": index,
                    "completion_tokens": 2048,
                    "usage_completion_tokens": 2048,
                    "finish_reason": "length",
                }
                for index in range(38)
            ],
            "observed_max_running": 38,
            "simultaneous_decoding_streams": 38,
            "running_metric_samples": 10,
            "kv_cache_usage_at_simultaneous_decode": 0.9,
            "simultaneous_resident_context_tokens_per_request": 6143,
            "kv_cache_usage_at_simultaneous_6143": 0.9,
            "observed_max_kv_cache_usage": 0.9,
            "kv_cache_usage_after_drain": 0.9,
            "kv_cache_block_bytes": 294912,
            "configured_kv_cache_blocks": 19342,
            "reserved_kv_cache_blocks": 1,
            "usable_kv_cache_blocks": 19341,
            "configured_kv_cache_bytes": 5704253440,
            "concurrent_6144_completion_proven": True,
            "concurrent_6143_residency_proven": True,
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
        "steady_window_prompt_tokens_delta": 0,
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


def prefill_interference(decoders: int, prefills: int) -> dict:
    return {
        "status": "pass",
        "scope": "current_profile_prefill_interference",
        "server_version": "0.28.0",
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "model_revision": MODEL_REVISION,
        "gpu_name": GPU_IDENTITY,
        "release_nonce": RELEASE_NONCE,
        "max_num_batched_tokens": 512,
        "decoders": decoders,
        "prefills": prefills,
        "submitted_requests": decoders + prefills,
        "decode_prompt_tokens": 4500,
        "decode_prompt_tokens_min": 4500,
        "decode_prompt_tokens_max": 4500,
        "decode_output_tokens": 1024,
        "prefill_prompt_tokens_min": 6000,
        "prefill_prompt_tokens_max": 6000,
        "prefill_output_tokens": 1,
        "baseline_seconds": 3.0,
        "metrics_interval_seconds": 0.05,
        "decode_prefill_overlap_proven": True,
        "all_decoders_unfinished_before_prefill": True,
        "all_decoders_unfinished_after_prefill": True,
        "baseline_decode_tok_s": 1200.0,
        "decode_tok_s_during_prefill": 600.0,
        "decode_retention_ratio": 0.5,
        "prefill_wall_seconds": 1.0 if prefills == 1 else 6.0,
        "integrated_decoder_deficit_tokens": (600.0 if prefills == 1 else 3600.0),
        "preemptions_delta": 0,
        "server_idle_after": True,
    }


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
        "vllm_binary_identity": {
            "native_extension": {
                "path": (
                    "/usr/local/lib/python3.12/site-packages/vllm/"
                    "_C_stable_libtorch.abi3.so"
                ),
                "bytes": 123456,
                "wheel_member": "vllm/_C_stable_libtorch.abi3.so",
                "sha256": "3" * 64,
            },
            "wheel_artifact": {
                "filename": (
                    "vllm-0.28.0+verse.test-cp38-abi3-manylinux_2_28_x86_64.whl"
                ),
                "sha256": "4" * 64,
                "manifest_sha256": "5" * 64,
            },
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
        for index in range(37)
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
        for index in range(18)
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
    kv_store = [
        {
            "node_id": node_id,
            "result": "passed",
            "suite": "kv_store_oracle",
        }
        for node_id in MODULE.CUDA_KV_STORE_TEST_NODE_IDS
    ]
    b12x = [
        {
            "node_id": node_id,
            "result": "passed",
            "suite": "b12x_oracle",
        }
        for node_id in MODULE.CUDA_B12X_TEST_NODE_IDS
    ]
    return [*routing, *gpu, *kv_store, *b12x]


def cuda_artifact_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((MODULE.ROOT / relative).read_bytes()).hexdigest()
        for relative in MODULE.CUDA_TEST_ARTIFACT_RELATIVE_PATHS
    }


def cuda_log(tests: list[dict] | None = None) -> str:
    test_inventory = cuda_tests() if tests is None else tests
    lines = [
        json.dumps(cuda_image_verification(), indent=2, sort_keys=True),
        f"VERSE_CUDA_GPU_IDENTITY={GPU_IDENTITY}",
    ]
    lines.extend(
        f"VERSE_CUDA_TEST_RESULT={test['result']} {test['suite']} {test['node_id']}"
        for test in test_inventory
    )
    lines.extend(
        f"VERSE_CUDA_TEST_ARTIFACT_SHA256={digest}  {relative}"
        for relative, digest in cuda_artifact_hashes().items()
    )
    lines.extend(MODULE.CUDA_MARKERS)
    return "\n".join(lines) + "\n"


def cuda_identity(log: str, tests: list[dict] | None = None) -> dict:
    return {
        "schema_version": 3,
        "status": "pass",
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "gpu_selector": GPU_UUID,
        "gpu": GPU,
        "host": HOST,
        "tests": cuda_tests() if tests is None else tests,
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
        f"profile={MODULE.EXPECTED_PROFILE_IDENTITY}\n"
        f"image_receipt_sha256={hashlib.sha256(json.dumps(image_receipt()).encode()).hexdigest()}\n"
        f"model_manifest_sha256={MODULE.EXPECTED_PROFILE['VERSE_MODEL_MANIFEST_SHA256']}\n"
        f"model_config_sha256={'6' * 64}\n"
        f"model_ready_marker_sha256={'7' * 64}\n"
        "model_file_count=22\n"
        "model_bytes=10403188777\n"
    )


def image_receipt() -> dict:
    binary = cuda_image_verification()["vllm_binary_identity"]
    native = binary["native_extension"]
    wheel = binary["wheel_artifact"]
    return {
        "schema_version": 1,
        "status": "approved",
        "approved_at": "2026-08-29T00:00:00+00:00",
        "image_digest": IMAGE_DIGEST,
        "fork_commit": FORK_COMMIT,
        "runtime_profile": MODULE.EXPECTED_PROFILE_IDENTITY,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "vllm_wheel_version": f"0.28.0+verse.{FORK_COMMIT[:12]}",
        "binary_identity": {
            "wheel_filename": wheel["filename"],
            "wheel_sha256": wheel["sha256"],
            "wheel_manifest_sha256": wheel["manifest_sha256"],
            "native_extension_member": native["wheel_member"],
            "native_extension_sha256": native["sha256"],
        },
    }


def churn() -> dict:
    return {
        "status": "pass",
        **qualification_identity(),
        "duration_seconds": 901,
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


def queue_stress() -> dict:
    def phase(name: str, clients: int, waiting: int) -> dict:
        before = {"running": 0, "waiting": 0, "preemptions": 0}
        after = {"running": 0, "waiting": 0, "preemptions": 0}
        return {
            "name": name,
            "clients": clients,
            "active_capacity": 38,
            "request_errors": 0,
            "observed_max_waiting": waiting,
            "metrics_evidence": {
                "observed_max_running": 38,
                "observed_generation_tokens_delta": 1000,
            },
            "metrics_before": before,
            "metrics_after": after,
        }

    return {
        "status": "pass",
        **qualification_identity(),
        "stress_seconds": 120,
        "prompt_token_targets": [5500, 5750, 6000],
        "max_completion_tokens": 128,
        "phases": [phase("max-active", 38, 0), phase("overflow-queue", 76, 38)],
    }


def user_latency() -> dict:
    def mode(name: str, background: int, waiting: int) -> dict:
        samples = []
        for target in (2000, 4000, 6000):
            for _ in range(5):
                samples.append(
                    {
                        "prompt_tokens": target,
                        "completion_tokens": 128,
                        "ttft_seconds": 0.5,
                        "end_to_end_seconds": 2.0,
                        "decode_tokens_per_second": 80.0,
                        "running_at_arrival": min(background, 38),
                        "waiting_at_arrival": waiting,
                    }
                )
        return {
            "mode": name,
            "background_clients": background,
            "measured_user_clients": 15,
            "samples_per_prompt": 5,
            "request_errors": 0,
            "preemptions_delta": 0,
            "samples": samples,
        }

    return {
        "status": "pass",
        **qualification_identity(),
        "completion_tokens_requested": 128,
        "modes": [mode("saturated", 14, 0), mode("overloaded", 52, 14)],
    }


def warm_latency() -> dict:
    def summary(samples: int) -> dict:
        return {
            "samples": samples,
            "ttft_p50_seconds": 0.5,
            "ttft_p95_seconds": 1.0,
            "end_to_end_p50_seconds": 2.0,
            "end_to_end_p95_seconds": 3.0,
            "decode_p05_tokens_per_second": 40.0,
        }

    grouped = {
        "2000": summary(13),
        "4000": summary(13),
        "6000": summary(12),
    }
    return {
        "status": "pass",
        **qualification_identity(),
        "clients": 38,
        "completion_tokens_requested": 100,
        "preemptions_delta": 0,
        "cold": {"pressure": {"max_running": 38}, "by_prompt_tokens": grouped},
        "warm_delta": {
            "pressure": {"max_running": 38},
            "by_prompt_tokens": grouped,
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
    write_json(release / "short/prefill-37x1.json", prefill_interference(37, 1))
    write_json(release / "short/prefill-30x8.json", prefill_interference(30, 8))
    write_json(release / "short/chat-contract.json", chat_contract())
    write_json(release / "queue-stress.json", queue_stress())
    write_json(release / "user-latency.json", user_latency())
    write_json(release / "warm-latency.json", warm_latency())
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
    write_json(release / "image-receipt.json", image_receipt())
    return release


def test_release_finalizer_accepts_one_unchanged_container(tmp_path: Path):
    result = MODULE.finalize(release_tree(tmp_path))

    assert result["status"] == "pass"
    assert result["scope"] == "pre_cutover_candidate_qualification"
    assert result["profile"] == "sm120-gemma4-nvfp4-v4"
    assert result["release_nonce"] == RELEASE_NONCE
    assert result["container"]["id"] == CONTAINER_ID
    assert result["cuda_oracle"]["gpu"]["compute_capability"] == [12, 0]
    assert result["cuda_oracle"]["gpu"]["uuid"] == GPU_UUID
    assert result["cuda_oracle"]["test_count"] == sum(
        MODULE.EXPECTED_CUDA_TEST_COUNTS.values()
    )
    assert result["disposable_host"]["identity_sha256"] == HOST_IDENTITY_SHA256
    assert set(result["artifacts_sha256"]) == set(
        MODULE.EXPECTED_ARTIFACT_RELATIVE_PATHS
    )


def test_release_finalizer_rejects_receipt_binary_identity_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "image-receipt.json"
    payload = json.loads(path.read_text())
    payload["binary_identity"]["native_extension_sha256"] = "f" * 64
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "runtime binary identity differs" in str(exc)
    else:
        raise AssertionError("receipt for unrelated native bytes was accepted")


def test_release_finalizer_rejects_server_receipt_hash_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "post-churn-server.txt"
    receipt_sha256 = hashlib.sha256(json.dumps(image_receipt()).encode()).hexdigest()
    path.write_text(path.read_text().replace(receipt_sha256, "f" * 64))

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "image receipt" in str(exc)
    else:
        raise AssertionError("server using another image receipt was accepted")


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


def test_release_finalizer_rejects_container_profile_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "container-after-churn.json"
    payload = json.loads(path.read_text())
    payload[0]["Config"]["Labels"]["ai.verse.runtime.profile"] = "sm120-gemma4-nvfp4-v1"
    write_json(path, payload)

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "wrong runtime profile" in str(exc)
    else:
        raise AssertionError("container evidence for another profile was accepted")


def test_release_finalizer_rejects_server_profile_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "short/postflight.txt"
    write_text(path, server_record() + "profile=sm120-gemma4-nvfp4-v1\n")

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "wrong runtime profile" in str(exc)
    else:
        raise AssertionError("ambiguous server profile evidence was accepted")


def test_release_finalizer_rejects_placeholder_chat_evidence(tmp_path: Path):
    release = release_tree(tmp_path)
    write_json(release / "short/chat-contract.json", {"status": "pass"})

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "belongs to another candidate" in str(exc)
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


def test_release_finalizer_rejects_incomplete_stream_evidence(tmp_path: Path):
    release = release_tree(tmp_path)
    payload = chat_contract()
    payload["exact_boundary_capacity"]["stream_completion_evidence"].pop()
    write_json(release / "short/chat-contract.json", payload)

    with pytest.raises(ValueError, match="per-stream completion evidence"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_capacity_byte_accounting_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    payload = chat_contract()
    payload["exact_boundary_capacity"]["kv_cache_block_bytes"] += 16
    write_json(release / "short/chat-contract.json", payload)

    with pytest.raises(ValueError, match="fixed KV byte accounting"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_missing_concurrent_completion_proof(tmp_path: Path):
    release = release_tree(tmp_path)
    payload = chat_contract()
    payload["exact_boundary_capacity"]["concurrent_6144_completion_proven"] = False
    write_json(release / "short/chat-contract.json", payload)

    with pytest.raises(ValueError, match="38 concurrent exact-6144 completions"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_missing_simultaneous_6143_residency(
    tmp_path: Path,
):
    release = release_tree(tmp_path)
    payload = chat_contract()
    payload["exact_boundary_capacity"]["concurrent_6143_residency_proven"] = False
    write_json(release / "short/chat-contract.json", payload)

    with pytest.raises(ValueError, match="simultaneous near-full 6K residents"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_prefill_gpu_identity_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "short/prefill-37x1.json"
    payload = json.loads(path.read_text())
    payload["gpu_name"] = GPU_IDENTITY.replace(
        GPU_UUID, "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
    )
    write_json(path, payload)

    with pytest.raises(ValueError, match="prefill interference GPU identities"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_prefill_scheduler_budget_drift(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "short/prefill-30x8.json"
    payload = json.loads(path.read_text())
    payload["max_num_batched_tokens"] = 768
    write_json(path, payload)

    with pytest.raises(ValueError, match="wrong scheduler token budget"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_nonrepresentative_prefill_shape(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "short/prefill-37x1.json"
    payload = json.loads(path.read_text())
    payload["prefill_prompt_tokens_min"] = 1_000
    payload["prefill_prompt_tokens_max"] = 1_000
    write_json(path, payload)

    with pytest.raises(ValueError, match="non-representative request shape"):
        MODULE.finalize(release)


def test_release_finalizer_rejects_poor_prefill_interference(tmp_path: Path):
    release = release_tree(tmp_path)
    path = release / "short/prefill-30x8.json"
    payload = json.loads(path.read_text())
    payload["decode_tok_s_during_prefill"] = 240.0
    payload["decode_retention_ratio"] = 0.2
    payload["integrated_decoder_deficit_tokens"] = 5_760.0
    write_json(path, payload)

    with pytest.raises(ValueError, match="fixed release threshold"):
        MODULE.finalize(release)


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


def test_release_finalizer_rejects_missing_b12x_oracle_node(tmp_path: Path):
    release = release_tree(tmp_path)
    tests = cuda_tests()[:-1]
    log = cuda_log(tests)
    write_text(release / "cuda-oracle.log", log)
    write_json(release / "cuda-oracle.json", cuda_identity(log, tests))

    try:
        MODULE.finalize(release)
    except ValueError as exc:
        assert "B12X" in str(exc) or "b12x_oracle" in str(exc)
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("release evidence without every B12X node was accepted")


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
    dockerfile = (MODULE.ROOT / "docker/Dockerfile").read_text()
    b12x_oracle = (
        MODULE.ROOT / "tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py"
    ).read_text()

    assert "require_env VERSE_VLLM_CONTAINER_ID" in release_runner
    assert "${VERSE_VLLM_CONTAINER:-" not in release_runner
    assert 'docker inspect "$VERSE_VLLM_CONTAINER_ID"' in release_runner
    assert 'docker exec "$VERSE_VLLM_CONTAINER_ID"' in release_runner
    assert "require_env VERSE_VLLM_GPU_UUID" in cuda_runner
    assert '--gpus "device=$VERSE_VLLM_GPU_UUID"' in cuda_runner
    assert "((KV_COUNT == 2))" in cuda_runner
    assert "VERSE_KV_STORE_ORACLE_PASSED" in cuda_runner
    assert "((B12X_COUNT == 6))" in cuda_runner
    assert "VERSE_B12X_ORACLE_PASSED" in cuda_runner
    assert "tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py" in (cuda_runner)
    assert (
        "COPY tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py" in dockerfile
    )
    assert (
        "COPY tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py" in dockerfile
    )
    assert "pytest.skip" not in b12x_oracle
    assert "torch.cuda.get_device_capability(0) == (12, 0)" in b12x_oracle
    assert 'importlib.util.find_spec("b12x") is None' in b12x_oracle
    assert "FlashInferB12xNvFp4LinearKernel" in b12x_oracle
    assert 'backend="b12x"' in b12x_oracle
