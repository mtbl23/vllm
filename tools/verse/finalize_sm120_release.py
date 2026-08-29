#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_sm120_acceptance import (
    EXPECTED_CONCURRENCY,
    MINIMUM_WEIGHTED_IMPROVEMENT,
    PRODUCTION_WEIGHTS,
    RELEASE_NONCE_RE,
)
from evaluate_sm120_acceptance import (
    EXPECTED_SCENARIOS as B01_EXPECTED_SCENARIOS,
)
from evaluate_sm120_acceptance import (
    evaluate as evaluate_b01,
)
from validate_sm120_profile import EXPECTED_PROFILE
from verify_sm120_image import (
    EXPECTED_DISTRIBUTIONS,
    EXPECTED_FLASHINFER_COMMIT,
)

EXPECTED_SCENARIOS = {
    str(target): expectation for target, expectation in B01_EXPECTED_SCENARIOS.items()
}
EXPECTED_PROFILE_IDENTITY = (
    f"sm120-gemma4-nvfp4-v{EXPECTED_PROFILE['VERSE_PROFILE_VERSION']}"
)
EXPECTED_TOTAL_KV_BYTES = int(EXPECTED_PROFILE["VERSE_KV_CACHE_MEMORY_BYTES"])
EXPECTED_BLOCK_KV_BYTES = int(EXPECTED_PROFILE["VERSE_KV_CACHE_BLOCK_BYTES"])
EXPECTED_RESERVED_KV_BLOCKS = int(EXPECTED_PROFILE["VERSE_KV_CACHE_RESERVED_BLOCKS"])
EXPECTED_PREFILL_SHAPES = {
    (37, 1): {
        "minimum_retention": 0.30,
        "maximum_wall_seconds": 1.50,
        "maximum_integrated_deficit_tokens": 1_000.0,
    },
    (30, 8): {
        "minimum_retention": 0.30,
        "maximum_wall_seconds": 8.00,
        "maximum_integrated_deficit_tokens": 6_000.0,
    },
}
ROOT = Path(__file__).parents[2]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
GPU_UUID_RE = re.compile(r"GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")
UUID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
DOCKER_HOST_ID_RE = re.compile(r"[A-Za-z0-9:._-]{8,256}")
CUDA_MARKERS = (
    "VERSE_ROUTING_GATES_PASSED",
    "VERSE_GPU_ORACLE_PASSED",
    "VERSE_KV_STORE_ORACLE_PASSED",
    "VERSE_B12X_ORACLE_PASSED",
    "SM120 image and native NVFP4 FA2/B12X correctness gates passed",
)
CUDA_TEST_ARTIFACT_RELATIVE_PATHS = (
    "tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py",
    "tests/v1/attention/test_nvfp4_flashinfer_vosplit.py",
    "tests/kernels/attention/test_flashinfer.py",
    "tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py",
    "tests/kernels/quantization/nvfp4_utils.py",
    "tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py",
)
CUDA_ROUTING_TEST_PREFIXES = (
    "tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py::",
    "tests/v1/attention/test_nvfp4_flashinfer_vosplit.py::",
)
CUDA_GPU_TEST_PREFIXES = (
    (
        "tests/kernels/attention/test_flashinfer.py::"
        "test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference"
    ),
    (
        "tests/kernels/attention/test_flashinfer.py::"
        "test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference"
    ),
)
CUDA_B12X_TEST_PREFIX = (
    "tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py::"
    "test_flashinfer_b12x_nvfp4_linear_matches_reference"
)
CUDA_B12X_PROFILE_TEST_NODE_IDS = tuple(
    f"{CUDA_B12X_TEST_PREFIX}[{case}]"
    for case in ("decode", "max-seqs", "max-batched-tokens")
)
CUDA_B12X_GEMMA_TEST_PREFIX = (
    "tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py::"
    "test_flashinfer_b12x_nvfp4_gemma_shapes_match_reference"
)
CUDA_B12X_GEMMA_TEST_NODE_IDS = tuple(
    f"{CUDA_B12X_GEMMA_TEST_PREFIX}[{case}]"
    for case in ("gemma-q-proj", "gemma-gate-proj", "gemma-down-proj")
)
CUDA_B12X_TEST_NODE_IDS = (
    CUDA_B12X_PROFILE_TEST_NODE_IDS + CUDA_B12X_GEMMA_TEST_NODE_IDS
)
CUDA_KV_STORE_TEST_PREFIX = (
    "tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py::"
    "test_verse_sm120_nvfp4_physical_hnd_roundtrip"
)
CUDA_KV_STORE_TEST_NODE_IDS = tuple(
    f"{CUDA_KV_STORE_TEST_PREFIX}[{case}-cuda:0]"
    for case in ("shape-regression", "gemma4-runtime")
)
EXPECTED_CUDA_TEST_COUNTS = {
    "routing": 55,
    "gpu_oracle": 6,
    "kv_store_oracle": len(CUDA_KV_STORE_TEST_NODE_IDS),
    "b12x_oracle": len(CUDA_B12X_TEST_NODE_IDS),
}
QUALIFICATION_TOOL_RELATIVE_PATHS = (
    "tools/verse/run_sm120_release_gates.sh",
    "tools/verse/run_sm120_cuda_gates.sh",
    "tools/verse/run_sm120_acceptance.sh",
    "tools/verse/run_sm120_churn.py",
    "tools/verse/run_sm120_queue_stress.py",
    "tools/verse/run_sm120_user_latency.py",
    "tools/verse/run_sm120_warm_latency.py",
    "tools/verse/sm120_evidence_identity.py",
    "tools/verse/sm120_image_receipt.py",
    "tools/verse/check_sm120_chat_contract.py",
    "benchmarks/verse/sm120_prefill_interference.py",
    "benchmarks/verse/SM120_PREFILL_SCHEDULER_RESULTS.md",
)
B01_REPORT_RELATIVE_PATHS = tuple(
    f"short/b01-{target}-run-{run}.json"
    for target in B01_EXPECTED_SCENARIOS
    for run in range(1, 4)
)
EXPECTED_ARTIFACT_RELATIVE_PATHS = (
    "cuda-oracle.log",
    "cuda-oracle.json",
    "container-before-cuda.json",
    "container-after-cuda.json",
    "short/preflight.txt",
    "short/postflight.txt",
    "short/chat-contract.json",
    *B01_REPORT_RELATIVE_PATHS,
    "short/b01-summary.json",
    "short/prefill-37x1.json",
    "short/prefill-30x8.json",
    "short/container-before.json",
    "short/container-after.json",
    "queue-stress.json",
    "user-latency.json",
    "warm-latency.json",
    "churn.json",
    "post-churn-chat-contract.json",
    "post-churn-server.txt",
    "container-after-churn.json",
    "candidate-host.json",
    "image-receipt.json",
    "image-attestation-verification.json",
)
ALLOWED_MANIFEST_OUTPUTS = {
    ".release-manifest.tmp",
    "release-manifest.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_artifacts(release_dir: Path) -> dict[str, bytes]:
    expected = set(EXPECTED_ARTIFACT_RELATIVE_PATHS)
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    missing = expected - actual
    require(not missing, f"missing release artifacts: {sorted(missing)}")
    unexpected = actual - expected - ALLOWED_MANIFEST_OUTPUTS
    require(not unexpected, f"untracked release artifacts: {sorted(unexpected)}")

    artifacts: dict[str, bytes] = {}
    for relative in EXPECTED_ARTIFACT_RELATIVE_PATHS:
        path = release_dir / relative
        require(
            path.is_file() and not path.is_symlink(),
            f"invalid evidence file: {path}",
        )
        data = path.read_bytes()
        require(data, f"empty evidence file: {path}")
        artifacts[relative] = data
    return artifacts


def parse_json(data: bytes, name: str) -> Any:
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON evidence: {name}") from exc


def one_container(payload: Any, name: str) -> dict[str, Any]:
    require(isinstance(payload, list) and len(payload) == 1, f"invalid {name}")
    container = payload[0]
    require(isinstance(container, dict), f"invalid {name} container")
    return container


def container_identity(container: dict[str, Any]) -> tuple[Any, ...]:
    state = container.get("State") or {}
    config = container.get("Config") or {}
    return (
        container.get("Id"),
        container.get("Image"),
        config.get("Image"),
        state.get("StartedAt"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_gpu_record(payload: Any, name: str) -> dict[str, Any]:
    require(isinstance(payload, dict), f"{name} GPU identity is absent")
    require(
        set(payload)
        == {
            "name",
            "uuid",
            "memory_total_mib",
            "driver_version",
            "compute_capability",
        },
        f"{name} GPU identity schema is invalid",
    )
    require(
        payload.get("name") == "NVIDIA GeForce RTX 5070 Ti",
        f"{name} did not run on an exact RTX 5070 Ti",
    )
    require(
        GPU_UUID_RE.fullmatch(str(payload.get("uuid", ""))) is not None,
        f"{name} GPU UUID is invalid",
    )
    memory = payload.get("memory_total_mib")
    require(
        isinstance(memory, int) and not isinstance(memory, bool) and memory > 0,
        f"{name} GPU memory identity is invalid",
    )
    require(
        isinstance(payload.get("driver_version"), str)
        and bool(payload["driver_version"])
        and "\n" not in payload["driver_version"],
        f"{name} GPU driver identity is invalid",
    )
    require(
        payload.get("compute_capability") == [12, 0],
        f"{name} did not run with exact SM120 compute capability",
    )
    return payload


def parse_gpu_csv(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, str) and "\n" not in value, f"invalid {name} GPU record")
    try:
        fields = next(csv.reader([value], skipinitialspace=True))
    except (csv.Error, StopIteration) as exc:
        raise ValueError(f"invalid {name} GPU record") from exc
    require(len(fields) == 4, f"invalid {name} GPU record")
    gpu_name, gpu_uuid, memory, driver = (field.strip() for field in fields)
    require(memory.isdigit(), f"invalid {name} GPU memory")
    return validate_gpu_record(
        {
            "name": gpu_name,
            "uuid": gpu_uuid,
            "memory_total_mib": int(memory),
            "driver_version": driver,
            "compute_capability": [12, 0],
        },
        name,
    )


def validate_host_record(payload: Any, name: str) -> dict[str, Any]:
    require(isinstance(payload, dict), f"{name} host identity is absent")
    require(
        set(payload)
        == {
            "identity_sha256",
            "machine_id_sha256",
            "boot_id",
            "docker_id",
            "docker_name",
        },
        f"{name} host identity schema is invalid",
    )
    machine_id_sha256 = str(payload.get("machine_id_sha256", ""))
    identity_sha256 = str(payload.get("identity_sha256", ""))
    boot_id = str(payload.get("boot_id", ""))
    docker_id = str(payload.get("docker_id", ""))
    docker_name = payload.get("docker_name")
    require(
        SHA256_RE.fullmatch(machine_id_sha256) is not None,
        f"{name} machine identity hash is invalid",
    )
    require(
        UUID_RE.fullmatch(boot_id) is not None,
        f"{name} boot identity is invalid",
    )
    require(
        DOCKER_HOST_ID_RE.fullmatch(docker_id) is not None,
        f"{name} Docker host identity is invalid",
    )
    require(
        isinstance(docker_name, str)
        and bool(docker_name)
        and len(docker_name) <= 255
        and "\n" not in docker_name,
        f"{name} Docker host name is invalid",
    )
    expected_identity = hashlib.sha256(
        f"{machine_id_sha256}\n{docker_id}\n{boot_id}\n".encode()
    ).hexdigest()
    require(
        identity_sha256 == expected_identity,
        f"{name} composite host identity hash is invalid",
    )
    return payload


def container_gpu_binding(container: dict[str, Any]) -> tuple[str, str]:
    labels = (container.get("Config") or {}).get("Labels") or {}
    gpu_uuid = str(labels.get("ai.verse.gpu.uuid", ""))
    require(
        GPU_UUID_RE.fullmatch(gpu_uuid) is not None,
        "container exact GPU UUID label is absent",
    )
    requests = (container.get("HostConfig") or {}).get("DeviceRequests") or []
    require(
        isinstance(requests, list)
        and len(requests) == 1
        and isinstance(requests[0], dict),
        "container does not have exactly one GPU request",
    )
    capabilities = requests[0].get("Capabilities") or []
    require(
        isinstance(capabilities, list)
        and any(isinstance(group, list) and "gpu" in group for group in capabilities),
        "container GPU request lacks the GPU capability",
    )
    selectors = requests[0].get("DeviceIDs")
    require(
        isinstance(selectors, list)
        and len(selectors) == 1
        and isinstance(selectors[0], str)
        and bool(selectors[0]),
        "container GPU selector is invalid",
    )
    return gpu_uuid, selectors[0]


def parse_utc_timestamp(value: Any, name: str) -> datetime:
    require(isinstance(value, str) and bool(value), f"{name} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} timestamp is invalid") from exc
    require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(None),
        f"{name} timestamp is not UTC",
    )
    return parsed


def validate_stream(payload: Any, minimum_content_chunks: int) -> None:
    require(isinstance(payload, dict), "stream evidence is not an object")
    require(payload.get("saw_done") is True, "stream did not prove [DONE]")
    require(
        int(payload.get("content_chunks", 0)) >= minimum_content_chunks,
        "stream has too few content-bearing chunks",
    )
    require(
        int(payload.get("content_characters", 0)) > 0,
        "stream has no model text",
    )


def validate_chat_contract(payload: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "chat contract did not pass")
    require(payload.get("scope") == "complete", "chat contract is startup-only")
    require(
        payload.get("model") == EXPECTED_PROFILE["VERSE_SERVED_MODEL_NAME"],
        "chat contract used the wrong model alias",
    )
    require(int(payload.get("ordinary_prompt_tokens", 0)) > 0, "missing prompt")
    validate_stream(payload.get("ordinary_stream"), 2)
    greedy = payload.get("greedy_decode_evidence")
    require(isinstance(greedy, dict), "missing finite greedy liveness evidence")
    require(
        greedy.get("scope") == "finite_liveness_only"
        and greedy.get("deterministic_equality_required") is False,
        "greedy evidence overclaims deterministic correctness",
    )

    def validate_fingerprint(value: Any, expected_tokens: int, name: str) -> None:
        require(isinstance(value, dict), f"{name} is not an object")
        require(
            int(value.get("completion_tokens", -1)) == expected_tokens,
            f"{name} used the wrong token count",
        )
        require(
            SHA256_RE.fullmatch(str(value.get("content_sha256", ""))) is not None
            and SHA256_RE.fullmatch(str(value.get("tokens_sha256", ""))) is not None,
            f"{name} hashes are invalid",
        )
        for key in ("minimum_logprob", "maximum_logprob"):
            require(
                isinstance(value.get(key), int | float)
                and math.isfinite(float(value[key])),
                f"{name} contains a non-finite logprob",
            )

    first_token_runs = greedy.get("finite_first_token_runs")
    require(
        isinstance(first_token_runs, list) and len(first_token_runs) == 2,
        "greedy decode evidence does not contain two first-token runs",
    )
    for index, run in enumerate(first_token_runs):
        validate_fingerprint(run, 1, f"greedy first-token run {index}")
    decode_runs = greedy.get("finite_decode_runs")
    require(
        isinstance(decode_runs, list) and len(decode_runs) == 2,
        "greedy decode evidence does not contain two runs",
    )
    for index, run in enumerate(decode_runs):
        validate_fingerprint(run, 16, f"greedy decode run {index}")
    semantic = payload.get("semantic_rp_evidence")
    require(isinstance(semantic, dict), "missing semantic RP evidence")
    require(
        semantic.get("scope") == "safe_rp_semantic_integrity"
        and semantic.get("raw_output_retained") is False,
        "semantic RP evidence has the wrong scope",
    )
    semantic_runs = semantic.get("runs")
    require(
        isinstance(semantic_runs, list)
        and len(semantic_runs) == 3
        and [run.get("seed") for run in semantic_runs] == [1103, 2207, 3301],
        "semantic RP evidence has the wrong run inventory",
    )
    for index, run in enumerate(semantic_runs):
        require(isinstance(run, dict), f"semantic RP run {index} is not an object")
        require(
            SHA256_RE.fullmatch(str(run.get("content_sha256", ""))) is not None,
            f"semantic RP run {index} hash is invalid",
        )
        require(
            int(run.get("character_count", 0)) >= 100
            and int(run.get("ascii_word_count", 0)) >= 24
            and int(run.get("unique_ascii_word_count", 0)) >= 14,
            f"semantic RP run {index} lacks English text",
        )
        require(
            int(run.get("replacement_character_count", -1)) == 0
            and int(run.get("non_latin_letter_count", -1)) == 0,
            f"semantic RP run {index} contains script corruption",
        )
        for key, minimum in (
            ("printable_fraction", 0.995),
            ("ascii_fraction", 0.97),
            ("alphabetic_fraction", 0.45),
            ("common_word_fraction", 0.15),
        ):
            value = run.get(key)
            require(
                isinstance(value, int | float)
                and math.isfinite(float(value))
                and float(value) >= minimum,
                f"semantic RP run {index} has invalid {key}",
            )
    require(
        int(payload.get("boundary_accepted_prompt_tokens", -1)) == 6143,
        "6143+1 boundary acceptance is absent",
    )
    validate_stream(payload.get("boundary_accepted_stream"), 1)
    require(
        int(payload.get("boundary_rejected_prompt_tokens", -1)) == 6144
        and int(payload.get("boundary_rejected_http_status", -1)) in {400, 413, 422},
        "6144+1 boundary rejection is absent",
    )
    capacity = payload.get("exact_boundary_capacity")
    require(isinstance(capacity, dict), "missing exact capacity evidence")
    expected_capacity = {
        "concurrency": EXPECTED_CONCURRENCY,
        "prompt_tokens_per_request": 4096,
        "max_tokens_per_request": 2048,
        "context_tokens_per_request": 6144,
        "observed_max_running": EXPECTED_CONCURRENCY,
    }
    for key, expected in expected_capacity.items():
        require(
            int(capacity.get(key, -1)) == expected,
            f"capacity evidence has the wrong {key}",
        )
    expected_completion_tokens = (
        EXPECTED_CONCURRENCY * expected_capacity["max_tokens_per_request"]
    )
    require(
        int(capacity.get("simultaneous_decoding_streams", -1)) == EXPECTED_CONCURRENCY,
        "capacity evidence did not prove all streams decoding simultaneously",
    )
    require(
        int(capacity.get("verified_exact_completion_streams", -1))
        == EXPECTED_CONCURRENCY,
        "capacity evidence did not verify every stream completion",
    )
    require(
        int(capacity.get("observed_completion_tokens_total", -1))
        == expected_completion_tokens,
        "capacity evidence has the wrong total completion token count",
    )
    require(
        int(capacity.get("observed_length_finish_streams", -1)) == EXPECTED_CONCURRENCY,
        "capacity evidence did not prove length finishes for every stream",
    )
    require(
        float(capacity.get("kv_cache_usage_at_simultaneous_decode", 0)) > 0,
        "capacity evidence has no positive KV occupancy at simultaneous decode",
    )
    require(
        capacity.get("concurrent_6144_completion_proven") is True,
        "capacity evidence did not prove 38 concurrent exact-6144 completions",
    )
    require(
        capacity.get("concurrent_6143_residency_proven") is True
        and int(capacity.get("simultaneous_resident_context_tokens_per_request", -1))
        == 6143
        and float(capacity.get("kv_cache_usage_at_simultaneous_6143", 0)) > 0,
        "capacity evidence did not prove 38 simultaneous near-full 6K residents",
    )
    configured_blocks = EXPECTED_TOTAL_KV_BYTES // EXPECTED_BLOCK_KV_BYTES
    usable_blocks = configured_blocks - EXPECTED_RESERVED_KV_BLOCKS
    require(
        int(capacity.get("kv_cache_block_bytes", -1)) == EXPECTED_BLOCK_KV_BYTES
        and int(capacity.get("configured_kv_cache_blocks", -1)) == configured_blocks
        and int(capacity.get("reserved_kv_cache_blocks", -1))
        == EXPECTED_RESERVED_KV_BLOCKS
        and int(capacity.get("usable_kv_cache_blocks", -1)) == usable_blocks
        and int(capacity.get("configured_kv_cache_bytes", -1))
        == EXPECTED_TOTAL_KV_BYTES,
        "capacity evidence uses the wrong fixed KV byte accounting",
    )
    stream_evidence = capacity.get("stream_completion_evidence")
    require(
        isinstance(stream_evidence, list)
        and len(stream_evidence) == EXPECTED_CONCURRENCY,
        "capacity evidence has incomplete per-stream completion evidence",
    )
    for index, stream in enumerate(stream_evidence):
        require(
            isinstance(stream, dict)
            and int(stream.get("stream_index", -1)) == index
            and int(stream.get("completion_tokens", -1))
            == expected_capacity["max_tokens_per_request"]
            and int(stream.get("usage_completion_tokens", -1))
            == expected_capacity["max_tokens_per_request"]
            and stream.get("finish_reason") == "length",
            f"capacity stream {index} has invalid completion evidence",
        )
    require(
        int(capacity.get("running_metric_samples", 0)) > 0,
        "capacity evidence has no scheduler samples",
    )
    require(
        float(capacity.get("preemptions_after", -1))
        == float(capacity.get("preemptions_before", -2)),
        "capacity evidence includes preemption",
    )
    require(
        float(capacity.get("scheduler_running_before", -1)) == 0
        and float(capacity.get("scheduler_waiting_before", -1)) == 0,
        "capacity evidence did not start from an idle scheduler",
    )
    require(
        float(capacity.get("scheduler_running_after", -1)) == 0
        and float(capacity.get("scheduler_waiting_after", -1)) == 0,
        "capacity evidence did not drain the scheduler",
    )


def validate_b01_summary(payload: dict[str, Any], container: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "B01 summary did not pass")
    require(
        RELEASE_NONCE_RE.fullmatch(str(payload.get("release_nonce", ""))) is not None,
        "B01 summary has an invalid release nonce",
    )
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    identity = payload.get("identity")
    require(isinstance(identity, dict), "B01 identity is absent")
    require(
        identity.get("image_digest") == config.get("Image"),
        "B01 image identity differs from the container",
    )
    require(
        identity.get("fork_commit") == labels.get("ai.vllm.build.commit"),
        "B01 source identity differs from the container",
    )
    require(
        identity.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "B01 used the wrong model revision",
    )
    parse_gpu_csv(identity.get("gpu_name"), "B01")
    scenarios = payload.get("scenarios")
    require(
        isinstance(scenarios, dict) and set(scenarios) == set(EXPECTED_SCENARIOS),
        "B01 scenario matrix is incomplete",
    )
    for target, expectation in EXPECTED_SCENARIOS.items():
        scenario = scenarios[target]
        require(isinstance(scenario, dict), f"B01 {target} result is invalid")
        require(scenario.get("status") == "pass", f"B01 {target} failed")
        require(int(scenario.get("runs", -1)) == 3, f"B01 {target} run count")
        require(
            int(scenario.get("disjoint_prefix_runs", -1)) == 1
            and int(scenario.get("explicitly_prewarmed_runs", -1)) == 2,
            f"B01 {target} prefix matrix is invalid",
        )
        steady = scenario.get("steady_tokens_per_second")
        wall = scenario.get("wall_tokens_per_second")
        require(
            isinstance(steady, list)
            and len(steady) == 3
            and isinstance(wall, list)
            and len(wall) == 3,
            f"B01 {target} run evidence is incomplete",
        )
        require(
            float(scenario.get("median_steady_tokens_per_second", 0))
            >= expectation["minimum"],
            f"B01 {target} steady throughput is below threshold",
        )
        require(
            float(scenario.get("median_wall_tokens_per_second", 0))
            >= expectation["minimum"] * 0.9,
            f"B01 {target} wall throughput is below threshold",
        )
        require(
            float(scenario.get("minimum_steady_tokens_per_second", -1))
            == expectation["minimum"]
            and float(scenario.get("legacy_baseline_tokens_per_second", -1))
            == expectation["baseline"],
            f"B01 {target} thresholds drifted",
        )
    expected_weights = {
        str(target): PRODUCTION_WEIGHTS[target] for target in B01_EXPECTED_SCENARIOS
    }
    weights = payload.get("production_weights")
    require(weights == expected_weights, "B01 production weights drifted")
    require(
        math.isclose(
            math.fsum(float(weight) for weight in weights.values()),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "B01 production weights do not sum to 1",
    )
    require(
        float(payload.get("weighted_improvement_over_legacy", 0))
        >= MINIMUM_WEIGHTED_IMPROVEMENT
        and float(payload.get("minimum_weighted_improvement", -1))
        == MINIMUM_WEIGHTED_IMPROVEMENT,
        "B01 weighted improvement is below threshold",
    )


def validate_prefill_interference(
    payload: dict[str, Any],
    container: dict[str, Any],
    *,
    decoders: int,
    prefills: int,
    release_nonce: str,
    expected_gpu: dict[str, Any],
) -> None:
    require(payload.get("status") == "pass", "prefill interference run failed")
    require(
        payload.get("scope") == "current_profile_prefill_interference",
        "prefill interference scope is invalid",
    )
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    require(
        payload.get("image_digest") == config.get("Image")
        and payload.get("fork_commit") == labels.get("ai.vllm.build.commit")
        and payload.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "prefill interference identity differs from the candidate",
    )
    require(
        parse_gpu_csv(payload.get("gpu_name"), "prefill interference") == expected_gpu,
        "CUDA and prefill interference GPU identities differ",
    )
    require(
        payload.get("release_nonce") == release_nonce,
        "prefill interference release nonce differs",
    )
    require(
        int(payload.get("max_num_batched_tokens", -1))
        == int(EXPECTED_PROFILE["VERSE_MAX_NUM_BATCHED_TOKENS"]),
        "prefill interference used the wrong scheduler token budget",
    )
    require(
        int(payload.get("decoders", -1)) == decoders
        and int(payload.get("prefills", -1)) == prefills,
        "prefill interference used the wrong workload shape",
    )
    require(
        int(payload.get("submitted_requests", -1)) == decoders + prefills
        and int(payload.get("decode_prompt_tokens", -1)) == 4_500
        and int(payload.get("decode_prompt_tokens_min", -1)) == 4_500
        and int(payload.get("decode_prompt_tokens_max", -1)) == 4_500
        and int(payload.get("decode_output_tokens", -1)) == 1_024
        and int(payload.get("prefill_prompt_tokens_min", -1)) == 6_000
        and int(payload.get("prefill_prompt_tokens_max", -1)) == 6_000
        and int(payload.get("prefill_output_tokens", -1)) == 1
        and math.isclose(
            float(payload.get("baseline_seconds", -1)),
            3.0,
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(payload.get("metrics_interval_seconds", -1)),
            0.05,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "prefill interference used a non-representative request shape",
    )
    require(
        payload.get("decode_prefill_overlap_proven") is True
        and payload.get("all_decoders_unfinished_before_prefill") is True
        and payload.get("all_decoders_unfinished_after_prefill") is True,
        "prefill interference did not prove decode and prefill overlap",
    )
    baseline_rate = float(payload.get("baseline_decode_tok_s", 0))
    mixed_decode_rate = float(payload.get("decode_tok_s_during_prefill", 0))
    retention = float(payload.get("decode_retention_ratio", 0))
    wall_seconds = float(payload.get("prefill_wall_seconds", 0))
    deficit = float(payload.get("integrated_decoder_deficit_tokens", -1))
    expected_deficit = (baseline_rate - mixed_decode_rate) * wall_seconds
    thresholds = EXPECTED_PREFILL_SHAPES[(decoders, prefills)]
    require(
        baseline_rate > 0
        and mixed_decode_rate > 0
        and 0 < retention <= 1
        and math.isclose(
            retention,
            mixed_decode_rate / baseline_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and wall_seconds > 0
        and math.isclose(
            deficit,
            expected_deficit,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ),
        "prefill interference has inconsistent throughput accounting",
    )
    require(
        retention >= thresholds["minimum_retention"]
        and wall_seconds <= thresholds["maximum_wall_seconds"]
        and deficit <= thresholds["maximum_integrated_deficit_tokens"],
        "prefill interference exceeded the fixed release threshold",
    )
    require(
        int(payload.get("preemptions_delta", -1)) == 0
        and payload.get("server_idle_after") is True,
        "prefill interference preempted or failed to drain",
    )


def parse_cuda_log(data: bytes) -> tuple[dict[str, Any], str]:
    try:
        text = data.decode("utf-8")
        payload, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CUDA oracle log lacks leading verification JSON") from exc
    require(isinstance(payload, dict), "CUDA image verification is not an object")
    return payload, text


def validate_cuda_identity(
    payload: dict[str, Any], log_data: bytes, container: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(
        set(payload)
        == {
            "schema_version",
            "status",
            "image_digest",
            "fork_commit",
            "gpu_selector",
            "gpu",
            "host",
            "tests",
            "test_artifacts_sha256",
            "test_log_sha256",
            "oracle_markers",
            "image_verification",
        },
        "CUDA identity schema is invalid",
    )
    require(payload.get("schema_version") == 3, "CUDA identity schema is invalid")
    require(payload.get("status") == "pass", "CUDA oracle did not pass")
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    image_digest = str(payload.get("image_digest", ""))
    require(
        IMAGE_DIGEST_RE.fullmatch(image_digest) is not None
        and image_digest == config.get("Image"),
        "CUDA image identity differs from the container",
    )
    require(
        payload.get("fork_commit") == labels.get("ai.vllm.build.commit"),
        "CUDA source identity differs from the container",
    )
    cuda_gpu = validate_gpu_record(payload.get("gpu"), "CUDA oracle")
    container_gpu_uuid, _ = container_gpu_binding(container)
    require(
        payload.get("gpu_selector") == cuda_gpu["uuid"] == container_gpu_uuid,
        "CUDA GPU UUID differs from the candidate container",
    )
    cuda_host = validate_host_record(payload.get("host"), "CUDA oracle")
    require(
        payload.get("test_log_sha256") == sha256_bytes(log_data),
        "CUDA oracle log hash does not match its identity record",
    )
    require(
        payload.get("oracle_markers") == list(CUDA_MARKERS),
        "CUDA oracle marker inventory is invalid",
    )

    verification, log_text = parse_cuda_log(log_data)
    lines = log_text.splitlines()
    for marker in CUDA_MARKERS:
        require(lines.count(marker) == 1, f"CUDA oracle marker is absent: {marker}")
    require(
        payload.get("image_verification") == verification,
        "CUDA image verification differs from the recorded oracle log",
    )
    require(verification.get("status") == "valid", "CUDA image verification failed")
    require(
        str(verification.get("python", "")).startswith("3.12."),
        "CUDA oracle used the wrong Python",
    )
    require(verification.get("torch") == "2.13.0+cu130", "CUDA oracle wrong Torch")
    require(verification.get("torch_cuda") == "13.0", "CUDA oracle wrong Torch CUDA")
    require(
        verification.get("flashinfer_commit") == EXPECTED_FLASHINFER_COMMIT,
        "CUDA oracle wrong FlashInfer commit",
    )
    require(
        verification.get("distributions") == EXPECTED_DISTRIBUTIONS,
        "CUDA oracle dependency tuple drifted",
    )
    binary_identity = verification.get("vllm_binary_identity")
    require(
        isinstance(binary_identity, dict)
        and set(binary_identity) == {"native_extension", "wheel_artifact"},
        "CUDA oracle lacks immutable vLLM binary identity",
    )
    native_extension = binary_identity.get("native_extension")
    wheel_artifact = binary_identity.get("wheel_artifact")
    require(
        isinstance(native_extension, dict)
        and set(native_extension) == {"path", "wheel_member", "bytes", "sha256"}
        and str(native_extension.get("path", "")).endswith(
            "/vllm/_C_stable_libtorch.abi3.so"
        )
        and str(native_extension.get("wheel_member", "")).endswith(
            "/_C_stable_libtorch.abi3.so"
        )
        and int(native_extension.get("bytes", 0)) > 0
        and SHA256_RE.fullmatch(str(native_extension.get("sha256", ""))) is not None,
        "CUDA oracle native extension identity is invalid",
    )
    require(
        isinstance(wheel_artifact, dict)
        and set(wheel_artifact) == {"filename", "sha256", "manifest_sha256"}
        and str(wheel_artifact.get("filename", "")).endswith(".whl")
        and SHA256_RE.fullmatch(str(wheel_artifact.get("sha256", ""))) is not None
        and SHA256_RE.fullmatch(str(wheel_artifact.get("manifest_sha256", "")))
        is not None,
        "CUDA oracle wheel identity is invalid",
    )
    gpu = verification.get("gpu")
    require(isinstance(gpu, dict), "CUDA oracle GPU identity is absent")
    require(
        str(gpu.get("name", "")).endswith("RTX 5070 Ti")
        and gpu.get("capability") == [12, 0],
        "CUDA oracle did not run on an exact RTX 5070 Ti SM120 GPU",
    )
    require(
        gpu.get("name") == cuda_gpu["name"],
        "CUDA torch and nvidia-smi GPU names differ",
    )

    gpu_lines = [
        line.removeprefix("VERSE_CUDA_GPU_IDENTITY=")
        for line in lines
        if line.startswith("VERSE_CUDA_GPU_IDENTITY=")
    ]
    require(len(gpu_lines) == 1, "CUDA log GPU identity inventory is invalid")
    require(
        parse_gpu_csv(gpu_lines[0], "CUDA log") == cuda_gpu,
        "CUDA log GPU identity differs from machine-readable evidence",
    )

    log_tests: list[dict[str, str]] = []
    for line in lines:
        if not line.startswith("VERSE_CUDA_TEST_RESULT="):
            continue
        fields = line.removeprefix("VERSE_CUDA_TEST_RESULT=").split(" ", 2)
        require(len(fields) == 3, "CUDA log test result is malformed")
        result, suite, node_id = fields
        log_tests.append({"node_id": node_id, "result": result, "suite": suite})
    tests = payload.get("tests")
    require(
        isinstance(tests, list) and tests == log_tests, "CUDA test inventory drifted"
    )
    require(
        len({test.get("node_id") for test in tests}) == len(tests),
        "CUDA test inventory contains duplicate node IDs",
    )
    suite_nodes: dict[str, list[str]] = {
        suite: [] for suite in EXPECTED_CUDA_TEST_COUNTS
    }
    for test in tests:
        require(
            isinstance(test, dict)
            and set(test) == {"node_id", "result", "suite"}
            and test.get("result") == "passed"
            and test.get("suite") in suite_nodes
            and isinstance(test.get("node_id"), str),
            "CUDA test result schema is invalid",
        )
        suite_nodes[test["suite"]].append(test["node_id"])
    for suite, expected_count in EXPECTED_CUDA_TEST_COUNTS.items():
        require(
            len(suite_nodes[suite]) == expected_count,
            f"CUDA {suite} test-node inventory is incomplete",
        )
    require(
        all(
            node.startswith(CUDA_ROUTING_TEST_PREFIXES)
            for node in suite_nodes["routing"]
        ),
        "CUDA routing inventory contains an unexpected test node",
    )
    require(
        all(
            node.startswith(CUDA_GPU_TEST_PREFIXES)
            for node in suite_nodes["gpu_oracle"]
        ),
        "CUDA GPU inventory contains an unexpected test node",
    )
    require(
        sum(
            node.startswith(CUDA_GPU_TEST_PREFIXES[0])
            for node in suite_nodes["gpu_oracle"]
        )
        == 4
        and sum(
            node.startswith(CUDA_GPU_TEST_PREFIXES[1])
            for node in suite_nodes["gpu_oracle"]
        )
        == 2,
        "CUDA GPU parameterized node inventory is incomplete",
    )
    require(
        set(suite_nodes["kv_store_oracle"]) == set(CUDA_KV_STORE_TEST_NODE_IDS),
        "CUDA KV-store parameterized node inventory is incomplete",
    )
    require(
        set(suite_nodes["b12x_oracle"]) == set(CUDA_B12X_TEST_NODE_IDS),
        "CUDA B12X parameterized node inventory is incomplete",
    )

    log_artifact_hashes: dict[str, str] = {}
    for line in lines:
        if not line.startswith("VERSE_CUDA_TEST_ARTIFACT_SHA256="):
            continue
        value = line.removeprefix("VERSE_CUDA_TEST_ARTIFACT_SHA256=")
        match = re.fullmatch(r"([0-9a-f]{64})  (tests/.+\.py)", value)
        require(match is not None, "CUDA test artifact log record is malformed")
        relative = match.group(2)
        require(
            relative not in log_artifact_hashes,
            "CUDA test artifact log contains a duplicate",
        )
        log_artifact_hashes[relative] = match.group(1)
    artifact_hashes = payload.get("test_artifacts_sha256")
    require(
        isinstance(artifact_hashes, dict)
        and artifact_hashes == log_artifact_hashes
        and set(artifact_hashes) == set(CUDA_TEST_ARTIFACT_RELATIVE_PATHS),
        "CUDA test artifact hash inventory is invalid",
    )
    expected_artifact_hashes = {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in CUDA_TEST_ARTIFACT_RELATIVE_PATHS
    }
    require(
        artifact_hashes == expected_artifact_hashes,
        "CUDA oracle ran test artifacts that differ from the release source",
    )
    return verification, cuda_gpu, cuda_host


def validate_bound_identity(
    payload: dict[str, Any],
    *,
    container: dict[str, Any],
    release_nonce: str,
    expected_gpu: dict[str, Any],
    label: str,
) -> None:
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    require(
        payload.get("image_digest") == config.get("Image")
        and payload.get("fork_commit") == labels.get("ai.vllm.build.commit")
        and payload.get("model_revision") == EXPECTED_PROFILE["VERSE_MODEL_REVISION"]
        and payload.get("release_nonce") == release_nonce
        and payload.get("container_id") == container.get("Id"),
        f"{label} belongs to another candidate or release run",
    )
    require(
        parse_gpu_csv(str(payload.get("gpu_name", "")), label) == expected_gpu,
        f"{label} has the wrong GPU identity",
    )


def validate_image_receipt(
    receipt: dict[str, Any],
    *,
    container: dict[str, Any],
    image_verification: dict[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "verified_at",
        "image_digest",
        "fork_commit",
        "runtime_profile",
        "source_archive_sha256",
        "vllm_wheel_version",
        "binary_identity",
    }
    require(
        set(receipt) == expected_keys
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "identity-verified",
        "image receipt schema is invalid",
    )
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    require(
        receipt.get("image_digest") == config.get("Image")
        and receipt.get("fork_commit") == labels.get("ai.vllm.build.commit")
        and receipt.get("runtime_profile") == EXPECTED_PROFILE_IDENTITY
        and receipt.get("source_archive_sha256")
        == labels.get("ai.verse.source.archive.sha256")
        and receipt.get("vllm_wheel_version")
        == labels.get("ai.verse.vllm.wheel.version"),
        "image receipt does not match the immutable container",
    )
    try:
        verified_at = datetime.fromisoformat(str(receipt.get("verified_at", "")))
    except ValueError as error:
        raise ValueError("image receipt verification time is invalid") from error
    require(
        verified_at.tzinfo is not None,
        "image receipt verification time must be timezone-aware",
    )
    binary = image_verification.get("vllm_binary_identity") or {}
    native = binary.get("native_extension") or {}
    wheel = binary.get("wheel_artifact") or {}
    expected_binary = {
        "wheel_filename": wheel.get("filename"),
        "wheel_sha256": wheel.get("sha256"),
        "wheel_manifest_sha256": wheel.get("manifest_sha256"),
        "native_extension_member": native.get("wheel_member"),
        "native_extension_sha256": native.get("sha256"),
    }
    require(
        receipt.get("binary_identity") == expected_binary,
        "runtime binary identity differs from the approved image receipt",
    )


def validate_image_attestation(
    payload: Any, *, container: dict[str, Any]
) -> dict[str, str]:
    require(isinstance(payload, list) and payload, "image attestation is absent")
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    image = str(config.get("Image", ""))
    require(IMAGE_DIGEST_RE.fullmatch(image) is not None, "image digest is invalid")
    image_name, image_sha256 = image.rsplit("@sha256:", 1)
    commit = str(labels.get("ai.vllm.build.commit", ""))
    require(SHA256_RE.fullmatch(image_sha256) is not None, "image digest is invalid")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "commit is invalid")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    require(image_name in serialized, "attestation has the wrong image repository")
    require(image_sha256 in serialized, "attestation has the wrong image digest")
    require(commit in serialized, "attestation has the wrong source commit")
    require(
        ".github/workflows/verse-sm120-image.yml" in serialized,
        "attestation has the wrong signer workflow",
    )
    require(
        "refs/heads/verse/v0.28-sm120-nvfp4-fa2" in serialized,
        "attestation has the wrong source ref",
    )
    return {
        "image_repository": image_name,
        "image_sha256": image_sha256,
        "source_commit": commit,
        "signer_workflow": ".github/workflows/verse-sm120-image.yml",
        "source_ref": "refs/heads/verse/v0.28-sm120-nvfp4-fa2",
    }


def validate_queue_stress(payload: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "queue stress did not pass")
    require(
        int(payload.get("stress_seconds", 0)) >= 120,
        "queue stress was too short",
    )
    require(
        payload.get("prompt_token_targets") == [5500, 5750, 6000]
        and int(payload.get("max_completion_tokens", -1)) == 128,
        "queue stress used the wrong request shape",
    )
    phases = payload.get("phases")
    require(
        isinstance(phases, list)
        and len(phases) == 2
        and [phase.get("name") for phase in phases] == ["max-active", "overflow-queue"],
        "queue stress has the wrong phase inventory",
    )
    expected = ((38, False), (76, True))
    for phase, (clients, require_waiting) in zip(phases, expected, strict=True):
        require(isinstance(phase, dict), "queue stress phase is not an object")
        metrics = phase.get("metrics_evidence")
        before = phase.get("metrics_before")
        after = phase.get("metrics_after")
        require(
            isinstance(metrics, dict)
            and isinstance(before, dict)
            and isinstance(after, dict),
            "queue stress phase lacks metrics evidence",
        )
        require(
            int(phase.get("clients", -1)) == clients
            and int(phase.get("active_capacity", -1)) == EXPECTED_CONCURRENCY
            and int(phase.get("request_errors", -1)) == 0
            and int(metrics.get("observed_max_running", -1)) == EXPECTED_CONCURRENCY
            and float(metrics.get("observed_generation_tokens_delta", 0)) > 0,
            "queue stress phase did not prove full healthy decode capacity",
        )
        require(
            float(before.get("running", -1)) == 0
            and float(before.get("waiting", -1)) == 0
            and float(after.get("running", -1)) == 0
            and float(after.get("waiting", -1)) == 0
            and float(after.get("preemptions", -1))
            == float(before.get("preemptions", -2)),
            "queue stress phase did not own, drain, or preserve the scheduler",
        )
        observed_waiting = int(phase.get("observed_max_waiting", -1))
        require(
            observed_waiting > 0 if require_waiting else observed_waiting == 0,
            "queue stress did not prove the expected waiting-queue behavior",
        )


def validate_user_latency(payload: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "user latency did not pass")
    require(
        int(payload.get("completion_tokens_requested", -1)) == 128,
        "user latency used the wrong completion length",
    )
    modes = payload.get("modes")
    require(
        isinstance(modes, list)
        and len(modes) == 2
        and [mode.get("mode") for mode in modes] == ["saturated", "overloaded"],
        "user latency has the wrong mode inventory",
    )
    for mode, background in zip(modes, (14, 52), strict=True):
        require(isinstance(mode, dict), "user latency mode is not an object")
        require(
            int(mode.get("background_clients", -1)) == background
            and int(mode.get("samples_per_prompt", -1)) == 5
            and int(mode.get("measured_user_clients", -1)) == 15
            and int(mode.get("request_errors", -1)) == 0
            and float(mode.get("preemptions_delta", math.inf)) == 0,
            "user latency used the wrong pressure or recorded a failure",
        )
        samples = mode.get("samples")
        require(
            isinstance(samples, list) and len(samples) == 15,
            "user latency has the wrong sample count",
        )
        for sample in samples:
            require(
                isinstance(sample, dict)
                and int(sample.get("prompt_tokens", -1)) in {2000, 4000, 6000}
                and int(sample.get("completion_tokens", -1)) == 128
                and 0
                < float(sample.get("ttft_seconds", 0))
                <= float(sample.get("end_to_end_seconds", -1))
                and float(sample.get("decode_tokens_per_second", 0)) > 0,
                "user latency contains an invalid measured request",
            )
        if mode.get("mode") == "overloaded":
            require(
                max(int(sample.get("waiting_at_arrival", 0)) for sample in samples) > 0,
                "overloaded user latency did not exercise queueing",
            )


def validate_warm_latency(payload: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "warm latency did not pass")
    require(
        int(payload.get("clients", -1)) == EXPECTED_CONCURRENCY
        and int(payload.get("completion_tokens_requested", -1)) == 100
        and float(payload.get("preemptions_delta", math.inf)) == 0,
        "warm latency used the wrong cohort or recorded preemption",
    )
    for phase_name in ("cold", "warm_delta"):
        phase = payload.get(phase_name)
        require(isinstance(phase, dict), f"warm latency lacks {phase_name}")
        pressure = phase.get("pressure")
        grouped = phase.get("by_prompt_tokens")
        require(
            isinstance(pressure, dict)
            and float(pressure.get("max_running", 0)) >= EXPECTED_CONCURRENCY
            and isinstance(grouped, dict)
            and set(grouped) == {"2000", "4000", "6000"},
            f"warm latency {phase_name} lacks full-pressure evidence",
        )
        for target, summary in grouped.items():
            require(
                isinstance(summary, dict)
                and int(summary.get("samples", 0)) >= 12
                and 0
                < float(summary.get("ttft_p50_seconds", 0))
                <= float(summary.get("ttft_p95_seconds", -1))
                and 0
                < float(summary.get("end_to_end_p50_seconds", 0))
                <= float(summary.get("end_to_end_p95_seconds", -1))
                and float(summary.get("decode_p05_tokens_per_second", 0)) > 0,
                f"warm latency {phase_name} {target} summary is invalid",
            )


def validate_churn(payload: dict[str, Any]) -> None:
    require(payload.get("status") == "pass", "churn did not pass")
    require(float(payload.get("duration_seconds", 0)) >= 7200, "churn was too short")
    require(
        int(payload.get("concurrency", -1)) == EXPECTED_CONCURRENCY,
        "churn used wrong concurrency",
    )
    require(int(payload.get("prompt_pool_size", -1)) == 64, "wrong prompt pool")
    require(
        int(payload.get("completed_requests", 0)) >= EXPECTED_CONCURRENCY,
        "churn completed too few requests",
    )
    require(int(payload.get("cancelled_requests", 0)) > 0, "no cancellations")
    require(int(payload.get("stream_chunks", 0)) > 0, "churn streamed no chunks")
    require(int(payload.get("request_errors", -1)) == 0, "churn had errors")
    before = payload.get("metrics_before")
    after = payload.get("metrics_after")
    require(isinstance(before, dict) and isinstance(after, dict), "missing metrics")
    require(
        float(before.get("running", -1)) == 0 and float(before.get("waiting", -1)) == 0,
        "churn did not own an idle server",
    )
    require(
        float(after.get("running", -1)) == 0 and float(after.get("waiting", -1)) == 0,
        "churn did not drain",
    )
    require(
        float(after.get("preemptions", -1)) == float(before.get("preemptions", -2)),
        "churn preempted requests",
    )
    require(
        float(after.get("prefix_hits", 0)) > float(before.get("prefix_hits", 0)),
        "churn did not prove prefix reuse",
    )


def validate_candidate_host(
    payload: dict[str, Any],
    container: dict[str, Any],
    cuda_gpu: dict[str, Any],
    cuda_host: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        set(payload)
        == {
            "schema_version",
            "status",
            "container_id",
            "image_digest",
            "fork_commit",
            "container_gpu_selector",
            "gpu",
            "host",
        }
        and payload.get("schema_version") == 1
        and payload.get("status") == "pass",
        "candidate host evidence schema is invalid",
    )
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    require(
        payload.get("container_id") == container.get("Id")
        and payload.get("image_digest") == config.get("Image")
        and payload.get("fork_commit") == labels.get("ai.vllm.build.commit"),
        "candidate host evidence belongs to another container",
    )
    gpu_uuid, selector = container_gpu_binding(container)
    require(
        payload.get("container_gpu_selector") == selector,
        "candidate host evidence has the wrong container GPU selector",
    )
    gpu = validate_gpu_record(payload.get("gpu"), "candidate container")
    host = validate_host_record(payload.get("host"), "candidate container")
    require(gpu["uuid"] == gpu_uuid, "candidate GPU UUID differs from its container")
    require(gpu == cuda_gpu, "CUDA and candidate container GPU identities differ")
    require(host == cuda_host, "CUDA and candidate container host identities differ")
    return gpu, host


def finalize(release_dir: Path) -> dict[str, Any]:
    release_dir = release_dir.resolve(strict=True)
    artifacts = load_artifacts(release_dir)

    def object_artifact(relative: str) -> dict[str, Any]:
        payload = parse_json(artifacts[relative], relative)
        require(isinstance(payload, dict), f"{relative} is not an object")
        return payload

    b01_reports: list[dict[str, Any]] = []
    for target in B01_EXPECTED_SCENARIOS:
        for run in range(1, 4):
            relative = f"short/b01-{target}-run-{run}.json"
            report = object_artifact(relative)
            require(
                int(report.get("requested_prompt_tokens", -1)) == target,
                f"{relative} has the wrong prompt target",
            )
            expected_prewarmed = run != 1
            expected_preparation = (
                "explicitly_prewarmed_this_run"
                if expected_prewarmed
                else "no_explicit_prewarm_this_run"
            )
            require(
                report.get("prewarmed") is expected_prewarmed
                and report.get("prefix_preparation") == expected_preparation,
                f"{relative} has the wrong prefix preparation",
            )
            b01_reports.append(report)

    recomputed_b01 = evaluate_b01(b01_reports)
    recorded_b01 = object_artifact("short/b01-summary.json")
    require(
        recorded_b01 == recomputed_b01,
        "B01 summary does not match the six raw benchmark reports",
    )
    evidence = {
        "initial_contract": object_artifact("short/chat-contract.json"),
        "prefill_37x1": object_artifact("short/prefill-37x1.json"),
        "prefill_30x8": object_artifact("short/prefill-30x8.json"),
        "queue_stress": object_artifact("queue-stress.json"),
        "user_latency": object_artifact("user-latency.json"),
        "warm_latency": object_artifact("warm-latency.json"),
        "churn": object_artifact("churn.json"),
        "post_churn_contract": object_artifact("post-churn-chat-contract.json"),
        "cuda_identity": object_artifact("cuda-oracle.json"),
        "candidate_host": object_artifact("candidate-host.json"),
        "image_receipt": object_artifact("image-receipt.json"),
        "image_attestation": parse_json(
            artifacts["image-attestation-verification.json"],
            "image-attestation-verification.json",
        ),
    }

    container_relatives = (
        "container-before-cuda.json",
        "container-after-cuda.json",
        "short/container-before.json",
        "short/container-after.json",
        "container-after-churn.json",
    )
    containers = [
        one_container(parse_json(artifacts[relative], relative), Path(relative).name)
        for relative in container_relatives
    ]
    identities = {container_identity(container) for container in containers}
    require(len(identities) == 1, "container identity changed during release gates")
    gpu_bindings = {container_gpu_binding(container) for container in containers}
    require(
        len(gpu_bindings) == 1, "container GPU binding changed during release gates"
    )
    for container in containers:
        state = container.get("State") or {}
        labels = (container.get("Config") or {}).get("Labels") or {}
        require(
            CONTAINER_ID_RE.fullmatch(str(container.get("Id", ""))) is not None,
            "candidate container ID is not exact",
        )
        require(
            labels.get("ai.verse.runtime.profile") == EXPECTED_PROFILE_IDENTITY,
            "candidate container has the wrong runtime profile",
        )
        require(state.get("Running") is True, "container is not running")
        require(state.get("OOMKilled") is False, "container was OOM-killed")
        require(container.get("RestartCount") == 0, "container restarted")

    cuda_verification, cuda_gpu, cuda_host = validate_cuda_identity(
        evidence["cuda_identity"], artifacts["cuda-oracle.log"], containers[0]
    )
    validate_image_receipt(
        evidence["image_receipt"],
        container=containers[0],
        image_verification=cuda_verification,
    )
    attestation = validate_image_attestation(
        evidence["image_attestation"], container=containers[0]
    )
    validate_b01_summary(recomputed_b01, containers[0])
    validate_prefill_interference(
        evidence["prefill_37x1"],
        containers[0],
        decoders=37,
        prefills=1,
        release_nonce=recomputed_b01["release_nonce"],
        expected_gpu=cuda_gpu,
    )
    validate_prefill_interference(
        evidence["prefill_30x8"],
        containers[0],
        decoders=30,
        prefills=8,
        release_nonce=recomputed_b01["release_nonce"],
        expected_gpu=cuda_gpu,
    )
    require(
        parse_gpu_csv(recomputed_b01["identity"]["gpu_name"], "B01") == cuda_gpu,
        "CUDA and B01 GPU identities differ",
    )
    candidate_gpu, candidate_host = validate_candidate_host(
        evidence["candidate_host"], containers[0], cuda_gpu, cuda_host
    )
    for label, key in (
        ("initial chat contract", "initial_contract"),
        ("queue stress", "queue_stress"),
        ("user latency", "user_latency"),
        ("warm latency", "warm_latency"),
        ("churn", "churn"),
        ("post-churn chat contract", "post_churn_contract"),
    ):
        validate_bound_identity(
            evidence[key],
            container=containers[0],
            release_nonce=recomputed_b01["release_nonce"],
            expected_gpu=cuda_gpu,
            label=label,
        )
    validate_chat_contract(evidence["initial_contract"])
    validate_queue_stress(evidence["queue_stress"])
    validate_user_latency(evidence["user_latency"])
    validate_warm_latency(evidence["warm_latency"])
    validate_churn(evidence["churn"])
    validate_chat_contract(evidence["post_churn_contract"])

    expected_commit = ((containers[0].get("Config") or {}).get("Labels") or {}).get(
        "ai.vllm.build.commit"
    )
    server_records = (
        "short/preflight.txt",
        "short/postflight.txt",
        "post-churn-server.txt",
    )
    for relative in server_records:
        try:
            server_record = artifacts[relative].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid server evidence: {relative}") from exc
        lines = server_record.splitlines()
        require("status=healthy" in lines, f"{relative} is not a healthy server record")
        require(
            f"commit={expected_commit}" in lines,
            f"{relative} has the wrong source identity",
        )
        profile_lines = [line for line in lines if line.startswith("profile=")]
        require(
            profile_lines == [f"profile={EXPECTED_PROFILE_IDENTITY}"],
            f"{relative} has the wrong runtime profile",
        )
        server_pairs = [line.split("=", 1) for line in lines if "=" in line]
        require(
            len(server_pairs) == len({key for key, _ in server_pairs}),
            f"{relative} contains duplicate evidence fields",
        )
        server_fields = dict(server_pairs)
        require(
            server_fields.get("image_receipt_sha256")
            == sha256_bytes(artifacts["image-receipt.json"]),
            f"{relative} has the wrong image receipt identity",
        )
        require(
            server_fields.get("model_manifest_sha256")
            == EXPECTED_PROFILE["VERSE_MODEL_MANIFEST_SHA256"]
            and SHA256_RE.fullmatch(server_fields.get("model_config_sha256", ""))
            is not None
            and SHA256_RE.fullmatch(server_fields.get("model_ready_marker_sha256", ""))
            is not None
            and int(server_fields.get("model_file_count", "0")) > 0
            and int(server_fields.get("model_bytes", "0")) > 0,
            f"{relative} lacks immutable model-byte evidence",
        )

    hashes = {relative: sha256_bytes(data) for relative, data in artifacts.items()}
    qualification_tool_hashes = {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in QUALIFICATION_TOOL_RELATIVE_PATHS
    }
    identity = next(iter(identities))
    return {
        "status": "pass",
        "scope": "pre_cutover_candidate_qualification",
        "profile": EXPECTED_PROFILE_IDENTITY,
        "release_nonce": recomputed_b01["release_nonce"],
        "production_weights": recomputed_b01["production_weights"],
        "prefill_interference": {
            "scheduler_max_num_batched_tokens": int(
                EXPECTED_PROFILE["VERSE_MAX_NUM_BATCHED_TOKENS"]
            ),
            "shapes": ["37x1", "30x8"],
            "scope": "current_profile_prefill_interference",
        },
        "queue_stress": {
            "active_capacity": EXPECTED_CONCURRENCY,
            "overflow_clients": 76,
            "minimum_stress_seconds": 120,
        },
        "user_latency": {
            "prompt_tokens": [2000, 4000, 6000],
            "samples_per_prompt_per_mode": 5,
            "modes": ["saturated", "overloaded"],
        },
        "warm_latency": {
            "prompt_tokens": [2000, 4000, 6000],
            "clients": EXPECTED_CONCURRENCY,
        },
        "container": {
            "id": identity[0],
            "image_id": identity[1],
            "image_digest": identity[2],
            "started_at": identity[3],
        },
        "disposable_host": candidate_host,
        "cuda_oracle": {
            "image_digest": evidence["cuda_identity"]["image_digest"],
            "fork_commit": evidence["cuda_identity"]["fork_commit"],
            "gpu": candidate_gpu,
            "flashinfer_commit": cuda_verification["flashinfer_commit"],
            "test_count": len(evidence["cuda_identity"]["tests"]),
        },
        "source_commit": expected_commit,
        "image_receipt_sha256": hashes["image-receipt.json"],
        "image_attestation": {
            **attestation,
            "verification_sha256": hashes["image-attestation-verification.json"],
        },
        "model_revision": EXPECTED_PROFILE["VERSE_MODEL_REVISION"],
        "artifacts_sha256": hashes,
        "qualification_tools_sha256": qualification_tool_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the complete Verse SM120 release evidence."
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = finalize(args.release_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
