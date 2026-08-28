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
    "SM120 image and native NVFP4 FA2 correctness gates passed",
)
CUDA_TEST_ARTIFACT_RELATIVE_PATHS = (
    "tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py",
    "tests/v1/attention/test_nvfp4_flashinfer_vosplit.py",
    "tests/kernels/attention/test_flashinfer.py",
)
CUDA_ROUTING_TEST_PREFIXES = (
    "tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py::",
    "tests/v1/attention/test_nvfp4_flashinfer_vosplit.py::",
)
CUDA_GPU_TEST_PREFIXES = (
    "tests/kernels/attention/test_flashinfer.py::"
    "test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference",
    "tests/kernels/attention/test_flashinfer.py::"
    "test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference",
)
EXPECTED_CUDA_TEST_COUNTS = {"routing": 47, "gpu_oracle": 6}
QUALIFICATION_TOOL_RELATIVE_PATHS = (
    "tools/verse/run_sm120_release_gates.sh",
    "tools/verse/run_sm120_cuda_gates.sh",
    "tools/verse/run_sm120_acceptance.sh",
    "tools/verse/run_sm120_churn.py",
    "tools/verse/check_sm120_chat_contract.py",
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
    "short/container-before.json",
    "short/container-after.json",
    "churn.json",
    "post-churn-chat-contract.json",
    "post-churn-server.txt",
    "container-after-churn.json",
    "candidate-host.json",
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
    deterministic = payload.get("deterministic_completion")
    require(isinstance(deterministic, dict), "missing deterministic evidence")
    require(
        int(deterministic.get("completion_tokens", -1)) == 16,
        "deterministic completion used the wrong token count",
    )
    require(
        SHA256_RE.fullmatch(str(deterministic.get("content_sha256", ""))) is not None
        and SHA256_RE.fullmatch(str(deterministic.get("tokens_sha256", "")))
        is not None,
        "deterministic completion hashes are invalid",
    )
    for key in ("minimum_logprob", "maximum_logprob"):
        require(
            isinstance(deterministic.get(key), int | float)
            and math.isfinite(float(deterministic[key])),
            "deterministic completion contains a non-finite logprob",
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
        "prompt_tokens_per_request": 6080,
        "max_tokens_per_request": 64,
        "context_tokens_per_request": 6144,
        "observed_max_running": EXPECTED_CONCURRENCY,
    }
    for key, expected in expected_capacity.items():
        require(
            int(capacity.get(key, -1)) == expected,
            f"capacity evidence has the wrong {key}",
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
    suite_nodes: dict[str, list[str]] = {"routing": [], "gpu_oracle": []}
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
        "churn": object_artifact("churn.json"),
        "post_churn_contract": object_artifact("post-churn-chat-contract.json"),
        "cuda_identity": object_artifact("cuda-oracle.json"),
        "candidate_host": object_artifact("candidate-host.json"),
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
        require(
            CONTAINER_ID_RE.fullmatch(str(container.get("Id", ""))) is not None,
            "candidate container ID is not exact",
        )
        require(state.get("Running") is True, "container is not running")
        require(state.get("OOMKilled") is False, "container was OOM-killed")
        require(container.get("RestartCount") == 0, "container restarted")

    cuda_verification, cuda_gpu, cuda_host = validate_cuda_identity(
        evidence["cuda_identity"], artifacts["cuda-oracle.log"], containers[0]
    )
    validate_b01_summary(recomputed_b01, containers[0])
    require(
        parse_gpu_csv(recomputed_b01["identity"]["gpu_name"], "B01") == cuda_gpu,
        "CUDA and B01 GPU identities differ",
    )
    candidate_gpu, candidate_host = validate_candidate_host(
        evidence["candidate_host"], containers[0], cuda_gpu, cuda_host
    )
    validate_chat_contract(evidence["initial_contract"])
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

    hashes = {relative: sha256_bytes(data) for relative, data in artifacts.items()}
    qualification_tool_hashes = {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in QUALIFICATION_TOOL_RELATIVE_PATHS
    }
    identity = next(iter(identities))
    return {
        "status": "pass",
        "scope": "pre_cutover_candidate_qualification",
        "release_nonce": recomputed_b01["release_nonce"],
        "production_weights": recomputed_b01["production_weights"],
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
