#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

require_env() {
  local name=$1
  [[ -n ${!name:-} ]] || {
    echo "$name is required" >&2
    exit 1
  }
}

require_env VERSE_VLLM_IMAGE
require_env VERSE_VLLM_EXPECTED_COMMIT
require_env VERSE_VLLM_GPU_UUID

tools/verse/verify_sm120_source.sh "$VERSE_VLLM_EXPECTED_COMMIT" >/dev/null

[[ $VERSE_VLLM_IMAGE =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_IMAGE must use an immutable sha256 digest" >&2
  exit 1
}
[[ $VERSE_VLLM_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo "VERSE_VLLM_EXPECTED_COMMIT must be a 40-character commit SHA" >&2
  exit 1
}
[[ $VERSE_VLLM_GPU_UUID =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] || {
  echo "VERSE_VLLM_GPU_UUID must be an exact full GPU UUID" >&2
  exit 1
}

uv run --script tools/verse/validate_sm120_profile.py \
  --image "$VERSE_VLLM_IMAGE" \
  --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" >/dev/null

docker image inspect "$VERSE_VLLM_IMAGE" >/dev/null
IMAGE_COMMIT=$(docker image inspect \
  --format '{{index .Config.Labels "ai.vllm.build.commit"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_PROFILE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.runtime.profile"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_RELEASE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.flashinfer.release"}}' \
  "$VERSE_VLLM_IMAGE")
EXPECTED_MANIFEST=$(shasum -a 256 \
  requirements/verse-sm120-flashinfer.lock | awk '{print $1}')
IMAGE_MANIFEST=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.flashinfer.manifest.sha256"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_BUILD_BASE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.base.build"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_RUNTIME_BASE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.base.runtime"}}' \
  "$VERSE_VLLM_IMAGE")
EXPECTED_BUILD_BASE='pytorch/manylinux2_28-builder:cuda13.0@sha256:7710cbc19d7ee951134e2e827f8ec89237c993095eb2581dd5e74f58e4e278c7'
EXPECTED_RUNTIME_BASE='nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4'

[[ $IMAGE_COMMIT == "$VERSE_VLLM_EXPECTED_COMMIT" ]] || {
  echo "candidate image has the wrong fork commit" >&2
  exit 1
}
[[ $IMAGE_PROFILE == sm120-gemma4-nvfp4-v1 ]] || {
  echo "candidate image has the wrong Verse profile" >&2
  exit 1
}
[[ $IMAGE_RELEASE == 0.6.18.dev20260819 ]] || {
  echo "candidate image has the wrong FlashInfer release" >&2
  exit 1
}
[[ $IMAGE_MANIFEST == "$EXPECTED_MANIFEST" ]] || {
  echo "candidate image has the wrong FlashInfer manifest" >&2
  exit 1
}
[[ $IMAGE_BUILD_BASE == "$EXPECTED_BUILD_BASE" ]] || {
  echo "candidate image has the wrong immutable build base" >&2
  exit 1
}
[[ $IMAGE_RUNTIME_BASE == "$EXPECTED_RUNTIME_BASE" ]] || {
  echo "candidate image has the wrong immutable runtime base" >&2
  exit 1
}

HOST_GPU_IDENTITY=$(LC_ALL=C nvidia-smi --id="$VERSE_VLLM_GPU_UUID" \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader,nounits)
[[ -n $HOST_GPU_IDENTITY && $HOST_GPU_IDENTITY != *$'\n'* ]] || {
  echo "GPU UUID did not resolve to exactly one host GPU" >&2
  exit 1
}
HOST_GPU_UUID=$(printf '%s\n' "$HOST_GPU_IDENTITY" | awk -F, \
  '{gsub(/[[:space:]]/, "", $2); print $2}')
[[ $HOST_GPU_UUID == "$VERSE_VLLM_GPU_UUID" ]] || {
  echo "nvidia-smi did not resolve the exact requested GPU UUID" >&2
  exit 1
}

CUDA_LOG_FILE=${VERSE_VLLM_CUDA_LOG_FILE:-}
CUDA_EVIDENCE_FILE=${VERSE_VLLM_CUDA_EVIDENCE_FILE:-}
EMIT_EVIDENCE=0
if [[ -z $CUDA_LOG_FILE && -z $CUDA_EVIDENCE_FILE ]]; then
  WORK_DIR=$(mktemp -d)
  CUDA_LOG_FILE="$WORK_DIR/cuda-oracle.log"
  CUDA_EVIDENCE_FILE="$WORK_DIR/cuda-oracle.json"
  EMIT_EVIDENCE=1
elif [[ -z $CUDA_LOG_FILE || -z $CUDA_EVIDENCE_FILE ]]; then
  echo "both CUDA evidence paths must be provided together" >&2
  exit 1
else
  [[ $CUDA_LOG_FILE == /* && $CUDA_EVIDENCE_FILE == /* ]] || {
    echo "CUDA evidence paths must be absolute" >&2
    exit 1
  }
  [[ ${CUDA_LOG_FILE%/*} == "${CUDA_EVIDENCE_FILE%/*}" ]] || {
    echo "CUDA evidence files must share one directory" >&2
    exit 1
  }
  [[ -d ${CUDA_LOG_FILE%/*} && ! -L ${CUDA_LOG_FILE%/*} ]] || {
    echo "CUDA evidence directory is invalid" >&2
    exit 1
  }
  [[ ! -e $CUDA_LOG_FILE && ! -L $CUDA_LOG_FILE ]] || {
    echo "CUDA log evidence already exists" >&2
    exit 1
  }
  [[ ! -e $CUDA_EVIDENCE_FILE && ! -L $CUDA_EVIDENCE_FILE ]] || {
    echo "CUDA identity evidence already exists" >&2
    exit 1
  }
  WORK_DIR=$(mktemp -d "${CUDA_LOG_FILE%/*}/.cuda-gates.XXXXXX")
fi
LOG_TMP="$WORK_DIR/cuda-oracle.log.tmp"
EVIDENCE_TMP="$WORK_DIR/cuda-oracle.json.tmp"
cleanup() {
  rm -f -- "$LOG_TMP" "$EVIDENCE_TMP"
  rmdir -- "$WORK_DIR" 2>/dev/null || true
}
trap cleanup EXIT
umask 027

docker run --rm \
  --pull never \
  --gpus "device=$VERSE_VLLM_GPU_UUID" \
  --shm-size 16g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=4g \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env XDG_CACHE_HOME=/tmp/xdg \
  --env FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer \
  --env VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/flashinfer-autotune \
  --env CUDA_CACHE_PATH=/tmp/cuda \
  --env TORCH_HOME=/tmp/torch \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  --entrypoint /bin/bash \
  "$VERSE_VLLM_IMAGE" \
  -euo pipefail -c '
    # FlashInfer 0.6.18 imports a deprecated CUTLASS compatibility alias.
    # Suppress only that pinned third-party warning; every other warning still
    # fails the clean-oracle check below.
    export PYTHONWARNINGS=ignore::DeprecationWarning:flashinfer.cute_dsl.utils
    /usr/local/bin/verify-verse-sm120-image --require-gpu
    GPU_IDENTITY=$(LC_ALL=C nvidia-smi \
      --query-gpu=name,uuid,memory.total,driver_version \
      --format=csv,noheader,nounits)
    [[ -n $GPU_IDENTITY && $(printf "%s\n" "$GPU_IDENTITY" | wc -l) -eq 1 ]]
    printf "VERSE_CUDA_GPU_IDENTITY=%s\n" "$GPU_IDENTITY"

    cd /opt/verse-tests
    ROUTING_COLLECT=$(mktemp)
    ROUTING_OUTPUT=$(mktemp)
    GPU_COLLECT=$(mktemp)
    GPU_OUTPUT=$(mktemp)
    python -m pytest --collect-only -q -p no:cacheprovider \
      tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
      tests/v1/attention/test_nvfp4_flashinfer_vosplit.py | tee "$ROUTING_COLLECT"
    ROUTING_COUNT=$(grep -Ec "^tests/.+::" "$ROUTING_COLLECT")
    ((ROUTING_COUNT > 0))
    ! grep -Eqi "skipped|deselected|xfailed|xpassed|warning|error" "$ROUTING_COLLECT"
    python -m pytest -q -p no:cacheprovider --maxfail=1 \
      tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
      tests/v1/attention/test_nvfp4_flashinfer_vosplit.py | tee "$ROUTING_OUTPUT"
    grep -Eq "^${ROUTING_COUNT} passed in [0-9.]+s$" "$ROUTING_OUTPUT"
    ! grep -Eqi "skipped|deselected|xfailed|xpassed|warning|error" "$ROUTING_OUTPUT"
    sed -n "s#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed routing \1#p" \
      "$ROUTING_COLLECT"
    echo VERSE_ROUTING_GATES_PASSED

    python -m pytest --collect-only -q -p no:cacheprovider \
      tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference \
      tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference | tee "$GPU_COLLECT"
    GPU_COUNT=$(grep -Ec "^tests/.+::" "$GPU_COLLECT")
    ((GPU_COUNT == 6))
    ! grep -Eqi "skipped|deselected|xfailed|xpassed|warning|error" "$GPU_COLLECT"
    python -m pytest -q -p no:cacheprovider --maxfail=1 \
      tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference \
      tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference | tee "$GPU_OUTPUT"
    grep -Eq "^${GPU_COUNT} passed in [0-9.]+s$" "$GPU_OUTPUT"
    ! grep -Eqi "skipped|deselected|xfailed|xpassed|warning|error" "$GPU_OUTPUT"
    sed -n "s#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed gpu_oracle \1#p" \
      "$GPU_COLLECT"
    echo VERSE_GPU_ORACLE_PASSED

    for artifact in \
      tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
      tests/v1/attention/test_nvfp4_flashinfer_vosplit.py \
      tests/kernels/attention/test_flashinfer.py; do
      sha256sum "$artifact" | sed "s#^#VERSE_CUDA_TEST_ARTIFACT_SHA256=#"
    done
    echo "SM120 image and native NVFP4 FA2 correctness gates passed"
  ' | tee "$LOG_TMP"

grep -Fxq 'VERSE_ROUTING_GATES_PASSED' "$LOG_TMP" || {
  echo "the static runtime routing gates did not pass in the candidate image" >&2
  exit 1
}
grep -Fxq 'VERSE_GPU_ORACLE_PASSED' "$LOG_TMP" || {
  echo "the exact SM120 NVFP4 FA2 oracle did not pass" >&2
  exit 1
}
if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$LOG_TMP"; then
  echo "the SM120 NVFP4 FA2 oracle produced a non-clean test result" >&2
  exit 1
fi

DOCKER_HOST_ID=$(docker info --format '{{.ID}}')
DOCKER_HOST_NAME=$(docker info --format '{{.Name}}')
uv run --no-project python - \
  "$LOG_TMP" "$VERSE_VLLM_IMAGE" "$VERSE_VLLM_EXPECTED_COMMIT" \
  "$VERSE_VLLM_GPU_UUID" "$DOCKER_HOST_ID" "$DOCKER_HOST_NAME" \
  >"$EVIDENCE_TMP" <<'PY'
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
image_digest = sys.argv[2]
fork_commit = sys.argv[3]
expected_gpu_uuid = sys.argv[4]
docker_host_id = sys.argv[5]
docker_host_name = sys.argv[6]
raw = log_path.read_bytes()
text = raw.decode("utf-8")
verification, _ = json.JSONDecoder().raw_decode(text.lstrip())
if not isinstance(verification, dict) or verification.get("status") != "valid":
    raise SystemExit("CUDA oracle log lacks valid image verification JSON")
if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image_digest):
    raise SystemExit("CUDA oracle image identity is not immutable")
if not re.fullmatch(r"[0-9a-f]{40}", fork_commit):
    raise SystemExit("CUDA oracle fork identity is invalid")


def one_prefixed_line(prefix: str) -> str:
    values = [
        line.removeprefix(prefix)
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(values) != 1:
        raise SystemExit(f"CUDA oracle log requires exactly one {prefix} record")
    return values[0]


fields = next(
    csv.reader(
        [one_prefixed_line("VERSE_CUDA_GPU_IDENTITY=")], skipinitialspace=True
    )
)
if len(fields) != 4:
    raise SystemExit("GPU identity must contain exactly four fields")
name, gpu_uuid, memory_total, driver_version = (field.strip() for field in fields)
if not re.fullmatch(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid
):
    raise SystemExit("GPU identity has an invalid UUID")
if gpu_uuid != expected_gpu_uuid:
    raise SystemExit("CUDA oracle ran on an unexpected GPU UUID")
if not memory_total.isdigit() or int(memory_total) <= 0:
    raise SystemExit("GPU identity has invalid memory")
if not name or not driver_version:
    raise SystemExit("GPU identity is incomplete")
verification_gpu = verification.get("gpu")
if not isinstance(verification_gpu, dict):
    raise SystemExit("CUDA image verification lacks GPU identity")
if verification_gpu.get("name") != name or verification_gpu.get("capability") != [12, 0]:
    raise SystemExit("CUDA image verification and nvidia-smi identity differ")

tests = []
for line in text.splitlines():
    if not line.startswith("VERSE_CUDA_TEST_RESULT="):
        continue
    fields = line.removeprefix("VERSE_CUDA_TEST_RESULT=").split(" ", 2)
    if len(fields) != 3:
        raise SystemExit("CUDA test result record is malformed")
    result, suite, node_id = fields
    tests.append({"node_id": node_id, "result": result, "suite": suite})
if not tests or len({test["node_id"] for test in tests}) != len(tests):
    raise SystemExit("CUDA test result inventory is empty or duplicated")
if any(test["result"] != "passed" for test in tests):
    raise SystemExit("CUDA test inventory contains a non-pass result")

artifact_hashes = {}
for line in text.splitlines():
    if not line.startswith("VERSE_CUDA_TEST_ARTIFACT_SHA256="):
        continue
    value = line.removeprefix("VERSE_CUDA_TEST_ARTIFACT_SHA256=")
    match = re.fullmatch(r"([0-9a-f]{64})  (tests/.+\.py)", value)
    if match is None or match.group(2) in artifact_hashes:
        raise SystemExit("CUDA test artifact hash record is malformed or duplicated")
    artifact_hashes[match.group(2)] = match.group(1)
if len(artifact_hashes) != 3:
    raise SystemExit("CUDA test artifact hash inventory is incomplete")

if not re.fullmatch(r"[A-Za-z0-9:._-]{8,256}", docker_host_id):
    raise SystemExit("Docker host identity is invalid")
if not docker_host_name or len(docker_host_name) > 255 or "\n" in docker_host_name:
    raise SystemExit("Docker host name is invalid")
machine_id = Path("/etc/machine-id").read_text().strip()
if not re.fullmatch(r"[0-9a-fA-F]{32}", machine_id):
    raise SystemExit("host machine identity is invalid")
boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip().lower()
if not re.fullmatch(
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id
):
    raise SystemExit("host boot identity is invalid")
machine_id_sha256 = hashlib.sha256(machine_id.lower().encode()).hexdigest()
host_identity_sha256 = hashlib.sha256(
    f"{machine_id_sha256}\n{docker_host_id}\n{boot_id}\n".encode()
).hexdigest()
markers = [
    "VERSE_ROUTING_GATES_PASSED",
    "VERSE_GPU_ORACLE_PASSED",
    "SM120 image and native NVFP4 FA2 correctness gates passed",
]
lines = text.splitlines()
if any(lines.count(marker) != 1 for marker in markers):
    raise SystemExit("CUDA oracle log has an invalid success-marker inventory")

print(
    json.dumps(
        {
            "schema_version": 3,
            "status": "pass",
            "image_digest": image_digest,
            "fork_commit": fork_commit,
            "gpu_selector": expected_gpu_uuid,
            "gpu": {
                "name": name,
                "uuid": gpu_uuid,
                "memory_total_mib": int(memory_total),
                "driver_version": driver_version,
                "compute_capability": [12, 0],
            },
            "host": {
                "identity_sha256": host_identity_sha256,
                "machine_id_sha256": machine_id_sha256,
                "boot_id": boot_id,
                "docker_id": docker_host_id,
                "docker_name": docker_host_name,
            },
            "tests": tests,
            "test_artifacts_sha256": artifact_hashes,
            "test_log_sha256": hashlib.sha256(raw).hexdigest(),
            "oracle_markers": markers,
            "image_verification": verification,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

[[ ! -e $CUDA_LOG_FILE && ! -L $CUDA_LOG_FILE ]] || {
  echo "CUDA log destination appeared during qualification" >&2
  exit 1
}
[[ ! -e $CUDA_EVIDENCE_FILE && ! -L $CUDA_EVIDENCE_FILE ]] || {
  echo "CUDA identity destination appeared during qualification" >&2
  exit 1
}
mv -- "$LOG_TMP" "$CUDA_LOG_FILE"
mv -- "$EVIDENCE_TMP" "$CUDA_EVIDENCE_FILE"

if ((EMIT_EVIDENCE == 1)); then
  cat "$CUDA_EVIDENCE_FILE"
  rm -f -- "$CUDA_LOG_FILE" "$CUDA_EVIDENCE_FILE"
fi
echo "SM120 image and native NVFP4 FA2 correctness gates passed" >&2
