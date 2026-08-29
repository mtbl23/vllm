#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

require_env() {
  local name=$1
  [[ -n ${!name:-} ]] || {
    echo "$name is required" >&2
    exit 1
  }
}

require_env VERSE_VLLM_RELEASE_DIR
require_env VERSE_VLLM_API_KEY_FILE
require_env VERSE_VLLM_EXPECTED_COMMIT
require_env VERSE_VLLM_IMAGE
require_env VERSE_VLLM_CONTAINER_ID
require_env VERSE_VLLM_GPU_UUID
require_env VERSE_VLLM_IMAGE_RECEIPT

"$ROOT/tools/verse/verify_sm120_source.sh" "$VERSE_VLLM_EXPECTED_COMMIT" \
  >/dev/null

[[ $VERSE_VLLM_CONTAINER_ID =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_CONTAINER_ID must be an exact 64-character container ID" >&2
  exit 1
}
[[ $VERSE_VLLM_GPU_UUID =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] || {
  echo "VERSE_VLLM_GPU_UUID must be an exact full GPU UUID" >&2
  exit 1
}
ACTUAL_CONTAINER_ID=$(docker inspect --format '{{.Id}}' \
  "$VERSE_VLLM_CONTAINER_ID")
[[ $ACTUAL_CONTAINER_ID == "$VERSE_VLLM_CONTAINER_ID" ]] || {
  echo "candidate container identity changed" >&2
  exit 1
}
CONTAINER_IMAGE=$(docker inspect --format '{{.Config.Image}}' \
  "$VERSE_VLLM_CONTAINER_ID")
CONTAINER_COMMIT=$(docker inspect \
  --format '{{index .Config.Labels "ai.vllm.build.commit"}}' \
  "$VERSE_VLLM_CONTAINER_ID")
CONTAINER_GPU_UUID=$(docker inspect \
  --format '{{index .Config.Labels "ai.verse.gpu.uuid"}}' \
  "$VERSE_VLLM_CONTAINER_ID")
[[ $CONTAINER_IMAGE == "$VERSE_VLLM_IMAGE" ]] || {
  echo "candidate container does not use VERSE_VLLM_IMAGE" >&2
  exit 1
}
[[ $CONTAINER_COMMIT == "$VERSE_VLLM_EXPECTED_COMMIT" ]] || {
  echo "candidate container does not use VERSE_VLLM_EXPECTED_COMMIT" >&2
  exit 1
}
[[ $CONTAINER_GPU_UUID == "$VERSE_VLLM_GPU_UUID" ]] || {
  echo "candidate container does not bind VERSE_VLLM_GPU_UUID" >&2
  exit 1
}
export VERSE_VLLM_CONTAINER="$VERSE_VLLM_CONTAINER_ID"

[[ $VERSE_VLLM_RELEASE_DIR == /* ]] || {
  echo "VERSE_VLLM_RELEASE_DIR must be absolute" >&2
  exit 1
}
if [[ -e $VERSE_VLLM_RELEASE_DIR ]] &&
  [[ -n $(find "$VERSE_VLLM_RELEASE_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "VERSE_VLLM_RELEASE_DIR must be absent or empty" >&2
  exit 1
fi
install -d -m 0750 "$VERSE_VLLM_RELEASE_DIR"
install -m 0600 "$VERSE_VLLM_IMAGE_RECEIPT" \
  "$VERSE_VLLM_RELEASE_DIR/image-receipt.json"

export VERSE_VLLM_RELEASE_NONCE
VERSE_VLLM_RELEASE_NONCE=$(uv run --no-project python -c \
  'import secrets; print(secrets.token_hex(32))')
[[ $VERSE_VLLM_RELEASE_NONCE =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to generate a valid release nonce" >&2
  exit 1
}

CUDA_LOG="$VERSE_VLLM_RELEASE_DIR/cuda-oracle.log"
CUDA_IDENTITY="$VERSE_VLLM_RELEASE_DIR/cuda-oracle.json"
export VERSE_VLLM_CUDA_LOG_FILE="$CUDA_LOG"
export VERSE_VLLM_CUDA_EVIDENCE_FILE="$CUDA_IDENTITY"
docker inspect "$VERSE_VLLM_CONTAINER_ID" \
  >"$VERSE_VLLM_RELEASE_DIR/container-before-cuda.json"
"$ROOT/tools/verse/run_sm120_cuda_gates.sh"
docker inspect "$VERSE_VLLM_CONTAINER_ID" \
  >"$VERSE_VLLM_RELEASE_DIR/container-after-cuda.json"

export VERSE_VLLM_ACCEPTANCE_DIR="$VERSE_VLLM_RELEASE_DIR/short"
"$ROOT/tools/verse/run_sm120_acceptance.sh"

ENDPOINT=$(awk -F= '$1 == "endpoint" {print $2}' \
  "$VERSE_VLLM_ACCEPTANCE_DIR/preflight.txt")
[[ $ENDPOINT =~ ^http://127\.0\.0\.1:[0-9]+$ ]] || {
  echo "short acceptance did not return a loopback endpoint" >&2
  exit 1
}

CANDIDATE_GPU_IDENTITY=$(docker exec "$VERSE_VLLM_CONTAINER_ID" nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader,nounits)
[[ -n $CANDIDATE_GPU_IDENTITY && $CANDIDATE_GPU_IDENTITY != *$'\n'* ]] || {
  echo "candidate container did not expose exactly one GPU identity" >&2
  exit 1
}
IDENTITY_ARGS=(
  --image-digest "$VERSE_VLLM_IMAGE"
  --fork-commit "$VERSE_VLLM_EXPECTED_COMMIT"
  --model-revision "$VERSE_MODEL_REVISION"
  --gpu-name "$CANDIDATE_GPU_IDENTITY"
  --release-nonce "$VERSE_VLLM_RELEASE_NONCE"
  --container-id "$VERSE_VLLM_CONTAINER_ID"
)

uv run --script "$ROOT/tools/verse/run_sm120_queue_stress.py" \
  --endpoint "$ENDPOINT" \
  --model verse-free \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --phase-seconds 60 \
  --active-capacity 38 \
  --overflow-clients 76 \
  --prompt-pool-size 96 \
  "${IDENTITY_ARGS[@]}" \
  >"$VERSE_VLLM_RELEASE_DIR/queue-stress.json"

uv run --script "$ROOT/tools/verse/run_sm120_user_latency.py" \
  --endpoint "$ENDPOINT" \
  --model verse-free \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --samples-per-prompt 5 \
  "${IDENTITY_ARGS[@]}" \
  >"$VERSE_VLLM_RELEASE_DIR/user-latency.json"

uv run --script "$ROOT/tools/verse/run_sm120_warm_latency.py" \
  --endpoint "$ENDPOINT" \
  --model verse-free \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --clients 38 \
  --max-tokens 100 \
  "${IDENTITY_ARGS[@]}" \
  >"$VERSE_VLLM_RELEASE_DIR/warm-latency.json"

uv run --script "$ROOT/tools/verse/run_sm120_churn.py" \
  --endpoint "$ENDPOINT" \
  --model verse-free \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --duration-seconds 7200 \
  --concurrency 38 \
  --prompt-pool-size 64 \
  "${IDENTITY_ARGS[@]}" \
  >"$VERSE_VLLM_RELEASE_DIR/churn.json"

uv run --script "$ROOT/tools/verse/check_sm120_chat_contract.py" \
  --endpoint "$ENDPOINT" \
  --model verse-free \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  "${IDENTITY_ARGS[@]}" \
  >"$VERSE_VLLM_RELEASE_DIR/post-churn-chat-contract.json"

"$ROOT/tools/verse/check_sm120_server.sh" \
  >"$VERSE_VLLM_RELEASE_DIR/post-churn-server.txt"
docker inspect "$VERSE_VLLM_CONTAINER_ID" \
  >"$VERSE_VLLM_RELEASE_DIR/container-after-churn.json"

DOCKER_HOST_ID=$(docker info --format '{{.ID}}')
DOCKER_HOST_NAME=$(docker info --format '{{.Name}}')
CANDIDATE_HOST_TMP="$VERSE_VLLM_RELEASE_DIR/.candidate-host.json.tmp"
[[ ! -e $CANDIDATE_HOST_TMP ]] || {
  echo "temporary candidate host evidence already exists" >&2
  exit 1
}
uv run --no-project python - \
  "$VERSE_VLLM_RELEASE_DIR/container-after-churn.json" \
  "$CUDA_IDENTITY" "$CANDIDATE_GPU_IDENTITY" "$DOCKER_HOST_ID" \
  "$DOCKER_HOST_NAME" >"$CANDIDATE_HOST_TMP" <<'PY'
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

containers = json.loads(Path(sys.argv[1]).read_bytes())
cuda = json.loads(Path(sys.argv[2]).read_bytes())
gpu_fields = next(csv.reader([sys.argv[3]], skipinitialspace=True))
docker_id = sys.argv[4]
docker_name = sys.argv[5]
if not isinstance(containers, list) or len(containers) != 1:
    raise SystemExit("candidate container evidence is invalid")
container = containers[0]
if not isinstance(container, dict) or not re.fullmatch(
    r"[0-9a-f]{64}", str(container.get("Id", ""))
):
    raise SystemExit("candidate container identity is invalid")
if len(gpu_fields) != 4:
    raise SystemExit("candidate GPU identity must have exactly four fields")
name, gpu_uuid, memory_total, driver_version = (field.strip() for field in gpu_fields)
if not re.fullmatch(
    r"GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}", gpu_uuid
):
    raise SystemExit("candidate GPU UUID is invalid")
if not memory_total.isdigit() or int(memory_total) <= 0 or not name or not driver_version:
    raise SystemExit("candidate GPU identity is incomplete")
gpu = {
    "name": name,
    "uuid": gpu_uuid,
    "memory_total_mib": int(memory_total),
    "driver_version": driver_version,
    "compute_capability": [12, 0],
}
labels = ((container.get("Config") or {}).get("Labels") or {})
if labels.get("ai.verse.gpu.uuid") != gpu_uuid:
    raise SystemExit("candidate GPU UUID differs from its immutable container label")
requests = (container.get("HostConfig") or {}).get("DeviceRequests") or []
if len(requests) != 1 or not isinstance(requests[0], dict):
    raise SystemExit("candidate container does not have exactly one GPU request")
selectors = requests[0].get("DeviceIDs")
if not isinstance(selectors, list) or len(selectors) != 1 or not selectors[0]:
    raise SystemExit("candidate container GPU selector is invalid")
if not re.fullmatch(r"[A-Za-z0-9:._-]{8,256}", docker_id):
    raise SystemExit("Docker host identity is invalid")
if not docker_name or len(docker_name) > 255 or "\n" in docker_name:
    raise SystemExit("Docker host name is invalid")
machine_id = Path("/etc/machine-id").read_text().strip().lower()
boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip().lower()
if not re.fullmatch(r"[0-9a-f]{32}", machine_id):
    raise SystemExit("host machine identity is invalid")
if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
    raise SystemExit("host boot identity is invalid")
machine_id_sha256 = hashlib.sha256(machine_id.encode()).hexdigest()
host_identity_sha256 = hashlib.sha256(
    f"{machine_id_sha256}\n{docker_id}\n{boot_id}\n".encode()
).hexdigest()
host = {
    "identity_sha256": host_identity_sha256,
    "machine_id_sha256": machine_id_sha256,
    "boot_id": boot_id,
    "docker_id": docker_id,
    "docker_name": docker_name,
}
if cuda.get("host") != host:
    raise SystemExit("CUDA and candidate container host identities differ")
if cuda.get("gpu") != gpu or cuda.get("gpu_selector") != gpu_uuid:
    raise SystemExit("CUDA and candidate container GPU identities differ")
print(
    json.dumps(
        {
            "schema_version": 1,
            "status": "pass",
            "container_id": container["Id"],
            "image_digest": (container.get("Config") or {}).get("Image"),
            "fork_commit": labels.get("ai.vllm.build.commit"),
            "container_gpu_selector": selectors[0],
            "gpu": gpu,
            "host": host,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
mv "$CANDIDATE_HOST_TMP" "$VERSE_VLLM_RELEASE_DIR/candidate-host.json"

MANIFEST_TMP="$VERSE_VLLM_RELEASE_DIR/.release-manifest.tmp"
[[ ! -e $MANIFEST_TMP ]] || {
  echo "temporary release manifest already exists" >&2
  exit 1
}
uv run --script "$ROOT/tools/verse/finalize_sm120_release.py" \
  --release-dir "$VERSE_VLLM_RELEASE_DIR" >"$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$VERSE_VLLM_RELEASE_DIR/release-manifest.json"

printf 'status=pass\nrelease_manifest=%s\nrelease_nonce=%s\ncontainer_id=%s\n' \
  "$VERSE_VLLM_RELEASE_DIR/release-manifest.json" \
  "$VERSE_VLLM_RELEASE_NONCE" \
  "$VERSE_VLLM_CONTAINER_ID"
