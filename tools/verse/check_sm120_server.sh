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

require_env VERSE_VLLM_IMAGE
require_env VERSE_VLLM_EXPECTED_COMMIT
require_env VERSE_VLLM_CACHE_DIR
require_env VERSE_MODEL_CACHE_DIR
require_env VERSE_VLLM_API_KEY_FILE
require_env VERSE_VLLM_GPU_UUID
require_env VERSE_VLLM_CONTAINER_ID

"$ROOT/tools/verse/verify_sm120_source.sh" "$VERSE_VLLM_EXPECTED_COMMIT" \
  >/dev/null

CONTAINER_ID=$VERSE_VLLM_CONTAINER_ID
GPU_DEVICE=${VERSE_VLLM_GPU_DEVICE:-0}
GPU_UUID=$VERSE_VLLM_GPU_UUID
READY_TIMEOUT=${VERSE_VLLM_READY_TIMEOUT_SECONDS:-900}
EXPECTED_RESTART_POLICY=${VERSE_VLLM_EXPECTED_RESTART_POLICY:-unless-stopped}

[[ $READY_TIMEOUT =~ ^[0-9]+$ ]] && ((READY_TIMEOUT >= 30)) || {
  echo "VERSE_VLLM_READY_TIMEOUT_SECONDS must be at least 30" >&2
  exit 1
}
[[ $EXPECTED_RESTART_POLICY == no || $EXPECTED_RESTART_POLICY == unless-stopped ]] || {
  echo "VERSE_VLLM_EXPECTED_RESTART_POLICY must be no or unless-stopped" >&2
  exit 1
}
[[ $GPU_DEVICE =~ ^[0-9]+$ ]] || {
  echo "VERSE_VLLM_GPU_DEVICE must be a non-negative integer" >&2
  exit 1
}
[[ $GPU_UUID =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] || {
  echo "VERSE_VLLM_GPU_UUID must be an exact full GPU UUID" >&2
  exit 1
}
[[ $CONTAINER_ID =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_CONTAINER_ID must be a full 64-character container ID" >&2
  exit 1
}

PROFILE_SHELL=$(uv run --script "$ROOT/tools/verse/validate_sm120_profile.py" \
  --profile "$ROOT/tools/verse/sm120_profile.env" \
  --image "$VERSE_VLLM_IMAGE" \
  --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
  --emit-shell)
eval "$PROFILE_SHELL"
SERVED_MODEL=$VERSE_SERVED_MODEL_NAME
MODEL_READY=$(uv run --script "$ROOT/tools/verse/prepare_sm120_model.py" \
  --cache-dir "$VERSE_MODEL_CACHE_DIR" --verify-ready --require-root-owner)
MODEL_DIRECTORY=$(uv run --no-project python -c \
  'import json,sys; print(json.load(sys.stdin)["model_directory"])' \
  <<<"$MODEL_READY")
MODEL_IDENTITY=$(uv run --no-project python -c '
import json,sys
payload=json.load(sys.stdin)
verification=payload["verification"]
print(verification["manifest_sha256"])
print(verification["config_sha256"])
print(verification["files"])
print(verification["bytes"])
' <<<"$MODEL_READY")
mapfile -t MODEL_IDENTITY_FIELDS <<<"$MODEL_IDENTITY"
MODEL_MANIFEST_SHA256=${MODEL_IDENTITY_FIELDS[0]}
MODEL_CONFIG_SHA256=${MODEL_IDENTITY_FIELDS[1]}
MODEL_FILE_COUNT=${MODEL_IDENTITY_FIELDS[2]}
MODEL_BYTES=${MODEL_IDENTITY_FIELDS[3]}
MODEL_READY_MARKER_SHA256=$(uv run --no-project python -c '
import hashlib,sys
with open(sys.argv[1], "rb") as handle:
    print(hashlib.file_digest(handle, "sha256").hexdigest())
' "$VERSE_MODEL_CACHE_DIR/.verse-sm120-model-ready.json")

INSPECT=$(mktemp)
VALIDATED=$(mktemp)
MODELS=$(mktemp)
AUTH_HEADER=$(mktemp)
trap 'rm -f "$INSPECT" "$VALIDATED" "$MODELS" "$AUTH_HEADER"' EXIT
chmod 600 "$AUTH_HEADER"

validate_container() {
  docker inspect "$CONTAINER_ID" >"$INSPECT"
  uv run --script "$ROOT/tools/verse/validate_sm120_container.py" \
    --container-id "$CONTAINER_ID" \
    --image "$VERSE_VLLM_IMAGE" \
    --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
    --served-model "$SERVED_MODEL" \
    --gpu-device "$GPU_DEVICE" \
    --gpu-uuid "$GPU_UUID" \
    --runtime-cache "$VERSE_VLLM_CACHE_DIR" \
    --model-cache "$VERSE_MODEL_CACHE_DIR" \
    --model-directory "$MODEL_DIRECTORY" \
    --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
    --restart-policy "$EXPECTED_RESTART_POLICY" \
    <"$INSPECT" >"$VALIDATED"
}

validate_container
HOST_PORT=$(uv run --no-project python -c \
  'import json,sys; print(json.load(sys.stdin)["host_port"])' <"$VALIDATED")
STARTED_AT=$(uv run --no-project python -c \
  'import json,sys; print(json.load(sys.stdin)["started_at"])' <"$VALIDATED")
ENDPOINT="http://127.0.0.1:${HOST_PORT}"
VISIBLE_GPU_UUID=$(LC_ALL=C docker exec "$CONTAINER_ID" nvidia-smi \
  --query-gpu=uuid --format=csv,noheader,nounits)
[[ -n $VISIBLE_GPU_UUID && $VISIBLE_GPU_UUID != *$'\n'* ]] || {
  echo "container does not expose exactly one GPU UUID" >&2
  exit 1
}
VISIBLE_GPU_UUID=${VISIBLE_GPU_UUID//[[:space:]]/}
[[ $VISIBLE_GPU_UUID == "$GPU_UUID" ]] || {
  echo "container-visible GPU UUID does not match VERSE_VLLM_GPU_UUID" >&2
  exit 1
}

deadline=$((SECONDS + READY_TIMEOUT))
until curl --noproxy '*' --fail --silent --show-error --max-time 5 \
  "$ENDPOINT/health" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "vLLM did not become healthy within ${READY_TIMEOUT}s" >&2
    docker logs --since "$STARTED_AT" "$CONTAINER_ID" 2>&1 | tail -200 >&2
    exit 1
  fi
  sleep 2
done

UNAUTH_STATUS=$(curl --noproxy '*' --silent --output /dev/null \
  --write-out '%{http_code}' \
  --max-time 10 --max-redirs 0 "$ENDPOINT/v1/models")
[[ $UNAUTH_STATUS == 401 ]] || {
  echo "unauthenticated model listing returned HTTP $UNAUTH_STATUS, expected 401" >&2
  exit 1
}

API_KEY=$(cat "$VERSE_VLLM_API_KEY_FILE")
printf 'Authorization: Bearer %s\n' "$API_KEY" >"$AUTH_HEADER"
unset API_KEY
curl --noproxy '*' --fail --silent --show-error --max-time 30 --max-redirs 0 \
  --header "@$AUTH_HEADER" "$ENDPOINT/v1/models" >"$MODELS"
uv run --no-project python -c '
import json, sys
served_model = sys.argv[1]
payload = json.load(sys.stdin)
ids = {entry.get("id") for entry in payload.get("data", [])}
if ids != {served_model}:
    raise SystemExit(f"served model IDs do not exactly match {served_model!r}: {ids}")
' "$SERVED_MODEL" <"$MODELS"

LOGS=$(docker logs --since "$STARTED_AT" "$CONTAINER_ID" 2>&1)
grep -Fq "Strict Verse Gemma 4 runtime validated" <<<"$LOGS" || {
  echo "strict Verse runtime validation marker is absent" >&2
  exit 1
}
grep -Fq "Using the FlashInfer FA2 paged wrapper" <<<"$LOGS" || {
  echo "native FlashInfer FA2 wrapper marker is absent" >&2
  exit 1
}
grep -Fq \
  "head_dim=512, page_size=64, speculative_tokens=0, xqa=True" \
  <<<"$LOGS" || {
  echo "exact Verse D512 NVFP4 XQA route marker is absent" >&2
  exit 1
}
if grep -Eiq \
  'fallback.*(triton|xqa)|using.*triton_attn|traceback|fatal|out of memory|oom-killed|cuda graph.*(fail|hang)|illegal memory access' \
  <<<"$LOGS"; then
  echo "current-start logs contain a forbidden fallback or fatal error" >&2
  exit 1
fi

# Revalidate after readiness so a crash cannot borrow stale startup evidence.
validate_container
CURRENT_STARTED_AT=$(uv run --no-project python -c \
  'import json,sys; print(json.load(sys.stdin)["started_at"])' <"$VALIDATED")
[[ $CURRENT_STARTED_AT == "$STARTED_AT" ]] || {
  echo "container restarted while readiness was being checked" >&2
  exit 1
}

printf 'status=healthy\ncontainer_id=%s\nendpoint=%s\ncommit=%s\nprofile=%s\ngpu_uuid=%s\nmodel_manifest_sha256=%s\nmodel_config_sha256=%s\nmodel_ready_marker_sha256=%s\nmodel_file_count=%s\nmodel_bytes=%s\n' \
  "$CONTAINER_ID" "$ENDPOINT" "$VERSE_VLLM_EXPECTED_COMMIT" \
  "$VERSE_RUNTIME_PROFILE" "$GPU_UUID" "$MODEL_MANIFEST_SHA256" \
  "$MODEL_CONFIG_SHA256" "$MODEL_READY_MARKER_SHA256" \
  "$MODEL_FILE_COUNT" "$MODEL_BYTES"
