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
require_env VERSE_VLLM_ACCEPTANCE_DIR
require_env VERSE_VLLM_IMAGE_RECEIPT

"$ROOT/tools/verse/verify_sm120_source.sh" "$VERSE_VLLM_EXPECTED_COMMIT" \
  >/dev/null

[[ $VERSE_VLLM_ACCEPTANCE_DIR == /* ]] || {
  echo "VERSE_VLLM_ACCEPTANCE_DIR must be absolute" >&2
  exit 1
}
if [[ -e $VERSE_VLLM_ACCEPTANCE_DIR ]] &&
  [[ -n $(find "$VERSE_VLLM_ACCEPTANCE_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "VERSE_VLLM_ACCEPTANCE_DIR must be absent or empty" >&2
  exit 1
fi
install -d -m 0750 "$VERSE_VLLM_ACCEPTANCE_DIR"

RELEASE_NONCE=${VERSE_VLLM_RELEASE_NONCE:-}
if [[ -z $RELEASE_NONCE ]]; then
  RELEASE_NONCE=$(uv run --no-project python -c \
    'import secrets; print(secrets.token_hex(32))')
fi
[[ $RELEASE_NONCE =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_RELEASE_NONCE must be 64 lowercase hex characters" >&2
  exit 1
}

PROFILE_SHELL=$(uv run --script "$ROOT/tools/verse/validate_sm120_profile.py" \
  --image "$VERSE_VLLM_IMAGE" \
  --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
  --emit-shell)
eval "$PROFILE_SHELL"

CONTAINER=${VERSE_VLLM_CONTAINER:-verse-vllm-sm120}
PREFLIGHT="$VERSE_VLLM_ACCEPTANCE_DIR/preflight.txt"
POSTFLIGHT="$VERSE_VLLM_ACCEPTANCE_DIR/postflight.txt"
"$ROOT/tools/verse/check_sm120_server.sh" | tee "$PREFLIGHT"
ENDPOINT=$(awk -F= '$1 == "endpoint" {print $2}' "$PREFLIGHT")
[[ $ENDPOINT =~ ^http://127\.0\.0\.1:[0-9]+$ ]] || {
  echo "preflight did not return a loopback endpoint" >&2
  exit 1
}

docker inspect "$CONTAINER" >"$VERSE_VLLM_ACCEPTANCE_DIR/container-before.json"
GPU_IDENTITY=$(docker exec "$CONTAINER" nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$GPU_IDENTITY" | wc -l | tr -d ' ') == 1 ]] || {
  echo "expected exactly one visible GPU identity" >&2
  exit 1
}

uv run --script "$ROOT/tools/verse/check_sm120_chat_contract.py" \
  --endpoint "$ENDPOINT" \
  --model "$VERSE_SERVED_MODEL_NAME" \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --image-digest "$VERSE_VLLM_IMAGE" \
  --fork-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
  --model-revision "$VERSE_MODEL_REVISION" \
  --gpu-name "$GPU_IDENTITY" \
  --release-nonce "$RELEASE_NONCE" \
  --container-id "$CONTAINER" \
  >"$VERSE_VLLM_ACCEPTANCE_DIR/chat-contract.json"

REPORTS=()
for TARGET in 1000 5500; do
  for RUN in 1 2 3; do
    REPORT="$VERSE_VLLM_ACCEPTANCE_DIR/b01-${TARGET}-run-${RUN}.json"
    EXTRA_ARGS=()
    if [[ $RUN == 1 ]]; then
      EXTRA_ARGS+=(--skip-warmup)
    fi
    uv run --script "$ROOT/benchmarks/verse/sm120_b01.py" \
      --endpoint "$ENDPOINT" \
      --model "$VERSE_SERVED_MODEL_NAME" \
      --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
      --concurrency 38 \
      --prompt-tokens "$TARGET" \
      --output-tokens 512 \
      --minimum-steady-seconds 10 \
      --minimum-steady-samples 50 \
      --minimum-aggregate 0 \
      --minimum-wall-ratio 0 \
      --image-digest "$VERSE_VLLM_IMAGE" \
      --fork-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
      --model-revision "$VERSE_MODEL_REVISION" \
      --gpu-name "$GPU_IDENTITY" \
      --release-nonce "$RELEASE_NONCE" \
      "${EXTRA_ARGS[@]}" \
      >"$REPORT"
    REPORTS+=("$REPORT")
  done
done

uv run --script "$ROOT/tools/verse/evaluate_sm120_acceptance.py" \
  "${REPORTS[@]}" >"$VERSE_VLLM_ACCEPTANCE_DIR/b01-summary.json"

for SHAPE in 37x1 30x8; do
  case "$SHAPE" in
    37x1) DECODERS=37; PREFILLS=1 ;;
    30x8) DECODERS=30; PREFILLS=8 ;;
    *) echo "unexpected prefill interference shape" >&2; exit 1 ;;
  esac
  uv run --script "$ROOT/benchmarks/verse/sm120_prefill_interference.py" \
    --endpoint "$ENDPOINT" \
    --model "$VERSE_SERVED_MODEL_NAME" \
    --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
    --decoders "$DECODERS" \
    --prefills "$PREFILLS" \
    --decode-prompt-tokens 4500 \
    --decode-output-tokens 1024 \
    --prefill-prompt-tokens 6000 \
    --baseline-seconds 3 \
    --metrics-interval 0.05 \
    --image-digest "$VERSE_VLLM_IMAGE" \
    --fork-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
    --model-revision "$VERSE_MODEL_REVISION" \
    --gpu-name "$GPU_IDENTITY" \
    --release-nonce "$RELEASE_NONCE" \
    --max-num-batched-tokens "$VERSE_MAX_NUM_BATCHED_TOKENS" \
    >"$VERSE_VLLM_ACCEPTANCE_DIR/prefill-${SHAPE}.json"
done

"$ROOT/tools/verse/check_sm120_server.sh" | tee "$POSTFLIGHT"
docker inspect "$CONTAINER" >"$VERSE_VLLM_ACCEPTANCE_DIR/container-after.json"

printf 'status=pass\nacceptance_dir=%s\nsummary=%s\nrelease_nonce=%s\n' \
  "$VERSE_VLLM_ACCEPTANCE_DIR" \
  "$VERSE_VLLM_ACCEPTANCE_DIR/b01-summary.json" \
  "$RELEASE_NONCE"
