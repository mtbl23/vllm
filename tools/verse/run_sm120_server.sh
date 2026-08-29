#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/tools/verse/sm120_sha256.bash"

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
require_env VERSE_VLLM_IMAGE_RECEIPT

"$ROOT/tools/verse/verify_sm120_source.sh" "$VERSE_VLLM_EXPECTED_COMMIT" \
  >/dev/null

if ((EUID != 0)); then
  echo "the Verse appliance launcher must run as root to create isolated mounts" >&2
  exit 1
fi

if [[ ! $VERSE_VLLM_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]]; then
  echo "VERSE_VLLM_EXPECTED_COMMIT must be a 40-character commit SHA" >&2
  exit 1
fi
if [[ ! $VERSE_VLLM_IMAGE =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "VERSE_VLLM_IMAGE must use an immutable sha256 digest" >&2
  exit 1
fi
if [[ $VERSE_VLLM_CACHE_DIR != /* ]]; then
  echo "VERSE_VLLM_CACHE_DIR must be absolute" >&2
  exit 1
fi
if [[ $VERSE_MODEL_CACHE_DIR != /* ]]; then
  echo "VERSE_MODEL_CACHE_DIR must be absolute" >&2
  exit 1
fi
[[ $VERSE_VLLM_IMAGE_RECEIPT == /* ]] || {
  echo "VERSE_VLLM_IMAGE_RECEIPT must be absolute" >&2
  exit 1
}

PROFILE_SHELL=$(uv run --script "$ROOT/tools/verse/validate_sm120_profile.py" \
  --profile "$ROOT/tools/verse/sm120_profile.env" \
  --image "$VERSE_VLLM_IMAGE" \
  --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
  --emit-shell)
eval "$PROFILE_SHELL"

CONTAINER=${VERSE_VLLM_CONTAINER:-verse-vllm-sm120}
HOST_PORT=${VERSE_VLLM_HOST_PORT:-8000}
GPU_DEVICE=${VERSE_VLLM_GPU_DEVICE:-0}
GPU_UUID=$VERSE_VLLM_GPU_UUID
SERVED_MODEL=$VERSE_SERVED_MODEL_NAME

[[ $HOST_PORT =~ ^[0-9]+$ ]] && ((HOST_PORT > 0 && HOST_PORT < 65536)) || {
  echo "VERSE_VLLM_HOST_PORT must be a valid TCP port" >&2
  exit 1
}
[[ $HOST_PORT == 8000 ]] || {
  echo "the fixed SM120 gateway contract requires VERSE_VLLM_HOST_PORT=8000" >&2
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
[[ $SERVED_MODEL =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "VERSE_SERVED_MODEL_NAME contains unsupported characters" >&2
  exit 1
}
[[ $VERSE_VLLM_API_KEY_FILE == /* ]] || {
  echo "VERSE_VLLM_API_KEY_FILE must be absolute" >&2
  exit 1
}
[[ -f $VERSE_VLLM_API_KEY_FILE && ! -L $VERSE_VLLM_API_KEY_FILE ]] || {
  echo "VERSE_VLLM_API_KEY_FILE must be a regular, non-symlink file" >&2
  exit 1
}
[[ -s $VERSE_VLLM_API_KEY_FILE ]] || {
  echo "VERSE_VLLM_API_KEY_FILE is empty" >&2
  exit 1
}
if LC_ALL=C grep -q $'\r' "$VERSE_VLLM_API_KEY_FILE" ||
  [[ $(awk 'END { print NR }' "$VERSE_VLLM_API_KEY_FILE") -ne 1 ]]; then
  echo "VERSE_VLLM_API_KEY_FILE must contain exactly one line" >&2
  exit 1
fi
[[ $VERSE_VLLM_CACHE_DIR != *,* ]] || {
  echo "VERSE_VLLM_CACHE_DIR cannot contain a comma" >&2
  exit 1
}
[[ $VERSE_MODEL_CACHE_DIR != *,* ]] || {
  echo "VERSE_MODEL_CACHE_DIR cannot contain a comma" >&2
  exit 1
}

verify_gpu_identity() {
  local identity detected_device detected_uuid extra
  identity=$(LC_ALL=C nvidia-smi --id="$GPU_DEVICE" \
    --query-gpu=index,uuid --format=csv,noheader,nounits) || {
    echo "failed to resolve VERSE_VLLM_GPU_DEVICE with nvidia-smi" >&2
    return 1
  }
  [[ -n $identity && $identity != *$'\n'* ]] || {
    echo "GPU ordinal did not resolve to exactly one GPU" >&2
    return 1
  }
  IFS=, read -r detected_device detected_uuid extra <<<"$identity"
  detected_device=${detected_device//[[:space:]]/}
  detected_uuid=${detected_uuid//[[:space:]]/}
  [[ -z ${extra//[[:space:]]/} && $detected_device == "$GPU_DEVICE" ]] || {
    echo "nvidia-smi returned an unexpected GPU ordinal" >&2
    return 1
  }
  [[ $detected_uuid == "$GPU_UUID" ]] || {
    echo "GPU ordinal $GPU_DEVICE does not resolve to VERSE_VLLM_GPU_UUID" >&2
    return 1
  }
}

require_idle_gpu() {
  local compute_processes
  compute_processes=$(LC_ALL=C nvidia-smi --id="$GPU_UUID" \
    --query-compute-apps=pid --format=csv,noheader,nounits) || {
    echo "failed to query existing GPU compute processes" >&2
    return 1
  }
  [[ -z ${compute_processes//[[:space:]]/} ]] || {
    echo "refusing to share GPU $GPU_UUID with a pre-existing compute process" >&2
    return 1
  }
}

for broad_path in / /bin /boot /dev /etc /home /lib /lib64 /opt /proc /root \
  /run /sbin /srv /sys /tmp /usr /var /var/lib; do
  if [[ $(realpath -m -- "$VERSE_VLLM_CACHE_DIR") == "$broad_path" ]] ||
    [[ $(realpath -m -- "$VERSE_MODEL_CACHE_DIR") == "$broad_path" ]]; then
    echo "cache paths must be dedicated subdirectories, not $broad_path" >&2
    exit 1
  fi
done

RUNTIME_CACHE=$(realpath -m -- "$VERSE_VLLM_CACHE_DIR")
MODEL_CACHE=$(realpath -m -- "$VERSE_MODEL_CACHE_DIR")
if [[ $VERSE_VLLM_CACHE_DIR != "$RUNTIME_CACHE" ]] ||
  [[ $VERSE_MODEL_CACHE_DIR != "$MODEL_CACHE" ]]; then
  echo "cache paths must be canonical and contain no symlink components" >&2
  exit 1
fi
if [[ $RUNTIME_CACHE == "$MODEL_CACHE" ]] ||
  [[ $RUNTIME_CACHE == "$MODEL_CACHE"/* ]] ||
  [[ $MODEL_CACHE == "$RUNTIME_CACHE"/* ]]; then
  echo "runtime and model caches must be disjoint" >&2
  exit 1
fi

API_KEY_FILE=$(uv run --no-project python - \
  "$VERSE_VLLM_API_KEY_FILE" "$RUNTIME_CACHE" "$MODEL_CACHE" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
runtime_cache = Path(sys.argv[2])
model_cache = Path(sys.argv[3])
if not path.is_absolute() or path.is_symlink() or not path.is_file():
    raise SystemExit("VERSE_VLLM_API_KEY_FILE must be an absolute regular non-symlink file")
resolved = path.resolve(strict=True)
if resolved != path:
    raise SystemExit("VERSE_VLLM_API_KEY_FILE must be canonical with no symlink components")
for cache in (runtime_cache, model_cache):
    try:
        resolved.relative_to(cache)
    except ValueError:
        pass
    else:
        raise SystemExit("the API key file must be outside both writable and model caches")
mode = stat.S_IMODE(resolved.stat().st_mode)
if mode & (stat.S_IRWXG | stat.S_IRWXO):
    raise SystemExit("the API key file must be owner-only")
if resolved.stat().st_uid != 2000 or not os.access(resolved, os.R_OK):
    raise SystemExit("the API key file must be owned by runtime UID 2000 and readable")
raw = resolved.read_bytes()
if len(raw) > 4096 or b"\r" in raw or b"\0" in raw:
    raise SystemExit("the API key file has invalid bytes")
if len(raw.splitlines()) != 1 or not raw.splitlines()[0]:
    raise SystemExit("the API key file must contain exactly one non-empty line")
print(resolved)
PY
)
VERSE_VLLM_API_KEY_FILE=$API_KEY_FILE

if [[ -e $RUNTIME_CACHE && ! -d $RUNTIME_CACHE ]]; then
  echo "VERSE_VLLM_CACHE_DIR exists but is not a directory" >&2
  exit 1
fi
if [[ -d $RUNTIME_CACHE ]] &&
  [[ -n $(find "$RUNTIME_CACHE" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "runtime cache is not empty; each candidate requires a fresh cache" >&2
  exit 1
fi

MODEL_READY=$(uv run --script "$ROOT/tools/verse/prepare_sm120_model.py" \
  --cache-dir "$VERSE_MODEL_CACHE_DIR" --verify-ready --require-root-owner)
MODEL_DIRECTORY=$(uv run --no-project python -c \
  'import json,sys; print(json.load(sys.stdin)["model_directory"])' \
  <<<"$MODEL_READY")

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container $CONTAINER already exists; refusing to replace it" >&2
  exit 1
fi

docker image inspect "$VERSE_VLLM_IMAGE" >/dev/null 2>&1 ||
  docker pull "$VERSE_VLLM_IMAGE"

IMAGE_COMMIT=$(docker image inspect \
  --format '{{index .Config.Labels "ai.vllm.build.commit"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_PROFILE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.runtime.profile"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_REVISION=$(docker image inspect \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_SOURCE_ARCHIVE=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.source.archive.sha256"}}' \
  "$VERSE_VLLM_IMAGE")
IMAGE_WHEEL_VERSION=$(docker image inspect \
  --format '{{index .Config.Labels "ai.verse.vllm.wheel.version"}}' \
  "$VERSE_VLLM_IMAGE")
EXPECTED_SOURCE_ARCHIVE=$(git -C "$ROOT" -c tar.umask=0000 archive --format=tar \
  "$VERSE_VLLM_EXPECTED_COMMIT" | verse_sha256 | awk '{print $1}')
EXPECTED_WHEEL_VERSION="0.28.0+verse.${VERSE_VLLM_EXPECTED_COMMIT:0:12}"
RECEIPT_ARGS=(
  --image "$VERSE_VLLM_IMAGE"
  --fork-commit "$VERSE_VLLM_EXPECTED_COMMIT"
  --runtime-profile "$VERSE_RUNTIME_PROFILE"
  --source-archive-sha256 "$EXPECTED_SOURCE_ARCHIVE"
  --vllm-wheel-version "$EXPECTED_WHEEL_VERSION"
)
uv run --script "$ROOT/tools/verse/sm120_image_receipt.py" verify \
  "${RECEIPT_ARGS[@]}" --receipt "$VERSE_VLLM_IMAGE_RECEIPT" >/dev/null
[[ $IMAGE_COMMIT == "$VERSE_VLLM_EXPECTED_COMMIT" ]] || {
  echo "image commit $IMAGE_COMMIT does not match expected commit" >&2
  exit 1
}
[[ $IMAGE_PROFILE == "$VERSE_RUNTIME_PROFILE" ]] || {
  echo "image is missing the Verse SM120 profile label" >&2
  exit 1
}
[[ $IMAGE_REVISION == "$VERSE_VLLM_EXPECTED_COMMIT" ]] || {
  echo "image OCI revision does not match expected commit" >&2
  exit 1
}
[[ $IMAGE_WHEEL_VERSION == "$EXPECTED_WHEEL_VERSION" ]] || {
  echo "image vLLM wheel version does not match expected commit" >&2
  exit 1
}
[[ $EXPECTED_SOURCE_ARCHIVE =~ ^[0-9a-f]{64}$ && \
  $IMAGE_SOURCE_ARCHIVE == "$EXPECTED_SOURCE_ARCHIVE" ]] || {
  echo "image source archive provenance does not match the expected commit" >&2
  exit 1
}

IMAGE_VERIFICATION=$(mktemp)
trap 'rm -f "$IMAGE_VERIFICATION"' EXIT
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges \
  --entrypoint /usr/local/bin/verify-verse-sm120-image \
  "$VERSE_VLLM_IMAGE" >"$IMAGE_VERIFICATION"
uv run --script "$ROOT/tools/verse/sm120_image_receipt.py" verify \
  "${RECEIPT_ARGS[@]}" --receipt "$VERSE_VLLM_IMAGE_RECEIPT" \
  --verification "$IMAGE_VERIFICATION" >/dev/null
IMAGE_RECEIPT_SHA256=$(verse_sha256 "$VERSE_VLLM_IMAGE_RECEIPT" | awk '{print $1}')
rm -f "$IMAGE_VERIFICATION"
trap - EXIT

verify_gpu_identity

install -d -o 0 -g 0 -m 0750 "$VERSE_VLLM_CACHE_DIR"
install -d -o 2000 -g 0 -m 0750 \
  "$VERSE_VLLM_CACHE_DIR/huggingface" \
  "$VERSE_VLLM_CACHE_DIR/cuda" \
  "$VERSE_VLLM_CACHE_DIR/flashinfer" \
  "$VERSE_VLLM_CACHE_DIR/flashinfer-autotune" \
  "$VERSE_VLLM_CACHE_DIR/vllm" \
  "$VERSE_VLLM_CACHE_DIR/xdg" \
  "$VERSE_VLLM_CACHE_DIR/triton"
install -d -o 2000 -g 0 -m 0750 \
  "$VERSE_VLLM_CACHE_DIR/torch" \
  "$VERSE_VLLM_CACHE_DIR/torch-extensions" \
  "$VERSE_VLLM_CACHE_DIR/torchinductor"

uv run --no-project python - \
  "$RUNTIME_CACHE" "$VERSE_VLLM_IMAGE" "$VERSE_VLLM_EXPECTED_COMMIT" \
  "$GPU_DEVICE" "$GPU_UUID" "$VERSE_RUNTIME_PROFILE" \
  "$IMAGE_RECEIPT_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

cache = Path(sys.argv[1])
marker = cache / ".verse-sm120-runtime.json"
temporary = cache / ".verse-sm120-runtime.json.tmp"
payload = {
    "schema_version": 1,
    "image": sys.argv[2],
    "fork_commit": sys.argv[3],
    "profile": sys.argv[6],
    "gpu_device": sys.argv[4],
    "gpu_uuid": sys.argv[5],
    "image_receipt_sha256": sys.argv[7],
}
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.chmod(0o640)
os.replace(temporary, marker)
PY

ENV_ARGS=(
  --env VLLM_API_KEY_FILE=/run/secrets/vllm_api_key
  --env VLLM_VERSE_RUNTIME_STRICT=1
  --env VLLM_NVFP4_KV_VOSPLIT=1
  --env VLLM_VERSE_NVFP4_XQA_DECODE="$VERSE_NVFP4_XQA_DECODE"
  --env VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="$VERSE_FLASHINFER_WORKSPACE_BUFFER_SIZE"
  --env VLLM_KV_CACHE_LAYOUT="$VERSE_KV_CACHE_LAYOUT"
  --env VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
  --env VLLM_USE_FLASHINFER_SAMPLER=0
  --env "VLLM_MAX_N_SEQUENCES=$VERSE_MAX_N_SEQUENCES"
  --env "VLLM_MAX_COMPLETION_PROMPTS=$VERSE_MAX_COMPLETION_PROMPTS"
  --env "VLLM_MAX_STOP_STRINGS=$VERSE_MAX_STOP_STRINGS"
  --env FLASHINFER_WORKSPACE_BASE=/cache/flashinfer
  --env CUDA_CACHE_PATH=/cache/cuda
  --env TORCH_HOME=/cache/torch
  --env TORCH_EXTENSIONS_DIR=/cache/torch-extensions
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
  --env HF_HOME=/cache/huggingface
  --env VLLM_CACHE_ROOT=/cache/vllm
  --env TRITON_CACHE_DIR=/cache/triton
  --env XDG_CACHE_HOME=/cache/xdg
  --env VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer-autotune
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
MOUNT_ARGS=(--mount "type=bind,src=$VERSE_VLLM_CACHE_DIR,dst=/cache")
MOUNT_ARGS+=(--mount "type=bind,src=$MODEL_DIRECTORY,dst=/models/model,readonly")
MOUNT_ARGS+=(--mount \
  "type=bind,src=$VERSE_VLLM_API_KEY_FILE,dst=/run/secrets/vllm_api_key,readonly")

CONTAINER_ID=
CREATED_CONTAINER=0
container_name_is_owned_by_id() {
  local identity
  identity=$(docker inspect --format '{{.Id}} {{.Name}}' \
    "$CONTAINER_ID" 2>/dev/null) || return 1
  [[ $identity == "$CONTAINER_ID /$CONTAINER" ]]
}

cleanup_failed_candidate() {
  local status=$?
  if ((status != 0 && CREATED_CONTAINER == 1)); then
    echo "candidate startup failed; inspecting captured container $CONTAINER_ID" >&2
    docker inspect --format \
      'state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}' \
      "$CONTAINER_ID" >&2 2>/dev/null || true
    docker logs --tail 200 "$CONTAINER_ID" 2>&1 | tail -200 >&2 || true
    if container_name_is_owned_by_id; then
      echo "removing newly-created container $CONTAINER_ID" >&2
      docker rm --force "$CONTAINER_ID" >/dev/null 2>&1 || true
    else
      echo "refusing cleanup because $CONTAINER_ID no longer owns /$CONTAINER" >&2
    fi
  fi
  exit "$status"
}
trap cleanup_failed_candidate EXIT

verify_gpu_identity
require_idle_gpu

CONTAINER_ID=$(docker create \
  --name "$CONTAINER" \
  --label ai.verse.gpu.ordinal="$GPU_DEVICE" \
  --label ai.verse.gpu.uuid="$GPU_UUID" \
  --restart no \
  --user 2000:0 \
  --gpus "device=$GPU_UUID" \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,nosuid,size=4g \
  --publish "127.0.0.1:${HOST_PORT}:8000" \
  --publish "127.0.0.1:8080:8080" \
  "${MOUNT_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "$VERSE_VLLM_IMAGE" \
  /models/model \
  --served-model-name "$SERVED_MODEL" \
  --quantization "$VERSE_QUANTIZATION" \
  --dtype bfloat16 \
  --linear-backend "$VERSE_LINEAR_BACKEND" \
  --max-model-len "$VERSE_MAX_MODEL_LEN" \
  --block-size "$VERSE_BLOCK_SIZE" \
  --max-num-seqs "$VERSE_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VERSE_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$VERSE_GPU_MEMORY_UTILIZATION" \
  --kv-cache-memory-bytes "$VERSE_KV_CACHE_MEMORY_BYTES" \
  --kv-cache-dtype "$VERSE_KV_CACHE_DTYPE" \
  --attention-backend "$VERSE_ATTENTION_BACKEND" \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --language-model-only \
  --no-enable-log-requests \
  --disable-uvicorn-access-log \
  --generation-config vllm \
  --compilation-config "$VERSE_COMPILATION_CONFIG" \
  --host 0.0.0.0 \
  --port 8000)

[[ $CONTAINER_ID =~ ^[0-9a-f]{64}$ ]] || {
  echo "docker create did not return a full immutable container ID" >&2
  exit 1
}
CREATED_CONTAINER=1
container_name_is_owned_by_id || {
  echo "captured container ID does not own the requested container name" >&2
  exit 1
}
docker start "$CONTAINER_ID" >/dev/null

VERSE_VLLM_CONTAINER_ID="$CONTAINER_ID" \
  VERSE_VLLM_EXPECTED_RESTART_POLICY=no \
  "$ROOT/tools/verse/check_sm120_server.sh"
uv run --script "$ROOT/tools/verse/check_sm120_chat_contract.py" \
  --endpoint "http://127.0.0.1:${HOST_PORT}" \
  --model "$SERVED_MODEL" \
  --api-key-file "$VERSE_VLLM_API_KEY_FILE" \
  --startup-only

VERSE_VLLM_CONTAINER_ID="$CONTAINER_ID" \
  VERSE_VLLM_EXPECTED_RESTART_POLICY=no \
  "$ROOT/tools/verse/check_sm120_server.sh"

trap - EXIT

printf 'status=ready\ncontainer=%s\ncontainer_id=%s\nendpoint=http://127.0.0.1:%s\n' \
  "$CONTAINER" "$CONTAINER_ID" "$HOST_PORT"
