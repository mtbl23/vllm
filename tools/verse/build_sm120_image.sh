#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
source "$ROOT/tools/verse/sm120_sha256.bash"

if [[ -n $(git status --short) ]]; then
  echo "refusing to build an uncommitted runtime" >&2
  exit 1
fi

COMMIT=$(git rev-parse --verify 'HEAD^{commit}')
VLLM_WHEEL_VERSION="0.28.0+verse.${COMMIT:0:12}"
BUILD_BASE_IMAGE='pytorch/manylinux2_28-builder:cuda13.0@sha256:7710cbc19d7ee951134e2e827f8ec89237c993095eb2581dd5e74f58e4e278c7'
FINAL_BASE_IMAGE='nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4'
IMAGE_REPOSITORY=${VERSE_VLLM_IMAGE_REPOSITORY:-verse-vllm}
IMAGE_TAG=${VERSE_VLLM_IMAGE_TAG:-sm120-${COMMIT:0:12}}
IMAGE=${IMAGE_REPOSITORY}:${IMAGE_TAG}
OUTPUT_MODE=${VERSE_VLLM_BUILD_OUTPUT:-load}
SOURCE_ARCHIVE=$(mktemp)
BUILD_METADATA=$(mktemp)
trap 'rm -f "$SOURCE_ARCHIVE" "$BUILD_METADATA"' EXIT

git -c tar.umask=0000 archive --format=tar --output="$SOURCE_ARCHIVE" "$COMMIT"
SOURCE_ARCHIVE_SHA256=$(verse_sha256 "$SOURCE_ARCHIVE" | awk '{print $1}')
FLASHINFER_MANIFEST_SHA256=$(git show \
  "$COMMIT:requirements/verse-sm120-flashinfer.lock" | \
  verse_sha256 | awk '{print $1}')
[[ $SOURCE_ARCHIVE_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to content-address the committed source archive" >&2
  exit 1
}
[[ $FLASHINFER_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to hash the committed FlashInfer manifest" >&2
  exit 1
}

case "$OUTPUT_MODE" in
  load) OUTPUT_ARGS=(--load) ;;
  push) OUTPUT_ARGS=(--push) ;;
  *)
    echo "VERSE_VLLM_BUILD_OUTPUT must be load or push" >&2
    exit 1
    ;;
esac

docker buildx build \
  --file docker/Dockerfile \
  --target vllm-openai-verse-sm120 \
  --platform linux/amd64 \
  --build-arg BUILD_BASE_IMAGE="$BUILD_BASE_IMAGE" \
  --build-arg FINAL_BASE_IMAGE="$FINAL_BASE_IMAGE" \
  --build-arg torch_cuda_arch_list=12.0 \
  --build-arg GIT_REPO_CHECK=0 \
  --build-arg GIT_REPO_MOUNT_SOURCE=tools/verse/archive-git-context \
  --build-arg VLLM_VERSION_OVERRIDE="$VLLM_WHEEL_VERSION" \
  --build-arg VLLM_VERSE_SM120_WHEEL=1 \
  --build-arg VLLM_BUILD_COMMIT="$COMMIT" \
  --build-arg VLLM_BUILD_PIPELINE=verse-sm120 \
  --build-arg VLLM_IMAGE_TAG="$IMAGE" \
  --label org.opencontainers.image.revision="$COMMIT" \
  --label ai.verse.source.archive.sha256="$SOURCE_ARCHIVE_SHA256" \
  --label ai.verse.vllm.wheel.version="$VLLM_WHEEL_VERSION" \
  --label ai.verse.runtime.profile=sm120-gemma4-nvfp4-v1 \
  --label ai.verse.flashinfer.release=0.6.18.dev20260819 \
  --label ai.verse.flashinfer.manifest.sha256="$FLASHINFER_MANIFEST_SHA256" \
  --label ai.verse.base.build="$BUILD_BASE_IMAGE" \
  --label ai.verse.base.runtime="$FINAL_BASE_IMAGE" \
  --metadata-file "$BUILD_METADATA" \
  --tag "$IMAGE" \
  "${OUTPUT_ARGS[@]}" \
  - <"$SOURCE_ARCHIVE"

if [[ "$OUTPUT_MODE" == load ]]; then
  IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  LABEL_COMMIT=$(docker image inspect \
    --format '{{index .Config.Labels "ai.vllm.build.commit"}}' "$IMAGE")
  LABEL_REVISION=$(docker image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$IMAGE")
  LABEL_SOURCE_ARCHIVE=$(docker image inspect \
    --format '{{index .Config.Labels "ai.verse.source.archive.sha256"}}' \
    "$IMAGE")
  LABEL_WHEEL_VERSION=$(docker image inspect \
    --format '{{index .Config.Labels "ai.verse.vllm.wheel.version"}}' \
    "$IMAGE")
  [[ "$LABEL_COMMIT" == "$COMMIT" ]] || {
    echo "built image commit label does not match $COMMIT" >&2
    exit 1
  }
  [[ "$LABEL_REVISION" == "$COMMIT" ]] || {
    echo "built image OCI revision does not match $COMMIT" >&2
    exit 1
  }
  [[ "$LABEL_SOURCE_ARCHIVE" == "$SOURCE_ARCHIVE_SHA256" ]] || {
    echo "built image source archive label does not match the build context" >&2
    exit 1
  }
  [[ "$LABEL_WHEEL_VERSION" == "$VLLM_WHEEL_VERSION" ]] || {
    echo "built image vLLM wheel label does not match $VLLM_WHEEL_VERSION" >&2
    exit 1
  }
  docker run --rm --entrypoint /usr/local/bin/verify-verse-sm120-image \
    "$IMAGE" >/dev/null
  printf 'image=%s\nimage_id=%s\ncommit=%s\nvllm_wheel_version=%s\nsource_archive_sha256=%s\nflashinfer_manifest_sha256=%s\n' \
    "$IMAGE" "$IMAGE_ID" "$COMMIT" "$VLLM_WHEEL_VERSION" "$SOURCE_ARCHIVE_SHA256" \
    "$FLASHINFER_MANIFEST_SHA256"
else
  DIGEST=$(uv run --no-project python -c '
import json, sys
digest = json.load(open(sys.argv[1]))["containerimage.digest"]
if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
    raise SystemExit("buildx did not return an immutable image digest")
print(digest)
' "$BUILD_METADATA")
  printf 'image=%s\nimage_digest=%s@%s\ncommit=%s\nvllm_wheel_version=%s\nsource_archive_sha256=%s\nflashinfer_manifest_sha256=%s\n' \
    "$IMAGE" "$IMAGE_REPOSITORY" "$DIGEST" "$COMMIT" \
    "$VLLM_WHEEL_VERSION" "$SOURCE_ARCHIVE_SHA256" \
    "$FLASHINFER_MANIFEST_SHA256"
fi
