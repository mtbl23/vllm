#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

if [[ -n $(git status --short) ]]; then
  echo "refusing to build an uncommitted runtime" >&2
  exit 1
fi

OUTPUT=${1:?usage: build_sm120_wheel_artifact.sh OUTPUT_DIRECTORY}
[[ $OUTPUT = /* ]] || {
  echo "output directory must be absolute" >&2
  exit 1
}
[[ ! -e $OUTPUT ]] || {
  echo "output directory already exists: $OUTPUT" >&2
  exit 1
}

COMMIT=$(git rev-parse --verify 'HEAD^{commit}')
VERSION="0.28.0+verse.${COMMIT:0:12}"
SOURCE_ARCHIVE=$(mktemp)
trap 'rm -f "$SOURCE_ARCHIVE"' EXIT
git -c tar.umask=0000 archive --format=tar --output="$SOURCE_ARCHIVE" "$COMMIT"

docker buildx build \
  --file docker/Dockerfile \
  --target vllm-verse-sm120-wheel-export \
  --platform linux/amd64 \
  --build-arg torch_cuda_arch_list=12.0 \
  --build-arg GIT_REPO_CHECK=0 \
  --build-arg GIT_REPO_MOUNT_SOURCE=tools/verse/archive-git-context \
  --build-arg VLLM_VERSION_OVERRIDE="$VERSION" \
  --build-arg VLLM_VERSE_SM120_WHEEL=1 \
  --output "type=local,dest=$OUTPUT" \
  - <"$SOURCE_ARCHIVE"

sha256sum "$OUTPUT"/dist/*.whl | tee "$OUTPUT/dist/wheel.sha256"
printf 'commit=%s\nversion=%s\n' "$COMMIT" "$VERSION"
