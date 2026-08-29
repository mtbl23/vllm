#!/usr/bin/env bash
set -euo pipefail

IMAGE=${VERSE_VLLM_IMAGE:-}
COMMIT=${VERSE_VLLM_EXPECTED_COMMIT:-}
TOKEN_FILE=${VERSE_VLLM_GITHUB_TOKEN_FILE:-}
OUTPUT=${VERSE_VLLM_ATTESTATION_VERIFICATION_OUTPUT:-}

[[ $IMAGE =~ ^ghcr\.io/mtbl23/verse-vllm@sha256:[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_IMAGE must be the exact immutable Verse image" >&2
  exit 1
}
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo "VERSE_VLLM_EXPECTED_COMMIT must be an exact commit" >&2
  exit 1
}
[[ $TOKEN_FILE == /* && -f $TOKEN_FILE && ! -L $TOKEN_FILE ]] || {
  echo "VERSE_VLLM_GITHUB_TOKEN_FILE must be an absolute regular file" >&2
  exit 1
}
uv run --no-project python -c '
import os, pathlib, stat, sys
path = pathlib.Path(sys.argv[1])
metadata = path.stat()
if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit(1)
' "$TOKEN_FILE" || {
  echo "GitHub token file must be caller-owned with exact mode 0600" >&2
  exit 1
}
[[ $OUTPUT == /* && ! -e $OUTPUT ]] || {
  echo "VERSE_VLLM_ATTESTATION_VERIFICATION_OUTPUT must be a new absolute path" >&2
  exit 1
}
for name in ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy; do
  [[ -z ${!name:-} ]] || {
    echo "proxy environment overrides are forbidden" >&2
    exit 1
  }
done
TOKEN=$(uv run --no-project python -c '
import pathlib, sys
raw = pathlib.Path(sys.argv[1]).read_bytes()
lines = raw.splitlines()
if len(lines) != 1 or not lines[0] or b"\r" in raw or b"\0" in raw:
    raise SystemExit("GitHub token must contain one clean line")
print(lines[0].decode())
' "$TOKEN_FILE")
TMP=$(mktemp "${OUTPUT}.tmp.XXXXXXXX")
cleanup() {
  local status=$?
  rm -f "$TMP"
  unset TOKEN
  exit "$status"
}
trap cleanup EXIT
GH_TOKEN=$TOKEN gh attestation verify "oci://$IMAGE" \
  --repo mtbl23/vllm \
  --signer-workflow \
    https://github.com/mtbl23/vllm/.github/workflows/verse-sm120-image.yml \
  --source-ref refs/heads/verse/v0.28-sm120-nvfp4-fa2 \
  --source-digest "$COMMIT" \
  --deny-self-hosted-runners \
  --format json >"$TMP"
unset TOKEN
uv run --no-project python -c '
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
if not isinstance(payload, list) or not payload:
    raise SystemExit("attestation verification output is absent")
' "$TMP"
chmod 0600 "$TMP"
mv "$TMP" "$OUTPUT"
trap - EXIT
echo "status=verified"
