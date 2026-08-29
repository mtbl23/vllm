#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CADDY_IMAGE='caddy:2.10.2-alpine@sha256:4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d'
CADDY_CONFIG="$ROOT/tools/verse/verse-sm120-gateway.Caddyfile"
GATEWAY_NAME=${VERSE_SM120_GATEWAY_CONTAINER:-verse-sm120-gateway}
VLLM_CONTAINER_ID=${VERSE_VLLM_CONTAINER_ID:-}
RELEASE_MANIFEST=${VERSE_VLLM_RELEASE_MANIFEST:-}
API_KEY_FILE=${VERSE_VLLM_API_KEY_FILE:-}

[[ $VLLM_CONTAINER_ID =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_CONTAINER_ID must be the exact 64-character candidate container ID" >&2
  exit 1
}
[[ -f $CADDY_CONFIG && ! -L $CADDY_CONFIG ]] || {
  echo "the tracked gateway Caddyfile is missing or is a symlink" >&2
  exit 1
}
[[ $RELEASE_MANIFEST == /* ]] || {
  echo "VERSE_VLLM_RELEASE_MANIFEST must be absolute" >&2
  exit 1
}
[[ $API_KEY_FILE == /* && -f $API_KEY_FILE && ! -L $API_KEY_FILE ]] || {
  echo "VERSE_VLLM_API_KEY_FILE must be an absolute regular non-symlink file" >&2
  exit 1
}
uv run --no-project python -c '
import os, pathlib, stat, sys
path = pathlib.Path(sys.argv[1])
metadata = path.stat()
if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit(1)
' "$API_KEY_FILE" || {
  echo "vLLM API key file must be caller-owned with exact mode 0600" >&2
  exit 1
}

ACTUAL_VLLM_ID=$(docker inspect --format '{{.Id}}' "$VLLM_CONTAINER_ID")
[[ $ACTUAL_VLLM_ID == "$VLLM_CONTAINER_ID" ]] || {
  echo "candidate container identity changed" >&2
  exit 1
}
[[ $(docker inspect --format '{{.State.Running}}' "$VLLM_CONTAINER_ID") == true ]] || {
  echo "candidate vLLM container is not running" >&2
  exit 1
}
CANDIDATE_IMAGE=$(docker inspect --format '{{.Config.Image}}' "$VLLM_CONTAINER_ID")
CANDIDATE_COMMIT=$(docker inspect \
  --format '{{index .Config.Labels "ai.vllm.build.commit"}}' \
  "$VLLM_CONTAINER_ID")
PORT_BINDINGS=$(docker inspect --format '{{json .NetworkSettings.Ports}}' \
  "$VLLM_CONTAINER_ID")
PORT_BINDING=$(uv run --no-project python -c '
import json, sys
ports = json.loads(sys.stdin.read())
if set(ports) != {"8000/tcp", "8080/tcp"}:
    raise SystemExit("candidate has unexpected published ports")
binding = ports["8000/tcp"]
if binding != [{"HostIp": "127.0.0.1", "HostPort": "8000"}]:
    raise SystemExit("candidate must publish only 127.0.0.1:8000")
gateway = ports["8080/tcp"]
if gateway != [{"HostIp": "127.0.0.1", "HostPort": "8080"}]:
    raise SystemExit("candidate must publish only 127.0.0.1:8080 for its gateway")
print("127.0.0.1:8000+127.0.0.1:8080")
' <<<"$PORT_BINDINGS")
[[ $PORT_BINDING == 127.0.0.1:8000+127.0.0.1:8080 ]] || {
  echo "candidate port identity is invalid" >&2
  exit 1
}
RELEASE_IDENTITY=$(uv run --script \
  "$ROOT/tools/verse/validate_sm120_gateway_release.py" \
  --manifest "$RELEASE_MANIFEST" \
  --container-id "$VLLM_CONTAINER_ID" \
  --image-digest "$CANDIDATE_IMAGE" \
  --fork-commit "$CANDIDATE_COMMIT")
mapfile -t RELEASE_FIELDS < <(uv run --no-project python -c '
import json, sys
payload = json.loads(sys.stdin.read())
for name in (
    "release_manifest_sha256",
    "release_nonce",
    "model_revision",
    "attestation_verification_sha256",
):
    print(payload[name])
' <<<"$RELEASE_IDENTITY")
[[ ${#RELEASE_FIELDS[@]} -eq 4 ]] || {
  echo "release identity is incomplete" >&2
  exit 1
}
RELEASE_MANIFEST_SHA256=${RELEASE_FIELDS[0]}
RELEASE_NONCE=${RELEASE_FIELDS[1]}
MODEL_REVISION=${RELEASE_FIELDS[2]}
ATTESTATION_VERIFICATION_SHA256=${RELEASE_FIELDS[3]}
API_KEY=$(cat "$API_KEY_FILE")
[[ -n $API_KEY && $API_KEY != *$'\n'* && ${#API_KEY} -le 4096 ]] || {
  echo "vLLM API key must contain one non-empty line" >&2
  exit 1
}
if docker container inspect "$GATEWAY_NAME" >/dev/null 2>&1; then
  echo "gateway container $GATEWAY_NAME already exists; refusing replacement" >&2
  exit 1
fi

docker image inspect "$CADDY_IMAGE" >/dev/null 2>&1 || docker pull "$CADDY_IMAGE"
docker run --rm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --mount "type=bind,src=$CADDY_CONFIG,dst=/etc/caddy/Caddyfile,readonly" \
  "$CADDY_IMAGE" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

GATEWAY_ID=""
MODEL_RESPONSE=""
AUTH_HEADER=""
cleanup_failed_gateway() {
  local status=$?
  [[ -z $MODEL_RESPONSE ]] || rm -f "$MODEL_RESPONSE"
  [[ -z $AUTH_HEADER ]] || rm -f "$AUTH_HEADER"
  if ((status != 0)) && [[ -n $GATEWAY_ID ]]; then
    local owned_id
    owned_id=$(docker inspect --format '{{.Id}}' "$GATEWAY_ID" 2>/dev/null || true)
    if [[ $owned_id == "$GATEWAY_ID" ]]; then
      docker logs --tail 100 "$GATEWAY_ID" >&2 2>/dev/null || true
      docker rm --force "$GATEWAY_ID" >/dev/null 2>&1 || true
    fi
  fi
  exit "$status"
}
trap cleanup_failed_gateway EXIT

GATEWAY_ID=$(docker create \
  --name "$GATEWAY_NAME" \
  --restart no \
  --network "container:$VLLM_CONTAINER_ID" \
  --user 65534:65534 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /data:rw,nosuid,nodev,noexec,size=16m \
  --tmpfs /config:rw,nosuid,nodev,noexec,size=4m \
  --label ai.verse.runtime.profile=sm120-gateway-v1 \
  --label "ai.verse.upstream.container=$VLLM_CONTAINER_ID" \
  --label "ai.verse.release.manifest.sha256=$RELEASE_MANIFEST_SHA256" \
  --env "VERSE_CANDIDATE_CONTAINER_ID=$VLLM_CONTAINER_ID" \
  --env "VERSE_IMAGE_DIGEST=$CANDIDATE_IMAGE" \
  --env "VERSE_FORK_COMMIT=$CANDIDATE_COMMIT" \
  --env "VERSE_MODEL_REVISION=$MODEL_REVISION" \
  --env "VERSE_RELEASE_NONCE=$RELEASE_NONCE" \
  --env "VERSE_RELEASE_MANIFEST_SHA256=$RELEASE_MANIFEST_SHA256" \
  --env "VERSE_ATTESTATION_VERIFICATION_SHA256=$ATTESTATION_VERIFICATION_SHA256" \
  --mount "type=bind,src=$CADDY_CONFIG,dst=/etc/caddy/Caddyfile,readonly" \
  "$CADDY_IMAGE" caddy run --config /etc/caddy/Caddyfile --adapter caddyfile)

[[ $(docker inspect --format '{{.Id}}' "$GATEWAY_ID") == "$GATEWAY_ID" ]] || {
  echo "gateway container identity changed after create" >&2
  exit 1
}
docker start "$GATEWAY_ID" >/dev/null

deadline=$((SECONDS + 30))
until curl --noproxy '*' --fail --silent --show-error --max-time 2 \
  http://127.0.0.1:8080/health >/dev/null 2>&1; do
  ((SECONDS < deadline)) || {
    echo "gateway did not become healthy" >&2
    exit 1
  }
  sleep 1
done

MODEL_RESPONSE=$(mktemp)
AUTH_HEADER=$(mktemp)
chmod 0600 "$AUTH_HEADER"
printf 'Authorization: Bearer %s\n' "$API_KEY" >"$AUTH_HEADER"
unset API_KEY
curl --noproxy '*' --fail --silent --show-error --max-time 5 \
  --header "@$AUTH_HEADER" \
  http://127.0.0.1:8080/v1/models >"$MODEL_RESPONSE"
uv run --no-project python -c '
import json, sys
payload = json.load(open(sys.argv[1]))
models = {item.get("id") for item in payload.get("data", [])}
if models != {"verse-free"}:
    raise SystemExit("gateway exposed the wrong model identity")
' "$MODEL_RESPONSE"

for guarded_path in /invocations /tokenize /docs /openapi.json; do
  status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 3 "http://127.0.0.1:8000${guarded_path}")
  [[ $status == 401 ]] || {
    echo "raw vLLM path $guarded_path bypassed authentication with HTTP $status" >&2
    exit 1
  }
done

for forbidden_path in /pause /abort_requests /invocations /metrics /docs /openapi.json; do
  status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 3 "http://127.0.0.1:8080${forbidden_path}")
  [[ $status == 404 ]] || {
    echo "gateway exposed forbidden path $forbidden_path with HTTP $status" >&2
    exit 1
  }
done

trap - EXIT
rm -f "$MODEL_RESPONSE" "$AUTH_HEADER"
printf 'status=ready\ngateway_container_id=%s\nupstream_container_id=%s\nrelease_manifest_sha256=%s\nendpoint=http://127.0.0.1:8080\n' \
  "$GATEWAY_ID" "$VLLM_CONTAINER_ID" "$RELEASE_MANIFEST_SHA256"
