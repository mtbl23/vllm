#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CADDY_IMAGE='caddy:2.10.2-alpine@sha256:4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d'
CADDY_CONFIG="$ROOT/tools/verse/verse-sm120-gateway.Caddyfile"
GATEWAY_NAME=${VERSE_SM120_GATEWAY_CONTAINER:-verse-sm120-gateway}
VLLM_CONTAINER_ID=${VERSE_VLLM_CONTAINER_ID:-}

[[ $VLLM_CONTAINER_ID =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_VLLM_CONTAINER_ID must be the exact 64-character candidate container ID" >&2
  exit 1
}
[[ -f $CADDY_CONFIG && ! -L $CADDY_CONFIG ]] || {
  echo "the tracked gateway Caddyfile is missing or is a symlink" >&2
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
cleanup_failed_gateway() {
  local status=$?
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
  --network host \
  --user 65534:65534 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /data:rw,nosuid,nodev,noexec,size=16m \
  --tmpfs /config:rw,nosuid,nodev,noexec,size=4m \
  --label ai.verse.runtime.profile=sm120-gateway-v1 \
  --label "ai.verse.upstream.container=$VLLM_CONTAINER_ID" \
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

for forbidden_path in /pause /abort_requests /invocations /metrics /docs /openapi.json; do
  status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 3 "http://127.0.0.1:8080${forbidden_path}")
  [[ $status == 404 ]] || {
    echo "gateway exposed forbidden path $forbidden_path with HTTP $status" >&2
    exit 1
  }
done

docker update --restart unless-stopped "$GATEWAY_ID" >/dev/null
trap - EXIT
printf 'status=ready\ngateway_container_id=%s\nupstream_container_id=%s\nendpoint=http://127.0.0.1:8080\n' \
  "$GATEWAY_ID" "$VLLM_CONTAINER_ID"
