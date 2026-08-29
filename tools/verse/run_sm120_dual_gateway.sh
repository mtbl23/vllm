#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$ROOT/tools/verse/verse-sm120-dual-gateway.Caddyfile"

require_env() {
  local name=$1
  [[ -n ${!name:-} ]] || {
    echo "$name is required" >&2
    exit 1
  }
}

require_env VERSE_CADDY_BINARY
require_env VERSE_CADDY_SHA256
require_env VERSE_GPU0_API_KEY_FILE
require_env VERSE_GPU1_API_KEY_FILE
require_env VERSE_GATEWAY_TOKEN_FILE
require_env VERSE_GPU0_KEY_UID
require_env VERSE_GPU1_KEY_UID

if ((EUID != 0)); then
  echo "the dual gateway launcher must run as root" >&2
  exit 1
fi
[[ -f $CONFIG && ! -L $CONFIG ]] || {
  echo "the tracked dual gateway Caddyfile is missing or is a symlink" >&2
  exit 1
}
[[ $VERSE_CADDY_BINARY == /* && -f $VERSE_CADDY_BINARY && \
  -x $VERSE_CADDY_BINARY && ! -L $VERSE_CADDY_BINARY ]] || {
  echo "VERSE_CADDY_BINARY must be an absolute executable regular file" >&2
  exit 1
}
[[ $VERSE_CADDY_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "VERSE_CADDY_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 1
}
ACTUAL_CADDY_SHA256=$(sha256sum "$VERSE_CADDY_BINARY" | awk '{print $1}')
[[ $ACTUAL_CADDY_SHA256 == "$VERSE_CADDY_SHA256" ]] || {
  echo "Caddy binary digest does not match VERSE_CADDY_SHA256" >&2
  exit 1
}

VERSE_GATEWAY_BIND=${VERSE_GATEWAY_BIND:-127.0.0.1:8080}
VERSE_GPU0_ORIGIN=${VERSE_GPU0_ORIGIN:-127.0.0.1:18001}
VERSE_GPU1_ORIGIN=${VERSE_GPU1_ORIGIN:-127.0.0.1:18002}
[[ $VERSE_GATEWAY_BIND =~ ^127\.0\.0\.1:[0-9]+$ ]] || {
  echo "VERSE_GATEWAY_BIND must use an explicit loopback IPv4 port" >&2
  exit 1
}
[[ $VERSE_GPU0_ORIGIN =~ ^127\.0\.0\.1:[0-9]+$ ]] || {
  echo "VERSE_GPU0_ORIGIN must use an explicit loopback IPv4 port" >&2
  exit 1
}
[[ $VERSE_GPU1_ORIGIN =~ ^127\.0\.0\.1:[0-9]+$ ]] || {
  echo "VERSE_GPU1_ORIGIN must use an explicit loopback IPv4 port" >&2
  exit 1
}
[[ $VERSE_GPU0_ORIGIN != "$VERSE_GPU1_ORIGIN" ]] || {
  echo "dual gateway origins must be distinct" >&2
  exit 1
}
[[ $VERSE_GATEWAY_BIND != "$VERSE_GPU0_ORIGIN" && \
  $VERSE_GATEWAY_BIND != "$VERSE_GPU1_ORIGIN" ]] || {
  echo "gateway bind address cannot overlap an upstream" >&2
  exit 1
}

read_secret() {
  python3 - "$1" "$2" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_uid = int(sys.argv[2])
if not path.is_absolute() or path.is_symlink() or not path.is_file():
    raise SystemExit("gateway key path must be an absolute regular non-symlink file")
resolved = path.resolve(strict=True)
if resolved != path:
    raise SystemExit("gateway key path must be canonical")
metadata = resolved.stat()
if metadata.st_uid != expected_uid:
    raise SystemExit("gateway key file owner does not match its pinned UID")
if stat.S_IMODE(metadata.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
    raise SystemExit("gateway key file must be owner-only")
raw = resolved.read_bytes()
if len(raw) > 4096 or b"\x00" in raw or b"\r" in raw:
    raise SystemExit("gateway key file contains invalid bytes")
lines = raw.splitlines()
if len(lines) != 1 or not lines[0]:
    raise SystemExit("gateway key file must contain one non-empty line")
value = lines[0].decode("utf-8")
if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
    raise SystemExit("gateway key must contain printable ASCII without spaces")
sys.stdout.write(value)
PY
}

[[ $VERSE_GPU0_KEY_UID =~ ^[0-9]+$ && $VERSE_GPU1_KEY_UID =~ ^[0-9]+$ ]] || {
  echo "worker key UIDs must be numeric" >&2
  exit 1
}
((VERSE_GPU0_KEY_UID > 0 && VERSE_GPU1_KEY_UID > 0)) || {
  echo "worker key UIDs must be unprivileged" >&2
  exit 1
}
[[ $VERSE_GPU0_KEY_UID != "$VERSE_GPU1_KEY_UID" ]] || {
  echo "dual worker key UIDs must be distinct" >&2
  exit 1
}
VERSE_GPU0_API_KEY=$(read_secret "$VERSE_GPU0_API_KEY_FILE" "$VERSE_GPU0_KEY_UID")
VERSE_GPU1_API_KEY=$(read_secret "$VERSE_GPU1_API_KEY_FILE" "$VERSE_GPU1_KEY_UID")
VERSE_GATEWAY_TOKEN=$(read_secret "$VERSE_GATEWAY_TOKEN_FILE" 0)
[[ $VERSE_GPU0_API_KEY != "$VERSE_GPU1_API_KEY" ]] || {
  echo "dual workers must use distinct API keys" >&2
  exit 1
}
[[ $VERSE_GATEWAY_TOKEN != "$VERSE_GPU0_API_KEY" && \
  $VERSE_GATEWAY_TOKEN != "$VERSE_GPU1_API_KEY" ]] || {
  echo "gateway credential must be distinct from worker API keys" >&2
  exit 1
}

gateway_port=${VERSE_GATEWAY_BIND##*:}
if ss -H -lnt "sport = :$gateway_port" | grep -q .; then
  echo "gateway bind port is already listening" >&2
  exit 1
fi

probe_worker() {
  local origin=$1
  local key=$2
  local status
  status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 5 "http://$origin/health")
  [[ $status == 200 ]] || {
    echo "worker health probe returned HTTP $status" >&2
    return 1
  }
  status=$(curl --noproxy '*' --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 5 -H "Authorization: Bearer $key" "http://$origin/v1/models")
  [[ $status == 200 ]] || {
    echo "authenticated worker model probe returned HTTP $status" >&2
    return 1
  }
}
probe_worker "$VERSE_GPU0_ORIGIN" "$VERSE_GPU0_API_KEY"
probe_worker "$VERSE_GPU1_ORIGIN" "$VERSE_GPU1_API_KEY"

VERSE_GATEWAY_UID=${VERSE_GATEWAY_UID:-65534}
VERSE_GATEWAY_GID=${VERSE_GATEWAY_GID:-65534}
[[ $VERSE_GATEWAY_UID =~ ^[0-9]+$ && $VERSE_GATEWAY_GID =~ ^[0-9]+$ ]] || {
  echo "gateway UID and GID must be numeric" >&2
  exit 1
}
((VERSE_GATEWAY_UID > 0 && VERSE_GATEWAY_GID > 0)) || {
  echo "gateway UID and GID must be unprivileged" >&2
  exit 1
}
[[ $VERSE_GATEWAY_UID != "$VERSE_GPU0_KEY_UID" && \
  $VERSE_GATEWAY_UID != "$VERSE_GPU1_KEY_UID" ]] || {
  echo "gateway UID must be distinct from worker key owners" >&2
  exit 1
}
RUNTIME_DIRECTORY=${VERSE_GATEWAY_RUNTIME_DIR:-/run/verse-sm120-gateway}
[[ $RUNTIME_DIRECTORY == /* && $RUNTIME_DIRECTORY != / ]] || {
  echo "VERSE_GATEWAY_RUNTIME_DIR must be a dedicated absolute directory" >&2
  exit 1
}
install -d -o "$VERSE_GATEWAY_UID" -g "$VERSE_GATEWAY_GID" -m 0700 \
  "$RUNTIME_DIRECTORY" "$RUNTIME_DIRECTORY/data" "$RUNTIME_DIRECTORY/config"

export VERSE_GATEWAY_BIND VERSE_GPU0_ORIGIN VERSE_GPU1_ORIGIN
export VERSE_GPU0_API_KEY VERSE_GPU1_API_KEY VERSE_GATEWAY_TOKEN
export XDG_DATA_HOME="$RUNTIME_DIRECTORY/data"
export XDG_CONFIG_HOME="$RUNTIME_DIRECTORY/config"

VERSE_GPU0_API_KEY=validation-placeholder-gpu0 \
VERSE_GPU1_API_KEY=validation-placeholder-gpu1 \
VERSE_GATEWAY_TOKEN=validation-placeholder-gateway \
  "$VERSE_CADDY_BINARY" validate --config "$CONFIG" --adapter caddyfile >/dev/null

ulimit -c 0
exec /usr/bin/setpriv \
  --reuid="$VERSE_GATEWAY_UID" \
  --regid="$VERSE_GATEWAY_GID" \
  --clear-groups \
  --no-new-privs \
  "$VERSE_CADDY_BINARY" run --config "$CONFIG" --adapter caddyfile
