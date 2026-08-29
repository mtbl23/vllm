#!/usr/bin/env bash
set -euo pipefail

# Qualify one already-running disposable Vast RTX 5070 Ti allocation. This
# driver never creates, starts, stops, or destroys provider resources and never
# opens a public listener. All client traffic crosses one SSH loopback forward.

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

usage() {
  cat >&2 <<'EOF'
usage: run_sm120_vast_release_gates.sh \
  --instance-id ID --image IMAGE@sha256:... --expected-commit SHA \
  --vast-api-key-file PATH --ssh-key-file PATH --hf-token-file PATH \
  --api-key-file PATH --image-receipt PATH --github-token-file PATH \
  --release-dir PATH
EOF
  exit 2
}

INSTANCE_ID=
IMAGE=
EXPECTED_COMMIT=
VAST_API_KEY_FILE=
SSH_KEY_FILE=
HF_TOKEN_FILE=
API_KEY_FILE=
IMAGE_RECEIPT=
GITHUB_TOKEN_FILE=
RELEASE_DIR=
while (($#)); do
  case "$1" in
    --instance-id) INSTANCE_ID=${2:-}; shift 2 ;;
    --image) IMAGE=${2:-}; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT=${2:-}; shift 2 ;;
    --vast-api-key-file) VAST_API_KEY_FILE=${2:-}; shift 2 ;;
    --ssh-key-file) SSH_KEY_FILE=${2:-}; shift 2 ;;
    --hf-token-file) HF_TOKEN_FILE=${2:-}; shift 2 ;;
    --api-key-file) API_KEY_FILE=${2:-}; shift 2 ;;
    --image-receipt) IMAGE_RECEIPT=${2:-}; shift 2 ;;
    --github-token-file) GITHUB_TOKEN_FILE=${2:-}; shift 2 ;;
    --release-dir) RELEASE_DIR=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ $INSTANCE_ID =~ ^[1-9][0-9]*$ ]] || usage
[[ $IMAGE =~ ^ghcr\.io/mtbl23/verse-vllm@sha256:[0-9a-f]{64}$ ]] || usage
[[ $EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || usage
[[ $(git rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || {
  echo "qualification source is not the expected commit" >&2
  exit 1
}
[[ -z $(git status --short) ]] || {
  echo "qualification source tree must be clean" >&2
  exit 1
}

validate_owner_file() {
  local path=$1 label=$2
  [[ $path == /* && -f $path && ! -L $path ]] || {
    echo "$label must be an absolute regular non-symlink file" >&2
    exit 1
  }
  [[ $(stat -f '%Su %Lp' "$path") == "$(id -un) 600" ]] || {
    echo "$label must be caller-owned with exact mode 0600" >&2
    exit 1
  }
  [[ -s $path ]] || {
    echo "$label must not be empty" >&2
    exit 1
  }
}

validate_secret_file() {
  local path=$1 label=$2
  validate_owner_file "$path" "$label"
  uv run --no-project python -c '
import pathlib, sys
raw = pathlib.Path(sys.argv[1]).read_bytes()
lines = raw.splitlines()
if len(lines) != 1 or not lines[0] or b"\r" in raw or b"\0" in raw:
    raise SystemExit(1)
' "$path" || {
    echo "$label must contain exactly one non-empty line" >&2
    exit 1
  }
}

for item in \
  "$VAST_API_KEY_FILE:Vast API key" \
  "$HF_TOKEN_FILE:Hugging Face token" \
  "$API_KEY_FILE:vLLM API key" \
  "$GITHUB_TOKEN_FILE:GitHub token"; do
  validate_secret_file "${item%%:*}" "${item#*:}"
done
validate_owner_file "$SSH_KEY_FILE" "SSH private key"
[[ $IMAGE_RECEIPT == /* && -f $IMAGE_RECEIPT && ! -L $IMAGE_RECEIPT ]] || {
  echo "image receipt must be an absolute regular non-symlink file" >&2
  exit 1
}
[[ $RELEASE_DIR == /* && ! -e $RELEASE_DIR ]] || {
  echo "release directory must be a new absolute path" >&2
  exit 1
}
install -d -m 0700 "$RELEASE_DIR"
install -m 0600 "$IMAGE_RECEIPT" "$RELEASE_DIR/image-receipt.json"
printf '%s\n' "$IMAGE" >"$RELEASE_DIR/image.txt"
printf '%s\n' "$EXPECTED_COMMIT" >"$RELEASE_DIR/commit.txt"
chmod 0600 "$RELEASE_DIR/image.txt" "$RELEASE_DIR/commit.txt"
VERSE_VLLM_IMAGE=$IMAGE \
VERSE_VLLM_EXPECTED_COMMIT=$EXPECTED_COMMIT \
VERSE_VLLM_GITHUB_TOKEN_FILE=$GITHUB_TOKEN_FILE \
VERSE_VLLM_ATTESTATION_VERIFICATION_OUTPUT="$RELEASE_DIR/attestation-verification.json" \
  tools/verse/verify_sm120_attestation.sh >/dev/null

export VAST_API_KEY
VAST_API_KEY=$(tr -d '\n' <"$VAST_API_KEY_FILE")

capture_provider() {
  local output=$1
  uvx --from vastai vastai show instance "$INSTANCE_ID" --raw --no-color | \
    uv run --no-project python -c '
import datetime, json, sys
payload = json.load(sys.stdin)
if isinstance(payload, list):
    if len(payload) != 1:
        raise SystemExit("provider returned the wrong instance count")
    payload = payload[0]
if not isinstance(payload, dict):
    raise SystemExit("provider returned a malformed instance")
names = (
    "id", "machine_id", "image_uuid", "gpu_name", "num_gpus",
    "actual_status", "intended_status", "cur_state", "ssh_host",
    "ssh_port", "start_date",
)
print(json.dumps({
    "schema_version": 1,
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "instance": {name: payload.get(name) for name in names},
}, indent=2, sort_keys=True))
' >"$output"
  chmod 0600 "$output"
}

capture_provider "$RELEASE_DIR/provider-before.json"
SSH_FIELDS=$(uv run --no-project python -c '
import json, sys
p = json.load(open(sys.argv[1]))["instance"]
if (
    p.get("id") != int(sys.argv[2])
    or p.get("image_uuid") != sys.argv[3]
    or p.get("gpu_name") != "RTX 5070 Ti"
    or p.get("num_gpus") != 1
    or p.get("actual_status") != "running"
    or p.get("intended_status") != "running"
    or p.get("cur_state") != "running"
):
    raise SystemExit("provider allocation does not match the fixed qualification target")
host, port = p.get("ssh_host"), p.get("ssh_port")
if not isinstance(host, str) or not host or not isinstance(port, int):
    raise SystemExit("provider SSH identity is unavailable")
print(host)
print(port)
' "$RELEASE_DIR/provider-before.json" "$INSTANCE_ID" "$IMAGE")
[[ $(printf '%s\n' "$SSH_FIELDS" | wc -l | tr -d ' ') -eq 2 ]] || exit 1
SSH_HOST=$(printf '%s\n' "$SSH_FIELDS" | sed -n '1p')
SSH_PORT=$(printf '%s\n' "$SSH_FIELDS" | sed -n '2p')

KNOWN_HOSTS="$RELEASE_DIR/ssh-known-hosts"
for _ in $(seq 1 30); do
  ssh-keyscan -T 5 -t ed25519 -p "$SSH_PORT" "$SSH_HOST" \
    >"$KNOWN_HOSTS.tmp" 2>/dev/null || true
  [[ -s $KNOWN_HOSTS.tmp ]] && break
  sleep 2
done
[[ -s $KNOWN_HOSTS.tmp ]] || {
  echo "provider SSH endpoint did not present a host key" >&2
  exit 1
}
mv "$KNOWN_HOSTS.tmp" "$KNOWN_HOSTS"
chmod 0600 "$KNOWN_HOSTS"
ssh-keygen -lf "$KNOWN_HOSTS" -E sha256 | awk '{print $2}' | sort -u \
  >"$RELEASE_DIR/ssh-host-fingerprint.txt"
[[ $(wc -l <"$RELEASE_DIR/ssh-host-fingerprint.txt" | tr -d ' ') -eq 1 ]] || {
  echo "provider SSH endpoint presented multiple host fingerprints" >&2
  exit 1
}
chmod 0600 "$RELEASE_DIR/ssh-host-fingerprint.txt"

SSH=(ssh -i "$SSH_KEY_FILE" -p "$SSH_PORT" -o BatchMode=yes \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$KNOWN_HOSTS" root@"$SSH_HOST")
SCP=(scp -i "$SSH_KEY_FILE" -P "$SSH_PORT" -o BatchMode=yes \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$KNOWN_HOSTS")
"${SSH[@]}" true

REMOTE_TAG=$(uv run --no-project python -c 'import secrets; print(secrets.token_hex(16))')
REMOTE_ROOT="/workspace/verse-qualification-$REMOTE_TAG"
REMOTE_EVIDENCE="$REMOTE_ROOT/evidence"
"${SSH[@]}" "install -d -m 0700 '$REMOTE_ROOT' && install -d -m 0750 '$REMOTE_EVIDENCE'"
"${SCP[@]}" \
  tools/verse/run_sm120_vast_outer.py \
  tools/verse/run_sm120_outer_cuda_gates.sh \
  tools/verse/snapshot_sm120_vast_outer.py \
  root@"$SSH_HOST":"$REMOTE_ROOT/"
"${SCP[@]}" "$HF_TOKEN_FILE" root@"$SSH_HOST":"$REMOTE_ROOT/hf-token"
"${SCP[@]}" "$API_KEY_FILE" root@"$SSH_HOST":"$REMOTE_ROOT/api-key"
"${SSH[@]}" "chmod 0700 '$REMOTE_ROOT/'*.py '$REMOTE_ROOT/'*.sh && chmod 0600 '$REMOTE_ROOT/hf-token' '$REMOTE_ROOT/api-key'"

QUALIFICATION_TOOLS=(
  benchmarks/verse/sm120_b01.py
  benchmarks/verse/sm120_prefill_interference.py
  tools/verse/check_sm120_chat_contract.py
  tools/verse/evaluate_sm120_acceptance.py
  tools/verse/finalize_sm120_vast_outer_release.py
  tools/verse/probe_sm120_outer_auth.py
  tools/verse/run_sm120_churn.py
  tools/verse/run_sm120_outer_cuda_gates.sh
  tools/verse/run_sm120_queue_stress.py
  tools/verse/run_sm120_user_latency.py
  tools/verse/run_sm120_vast_outer.py
  tools/verse/run_sm120_vast_release_gates.sh
  tools/verse/run_sm120_warm_latency.py
  tools/verse/sm120_evidence_identity.py
  tools/verse/snapshot_sm120_vast_outer.py
  tools/verse/verify_sm120_attestation.sh
)
uv run --no-project python - "${QUALIFICATION_TOOLS[@]}" \
  >"$RELEASE_DIR/qualification-tools.json" <<'PY'
import hashlib, json, pathlib, sys
print(json.dumps({
    "schema_version": 1,
    "sha256": {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in sys.argv[1:]
    },
}, indent=2, sort_keys=True))
PY
chmod 0600 "$RELEASE_DIR/qualification-tools.json"

"${SSH[@]}" "$REMOTE_ROOT/run_sm120_outer_cuda_gates.sh '$REMOTE_EVIDENCE/cuda-oracle.log'"
"${SCP[@]}" root@"$SSH_HOST":"$REMOTE_EVIDENCE/cuda-oracle.log" \
  "$RELEASE_DIR/cuda-oracle.log"

"${SSH[@]}" "uv run --script '$REMOTE_ROOT/run_sm120_vast_outer.py' \
  --hf-token-file '$REMOTE_ROOT/hf-token' \
  --api-key-file '$REMOTE_ROOT/api-key' \
  --model-cache '/models/verse-model-cache-$REMOTE_TAG' \
  --runtime-cache '$REMOTE_ROOT/runtime-cache' \
  --evidence-dir '$REMOTE_EVIDENCE' \
  --server-pid-file '$REMOTE_ROOT/server.pid' \
  --server-log '$REMOTE_ROOT/server.log' \
  --expected-commit '$EXPECTED_COMMIT' --port 8000"
"${SCP[@]}" root@"$SSH_HOST":"$REMOTE_EVIDENCE/live-process.json" \
  "$RELEASE_DIR/live-process-before.json"
"${SCP[@]}" root@"$SSH_HOST":"$REMOTE_EVIDENCE/image-verification.json" \
  "$RELEASE_DIR/image-verification.json"

LOCAL_PORT=$(uv run --no-project python -c '
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
')
ssh -i "$SSH_KEY_FILE" -p "$SSH_PORT" -o BatchMode=yes \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$KNOWN_HOSTS" -o ExitOnForwardFailure=yes \
  -N -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:8000" root@"$SSH_HOST" &
TUNNEL_PID=$!
MANIFEST_TMP=
cleanup() {
  local status=$?
  [[ -z $MANIFEST_TMP ]] || rm -f "$MANIFEST_TMP"
  kill "$TUNNEL_PID" >/dev/null 2>&1 || true
  wait "$TUNNEL_PID" >/dev/null 2>&1 || true
  unset VAST_API_KEY
  exit "$status"
}
trap cleanup EXIT
sleep 2
kill -0 "$TUNNEL_PID"
ENDPOINT="http://127.0.0.1:$LOCAL_PORT"

uv run --script tools/verse/probe_sm120_outer_auth.py \
  --endpoint "$ENDPOINT" --api-key-file "$API_KEY_FILE" \
  >"$RELEASE_DIR/auth-proof.json"

IDENTITY_FIELDS=$(uv run --no-project python -c '
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "tools/verse")
from finalize_sm120_vast_outer_release import live_identity, provider_identity
p = json.load(open(sys.argv[1]))
l = json.load(open(sys.argv[2]))
commit, image = sys.argv[3:5]
fingerprint = Path(sys.argv[5]).read_text().strip()
runtime_id = hashlib.sha256(json.dumps({
    "provider": provider_identity(p, image),
    "live": live_identity(l, commit),
    "ssh": fingerprint,
}, sort_keys=True).encode()).hexdigest()
gpu = l["gpu"]
print(runtime_id)
print("{name}, {uuid}, {memory_total_mib}, {driver_version}".format(**gpu))
' "$RELEASE_DIR/provider-before.json" "$RELEASE_DIR/live-process-before.json" \
  "$EXPECTED_COMMIT" "$IMAGE" "$RELEASE_DIR/ssh-host-fingerprint.txt")
[[ $(printf '%s\n' "$IDENTITY_FIELDS" | wc -l | tr -d ' ') -eq 2 ]] || exit 1
RUNTIME_ID=$(printf '%s\n' "$IDENTITY_FIELDS" | sed -n '1p')
GPU_IDENTITY=$(printf '%s\n' "$IDENTITY_FIELDS" | sed -n '2p')
RELEASE_NONCE=$(uv run --no-project python -c 'import secrets; print(secrets.token_hex(32))')
MODEL_REVISION=e2c6cd9c3302e91c032a378a607009c82ba16fac
BASE_IDENTITY=(
  --image-digest "$IMAGE" --fork-commit "$EXPECTED_COMMIT"
  --model-revision "$MODEL_REVISION" --gpu-name "$GPU_IDENTITY"
  --release-nonce "$RELEASE_NONCE"
)
BOUND_IDENTITY=("${BASE_IDENTITY[@]}" --container-id "$RUNTIME_ID")

uv run --script tools/verse/check_sm120_chat_contract.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  "${BOUND_IDENTITY[@]}" >"$RELEASE_DIR/chat-contract.json"

REPORTS=()
for target in 1000 5500; do
  for run in 1 2 3; do
    report="$RELEASE_DIR/b01-${target}-run-${run}.json"
    warm=()
    [[ $run == 1 ]] && warm+=(--skip-warmup)
    uv run --script benchmarks/verse/sm120_b01.py \
      --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
      --concurrency 38 --prompt-tokens "$target" --output-tokens 512 \
      --minimum-steady-seconds 10 --minimum-steady-samples 50 \
      --minimum-aggregate 0 --minimum-wall-ratio 0 \
      "${BASE_IDENTITY[@]}" "${warm[@]}" >"$report"
    REPORTS+=("$report")
  done
done
uv run --script tools/verse/evaluate_sm120_acceptance.py "${REPORTS[@]}" \
  >"$RELEASE_DIR/b01-summary.json"

for shape in 37x1 30x8; do
  [[ $shape == 37x1 ]] && decoders=37 || decoders=30
  [[ $shape == 37x1 ]] && prefills=1 || prefills=8
  uv run --script benchmarks/verse/sm120_prefill_interference.py \
    --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
    --decoders "$decoders" --prefills "$prefills" \
    --decode-prompt-tokens 4500 --decode-output-tokens 1024 \
    --prefill-prompt-tokens 6000 --baseline-seconds 3 --metrics-interval 0.05 \
    --max-num-batched-tokens 512 "${BASE_IDENTITY[@]}" \
    >"$RELEASE_DIR/prefill-${shape}.json"
done

uv run --script tools/verse/run_sm120_queue_stress.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  --phase-seconds 60 --active-capacity 38 --overflow-clients 76 \
  --prompt-pool-size 96 "${BOUND_IDENTITY[@]}" \
  >"$RELEASE_DIR/queue-stress.json"
uv run --script tools/verse/run_sm120_user_latency.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  --samples-per-prompt 5 "${BOUND_IDENTITY[@]}" \
  >"$RELEASE_DIR/user-latency.json"
uv run --script tools/verse/run_sm120_warm_latency.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  --clients 38 --max-tokens 100 "${BOUND_IDENTITY[@]}" \
  >"$RELEASE_DIR/warm-latency.json"
uv run --script tools/verse/run_sm120_churn.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  --duration-seconds 7200 --concurrency 38 --prompt-pool-size 64 \
  "${BOUND_IDENTITY[@]}" >"$RELEASE_DIR/churn.json"
uv run --script tools/verse/check_sm120_chat_contract.py \
  --endpoint "$ENDPOINT" --model verse-free --api-key-file "$API_KEY_FILE" \
  "${BOUND_IDENTITY[@]}" >"$RELEASE_DIR/post-churn-chat-contract.json"

"${SSH[@]}" "uv run --script '$REMOTE_ROOT/snapshot_sm120_vast_outer.py' \
  --pid-file '$REMOTE_ROOT/server.pid' \
  --output '$REMOTE_EVIDENCE/live-process-after.json' \
  --expected-commit '$EXPECTED_COMMIT'"
"${SCP[@]}" root@"$SSH_HOST":"$REMOTE_EVIDENCE/live-process-after.json" \
  "$RELEASE_DIR/live-process-after.json"
"${SCP[@]}" root@"$SSH_HOST":"$REMOTE_ROOT/server.log" \
  "$RELEASE_DIR/server.log"
capture_provider "$RELEASE_DIR/provider-after.json"

MANIFEST_TMP=$(mktemp "${RELEASE_DIR}.manifest.XXXXXXXX")
uv run --script tools/verse/finalize_sm120_vast_outer_release.py \
  --release-dir "$RELEASE_DIR" >"$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$RELEASE_DIR/release-manifest.json"
MANIFEST_TMP=
chmod 0600 "$RELEASE_DIR/release-manifest.json"
echo "SM120 Vast release qualification passed: $RELEASE_DIR/release-manifest.json"
