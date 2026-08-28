#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXPECTED_COMMIT=${1:-${VERSE_VLLM_EXPECTED_COMMIT:-}}

[[ $EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo "expected source commit must be a 40-character SHA" >&2
  exit 1
}

TOPLEVEL=$(git -C "$ROOT" rev-parse --show-toplevel)
[[ $TOPLEVEL == "$ROOT" ]] || {
  echo "Verse validation tools are not running from their repository root" >&2
  exit 1
}

HEAD_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
[[ $HEAD_COMMIT == "$EXPECTED_COMMIT" ]] || {
  echo "source HEAD $HEAD_COMMIT does not match candidate $EXPECTED_COMMIT" >&2
  exit 1
}

STATUS=$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)
[[ -z $STATUS ]] || {
  echo "candidate validation requires a completely clean source tree" >&2
  printf '%s\n' "$STATUS" >&2
  exit 1
}

printf 'status=clean\ncommit=%s\n' "$HEAD_COMMIT"
