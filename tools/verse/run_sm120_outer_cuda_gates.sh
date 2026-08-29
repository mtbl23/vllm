#!/usr/bin/env bash
set -euo pipefail

# Qualification-only counterpart to the fork's Docker CUDA gate. Vast runs
# the immutable OCI image as the outer container, so these commands execute
# directly against the image's pinned wheel and embedded test artifacts.

MODE=${1:-}
if [[ $MODE == --worker ]]; then
  OUTPUT=${2:?absolute evidence log path is required}
  CACHE_ROOT=${3:?absolute fresh cache root is required}
  [[ $(id -u) == 2000 && $(id -g) == 0 ]] || {
    echo "CUDA gate worker must run as UID 2000:GID 0" >&2
    exit 1
  }
else
  OUTPUT=${1:?absolute evidence log path is required}
  [[ $(id -u) == 0 ]] || {
    echo "CUDA gate bootstrap must run as root" >&2
    exit 1
  }
  [[ $OUTPUT == /* && ! -e $OUTPUT ]] || {
    echo "evidence log must be a new absolute path" >&2
    exit 1
  }
  install -d -m 0750 -o 2000 -g 0 "${OUTPUT%/*}"
  install -m 0640 -o 2000 -g 0 /dev/null "$OUTPUT"
  CACHE_ROOT=$(mktemp -d /tmp/verse-cuda-gates.XXXXXXXX)
  chown 2000:0 "$CACHE_ROOT"
  chmod 0700 "$CACHE_ROOT"
  exec /usr/bin/env -i \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/setpriv --reuid 2000 --regid 0 --clear-groups --no-new-privs \
    "$0" --worker "$OUTPUT" "$CACHE_ROOT"
fi

[[ $OUTPUT == /* && -f $OUTPUT && ! -L $OUTPUT ]] || {
  echo "evidence log must be a new absolute path" >&2
  exit 1
}
umask 027

export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$CACHE_ROOT/flashinfer-autotune
export CUDA_CACHE_PATH=$CACHE_ROOT/cuda
export TORCH_HOME=$CACHE_ROOT/torch
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch-extensions
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export PYTHONWARNINGS=ignore::DeprecationWarning:flashinfer.cute_dsl.utils

{
  /usr/local/bin/verify-verse-sm120-image --require-gpu
  GPU_IDENTITY=$(LC_ALL=C nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,compute_cap \
    --format=csv,noheader,nounits)
  [[ -n $GPU_IDENTITY && $GPU_IDENTITY != *$'\n'* ]]
  [[ $GPU_IDENTITY == 'NVIDIA GeForce RTX 5070 Ti,'*', 12.0' ]]
  printf 'VERSE_CUDA_GPU_IDENTITY=%s\n' "$GPU_IDENTITY"

  cd /opt/verse-tests
  ROUTING_COLLECT=$(mktemp)
  ROUTING_OUTPUT=$(mktemp)
  GPU_COLLECT=$(mktemp)
  GPU_OUTPUT=$(mktemp)
  KV_COLLECT=$(mktemp)
  KV_OUTPUT=$(mktemp)
  B12X_COLLECT=$(mktemp)
  B12X_OUTPUT=$(mktemp)

  python -m pytest --collect-only -q -p no:cacheprovider \
    tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
    tests/v1/attention/test_nvfp4_flashinfer_vosplit.py | tee "$ROUTING_COLLECT"
  ROUTING_COUNT=$(grep -Ec '^tests/.+::' "$ROUTING_COLLECT")
  ((ROUTING_COUNT == 55))
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$ROUTING_COLLECT"; then
    exit 1
  fi
  python -m pytest -q -p no:cacheprovider --maxfail=1 \
    tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
    tests/v1/attention/test_nvfp4_flashinfer_vosplit.py | tee "$ROUTING_OUTPUT"
  grep -Eq "^${ROUTING_COUNT} passed in [0-9.]+s$" "$ROUTING_OUTPUT"
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$ROUTING_OUTPUT"; then
    exit 1
  fi
  sed -n 's#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed routing \1#p' \
    "$ROUTING_COLLECT"
  echo VERSE_ROUTING_GATES_PASSED

  python -m pytest --collect-only -q -p no:cacheprovider \
    tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference \
    tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference | tee "$GPU_COLLECT"
  GPU_COUNT=$(grep -Ec '^tests/.+::' "$GPU_COLLECT")
  ((GPU_COUNT == 6))
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$GPU_COLLECT"; then
    exit 1
  fi
  python -m pytest -q -p no:cacheprovider --maxfail=1 \
    tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_vo_split_hnd_matches_reference \
    tests/kernels/attention/test_flashinfer.py::test_flashinfer_fa2_nvfp4_gemma4_sliding_hnd_matches_reference | tee "$GPU_OUTPUT"
  grep -Eq "^${GPU_COUNT} passed in [0-9.]+s$" "$GPU_OUTPUT"
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$GPU_OUTPUT"; then
    exit 1
  fi
  sed -n 's#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed gpu_oracle \1#p' \
    "$GPU_COLLECT"
  echo VERSE_GPU_ORACLE_PASSED

  python -m pytest --collect-only -q -p no:cacheprovider \
    tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py::test_verse_sm120_nvfp4_physical_hnd_roundtrip | tee "$KV_COLLECT"
  KV_COUNT=$(grep -Ec '^tests/.+::' "$KV_COLLECT")
  ((KV_COUNT == 2))
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$KV_COLLECT"; then
    exit 1
  fi
  python -m pytest -q -p no:cacheprovider --maxfail=1 \
    tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py::test_verse_sm120_nvfp4_physical_hnd_roundtrip | tee "$KV_OUTPUT"
  grep -Eq "^${KV_COUNT} passed in [0-9.]+s$" "$KV_OUTPUT"
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$KV_OUTPUT"; then
    exit 1
  fi
  sed -n 's#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed kv_store_oracle \1#p' \
    "$KV_COLLECT"
  echo VERSE_KV_STORE_ORACLE_PASSED

  python -m pytest --collect-only -q -p no:cacheprovider \
    tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py | tee "$B12X_COLLECT"
  B12X_COUNT=$(grep -Ec '^tests/.+::' "$B12X_COLLECT")
  ((B12X_COUNT == 6))
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$B12X_COLLECT"; then
    exit 1
  fi
  python -m pytest -q -p no:cacheprovider --maxfail=1 \
    tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py | tee "$B12X_OUTPUT"
  grep -Eq "^${B12X_COUNT} passed in [0-9.]+s$" "$B12X_OUTPUT"
  if grep -Eqi 'skipped|deselected|xfailed|xpassed|warning|error' "$B12X_OUTPUT"; then
    exit 1
  fi
  sed -n 's#^\(tests/.*::.*\)$#VERSE_CUDA_TEST_RESULT=passed b12x_oracle \1#p' \
    "$B12X_COLLECT"
  echo VERSE_B12X_ORACLE_PASSED

  for artifact in \
    tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py \
    tests/v1/attention/test_nvfp4_flashinfer_vosplit.py \
    tests/kernels/attention/test_flashinfer.py \
    tests/kernels/attention/test_verse_sm120_nvfp4_kv_cache.py \
    tests/kernels/quantization/nvfp4_utils.py \
    tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py; do
    sha256sum "$artifact" | sed 's#^#VERSE_CUDA_TEST_ARTIFACT_SHA256=#'
  done
  echo SM120_HOSTED_CUDA_GATES_PASSED
} | tee "$OUTPUT"

for marker in \
  VERSE_ROUTING_GATES_PASSED \
  VERSE_GPU_ORACLE_PASSED \
  VERSE_KV_STORE_ORACLE_PASSED \
  VERSE_B12X_ORACLE_PASSED \
  SM120_HOSTED_CUDA_GATES_PASSED; do
  grep -Fxq "$marker" "$OUTPUT"
done
