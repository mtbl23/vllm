#!/usr/bin/env bash
set -euo pipefail

# Disposable source-checkout launcher for SM120 benchmark and profiler runs.
# Production must use tools/verse/run_sm120_server.sh with an immutable image.

MODEL_PATH="${MODEL_PATH:-/var/lib/verse-model-cache/verse-sm120-materialized/e2c6cd9c3302e91c032a378a607009c82ba16fac-159e4219ea3ffd63}"
VLLM_SOURCE="${VLLM_SOURCE:-/root/vllm-verse-sm120}"
SERVER_LOG="${SERVER_LOG:-/root/server-xqa.log}"
SERVER_PID="${SERVER_PID:-/root/server-xqa.pid}"

export PYTHONPATH="$VLLM_SOURCE"
export VLLM_VERSE_RUNTIME_STRICT=1
export VLLM_NVFP4_KV_VOSPLIT=1
export VLLM_VERSE_NVFP4_XQA_DECODE=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((64 * 1024 * 1024))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROFILE_PREFIX=()
PROFILER_ARGS=()
if [[ -n ${VERSE_NSYS_OUTPUT:-} ]]; then
  PROFILE_PREFIX=(
    /usr/local/bin/nsys profile
    --force-overwrite=true
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --output="$VERSE_NSYS_OUTPUT"
  )
  PROFILER_ARGS=(--profiler-config '{"profiler":"cuda"}')
fi

nohup "${PROFILE_PREFIX[@]}" /usr/bin/python3 /usr/local/bin/vllm serve \
  "$MODEL_PATH" \
  --served-model-name verse-free \
  --quantization modelopt_fp4 \
  --dtype bfloat16 \
  --linear-backend flashinfer_b12x \
  --max-model-len 6144 \
  --block-size 16 \
  --max-num-seqs 38 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.94 \
  --kv-cache-memory-bytes 5704253440 \
  --kv-cache-dtype nvfp4 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --language-model-only \
  --no-enable-log-requests \
  --disable-uvicorn-access-log \
  --generation-config vllm \
  "${PROFILER_ARGS[@]}" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,8,16,24,32,38],"max_cudagraph_capture_size":38}' \
  --host 127.0.0.1 \
  --port 8000 \
  >"$SERVER_LOG" 2>&1 </dev/null &

echo "$!" >"$SERVER_PID"
cat "$SERVER_PID"
