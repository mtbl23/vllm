#!/bin/sh
set -eu

/usr/local/bin/verify-verse-sm120-image --require-gpu

for requirement in \
    'VLLM_VERSE_RUNTIME_STRICT=1' \
    'VLLM_NVFP4_KV_VOSPLIT=1' \
    'VLLM_KV_CACHE_LAYOUT=HND' \
    'VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0' \
    'VLLM_USE_FLASHINFER_SAMPLER=0' \
    'FLASHINFER_WORKSPACE_BASE=/cache/flashinfer' \
    'CUDA_CACHE_PATH=/cache/cuda' \
    'TORCH_HOME=/cache/torch' \
    'TORCH_EXTENSIONS_DIR=/cache/torch-extensions' \
    'TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor'; do
    name=${requirement%%=*}
    expected=${requirement#*=}
    eval "actual=\${$name-}"
    if [ "$actual" != "$expected" ]; then
        echo "$name must equal $expected" >&2
        exit 1
    fi
done

python /opt/verse-tools/prepare_sm120_model.py \
    --model-directory /models/model \
    --require-root-owner \
    --verify-mounted-model >/dev/null

key_file=${VLLM_API_KEY_FILE:-}
if [ -z "$key_file" ] || [ ! -f "$key_file" ]; then
    echo "VLLM_API_KEY_FILE must name a mounted secret file" >&2
    exit 1
fi

api_key=$(cat "$key_file")
if [ -z "$api_key" ]; then
    echo "VLLM API key secret is empty" >&2
    exit 1
fi
case "$api_key" in
    *'
'*)
        echo "VLLM API key secret contains an embedded newline" >&2
        exit 1
        ;;
esac

export VLLM_API_KEY=$api_key
unset api_key key_file
exec vllm serve "$@"
