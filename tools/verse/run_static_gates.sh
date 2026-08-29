#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

PYTHON_FILES=(
  setup.py
  benchmarks/verse/sm120_b01.py
  benchmarks/verse/sm120_b12x_tactics.py
  benchmarks/verse/sm120_lm_head.py
  benchmarks/verse/sm120_nvfp4_decode_backends.py
  benchmarks/verse/sm120_prefill_interference.py
  tests/benchmarks/test_verse_sm120_b01.py
  tests/benchmarks/test_verse_sm120_prefill_interference.py
  tests/kernels/attention/test_flashinfer.py
  tests/kernels/quantization/nvfp4_utils.py
  tests/kernels/quantization/test_verse_sm120_b12x_nvfp4.py
  tests/tools/test_verse_sm120_container.py
  tests/tools/test_verse_sm120_cutover.py
  tests/tools/test_verse_sm120_acceptance.py
  tests/tools/test_verse_sm120_chat_contract.py
  tests/tools/test_verse_sm120_churn.py
  tests/tools/test_verse_sm120_model.py
  tests/tools/test_verse_sm120_profile.py
  tests/tools/test_verse_sm120_release.py
  tests/tools/test_verse_sm120_source.py
  tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py
  tests/v1/attention/test_nvfp4_flashinfer_vosplit.py
  tools/verse/prepare_sm120_model.py
  tools/verse/build_sm120_wheel_identity.py
  tools/verse/finalize_sm120_release.py
  tools/verse/evaluate_sm120_acceptance.py
  tools/verse/check_sm120_chat_contract.py
  tools/verse/download_sm120_model.py
  tools/verse/run_sm120_churn.py
  tools/verse/run_sm120_queue_stress.py
  tools/verse/run_sm120_user_latency.py
  tools/verse/run_sm120_warm_latency.py
  tools/verse/sm120_evidence_identity.py
  tools/verse/sm120_image_receipt.py
  tools/verse/validate_sm120_container.py
  tools/verse/validate_sm120_profile.py
  tools/verse/verify_sm120_image.py
  tools/verse/switch_sm120_cloudflare_route.py
  tools/verse/verify_sm120_public_gateway.py
  vllm/model_executor/models/config.py
  vllm/v1/attention/backends/flashinfer.py
)

bash -n tools/verse/*.sh docker/entrypoints/verse-sm120-entrypoint.sh \
  benchmarks/verse/c22_launch_sm120_xqa.sh
for executable in tools/verse/*.sh docker/entrypoints/verse-sm120-entrypoint.sh; do
  [[ -x $executable ]] || {
    echo "$executable must be executable" >&2
    exit 1
  }
done
uvx --from ruff==0.15.12 ruff check "${PYTHON_FILES[@]}"
uvx --from ruff==0.15.12 ruff format --check "${PYTHON_FILES[@]}"
uv run --no-project --with dockerfile-parse==2.0.1 python tools/generate_versions_json.py --check

uv run --no-project --with pytest==9.1.0 python -m pytest -q \
  --confcutdir=tests/tools \
  tests/tools/test_verse_sm120_profile.py \
  tests/tools/test_verse_sm120_release.py \
  tests/tools/test_verse_sm120_source.py \
  tests/tools/test_verse_sm120_model.py \
  tests/tools/test_verse_sm120_container.py \
  tests/tools/test_verse_sm120_cutover.py \
  tests/tools/test_verse_sm120_acceptance.py \
  tests/tools/test_verse_sm120_chat_contract.py \
  tests/tools/test_verse_sm120_churn.py
uv run --no-project --with pytest==9.1.0 python -m pytest -q \
  --confcutdir=tests/benchmarks \
  tests/benchmarks/test_verse_sm120_b01.py \
  tests/benchmarks/test_verse_sm120_prefill_interference.py

grep -Fxq \
  'flashinfer-python @ https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260819/flashinfer_python-0.6.18.dev20260819-py3-none-any.whl#sha256=50ad966220b5160f17fcb9e064bdfbcda726ec779fb0c74fd3449b3c48c66600' \
  requirements/verse-sm120-flashinfer.lock
grep -Fxq \
  'flashinfer-cubin @ https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260819/flashinfer_cubin-0.6.18.dev20260819-py3-none-any.whl#sha256=277c3f2ef478dd8da5f315f21c3ce56c4437dbb47b170ceb4b47185f9c46560b' \
  requirements/verse-sm120-flashinfer.lock
grep -Fxq \
  'flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260819/flashinfer_jit_cache-0.6.18.dev20260819%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=6c44aabbe7f225b97a546f92fe65b51cbd4ef83838026e4a5d6d6d35486d2ea1' \
  requirements/verse-sm120-flashinfer.lock
[[ $(grep -Ec '^flashinfer-(python|cubin|jit-cache) @ ' \
  requirements/verse-sm120-flashinfer.lock) -eq 3 ]]
grep -Fxq 'flashinfer-python==0.6.17' requirements/cuda.txt
grep -Fxq 'flashinfer-cubin==0.6.17' requirements/cuda.txt
grep -Fxq 'nvidia-cutlass-dsl[cu13]==4.6.2' requirements/cuda.txt
grep -Fq 'quack-kernels==0.6.4' requirements/cuda.txt
grep -Fq -- '-r /opt/verse/requirements/verse-sm120-flashinfer.lock' \
  docker/Dockerfile
grep -Fq "'nvidia-cutlass-dsl[cu13]==4.7.0'" docker/Dockerfile
grep -Fq 'nvidia-cudnn-frontend==1.27.0' docker/Dockerfile
grep -Fq 'nvidia-nccl-cu13==2.29.7' docker/Dockerfile
grep -Fq 'ENV UV_OVERRIDE=/etc/uv-overrides-verse-sm120.txt' docker/Dockerfile
grep -Fq 'deep_ep' docker/Dockerfile
grep -Fq '        b12x \' docker/Dockerfile
grep -Fq 'VLLM_VERSE_SM120_WHEEL=1' tools/verse/build_sm120_image.sh
grep -Fq 'flashinfer-python @ https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260819/' \
  requirements/verse-sm120-wheel.txt
grep -Fq 'uv pip uninstall --system' docker/Dockerfile
grep -Fq 'quack-kernels' docker/Dockerfile
grep -Fq '/opt/verse-tools/sm120_profile.env' docker/Dockerfile
grep -Fq -- '--verify-mounted-model' docker/entrypoints/verse-sm120-entrypoint.sh
grep -Fq -- '--restart no' tools/verse/run_sm120_server.sh
grep -Fq 'verify_sm120_source.sh' tools/verse/run_sm120_server.sh
grep -Fq 'verify_sm120_source.sh' tools/verse/run_sm120_acceptance.sh
grep -Fq 'verify_sm120_source.sh' tools/verse/run_sm120_release_gates.sh
grep -Fq 'respond 404' tools/verse/verse-sm120-gateway.Caddyfile
grep -Fq '/v1/chat/completions' tools/verse/verse-sm120-gateway.Caddyfile
grep -Fq 'cloudflare_route.py' tools/verse/SM120_CUTOVER_RUNBOOK.md
grep -Fq -- '--cap-drop ALL' tools/verse/run_sm120_server.sh
grep -Fq -- '--user 2000:0' tools/verse/run_sm120_server.sh
grep -Fq 'USER 2000:0' docker/Dockerfile
grep -Fq 'rm -rf /vllm-workspace/benchmarks /vllm-workspace/examples' \
  docker/Dockerfile
grep -Fq 'whiteout intentionally narrows the visible appliance surface' \
  docker/Dockerfile
grep -Fq 'FLASHINFER_WORKSPACE_BASE=/cache/flashinfer' docker/Dockerfile
grep -Fq 'VLLM_MAX_N_SEQUENCES=1' docker/Dockerfile
grep -Fq 'VLLM_MAX_COMPLETION_PROMPTS=1' docker/Dockerfile
if grep -Eq -- '--ipc host' \
  tools/verse/run_sm120_server.sh tools/verse/run_sm120_cuda_gates.sh; then
  echo "host IPC remains in the Verse appliance" >&2
  exit 1
fi
grep -Fq 'self.disable_split_kv = True' \
  vllm/v1/attention/backends/flashinfer.py
grep -Fq 'return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE' \
  vllm/v1/attention/backends/flashinfer.py
grep -Fq \
  'pytorch/manylinux2_28-builder:cuda13.0@sha256:7710cbc19d7ee951134e2e827f8ec89237c993095eb2581dd5e74f58e4e278c7' \
  tools/verse/build_sm120_image.sh
grep -Fq \
  'nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4' \
  tools/verse/build_sm120_image.sh
if grep -REq --exclude=run_static_gates.sh \
  'ALLOW_MUTABLE|allow-mutable' tools/verse; then
  echo "mutable-image bypass remains in the Verse appliance" >&2
  exit 1
fi

git diff HEAD --check

echo "Verse SM120 static gates passed"
