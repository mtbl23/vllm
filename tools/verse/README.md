# Verse SM120 Gemma 4 runtime

This directory defines the only supported first-release profile for the Verse
free-model runtime:

- RTX 5070 Ti with exact SM120 compute capability
- the pinned Campaign 22 Gemma 4 12B ModelOpt NVFP4 W4A4 checkpoint
- NVFP4 KV cache through FlashInfer FA2
- explicit HND physical KV layout
- 6,144-token context
- synchronous B01 scheduling with at most 38 active sequences
- 512-token chunked-prefill batches, the provisionally qualified scheduler budget
- FlashInfer's CUTLASS-4.7-compatible SM120 B12X backend for native W4A4 GEMM
- prefix caching and the hybrid KV manager enabled
- full-decode-only CUDA graphs at fixed capture sizes through 38 requests

The profile deliberately excludes multimodal inputs, Triton attention, BoN,
ranking, speculative decoding, and KV offload. Exact D512 Gemma 4 decode uses
the Verse-qualified FlashInfer XQA route; prefill and unsupported shapes remain
on the pinned FA2 path. The Verse gateway remains responsible for
authentication, quotas, queue admission, prompt construction, streaming, and
retries.

CUDA graphs are restricted to decode intentionally. The pinned native FA2
VO-split path handles prefill outside graph capture, while fixed decode-only
capture sizes cover 1, 8, 16, 24, 32, and 38 active requests. Other graph modes
cannot be enabled by a launch override.

## Build

Builds must come from a clean committed tree. The derived image selects only
SM120 kernels at runtime and is labeled with the exact fork commit and the
SHA-256 of the FlashInfer dependency manifest. The inherited upstream OCI
layers still contain the upstream source, examples, and benchmarks; deleting
their visible paths in a later layer would not remove those bytes from image
history. The Python, cubin, and CUDA 13 JIT-cache wheels are all pinned to
nightly-v0.6.18-20260819; Cutlass DSL is pinned to 4.7.0. The Linux/amd64 build
and runtime base images are pinned by manifest digest.

```bash
VERSE_VLLM_IMAGE_REPOSITORY=registry.example/verse-vllm \
VERSE_VLLM_BUILD_OUTPUT=push \
tools/verse/build_sm120_image.sh
```

The launcher never accepts mutable image tags. The build workflow also emits an
owner-controlled image receipt that binds the registry digest, fork commit,
source archive, wheel artifact, and native extension bytes. Install that
receipt as a root-owned mode-0600 file on every qualification or production
host. Labels stored inside the candidate image are not accepted as independent
provenance.

## Launch

Prepare the exact immutable Hugging Face subdirectory once. The preparer
verifies every file against the Campaign 22 build manifest before writing a
ready marker. It resolves Hugging Face blob symlinks into a content-addressed,
read-only materialization and the serving container mounts only that verified
tree, never the shared blob cache or Hugging Face token. Account for one full
extra copy of the candidate weights on the validation host.

```bash
export VERSE_MODEL_CACHE_DIR=/var/lib/verse-model-cache
sudo tools/verse/prepare_sm120_model.py \
  --cache-dir "$VERSE_MODEL_CACHE_DIR" \
  --token-file /run/verse-secrets/huggingface-token
```

The launcher refuses to replace an existing container. It binds the API to
localhost, mounts an owner-only API key file from outside both cache trees,
checks the image's fork-commit label, and accepts only the verified model
snapshot pinned in `sm120_profile.env`. Every candidate must receive a fresh,
empty runtime cache so compiled artifacts cannot leak between candidates or a
rollback. The only served model alias is `verse-free`, matching the current
Verse Free backend contract.

```bash
export VERSE_VLLM_IMAGE='registry.example/verse-vllm@sha256:...'
export VERSE_VLLM_EXPECTED_COMMIT='<40-character fork commit>'
export VERSE_VLLM_IMAGE_RECEIPT='/run/verse-release/verse-sm120-image-receipt.json'
export VERSE_MODEL_CACHE_DIR='/var/lib/verse-model-cache'
export VERSE_VLLM_CACHE_DIR='/var/lib/verse-vllm-cache'
export VERSE_VLLM_API_KEY_FILE='/run/verse-secrets/vllm-api-key'
sudo chown 2000:0 "$VERSE_VLLM_API_KEY_FILE"
sudo chmod 0600 "$VERSE_VLLM_API_KEY_FILE"
sudo chown root:root "$VERSE_VLLM_IMAGE_RECEIPT"
sudo chmod 0600 "$VERSE_VLLM_IMAGE_RECEIPT"
sudo --preserve-env=VERSE_VLLM_IMAGE,VERSE_VLLM_EXPECTED_COMMIT,VERSE_VLLM_IMAGE_RECEIPT,VERSE_MODEL_CACHE_DIR,VERSE_VLLM_CACHE_DIR,VERSE_VLLM_API_KEY_FILE \
  tools/verse/run_sm120_server.sh
```

The API key file must be owned by UID 2000 with owner-only permissions. The
launcher verifies the external receipt before executing the image, then runs
the image verifier in a networkless read-only container and requires the
runtime wheel/native hashes to match that receipt. It runs as root only to
establish root-owned mount boundaries; the serving process runs as UID 2000/GID
0 with every Linux capability dropped.

The container first starts with automatic restart disabled. The launcher waits
for health, checks authenticated model discovery and strict kernel markers, and
runs both streaming and repeated greedy inference. Only after those checks
pass does it enable `unless-stopped` restart behavior and print `status=ready`.
A failed startup removes only the newly-created candidate container. The model
mount remains read-only and its full manifest inventory is rehashed on every
container start, including Docker-initiated restarts.

The launch profile is fixed in `sm120_profile.env`. Capacity or kernel changes
must create a new profile version rather than silently overriding values.

## Public gateway boundary

Never point a Cloudflare Tunnel at the raw vLLM listener. The tracked Caddy
gateway binds only to `127.0.0.1:8080`, proxies to the loopback vLLM listener,
and exposes exactly `/health`, `/v1/models`, `/tokenize`, and
`/v1/chat/completions`. Every other path receives a local 404. Cloudflare
Access remains the public authentication boundary and the vLLM bearer key
remains a second, independent application credential.

Start the gateway only after the launcher has returned the immutable
64-character candidate container ID:

```bash
export VERSE_VLLM_CONTAINER_ID='<exact-container-id>'
export VERSE_VLLM_RELEASE_MANIFEST='<absolute-qualified-release-manifest>'
export VERSE_VLLM_API_KEY_FILE='<absolute-owner-only-vllm-key-file>'
tools/verse/run_sm120_gateway.sh
```

The gateway refuses a mutable or mismatched upstream. The qualified manifest
must bind the exact 64-character container ID, immutable image digest, fork
commit, model revision, and release nonce. The candidate must publish exactly
`127.0.0.1:8000`, and all raw vLLM application routes other than `/health` and
`/metrics` remain bearer-authenticated even though the public gateway exposes
neither management surface. Gateway responses include immutable candidate
identity headers so the Access-protected public proof can bind a tunnel to the
same qualified process before a route change.

Qualification and production must use distinct Cloudflare Tunnel UUIDs. Do
not attach a candidate connector as another replica of the production tunnel:
Cloudflare can route traffic to any replica of one tunnel, so that topology
cannot provide deterministic candidate isolation or rollback. Validate the
candidate through a separate Access-protected hostname, then switch the one
exact stable proxied CNAME only after all release gates pass. The full
compare-before-write procedure and its exact inverse are in
`SM120_CUTOVER_RUNBOOK.md`.

Before qualification, verify the registry digest's GitHub artifact attestation
against the exact fork repository, image workflow, source branch, and source
commit. The image workflow signs provenance from a GitHub-hosted runner and
publishes both the bundle and an exact-policy verification result beside the
image identity receipt. Version-only appliance additions are installed from
`requirements/verse-sm120-runtime.lock` with hashes and without dependency
resolution; the pre-existing image dependency set is checked afterward.

## Verification

Before routing traffic, run both checks on the disposable host:

```bash
tools/verse/run_static_gates.sh
tools/verse/run_sm120_cuda_gates.sh
tools/verse/check_sm120_server.sh
```

The CUDA gate runs inside the immutable candidate image. It checks the exact
FlashInfer/Cutlass/Torch tuple and executes the production metadata-builder,
NVFP4 cache update, and `FlashInferImpl.forward` path at short, 1K, 5.5K, and
6,144-token KV lengths. The output is compared with attention computed directly
from the original BF16 tensors, independently of the packed cache views. A
skipped hardware test fails the gate. The live check verifies the image commit,
localhost binding, exact runtime tuple, health endpoint, served model, strict
startup marker, and absence of a silent Triton or XQA fallback.

## Performance acceptance

Run the fixed acceptance matrix after the image, CUDA, and live gates:

```bash
export VERSE_VLLM_ACCEPTANCE_DIR=/var/lib/verse-acceptance/<candidate-id>
tools/verse/run_sm120_acceptance.sh
```

The runner first exercises Verse's real message-based `/tokenize` and
streaming `/v1/chat/completions` shapes, including sampling extensions, SSE
termination, repeated exact-length greedy runs with finite logprobs, exact
6,144-token admission, 6,145-token rejection, and 38 distinct concurrent
boundary requests without preemption. It then runs one disjoint-prefix trial
without an explicit prewarm and two explicitly prewarmed B01 trials at 1K and
at 5.5K, all with exactly 38 active requests and 512 generated tokens.
Generated text is never written to disk.
Identity is derived from the validated container and `nvidia-smi`, not typed
into the report by an operator.

The first-release gates are:

- at least 1,074 aggregate decode tokens/second at 38 users and about 1K context
- at least 992 aggregate decode tokens/second at 38 users and about 5.5K context
- at least 30 percent improvement on the production-weighted 1K/5.5K mix
- 38 active slots without preemption, OOM, restart, or request loss
- no material output divergence beyond the accepted NVFP4 KV tolerance
- no crash or cache corruption during the separate two-hour heavy churn gate

The old Triton INT4 measurements were 895 tok/s at 1K and 734.7 tok/s at 5.5K
with 40 active requests. The acceptance evaluator checks both scenario medians
and the mean relative improvement across the two legacy baselines. Missing any
gate keeps the old image in service.

For a complete release decision, run the wrapper below. It performs the short
matrix, the fixed two-hour heavy churn gate, a second complete chat-contract check,
and proves from Docker state that the same immutable container remained alive
without a restart for the whole campaign:

```bash
export VERSE_VLLM_RELEASE_DIR=/var/lib/verse-acceptance/<candidate-id>-release
tools/verse/run_sm120_release_gates.sh
```

The churn phase keeps 38 workers cycling through 64 distinct 1K, 5.5K, and 6K
prefixes, cancels a deterministic fraction mid-stream, discards all generated
text, and fails on request errors, preemptions, missing prefix hits, or queues
that do not drain. Only `release-manifest.json` with `status: pass` is
pre-cutover candidate evidence. It does not claim that production routing or a
rollback drill occurred. The wrapper deliberately does not start a public
gateway, connect Cloudflare, or require routing credentials. Gateway proof and
the non-production cutover/rollback drill are separate deployment-readiness
steps in the runbook and must not block or contaminate disposable GPU
qualification.

## Cutover and rollback boundary

This fork does not automate production routing. Build and validate a new image,
start it beside the old service, run the gates, and then change the gateway or
tunnel target separately. Keep the prior image digest and container definition
until the new runtime has passed the observation window. Rollback means routing
back to that immutable prior service, not modifying the running container.
Follow `SM120_CUTOVER_RUNBOOK.md`; a passing release manifest is candidate
qualification, not evidence that cutover or rollback happened.

BoN and a ranker are a later release. They must not be enabled until B01 passes
correctness, capacity, throughput, and churn gates on the intended GPU.
