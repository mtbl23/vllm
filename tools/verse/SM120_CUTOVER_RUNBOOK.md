# Verse SM120 cutover and rollback runbook

This runbook begins only after qualification on a disposable exact RTX 5070 Ti.
Qualification must not connect a Cloudflare tunnel, alter a Verse gateway, or
touch the currently serving Free model.

The stable public hostname must be a proxied CNAME to one exact Cloudflare
Tunnel UUID. The old and candidate hosts use different tunnel UUIDs. Never run
the candidate as a replica of the production tunnel: replicas do not provide
deterministic traffic steering.

Each tunnel must point only to the tracked loopback Caddy gateway on
`http://127.0.0.1:8080`. It must never point to vLLM on port 8000. The gateway
allows only `/health`, `/v1/models`, `/tokenize`, and `/v1/chat/completions` and
returns 404 for every other path. Cloudflare Access must keep the existing
Service Auth policy on the stable hostname.

## Required records

Create an owner-only change record outside the repository. Record:

- candidate image by registry digest
- candidate fork commit and Campaign 22 model revision
- successful GitHub artifact-attestation verification for that exact digest,
  source commit, protected workflow path, and branch ref
- owner-only GitHub token file used only for artifact-attestation verification
- absolute path and SHA-256 of the candidate `release-manifest.json`
- old service endpoint, health URL, image or artifact identity, and start command
- new service endpoint and health URL
- Cloudflare account ID, zone ID, DNS record ID, old tunnel UUID, and candidate
  tunnel UUID
- owner-only Cloudflare proof token file with DNS Read and Cloudflare Tunnel Read
  permissions, plus a separate DNS Edit token used only for the route switch
- owner-only Access Client ID, Access Client Secret, and vLLM API key files
- operator and UTC start time

Do not put API keys, Hugging Face tokens, tunnel credentials, or generated text
in the record.

## Hard preconditions

All conditions must be true before routing one production request:

1. The candidate binding manifest has `status: pass` and
   `scope: pre_cutover_candidate_binding`.
2. The exact image digest has a valid GitHub artifact attestation from
   `.github/workflows/verse-sm120-image.yml`, the expected source commit and
   `refs/heads/verse/v0.28-sm120-nvfp4-fa2`, and a GitHub-hosted runner.
3. The CUDA, chat-contract, 38-slot, B01, and two-hour heavy churn gates all
   came from one image digest, fork commit, model revision, exact disposable
   RTX 5070 Ti, and process with zero restarts. The resulting
   `disposable_image_qualification` manifest is then bound to the exact
   production candidate container by `bind_sm120_candidate_release.py`.
4. The old service is healthy and remains running. A fresh public-route proof
   binds its distinct rollback hostname to the recorded old tunnel UUID.
5. The candidate is on a distinct disposable host and exact GPU UUID, with no
   production or unrelated compute process sharing that GPU.
6. The candidate has a different Cloudflare Tunnel UUID from production.
7. The tracked gateway is running, and cloudflared points only to its loopback
   port 8080.
8. The exact rollback target is the recorded healthy old tunnel UUID.
9. The stable DNS record is the exact expected proxied CNAME and no deployment
   or DNS automation can race the cutover. All Verse route changes use the same
   owner-only local deployment lock.
10. No lifecycle command used below can select an instance by a mutable label,
   partial name, wildcard, or unresolved environment variable.

If any condition is false, stop before cutover.

## Side-by-side candidate check

Run these checks from the candidate fork's exact clean commit:

```bash
umask 077
export VERSE_VLLM_GITHUB_TOKEN_FILE=\
"$CHANGE_RECORD_DIR/github-attestation-token"
VERSE_VLLM_ATTESTATION_VERIFICATION_OUTPUT=\
"$CHANGE_RECORD_DIR/image-attestation-verification.json" \
  tools/verse/verify_sm120_attestation.sh
tools/verse/verify_sm120_source.sh "$VERSE_VLLM_EXPECTED_COMMIT"
tools/verse/check_sm120_server.sh
VERSE_VLLM_VALIDATION_OUTPUT="$CHANGE_RECORD_DIR/candidate-validation.json" \
  tools/verse/check_sm120_server.sh
uv run --script tools/verse/bind_sm120_candidate_release.py \
  --qualification-manifest "$IMAGE_QUALIFICATION_MANIFEST" \
  --candidate-validation "$CHANGE_RECORD_DIR/candidate-validation.json" \
  --container-id "$EXACT_CANDIDATE_CONTAINER_ID" \
  --image "$VERSE_VLLM_IMAGE" \
  --expected-commit "$VERSE_VLLM_EXPECTED_COMMIT" \
  >"$CANDIDATE_RELEASE_MANIFEST"
VERSE_VLLM_CONTAINER_ID="$EXACT_CANDIDATE_CONTAINER_ID" \
VERSE_VLLM_RELEASE_MANIFEST="$CANDIDATE_RELEASE_MANIFEST" \
VERSE_VLLM_API_KEY_FILE="$VLLM_API_KEY_FILE" \
VERSE_VLLM_GITHUB_TOKEN_FILE="$VERSE_VLLM_GITHUB_TOKEN_FILE" \
  tools/verse/run_sm120_gateway.sh
curl --noproxy '*' --fail --silent --show-error "$NEW_HEALTH_URL"
curl --noproxy '*' --fail --silent --show-error "$OLD_HEALTH_URL"
```

Resolve `NEW_HEALTH_URL` and `OLD_HEALTH_URL` to explicit loopback or private
origins before invoking `curl`. Never embed credentials in either URL.

Capture immediately before cutover:

- candidate container ID, image digest, start time, and restart count
- old service identity and health response
- candidate GPU memory, utilization, scheduler running/waiting, preemptions,
  prompt throughput, and generation throughput
- gateway's currently active old target

Configure the candidate's distinct tunnel to send the stable hostname to
`http://127.0.0.1:8080`, with Protect with Access enabled. Before any DNS
change, run the complete public-route proof on a temporary Access-protected
candidate hostname:

```bash
uv run --script tools/verse/verify_sm120_public_gateway.py \
  --endpoint "$CANDIDATE_ACCESS_ORIGIN" \
  --model verse-free \
  --target-mode qualified-candidate \
  --target-tunnel "$NEW_TUNNEL_ID" \
  --account-id "$CF_ACCOUNT_ID" \
  --zone-id "$CF_ZONE_ID" \
  --record-id "$CF_CANDIDATE_PROOF_RECORD_ID" \
  --stable-record-id "$CF_FREE_DNS_RECORD_ID" \
  --hostname "$CANDIDATE_PROOF_HOSTNAME" \
  --production-hostname "$STABLE_FREE_HOSTNAME" \
  --expected-current-tunnel "$OLD_TUNNEL_ID" \
  --cloudflare-api-token-file "$CF_DNS_READ_TOKEN_FILE" \
  --release-manifest "$CANDIDATE_RELEASE_MANIFEST" \
  --api-key-file "$VLLM_API_KEY_FILE" \
  --access-client-id-file "$ACCESS_CLIENT_ID_FILE" \
  --access-client-secret-file "$ACCESS_CLIENT_SECRET_FILE" \
  >"$CHANGE_RECORD_DIR/candidate-public-proof.json"

uv run --script tools/verse/switch_sm120_cloudflare_route.py inspect \
  --zone-id "$CF_ZONE_ID" \
  --record-id "$CF_FREE_DNS_RECORD_ID" \
  --hostname free-inference.verse-rp.com \
  --expected-current-tunnel "$OLD_TUNNEL_ID" \
  --api-token-file "$CF_DNS_EDIT_TOKEN_FILE"
```

The public-route proof must show that an unauthenticated request is rejected,
all four required API paths work, and vLLM lifecycle, documentation, metrics,
and invocation routes all return 404 through the public origin. Create
`CHANGE_RECORD_DIR` with mode 0700 under the operator account, set `umask 077`,
and require every proof file to be owned by that account and not writable by
group or other users. The candidate proof must also match all immutable
identity headers emitted by the exact qualified gateway. Its exact proof and
production ingress rules must be the first matching Cloudflare rules, and the
proof binds the stable DNS record's current tunnel and `modified_on` value.

## Cutover

1. Pause DNS, tunnel, and deployment automation for the one Free hostname.
2. Execute this exact compare-before-write route switch once:

   ```bash
   uv run --script tools/verse/switch_sm120_cloudflare_route.py switch \
     --zone-id "$CF_ZONE_ID" \
     --record-id "$CF_FREE_DNS_RECORD_ID" \
     --hostname free-inference.verse-rp.com \
     --expected-current-tunnel "$OLD_TUNNEL_ID" \
     --target-tunnel "$NEW_TUNNEL_ID" \
     --target-mode qualified-candidate \
     --target-release-manifest "$CANDIDATE_RELEASE_MANIFEST" \
     --target-public-proof "$CHANGE_RECORD_DIR/candidate-public-proof.json" \
     --target-proof-zone-id "$CF_ZONE_ID" \
     --target-proof-record-id "$CF_CANDIDATE_PROOF_RECORD_ID" \
     --target-proof-hostname "$CANDIDATE_PROOF_HOSTNAME" \
     --target-proof-account-id "$CF_ACCOUNT_ID" \
     --target-proof-api-token-file "$CF_DNS_READ_TOKEN_FILE" \
     --api-token-file "$CF_DNS_EDIT_TOKEN_FILE" \
     --receipt "$CHANGE_RECORD_DIR/cutover-route.json" \
     --lock "$CHANGE_RECORD_DIR/free-route.lock" \
     --apply
   ```

3. Require the command's verified readback and owner-only receipt. The tool
   reserves a durable pending receipt before mutation, acquires the deployment
   lock, and repeats the exact record and `modified_on` check immediately before
   PATCH. A mismatch aborts without mutation. Cloudflare's DNS record API does
   not expose a server-side compare-and-swap precondition, so no operator or
   external automation may edit this record outside the locked tool.
4. Send one authenticated synthetic canary through the public Verse Free path.
5. Confirm streaming content, `[DONE]`, the expected model alias, and no retry
   to another model.
6. Resume ordinary traffic. Keep the old service running and untouched.

The route change is the only state mutation in this phase. Do not stop, delete,
reconfigure, or reclaim the old service.

## Ten-minute observation window

Observe gateway and candidate telemetry continuously. Roll back immediately if
any of these occurs:

- candidate process or container restart
- OOM, CUDA error, non-finite output, cache corruption, or kernel fallback
- any KV preemption increase
- sustained scheduler growth without drain
- repeated HTTP 5xx, malformed SSE, missing `[DONE]`, or request loss
- loss of the expected model/image/fork identity
- aggregate decode materially below the accepted B01 envelope under a
  comparable workload
- a user-visible regression severe enough that continuing increases impact

Do not diagnose in place while users remain routed to a failing candidate.

## Rollback

1. After cutover, create a fresh rollback proof. A pre-cutover proof is
   intentionally invalid because it cannot bind the new stable-route state:

   ```bash
   uv run --script tools/verse/verify_sm120_public_gateway.py \
     --endpoint "$OLD_ACCESS_ORIGIN" \
     --model verse-free \
     --target-mode recorded-rollback \
     --target-tunnel "$OLD_TUNNEL_ID" \
     --account-id "$CF_ACCOUNT_ID" \
     --zone-id "$CF_ZONE_ID" \
     --record-id "$CF_ROLLBACK_PROOF_RECORD_ID" \
     --stable-record-id "$CF_FREE_DNS_RECORD_ID" \
     --hostname "$ROLLBACK_PROOF_HOSTNAME" \
     --production-hostname "$STABLE_FREE_HOSTNAME" \
     --expected-current-tunnel "$NEW_TUNNEL_ID" \
     --cloudflare-api-token-file "$CF_DNS_READ_TOKEN_FILE" \
     --api-key-file "$OLD_VLLM_API_KEY_FILE" \
     --access-client-id-file "$ACCESS_CLIENT_ID_FILE" \
     --access-client-secret-file "$ACCESS_CLIENT_SECRET_FILE" \
     >"$CHANGE_RECORD_DIR/old-public-proof.json"
   ```

2. Execute the inverse exact-ID switch once:

   ```bash
   uv run --script tools/verse/switch_sm120_cloudflare_route.py switch \
     --zone-id "$CF_ZONE_ID" \
     --record-id "$CF_FREE_DNS_RECORD_ID" \
     --hostname free-inference.verse-rp.com \
     --expected-current-tunnel "$NEW_TUNNEL_ID" \
     --target-tunnel "$OLD_TUNNEL_ID" \
     --target-mode recorded-rollback \
     --target-public-proof "$CHANGE_RECORD_DIR/old-public-proof.json" \
     --target-proof-zone-id "$CF_ZONE_ID" \
     --target-proof-record-id "$CF_ROLLBACK_PROOF_RECORD_ID" \
     --target-proof-hostname "$ROLLBACK_PROOF_HOSTNAME" \
     --target-proof-account-id "$CF_ACCOUNT_ID" \
     --target-proof-api-token-file "$CF_DNS_READ_TOKEN_FILE" \
     --api-token-file "$CF_DNS_EDIT_TOKEN_FILE" \
     --receipt "$CHANGE_RECORD_DIR/rollback-route.json" \
     --lock "$CHANGE_RECORD_DIR/free-route.lock" \
     --apply
   ```

3. Require the command's verified readback and owner-only receipt.
4. Send one authenticated synthetic canary through the public Verse Free path.
5. Confirm the old service receives requests and the candidate drains to zero
   running and waiting requests.
6. Record the trigger, UTC time, route readback, and both service identities.
7. Leave both services and all evidence intact for diagnosis.

Rollback never means changing the candidate image, model tree, launch flags, or
container in place. It means routing back to the already healthy old service.

Before production, drill the same pair of switch commands against a dedicated
non-production proxied CNAME and two disposable tunnel UUIDs. Preserve both
verified receipts as the rollback-drill artifact. A syntax check or dry-run is
not a rollback drill.

## Successful closeout

After at least 24 hours of healthy candidate service:

1. Preserve the old image or artifact identity and its launch definition.
2. Archive the cutover record and qualification manifest without secrets.
3. Stop the old service by its exact provider or container ID.
4. Verify production remains healthy before deleting anything.
5. Delete the old disposable resource only by its exact ID and only after its
   retained artifact can recreate the rollback service.

Model-cache cleanup, old-image deletion, and BoN/ranker work are separate
changes. None belongs in the initial cutover.
