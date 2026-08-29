# Security Hardening Proposal: Outbound Salad Worker Broker

## Decision

Choose how Verse should authenticate, route to, and revoke inference workers
running on consumer-owned SaladCloud machines.

## Executive Recommendation

I recommend an outbound signed-lease broker. Each worker retrieves Salad's
five-minute workload JWT from local IMDS and connects outbound to a Verse-owned
broker. The broker validates the JWT signature and exact production claims,
then leases one bounded request at a time over that connection. The worker never
receives a Cloudflare tunnel token, Verse master API key, database credential,
Hugging Face token, or provider-management key.

The same broker should own per-chat affinity, the measured 38-slot admission
cap, queueing, worker health, and replacement warmup. That gives the security
boundary and the two-worker availability design one source of truth. It also
prevents a load balancer from randomly moving a warm chat between GPUs and
turning a recoverable worker loss into persistent prefix-cache churn.

## Evidence

The existing runtime already provides a strong inner boundary. The launcher
binds vLLM to loopback, runs it non-root, drops capabilities, mounts immutable
weights, and requires an owner-only API key. The Caddy gateway exposes only the
four required routes and discards access logs. Those controls should remain.

What changes is the host assumption. The current cutover design gives an owned
host a tunnel identity and a long-lived application credential. Cloudflare
explicitly warns that anyone with a remotely managed tunnel token can run the
tunnel. Salad, by contrast, documents a five-minute node JWT obtained from IMDS
with organization, project, container-group, instance, and machine claims. That
is a better bootstrap identity because it can be verified centrally without
preloading a permanent Verse secret.

We should also preserve vLLM's warning that internal distributed services are
not authenticated or encrypted by default. The production profile is one GPU
and does not need cross-host tensor, pipeline, data-parallel, or KV-transfer
ports. None should be enabled on Salad.

## Current Design And Failure Mode

The owned-host appliance assumes that root and the local secret store belong to
Verse. That is reasonable for Ultra or a controlled cloud VM. It is not a sound
confidentiality assumption for a gaming user's PC. If we copy the design
directly, the machine owner can read the tunnel token and vLLM key, impersonate
the worker while those credentials remain valid, inspect prompt plaintext, and
copy model weights.

Shortening a tunnel token reduces the impersonation window but does not settle
who owns routing, admission, or chat affinity. Salad's group gateway load
balances across replicas, while a long-running RP chat benefits from returning
to the same GPU's prefix cache. A separate backend semaphore plus provider load
balancer can also disagree about whether a worker is full or healthy.

The structural issue is therefore broader than secret storage: Verse needs one
trusted component that owns worker identity, request authorization, capacity,
affinity, failure, and replacement.

## Desired Invariants

- Workers initiate outbound connections and expose no public inference,
  metrics, profiler, SSH, or management port.
- The broker validates the Salad JWT signature and exact organization, project,
  container-group, instance, and machine claims.
- Authentication is refreshed before the five-minute Salad identity expires.
- A request lease binds request ID, chat affinity key, worker session, model
  digest, prompt digest, maximum input and output size, nonce, and expiry.
- A lease can complete once. Retry or reassignment invalidates the old lease.
- The broker admits at most 38 active requests per 5070 Ti and queues the rest.
- Prompt and output bytes are streamed with backpressure and never logged.
- Replacement nodes remain `starting` or `warming` until exact image, model,
  kernel, health, and synthetic generation checks pass.
- Losing one worker cannot cause the survivor to exceed 38 active requests.
- A worker has no access to the database, object store, payment system,
  Cloudflare control plane, or provider control plane.

## Constraints And Non-Goals

The first cohort is two logical workers. Salad is expected to recreate a lost
worker in about 10 minutes and at worst about 20 minutes. We do not need to
solve provider placement or recreate the physical GPU ourselves.

The broker does not make a consumer machine confidential. Inference requires
plaintext on that host. A machine administrator can inspect memory, instrument
the process, alter outputs, or copy loaded weights. The proposed controls limit
network exposure, credential reuse, blast radius, and routing ambiguity. They
cannot provide hardware attestation or encrypted execution where the provider
does not offer it.

BoN, ranker behavior, speculative decoding, and model-quality decisions are
outside this proposal. The protocol should carry a parent request cleanly so
those can be added later without changing the trust boundary.

## Before Architecture

The [before diagram](../diagrams/outbound-worker-broker-before.mmd) shows the
owned-host topology being copied onto consumer machines. Each worker owns a
public routing identity and reusable credentials. Cloudflare may protect the
network path, but the worker host remains able to steal those identities, and
routing is separate from backend admission and KV affinity.

## Options

### Option 1: Reuse One Cloudflare Tunnel Per Worker

This option preserves nearly all of the current appliance. vLLM remains on
loopback behind the local allowlisting gateway, and each worker runs its own
Cloudflare connector. It is attractive because the image and runbook already
understand this path, so the first canary can be assembled quickly.

The principal concern is that a remotely managed tunnel token is itself the
ability to run the connector. A host administrator can copy it. The vLLM bearer
key is also readable by root. Revocation can contain the event, but every worker
adds secret lifecycle and public-route state. This option also leaves affinity
and capacity split between Cloudflare, Salad, and Verse.

I would use this only for a short non-production qualification or for machines
Verse owns. I would not use it as the durable consumer-node design.

### Option 2: Issue Ephemeral Per-Worker Tunnel Credentials

An identity broker can validate the Salad JWT and issue temporary routing
credentials. This materially shortens the useful lifetime of stolen secrets and
automates replacement. It also preserves the ordinary inbound HTTP flow, which
makes migration easier.

What gives me pause is that we would add a broker without moving routing into
it. The worker still receives a credential, and the provider gateway or
Cloudflare still chooses a replica separately from Verse's queue and KV
affinity. We would own two overlapping control planes and need to debug their
interaction during the exact failure case this design is meant to simplify.

This option becomes preferable only if changing backend dispatch is currently
impossible and the deployment must remain inbound HTTP.

### Option 3: Use An Outbound Signed-Lease Broker

The worker starts with no Verse secret. It retrieves a short Salad workload JWT
from IMDS and opens a TLS connection to the broker. The broker validates the
signature through Salad's JWKS and rejects any identity outside the exact
production organization, project, and container group. The worker periodically
reauthenticates before the five-minute JWT expires.

After warmup, the broker assigns bounded leases over the connection. A lease is
not a general API key: it names one request, one worker session, one model
digest, one nonce, one expiry, and explicit input/output limits. The response is
accepted only against that lease. If the connection dies and a request is
reassigned, the old lease is revoked so a delayed or malicious response cannot
win a race.

This option has the best security structure because the worker exposes nothing
inbound and holds no reusable central credential. It also gives us the best
reliability structure. The broker already knows that each worker has 38 slots,
so it can preserve chat affinity while healthy, queue over capacity, stop
leasing immediately on failure, and admit a replacement only after warmup.

The cost is real engineering work. The broker must be replicated, streaming
must apply backpressure rather than buffer prompts and outputs, and lease state
must remain correct during broker failover. The protocol needs explicit replay,
expiry, cancellation, and duplicate-completion tests. This is still a bounded
Verse-specific service rather than a vLLM fork feature; vLLM remains an inner
loopback engine behind the worker sidecar.

I recommend this option under the current constraints.

## Option Delta

| Dimension | Existing tunnels | Ephemeral tunnels | Outbound lease broker |
| --- | --- | --- | --- |
| Public worker ingress | Through Cloudflare | Through Cloudflare | None |
| Reusable central secret on worker | Yes | Temporarily | No |
| Native Salad identity | Optional | Bootstrap | Required and continuously refreshed |
| 38-slot admission owner | Split | Split | Broker |
| Per-chat KV affinity | Additional layer | Additional layer | Broker-native |
| Worker replacement | Secret and route lifecycle | Broker plus route lifecycle | Reauthenticate and warm |
| Consumer-host plaintext risk | Remains | Remains | Remains |
| Migration effort | Low | Medium | High |

## Availability And Failure Semantics

Both logical workers are normally `ready`, and the broker splits chats while
keeping each chat sticky to its assigned worker. The assignment is a cache
optimization, not durable state: the backend retains the canonical request and
can replay it after failure.

When worker A disconnects or fails a strict health deadline, the broker marks it
`unhealthy`, revokes its active leases, and stops assigning it work. Worker B
continues up to 38 active requests. Additional requests remain queued rather
than forcing B beyond the measured capacity. In-flight work from A may be
retried on B with an explicit new lease; this costs prefill and latency but does
not risk duplicate acceptance.

When Salad creates replacement A2, A2 authenticates as a new instance and
machine. It enters `starting`, proves the immutable image and model identity,
then enters `warming` for kernel and synthetic inference gates. Only then does
the broker mark it `ready` and gradually restore ordinary traffic. We should not
reuse the previous worker session merely because the container-group identity
matches.

The probability of both nodes disappearing together can be low without being a
security assumption. If both are unavailable, requests queue or fail cleanly;
the broker never routes to an unverified replacement.

## Rollout And Rollback

We should first deploy the broker and one worker against synthetic traffic only.
The worker image must have no Cloudflare, Hugging Face, database, or provider
management secret. We then exercise expired and wrong-group Salad identities,
nonce replay, request tampering, cancellation, connection loss, delayed old
responses, content-free logs, and 38-slot backpressure.

Next, run two disposable workers and repeatedly kill either one while 76 active
plus queued synthetic streams are present. We should prove that the survivor
never exceeds 38, no response is accepted twice, the replacement receives no
traffic before warmup, and per-chat affinity returns after recovery.

Canary production only after those gates pass. Keep the current Free provider
healthy as rollback. Rollback is a backend route change that stops issuing
leases and sends Free traffic to the old provider. No worker credential needs
to survive the rollback.

## Residual Risk

Consumer-host access to prompts, outputs, and weights remains. A compromised
worker can return plausible but wrong text, selectively fail requests, or leak
content through a covert outbound channel if provider egress cannot be
restricted. We can detect some integrity failures with canaries, model identity
checks, timing and health anomalies, and limited retry comparisons, but we
cannot prove arbitrary generated text was honestly computed.

If the content confidentiality requirement excludes consumer-host visibility,
we should not route that content to Salad gaming PCs. The appropriate alternative
is an owned host or a provider offering a trusted confidential-computing
boundary.

## Validation Plan

1. Validate Salad JWTs against the documented JWKS and reject wrong issuer,
   audience, expiry, organization, project, group, instance, and machine.
2. Reauthenticate active worker sessions before every five-minute identity
   expiry.
3. Fuzz lease framing and test tampering, replay, cancellation, timeout, stale
   completion, duplicate completion, and broker restart.
4. Prove the production image exposes no listening public socket and contains
   no permanent secret.
5. Scan logs and telemetry for prompt, output, persona, character-card, bearer,
   JWT, and request-body material.
6. Load test 38 active requests per worker plus queue pressure with streaming
   backpressure.
7. Kill either worker repeatedly and verify the survivor is capped at 38 while
   the replacement completes all warmup gates.
8. Verify the backend can disable one worker or the whole Salad cohort without
   changing provider credentials.

## Implementation Handoff

Implementation planning should begin only after the outbound broker option and
the consumer-host confidentiality residual risk are explicitly accepted. The
first work packages are protocol definition, broker ownership, Salad JWT
validation, worker sidecar isolation, and the two-worker failure-state test
harness. The vLLM kernel fork itself should not gain provider credentials or
public networking responsibilities.
