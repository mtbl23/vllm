# Security Hardening Review: Verse SM120 on SaladCloud

## Evidence Basis

I inspected the fixed SM120 image, launcher, loopback gateway, cutover runbook,
and vLLM security guidance at revision
`30aac553edceddcaeec8a028b2a56be3ddf2a1c2`. I also reviewed SaladCloud's
documented outbound networking and five-minute workload JWT, plus Cloudflare's
tunnel credential model. The existing appliance is careful about an owned host,
but its long-lived local secret and per-host tunnel pattern should not be copied
unchanged to consumer-owned Salad nodes.

## Constraints

We want two inexpensive RTX 5070 Ti workers, 38 active slots each, and rapid
provider replacement. We do not assume a gaming-PC administrator, kernel, or
container runtime is confidential. We must preserve private request handling as
far as this provider model permits, but we cannot promise confidentiality from
the machine owner without confidential-computing hardware. Production remains
untouched while this design is evaluated.

## Opportunity Portfolio

| Opportunity | Evidence | Options | Recommendation | Proposal |
| --- | --- | --- | --- | --- |
| Move worker trust and routing into a Verse-owned outbound broker | Current host-bound Cloudflare and bearer credentials; Salad outbound networking and five-minute workload JWT | Existing tunnel topology; ephemeral per-worker tunnels; outbound broker with signed leases | Use the outbound broker and Salad workload identity | [Full proposal](proposals/outbound-worker-broker.md) |

## Recommendation Summary

I recommend that each worker obtain Salad's short-lived workload JWT and open
an outbound authenticated connection to a Verse-owned broker. The broker should
validate the Salad signature and exact organization, project, container-group,
instance, and machine claims before admitting the worker. It should then issue
short request leases instead of giving the worker a Verse API key, Cloudflare
tunnel token, database credential, or Hugging Face token.

This design is also the cleanest fit for two logical workers. The broker owns
session affinity for KV reuse, caps each worker at its measured 38 active
requests, and queues excess traffic. If one worker disappears, the broker stops
leasing to it and sends admissible work to the survivor. A replacement joins in
`warming` state and receives traffic only after identity, model digest, kernel,
health, and synthetic inference checks pass.

We should be candid about the residual risk. TLS, short-lived identity, and
container hardening protect against network attackers and credential reuse.
They do not prevent the owner of a Salad machine from observing prompts and
outputs in host memory or copying the loaded model. If that residual risk is
unacceptable for private user content, the correct control is a trusted or
confidential GPU provider, not another credential layer inside the same host.

## Next Decisions

1. Confirm that consumer-host plaintext exposure is an accepted provider risk
   for Free-model traffic.
2. Select the outbound broker option for implementation planning.
3. Decide whether the broker runs in the existing Verse backend or as a small,
   separately scaled service.
4. Define the exact Salad organization, project, and container-group claims
   that may register as production workers.
5. Preserve the current direct Cloudflare appliance path only for owned hosts,
   such as Ultra, rather than forcing both trust models into one deployment.
