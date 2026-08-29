# Working Context

This analysis was prepared from the local source tree at
`/Users/mtbl23/.codex/worktrees/vllm-verse-sm120` on revision
`30aac553edceddcaeec8a028b2a56be3ddf2a1c2`.

## Source inventory

| Evidence ID | Artifact | SHA-256 |
| --- | --- | --- |
| `runtime-launcher` | `tools/verse/run_sm120_server.sh` | `8a7ce0963f45451e8581cb8f3f909c40089ef4a50a3214c51bccca91ecbae4e0` |
| `local-gateway-launcher` | `tools/verse/run_sm120_gateway.sh` | `f2088364feec0af4ef0068ccbe3e080880176b7946637f19abe711d1d5e2f8b1` |
| `local-gateway-policy` | `tools/verse/verse-sm120-gateway.Caddyfile` | `45e1275ab57c2a6bb3fa83aac9efdc5aa6799f78ebcf266e907bb3120d58854e` |
| `current-cutover` | `tools/verse/SM120_CUTOVER_RUNBOOK.md` | `dcb8abeb6426329fe56b661d1093fc332ad566b1c46fc20f73a5ad5df3339b10` |
| `runtime-entrypoint` | `docker/entrypoints/verse-sm120-entrypoint.sh` | `80ce712637dbfae3c96218d0f70ba154237db3c6d74d6b5161a1a5736e799708` |
| `runtime-image` | `docker/Dockerfile` | `a60fb061fd2463e15fe3025ef59b179e9c6618b532e28eabf2a42ce45638f633` |
| `vllm-security` | `docs/usage/security.md` | `a89426d6868893c3679e5444d368876f013c12c4a87183ca0eac852edfe65b35` |
| `salad-networking` | <https://docs.salad.com/container-engine/explanation/infrastructure-platform/networking> | Current external document |
| `salad-workload-jwt` | <https://docs.salad.com/container-engine/tutorials/security/jwt-authentication> | Current external document |
| `cloudflare-tunnel-token` | <https://developers.cloudflare.com/tunnel/advanced/tunnel-tokens/> | Current external document |
| `cloudflare-tunnel` | <https://developers.cloudflare.com/tunnel/> | Current external document |

The normalized inventory digest is
`4ce00c3db75f8a76e4a9eb84e6bde83726cdf6115f7604f20227df2df43c7bba`.

## Supplied deployment constraints

- The provider is expected to be SaladCloud at approximately $0.10 per RTX
  5070 Ti hour.
- The physical machines belong to consumer gaming users and are not a trusted
  administrative boundary.
- The initial service cohort contains two logical workers.
- Salad replaces a lost worker with an equivalent worker at the same rate,
  normally within 10 minutes and at worst approximately 20 minutes.
- During replacement, all admissible traffic may move to the surviving worker.
- The runtime capacity proven elsewhere is 38 active 6,144-token requests per
  RTX 5070 Ti. Failover must queue above that limit rather than over-admit.
- Availability orchestration is not the immediate implementation target. This
  analysis records the two-worker behavior so security choices do not prevent
  it later.

## Important evidentiary boundary

This is a source and architecture review. It does not claim that Salad host
isolation, provider failover timing, worker JWT validation, or the proposed
broker has been exercised. Consumer-host confidentiality is not available from
ordinary container isolation: a machine administrator can observe plaintext
needed by inference and can copy model weights after they are loaded.
