# SM120 Semantic Parity Report

## Scope

This gate isolates the corrected FlashInfer/NVFP4 runtime from a Triton/int4
control on the same dual RTX 5070 Ti host. Both arms used the same Campaign 22
model, prompts, production sampler, seeds, output cap, and GPU generation. The
only intentional runtime difference was the attention/KV implementation.

The packet contained 576 blinded pairs covering continuity, character role,
NPC ownership, scene pivots, long context, mobile formatting, SFW preservation,
and adult/dark tone. Reviewers did not receive the arm key.

## Immutable inputs

- FlashInfer output SHA-256:
  `cc2396696ad3f0cbcba2700dac71732b05ba65ca0c2e9601f5248d49572b46cd`
- Triton output SHA-256:
  `fdf9f1a1bc49e5b59d3ebfc4c24823fef1b750f6fe2c152f9eafff253fc66076`
- Blind packet SHA-256:
  `93bdb7c92fed7db09c30ec283e5cd3a2ea648ee428b79fa57671a62efbd2553e`
- Blind key SHA-256:
  `25a6dbeb58ddb91a964e17130e5e1c72179059316086e962216828f56dd126a9`
- Reviewer 2 assessment SHA-256:
  `c9f4430452d92b05218f12b89f23a2af8e36ecd1ee7e23a11d72d56227b4c844`

## Unblinded results

Reviewer 1 performed an overall comparative review:

- corrected FlashInfer/NVFP4 wins: 184
- Triton/int4 control wins: 181
- equivalent or ordinary stochastic variation: 211
- annotated issue instances: 398 on FlashInfer, 404 on Triton

Reviewer 2 independently annotated hard behavioral failures:

- outputs with at least one hard issue: 231 on FlashInfer, 220 on Triton
- total hard-issue instances: 306 on FlashInfer, 319 on Triton
- factual forgetting or contradiction: 86 on FlashInfer, 88 on Triton
- unexplained character introduction: 4 on FlashInfer, 5 on Triton
- alphabet/token-soup annotations: 62 on FlashInfer, 72 on Triton

Response lengths were also matched: both arms had a median of 65 generated
tokens and a P95 of 86 generated tokens.

## Verdict

The corrected SM120 path passes the semantic parity gate. Neither independent
review shows a directional quality regression. The near-even preference split,
matched continuity rates, and matched response-length distribution are
consistent with ordinary numerical and sampling variance rather than a hidden
fork-quality loss.

This verdict applies only to the corrected native binary and immutable runtime
qualified by the release gates. It does not bless a rebuilt image with a
different native-extension hash, model revision, profile, or sampler.
