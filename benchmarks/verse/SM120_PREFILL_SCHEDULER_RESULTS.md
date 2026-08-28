# SM120 prefill scheduler sweep

These measurements use the Campaign 22 W4A4 model with NVFP4 KV on one RTX
5070 Ti. Decoder prompts are prewarmed at 4500 tokens and request 1024 output
tokens. Cold prefill prompts contain 6000 tokens and request one output token.
The benchmark proves that decoder and prefill requests overlap, rejects
preemption, and requires an idle scheduler after every run.

## Results

| Token budget | Workload | Baseline decode tok/s | Decode during prefill tok/s | Retention | Prefill wall time | Prefill tok/s | Integrated decoder deficit |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 37 decoders + 1 prefill | 1480.05 | 781.88 | 52.83% | 3.132 s | 1915.44 | 2187 tokens |
| 128 | 30 decoders + 8 prefills | 1314.36 | 633.10 | 48.17% | 23.05 s | 2082.01 | 15703 tokens |
| 256 | 37 decoders + 1 prefill | 1455.68 | 782.17 | 53.73% | 1.406 s | 4267.41 | 947 tokens |
| 256 | 30 decoders + 8 prefills | 1318.72 | 618.11 | 46.87% | 10.256 s | 4679.03 | 7185 tokens |
| 384 | 37 decoders + 1 prefill | 1496.05 | 666.80 | 44.57% | 0.968 s | 6198.96 | 803 tokens |
| 384 | 30 decoders + 8 prefills | 1314.63 | 599.05 | 45.57% | 6.846 s | 7011.04 | 4899 tokens |
| 512 | 37 decoders + 1 prefill | 1491.71 | 615.25 | 41.24% | 0.776 s | 7732.80 | 680 tokens |
| 512 | 30 decoders + 8 prefills | 1318.77 | 534.38 | 40.52% | 5.56 s | 8628.03 | 4361 tokens |
| 768 | 37 decoders + 1 prefill | 1479.60 | 480.88 | 32.50% | 0.694 s | 8644.29 | 693 tokens |
| 768 | 30 decoders + 8 prefills | 1320.80 | 397.05 | 30.06% | 4.927 s | 9741.50 | 4552 tokens |
| 1024 | 37 decoders + 1 prefill | 1471.98 | 363.14 | 24.67% | 0.663 s | 9045.37 | 735 tokens |
| 1024 | 30 decoders + 8 prefills | 1313.74 | 312.88 | 23.82% | 4.59 s | 10444.38 | 4594 tokens |

The integrated decoder deficit is `(baseline decode - mixed decode) * prefill
wall time`. It measures the decoder work displaced over the entire prefill
event, so a small instantaneous decode rate is not rewarded merely for ending
the interference window quickly.

## Provisional decision

512 tokens is the qualified first-release budget. In this single exploratory
sweep it produced the lowest observed integrated decoder deficit for both one
cold arrival and an eight-request prefill storm. The 512-versus-768 margins are
small enough that these measurements do not establish a statistically stable
global optimum. Release evidence therefore proves only that the pinned
512-token profile meets the fixed interference gates on the candidate image;
it does not claim that 512 is universally optimal. Lower budgets preserved
more instantaneous decode but prolonged the measured storm, while higher
budgets shortened the measured prefill window and displaced more decode in
this sweep.
