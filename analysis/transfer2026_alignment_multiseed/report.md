# Cross-alignment multi-seed analysis

This report is generated from the three original per-run `summary.csv` files.
Reranker-seed combinations are paired within each alignment checkpoint; cross-alignment
statistics use the three alignment-level estimates rather than flattening nine runs.

## Provenance

| Alignment seed | Source commit | Params SHA256 | Checkpoint SHA256 |
| ---: | --- | --- | --- |
| 42 | `a2280d2828ce872da1f69319b49e0ef7f1bed572` | `b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6` | `ad5d1eb76805c51563349f259a4b4c935336064b6171a4e650472b78aeeaa06f` |
| 43 | `28bdce232402cea3d656c790b12a9f32d1cc5382` | `b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6` | `898966d71ed28da7f4716f3c2b699128638506f3d6f710d20a6af1a0dfab2f8b` |
| 44 | `28bdce232402cea3d656c790b12a9f32d1cc5382` | `b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6` | `d007f832cfde8c02f61c053ee2b4098664c631868571f23ee83a96e107935a5c` |

## Alignment-level results

Top-1 differences are percentage points; MRR differences use the raw [0, 1] scale.

| Alignment | Candidate | Base Top-1 | Pointwise Top-1 | Transformer Top-1 | ΔTop-1 | Pointwise MRR | Transformer MRR | ΔMRR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | mass | 47.4596 | 67.4432 | 68.5824 | 1.1392 | 0.717900 | 0.725967 | 0.008067 |
| 42 | formula | 63.1009 | 73.9709 | 74.5671 | 0.5962 | 0.776367 | 0.780167 | 0.003800 |
| 43 | mass | 35.9193 | 66.9211 | 67.1888 | 0.2677 | 0.704200 | 0.705833 | 0.001633 |
| 43 | formula | 45.0786 | 69.6400 | 69.3021 | -0.3379 | 0.721433 | 0.719600 | -0.001833 |
| 44 | mass | 23.9121 | 57.7827 | 57.4239 | -0.3588 | 0.612167 | 0.609067 | -0.003100 |
| 44 | formula | 28.7309 | 59.7574 | 59.8162 | 0.0589 | 0.631367 | 0.633433 | 0.002067 |

## Cross-alignment descriptive summary

The mean and sample standard deviation below use the three alignment-level
paired estimates. They are descriptive statistics, not confidence intervals
or statistical-significance tests.

| Candidate | Mean ΔTop-1 | SD ΔTop-1 | Mean ΔMRR | SD ΔMRR | Positive alignments (Top-1 / MRR) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mass | 0.349344 | 0.752349 | 0.002200 | 0.005605 | 2/3 / 2/3 |
| formula | 0.105711 | 0.468825 | 0.001344 | 0.002885 | 2/3 / 2/3 |

## Claim gate

- Status: `fail`.
- Fine-grained pair directions (positive / zero / negative): Top-1 10/0/8; MRR 11/0/7.
- Decision: do not claim that Set Transformer consistently outperforms Pointwise across alignment checkpoints.
- Residual-vs-base descriptive gate: `pass_descriptive` (12/12 aggregate cells improve both Top-1 and MRR).
- Supported boundary: supervised residual reranking improves over the base retriever in these internal descriptive results, while the independent benefit of candidate interaction remains unestablished.
- The three alignments were produced by 2 source commit(s); results retain per-alignment provenance.
- No confidence intervals, hypothesis tests, cross-dataset evidence, or causal claims are provided by this analysis.
