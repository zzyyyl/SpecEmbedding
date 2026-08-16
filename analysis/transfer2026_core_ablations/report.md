# Core component ablation analysis

This report compares the frozen full Set Transformer with five pre-registered
component removals at canonical alignment seed 42. Reranker seeds are paired
within candidate protocol; statistics are descriptive sample means/SDs (n=3).

Delta is defined as `full - ablation`: positive values favor the full model.
Top-1 differences are percentage points; MRR uses the raw [0, 1] scale.

## Provenance and compatibility

- Full source commit: `a2280d2828ce872da1f69319b49e0ef7f1bed572`.
- Ablation source commit: `8bd2ca439bc972c0ada88b6ad8b746372040df14`.
- Shared params SHA-256: `b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6`.
- Execution-relevant changed paths between source commits: `[]`.
- Both runs use the same canonical top-40 caches, validation exclusions,
  selection metric, training budget, and 17,556-query test evaluation.
- The final supervisor batch reports 29 reused completed attempts (`skipped`) and 1 completed in that invocation; every selected attempt status is `complete`.

## Paired component results

| Candidate | Removal | Full Top-1 | Removal Top-1 | ΔTop-1 (P/Z/N) | Full MRR | Removal MRR | ΔMRR (P/Z/N) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mass | no_residual | 68.5824 ± 0.2035 | 68.1609 ± 0.0609 | 0.4215 ± 0.1882 (3/0/0) | 0.725967 ± 0.002499 | 0.723267 ± 0.000635 | 0.002700 ± 0.003005 (2/0/1) |
| mass | no_base_score | 68.5824 ± 0.2035 | 68.0660 ± 0.2006 | 0.5164 ± 0.3982 (3/0/0) | 0.725967 ± 0.002499 | 0.722800 ± 0.001808 | 0.003167 ± 0.003402 (2/0/1) |
| mass | no_rank_embedding | 68.5824 ± 0.2035 | 70.1204 ± 0.3019 | -1.5379 ± 0.4846 (0/0/3) | 0.725967 ± 0.002499 | 0.737467 ± 0.002818 | -0.011500 ± 0.005311 (0/0/3) |
| mass | no_interaction_features | 68.5824 ± 0.2035 | 68.0983 ± 0.8464 | 0.4841 ± 0.6522 (3/0/0) | 0.725967 ± 0.002499 | 0.722867 ± 0.006646 | 0.003100 ± 0.005147 (2/0/1) |
| mass | listwise_only | 68.5824 ± 0.2035 | 68.9071 ± 0.7553 | -0.3247 ± 0.9008 (1/0/2) | 0.725967 ± 0.002499 | 0.728633 ± 0.004594 | -0.002667 ± 0.005493 (1/0/2) |
| formula | no_residual | 74.5671 ± 0.2819 | 75.0057 ± 0.5433 | -0.4386 ± 0.2970 (0/0/3) | 0.780167 ± 0.002554 | 0.783800 ± 0.004223 | -0.003633 ± 0.001701 (0/0/3) |
| formula | no_base_score | 74.5671 ± 0.2819 | 74.3943 ± 0.5501 | 0.1728 ± 0.7115 (1/0/2) | 0.780167 ± 0.002554 | 0.779333 ± 0.003787 | 0.000833 ± 0.005501 (1/0/2) |
| formula | no_rank_embedding | 74.5671 ± 0.2819 | 76.6177 ± 0.6918 | -2.0506 ± 0.4485 (0/0/3) | 0.780167 ± 0.002554 | 0.793767 ± 0.005784 | -0.013600 ± 0.003928 (0/0/3) |
| formula | no_interaction_features | 74.5671 ± 0.2819 | 75.7557 ± 0.5506 | -1.1886 ± 0.8306 (0/0/3) | 0.780167 ± 0.002554 | 0.789000 ± 0.003831 | -0.008833 ± 0.006296 (0/0/3) |
| formula | listwise_only | 74.5671 ± 0.2819 | 74.2386 ± 0.2456 | 0.3285 ± 0.3434 (3/0/0) | 0.780167 ± 0.002554 | 0.777100 ± 0.002464 | 0.003067 ± 0.004102 (2/0/1) |

`P/Z/N` counts positive/zero/negative full-minus-ablation deltas across
reranker seeds 42/43/44.

## Component claim gate

- Overall status: `component_claims_require_narrowing`; 0/5 components pass the strict descriptive gate.
- `no_residual` (residual base-score shortcut): `candidate_dependent`. Only the residual base-score addition is removed; the base score remains an MLP input.
- `no_base_score` (joint base-score feature and residual pathway): `mean_favors_full_seed_mixed`. Both the base-score MLP feature and residual pathway are removed, so this is not a single-factor feature ablation.
- `no_rank_embedding` (rank embedding): `removal_consistently_better`. Only the rank embedding is zeroed at the fixed MLP width.
- `no_interaction_features` (product and absolute-difference features): `candidate_dependent`. Product and absolute-difference features are removed; candidate self-attention remains enabled.
- `listwise_only` (pairwise loss term): `candidate_dependent`. The pairwise loss weight is set to zero; the model architecture is unchanged.

The rank embedding is not supported under this frozen protocol: removing it improves both Top-1 and MRR in all six candidate/seed cells.
Candidate-dependent components under the frozen gate: residual base-score shortcut, product and absolute-difference features, pairwise loss term.
Removing the joint base-score pathway lowers the mean MRR in both protocols, but paired-seed directions are mixed.

These results do not justify selecting a new post-hoc main model or claiming
statistical significance, causal component effects, cross-dataset generalization,
or cross-alignment component stability. The supported paper-level boundary remains
supervised non-generative second-stage reranking; individual component claims must
be narrowed.
