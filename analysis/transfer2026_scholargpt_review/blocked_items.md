# ScholarGPT follow-up feasibility record

This record separates reviewer-suggested analyses completed with the local
artifacts from comparisons that remain outside the current evidence boundary.
It is not a scientific result table.

## Completed in the second-stage pilot

- Same-cache K=20/40/80 evaluation-only sensitivity was recorded from the
  256-candidate test cache; it is not K-specific retraining.
- Capacity-matched no-rank pointwise was trained with the same 5,000-query,
  three-epoch pilot configuration and seeds 42--44.
- Query-level paired predictions and deterministic bootstrap summaries were
  saved for mass/formula relative and pointwise models.
- Score-only, embedding-only, embedding+base, candidate-only,
  no-spectrum-conditioning, no-molecular-relation, and no-antisymmetric controls
  were trained once per pool and evaluated on the same test cache.
- The new relative full-pool cache was audited under the MassSpecGym 1.3.1
  2D-InChIKey transform; neither saved test cache contained multiple positives
  or candidate identity collisions.
- RTX 4090 forward-only latency and peak allocation were measured for the
  primary relative/pointwise pair.

## Remaining outside scope

- A retriever-agnostic external baseline requires a second retriever and matched
  candidate construction; reported-only JESTR/GLMR numbers are not substitutes.
- A full official-loader rerun requires the official data/loader environment;
  the current identity result is a cache-level audit.
- Candidate-aware alignment retraining and alignment-level confidence estimates
  were not added to this pilot.

No external-baseline or full-loader claim is converted into a number in the
manuscript.
