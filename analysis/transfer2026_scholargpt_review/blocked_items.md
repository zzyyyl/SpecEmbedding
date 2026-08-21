# ScholarGPT follow-up feasibility record

This record separates reviewer-suggested analyses that can be audited from the
current tracked artifacts from analyses that require unavailable inputs or new
training. It is not a scientific result table.

## Feasible with current artifacts

- A read-only identity audit can run the saved canonical Transformer checkpoint
  on the tracked top-40 rerank caches and recompute ranks under local exact
  SMILES and a 2D InChIKey-prefix identity. Any resulting values are reported
  separately from the local protocol and are not called official-evaluator
  results until the evaluator implementation is verified.
- Cache-only inventories can count identity collisions and multiple-positive
  candidate lists without changing checkpoints or training artifacts.

## Blocked without expanding the project scope

- K=20/80/full-pool sensitivity requires the original full candidate pools and
  labels; truncating the K=40 cache is not an equivalent experiment.
- Parameter-matched pointwise requires a new checkpoint and training run.
- A retriever-agnostic external baseline requires a second retriever and
  matched candidate construction; reported-only JESTR/GLMR numbers are not
  substitutes.
- Query-level paired bootstrap requires per-query predictions for every model;
  the released aggregate tables and seed summaries do not contain them.
- Isolated embeddings-only, base-score-only, and embeddings-plus-score
  rerankers require new training. Existing removal rows are not those
  controlled feature-source experiments.

No blocked item is converted into a number or claim in the manuscript.
