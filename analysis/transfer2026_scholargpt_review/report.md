# ScholarGPT identity-rule audit

## Scope

This is an evaluation-only audit of the saved canonical Transformer seed-42
reranker checkpoints on the tracked overlap-clean top-40 test caches. It does
not retrain a model, reconstruct the full candidate pool, or invoke the
MassSpecGym official loader. The reference protocol tested here is the first
block of the RDKit-generated 2D InChIKey, with all candidates sharing that key
treated as positive. It is therefore a reference-identity audit, not an
official-evaluator-equivalent rerun.

Command:

```text
PYTHONPATH=. conda run -n specembedding python \
  analysis/transfer2026_scholargpt_review/official_identity_eval.py \
  --output analysis/transfer2026_scholargpt_review/official_identity_eval.json \
  --device cuda:0 --batch-size 64
```

Environment: PyTorch 2.5.1, CUDA-enabled NVIDIA GeForce RTX 4090. The run was
executed outside the restricted container in a detached `tmux` session.

## Results

| pool | protocol | base R@1 | rerank R@1 | base MRR | rerank MRR | positive queries | multiple-positive queries | candidate collisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mass | local exact SMILES | 47.4596% | 68.7400% | 55.0349% | 72.8710% | 14,726/17,556 | 0 | 0 |
| mass | 2D InChIKey prefix | 47.4596% | 68.7400% | 55.0349% | 72.8710% | 14,726/17,556 | 0 | 0 |
| formula | local exact SMILES | 63.1009% | 74.7494% | 68.8556% | 78.2249% | 15,670/17,556 | 0 | 0 |
| formula | 2D InChIKey prefix | 63.1009% | 74.7494% | 68.8556% | 78.2249% | 15,670/17,556 | 0 | 0 |

The corresponding R@5/R@20 values are mass 63.5111%/77.7626% for base and
77.7740%/82.1770% for rerank; formula 75.7861%/84.9339% for base and
82.2511%/86.8934% for rerank. These are single-checkpoint seed-42 audit
values, not replacements for the three-seed main tables.

Within the stored top-40 lists, no query had more than one candidate matching
the tested 2D identity, and no top-rank metric changed between the two identity
rules. The reference-positive count is lower than the total query count when
the equivalent identity is absent from the stored top-40 list; this is a
coverage observation, not evidence that the full official candidate pool has
the same multiplicity structure.

Artifacts:

- `official_identity_eval.py` — evaluator implementation;
- `official_identity_eval.json` — raw output;
- `identity_inventory_report.md` — cache-only inventory;
- `blocked_items.md` — analyses requiring unavailable inputs or new training.
