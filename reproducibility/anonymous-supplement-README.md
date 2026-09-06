# Anonymous source supplement

This archive contains the source code needed to inspect the spectrum--molecule
alignment and offline candidate-reranking pipeline. It was assembled from an
explicit allowlist and intentionally contains no Git history, paper sources,
raw spectra, candidate pools, model checkpoints, rerank caches, experiment
logs, or internal artifact inventories.

## Environment

The human-readable environment recipe is `environment.yml`. Linux x86-64 users
can instead reconstruct the audited post-hoc validation environment from the
explicit Conda and exact-version PyPI locks under `reproducibility/`:

```bash
conda create -n specembedding-anonymous --file reproducibility/conda-linux-64.lock
conda run -n specembedding-anonymous python -m pip install \
  -r reproducibility/pip-linux-64.lock
```

The locks reproduce the validation environment, not the exact historical
training environment. The PyPI lock fixes versions but does not contain wheel
hashes.

## Validation

Run the source-only unit tests from the extracted archive root:

```bash
conda run -n specembedding-anonymous python -m pytest -q
```

Then run the self-contained CPU smoke test:

```bash
conda run -n specembedding-anonymous \
  python reproducibility/run_anonymous_smoke.py
```

The smoke test creates tiny deterministic MatchMS spectra and candidate sets,
instantiates a reduced randomly initialized alignment model, and runs the
packaged `prepare_rerank_cache.py` -> `train_rerank.py` -> `eval_rerank.py`
entry points on CPU. It validates spectrum and molecule encoding, cache
serialization, reranker training, checkpoint compatibility, and metric-only
evaluation from a working directory outside the extracted package. It requires
no external data, GPU, Git metadata, or network access. The random alignment
checkpoint and one-epoch tiny relative reranker are execution fixtures only: this
smoke does not reproduce or support any reported scientific metric.

## Current method and real-data entry points

The current method is a rank-free, non-generative relative reranker. It uses a
no-forcing candidate cache, pointwise coarse scoring over the supplied pool,
and a spectrum-conditioned relative branch on the coarse top-40. The default
configuration is `model_type=relative`, `train_k=256`, and
`use_rank_embedding=false`. The synthetic smoke test exercises this current
relative path; it is an executable integration check, not a paper experiment.

`run_rerank_pipeline.py` connects offline cache preparation, reranker training,
and evaluation around a frozen alignment checkpoint. Its default is
no-forcing. The legacy train-cache behavior can be selected explicitly with
`--force-include-positive`; validation and test caches never force positives.
Real runs require separately supplied datasets, candidate files, and
checkpoints; none are bundled in this anonymous archive.

The source defaults do not certify the protocol of an existing checkpoint.
Reported top-40 runs used limited training queries and training-positive
forcing; the separate top-256 no-forcing pilot has a different scope. Inspect
the run's configuration and cache metadata before interpreting its metrics.
Saved-embedding candidate/identity audits do not reproduce the full official
loader. The reranker remains closed-library, and `formula` assumes a known
molecular formula. Formal training uses all eligible training samples and an
explicit CUDA device; the CPU path above is only a synthetic integration test.

`ANONYMOUS_MANIFEST.json` records the byte size, SHA-256 digest, and mode of
every archive member. `SOURCE_ALLOWLIST.txt` records the exact source-file
selection used to build the archive.
