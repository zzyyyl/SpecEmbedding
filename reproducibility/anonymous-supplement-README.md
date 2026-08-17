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
checkpoint and one-epoch tiny Transformer are execution fixtures only: this
smoke does not reproduce or support any reported scientific metric.

## Real-data entry points

The orchestration entry point is `run_rerank_pipeline.py`; it connects offline
cache preparation, reranker training, and evaluation around a frozen alignment
checkpoint. Real runs require separately supplied datasets, candidate files,
and checkpoints; none are bundled in this anonymous archive. Alignment model
definitions are included so the cache-preparation path can be inspected, but
the historical training runs and their artifacts are intentionally absent.

Recall and MRR in the accompanying study use a local exact-target-SMILES
single-positive rule over supplied candidate files. This is not equivalent to
the reference evaluator's default two-dimensional InChIKey equivalence and
possible multiple positives. External reported baselines are therefore not
directly comparable to the local results.

`ANONYMOUS_MANIFEST.json` records the byte size, SHA-256 digest, and mode of
every archive member. `SOURCE_ALLOWLIST.txt` records the exact source-file
selection used to build the archive.
