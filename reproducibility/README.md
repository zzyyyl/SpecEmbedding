# Reproducibility snapshot

This directory records the post-hoc validation environment used to audit the
current repository. It does not claim to reconstruct the historical training
environment bit for bit.

For Linux x86-64, create the exact Conda layer first, then install the locked
PyPI layer:

```bash
conda create -n specembedding-repro --file reproducibility/conda-linux-64.lock
conda run -n specembedding-repro python -m pip install \
  -r reproducibility/pip-linux-64.lock
```

`environment.yml` remains the human-maintained, directly pinned environment
recipe. `runtime-snapshot.yaml` records the audited host capabilities, package
versions, validation commands, and lock-file digests without usernames,
hostnames, absolute local paths, or GPU UUIDs.

To validate both lock layers in a fresh environment, run:

```bash
bash reproducibility/validate_linux_lock.sh <unused-environment-name>
```

## Mentor-review experiment index

`mentor2026_experiment_index.json` records the repository-relative paths,
source commits, resolved run metadata, SHA-256 digests, and local availability
of the 36 pointwise/relative main runs and 30 relative ablation runs used for
the 2026-08-23 paper revision. The corresponding ignored checkpoint, cache,
training-log, and evaluation-log files remain local artifacts and must be
copied into the controlled submission archive before release.

The tracked `analysis/mentor2026_experiments/` files contain the alignment
summary, ablation summary, candidate-cutoff results, forward-only reference,
and selected-query hard-case ranking outputs. The hard-case JSON files are
ranking/prediction outputs for alignment-42 seed-42 pointwise and relative
runs; they are not a claim that every ignored checkpoint has a full prediction
dump.
