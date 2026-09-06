# Reproducibility and release records

This directory records a post-hoc validation environment and versioned artifact
inventories. These snapshots do not reconstruct every historical training
environment or validate the latest source automatically.

## Environment

[environment.yml](../environment.yml) is the maintained recipe. Linux x86-64
users can reconstruct the recorded Conda layer and version-pinned PyPI layer:

```bash
conda create -n specembedding-repro --file reproducibility/conda-linux-64.lock
conda run -n specembedding-repro python -m pip install \
  -r reproducibility/pip-linux-64.lock
bash reproducibility/validate_linux_lock.sh <unused-environment-name>
```

The final command independently validates both layers in a new environment.
[runtime-snapshot.yaml](runtime-snapshot.yaml) records the audited versions,
capabilities, commands and lock digests. The PyPI lock has versions, not wheel hashes.

## Experiment evidence

[mentor2026_experiment_index.json](mentor2026_experiment_index.json) records source
commits, commands, local availability and hashes for 36 pointwise/relative runs
and 30 relative ablations. These runs used top-40 caches and a 20,000-query cap;
training caches forced positives, while validation/test caches did not.
They are distinct from the top-256 no-forcing pilot and pending full-training runs.

The [analysis index](../analysis/README.md) links the corresponding reports,
cutoff results, forward measurements and selected-query outputs. Selected cases
are not complete prediction dumps. The index is internal: it includes local paths
and references private checkpoints, caches and logs, so it is not an anonymous
submission manifest.

## Anonymous source supplement

Build from the explicit [allowlist](anonymous-allowlist.txt):

```bash
python reproducibility/build_anonymous_supplement.py
```

The generated archive defaults to `dist/specembedding-anonymous-supplement.zip`.
An older copy is tracked for handoff despite `dist/` being ignored for new files.
Its [validation record](anonymous-supplement-validation.yaml) applies only to the
stated source version. Rebuilding the archive requires new independent validation;
do not relabel the old record as a check of current code.

The packaged [README](anonymous-supplement-README.md) documents installation and
synthetic CPU smoke. The archive excludes Git history, paper sources, real data,
checkpoints, caches, logs and internal experiment inventories. It supports source
inspection and executable integration, not reproduction of paper metrics.

For a new release, validate identity scanning, independent extraction, package
tests, shell syntax, Ruff, compileall and synthetic smoke; record source commit,
archive hash and results in the validation YAML. Final joint PDF/source/archive
and venue checks remain in the [submission checklist](../paper/TRANSFER_2026_PLAN.md).
