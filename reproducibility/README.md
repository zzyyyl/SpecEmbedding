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
