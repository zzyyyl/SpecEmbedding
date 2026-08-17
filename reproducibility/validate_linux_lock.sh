#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
environment_name="${1:-specembedding-lock-validation}"

conda create -y \
  --name "${environment_name}" \
  --file "${script_dir}/conda-linux-64.lock"
conda run -n "${environment_name}" python -m pip install \
  -r "${script_dir}/pip-linux-64.lock"
conda run -n "${environment_name}" python -m pytest -q "${repository_root}/tests"
conda run -n "${environment_name}" python -m ruff check "${repository_root}"

printf 'Explicit lock validation completed successfully.\n'
