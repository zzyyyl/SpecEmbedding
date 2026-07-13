#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-specembedding}"
DEVICE="${DEVICE:-cuda:1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
FORCE_RERUN="${FORCE_RERUN:-false}"

SOURCE_COMMIT="a2280d2828ce872da1f69319b49e0ef7f1bed572"
SOURCE_PREFIX="a2280d2_massspecgym_nopretrain_valoverlapclean"
SOURCE_ROOT="checkpoints_rerank/${SOURCE_PREFIX}_topk40_multiseed"
OUTPUT_ROOT="${MCES_OUTPUT_ROOT:-checkpoints_rerank/${SOURCE_PREFIX}_topk40_mces_seed42_transformer}"
EXPECTED_PARAMS_SHA256="b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6"
EXPECTED_ALIGNMENT_SHA256="ad5d1eb76805c51563349f259a4b4c935336064b6171a4e650472b78aeeaa06f"
EXPECTED_VAL_INDICES=(7686 7687 7688 8464 8465 8466)
CANDIDATES=(mass formula)

declare -A CACHE_PATHS=(
  [mass]="rerank_cache/${SOURCE_PREFIX}_mass_topk40/massspecgym_mass_test.pt"
  [formula]="rerank_cache/${SOURCE_PREFIX}_formula_topk40/massspecgym_formula_test.pt"
)
declare -A CACHE_SHA256=(
  [mass]="6842fa615b7cf07fbf3c82825d75e72cddcd2e0da86dcdb3c5d8e3679f931ea6"
  [formula]="f2459430637f3495d22f40d242fe7c6210d35993aeca5c9667229a917372b0d0"
)
declare -A CHECKPOINT_PATHS=(
  [mass]="${SOURCE_ROOT}/mass/transformer/seed42/attempt_001/best_reranker.pth"
  [formula]="${SOURCE_ROOT}/formula/transformer/seed42/attempt_001/best_reranker.pth"
)
declare -A CHECKPOINT_SHA256=(
  [mass]="aedf5e82089ec8ec350da2ab04718a879f5019756f701c1a1e1da3559e541c57"
  [formula]="ca13cd793ba87b6eceefb6aa3c1dc46e95eb8d36a92ddeb31cca7d04cf552ba8"
)
declare -A STATUS_PATHS=(
  [mass]="${SOURCE_ROOT}/mass/transformer/seed42/attempt_001/status.json"
  [formula]="${SOURCE_ROOT}/formula/transformer/seed42/attempt_001/status.json"
)
declare -A CANDIDATE_SHA256=(
  [mass]="b4ddf39783ea2dceb32217eab68b71133572f63cc482ba9d17ed9a530c906a8f"
  [formula]="0b0d8ffff166eeda9c9dc4060618f823ae4984a71d045119b13b01b5dd5f8194"
)
declare -A EXPECTED_BASE_R1=(
  [mass]="47.4596"
  [formula]="63.1009"
)
declare -A EXPECTED_RERANK_R1=(
  [mass]="68.7400"
  [formula]="74.7494"
)

PYTHON=(conda run --no-capture-output -n "$CONDA_ENV" python)
DRY_RUN=false
FORCE=false
CURRENT_STATUS=""
CURRENT_CANDIDATE=""
RESULT_ATTEMPT=""

usage() {
  cat <<'EOF'
Usage: ./run_overlap_clean_mces.sh [--dry-run] [--force] [--help]

Recompute MCES@1 only for the overlap-clean a2280d2 seed-42 Transformer
checkpoints on the mass and formula test caches. This script never trains a
model and writes each evaluation to a dedicated, numbered attempt directory.

Options:
  --dry-run  Validate all source artifacts and print the planned GPU checks and
             evaluation commands without calling nvidia-smi or eval_rerank.py.
  --force    Run new attempts even when matching completed attempts exist.
  --help     Show this help message.

Environment overrides:
  CONDA_ENV          Conda environment (default: specembedding)
  DEVICE             Explicit CUDA device (default: cuda:1)
  MIN_FREE_MIB       Required free GPU memory (default: 16000)
  MAX_UTILIZATION    Maximum allowed GPU utilization (default: 20)
  MCES_OUTPUT_ROOT   Dedicated MCES result root
  FORCE_RERUN        true/false alternative to --force
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --force)
      FORCE=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$FORCE_RERUN" in
  true|TRUE|1|yes|YES)
    FORCE=true
    ;;
  false|FALSE|0|no|NO)
    ;;
  *)
    echo "FORCE_RERUN must be a true/false value, got: $FORCE_RERUN" >&2
    exit 2
    ;;
esac

if [[ ! "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  echo "DEVICE must use the explicit cuda:N form, got: $DEVICE" >&2
  exit 2
fi
GPU_INDEX="${BASH_REMATCH[1]}"
if ! [[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_MIB must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$MAX_UTILIZATION" =~ ^[0-9]+$ ]] || (( MAX_UTILIZATION > 100 )); then
  echo "MAX_UTILIZATION must be an integer between 0 and 100." >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH." >&2
  exit 1
fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

atomic_status_update() {
  local status_path="$1"
  local state="$2"
  local error_message="${3:-}"
  python3 - "$status_path" "$state" "$error_message" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
state = sys.argv[2]
error = sys.argv[3]
payload = json.loads(path.read_text(encoding="utf-8"))
payload["state"] = state
payload[f"{state}_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
if error:
    payload["error"] = error
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

write_batch_status() {
  local state="$1"
  local error_message="${2:-}"
  local mass_attempt="${3:-}"
  local formula_attempt="${4:-}"
  mkdir -p "$OUTPUT_ROOT"
  python3 - "$OUTPUT_ROOT/batch_status.json" "$state" "$error_message" \
    "$mass_attempt" "$formula_attempt" "$SOURCE_COMMIT" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
state, error, mass_attempt, formula_attempt, source_commit = sys.argv[2:]
payload = {
    "schema_version": 1,
    "task": "overlap_clean_seed42_transformer_mces",
    "state": state,
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "source_commit": source_commit,
    "results": {
        "mass": mass_attempt or None,
        "formula": formula_attempt or None,
    },
}
if error:
    payload["error"] = error
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

handle_signal() {
  local signal_name="$1"
  trap - INT TERM
  if [[ -n "$CURRENT_STATUS" && -f "$CURRENT_STATUS" ]]; then
    atomic_status_update "$CURRENT_STATUS" interrupted "Received ${signal_name}"
  fi
  if [[ "$DRY_RUN" == false ]]; then
    write_batch_status interrupted "Received ${signal_name} while evaluating ${CURRENT_CANDIDATE:-unknown}."
  fi
  exit 130
}
trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM

validate_runtime_source() {
  git cat-file -e "${SOURCE_COMMIT}^{commit}"
  if ! git diff --quiet "$SOURCE_COMMIT" -- eval_rerank.py params.yaml SpecEmbedding; then
    echo "Evaluation runtime differs from source commit $SOURCE_COMMIT." >&2
    echo "Audit the runtime changes before recomputing paper metrics." >&2
    return 1
  fi
  local params_sha256
  params_sha256="$(sha256sum -- params.yaml)"
  params_sha256="${params_sha256%% *}"
  if [[ "$params_sha256" != "$EXPECTED_PARAMS_SHA256" ]]; then
    echo "params.yaml fingerprint mismatch: $params_sha256" >&2
    return 1
  fi
}

validate_source_artifact() {
  local candidate="$1"
  local cache="${CACHE_PATHS[$candidate]}"
  local checkpoint="${CHECKPOINT_PATHS[$candidate]}"
  local source_status="${STATUS_PATHS[$candidate]}"

  "${PYTHON[@]}" - "$REPO_ROOT" "$candidate" "$cache" "${CACHE_SHA256[$candidate]}" \
    "$checkpoint" "${CHECKPOINT_SHA256[$candidate]}" "$source_status" \
    "${CANDIDATE_SHA256[$candidate]}" "$SOURCE_COMMIT" "$SOURCE_PREFIX" \
    "$EXPECTED_PARAMS_SHA256" "$EXPECTED_ALIGNMENT_SHA256" \
    "${EXPECTED_VAL_INDICES[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

(
    repo_root,
    candidate,
    cache_raw,
    cache_sha256,
    checkpoint_raw,
    checkpoint_sha256,
    status_raw,
    candidate_sha256,
    source_commit,
    source_prefix,
    params_sha256,
    alignment_sha256,
    *raw_indices,
) = sys.argv[1:]
repo_root = Path(repo_root).resolve()
cache = (repo_root / cache_raw).resolve()
checkpoint = (repo_root / checkpoint_raw).resolve()
status_path = (repo_root / status_raw).resolve()
indices = [int(value) for value in raw_indices]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def resolve_recorded(path_value: str) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else repo_root / path).resolve()

for path, label in (
    (cache, "test cache"),
    (checkpoint, "best checkpoint"),
    (status_path, "source status"),
):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty {label}: {path}")
if digest(cache) != cache_sha256:
    raise RuntimeError(f"Test-cache fingerprint mismatch: {cache}")
if digest(checkpoint) != checkpoint_sha256:
    raise RuntimeError(f"Checkpoint fingerprint mismatch: {checkpoint}")

status = json.loads(status_path.read_text(encoding="utf-8"))
assert status["state"] == "complete", status_path
assert status["candidate_type"] == candidate, status_path
assert status["model_type"] == "transformer", status_path
assert status["ablation"] == "full", status_path
assert status["seed"] == 42, status_path
assert status["exclude_val_query_indices"] == indices, status_path
assert status["git_commit"] == source_commit, status_path
assert status["params_sha256"] == params_sha256, status_path
fingerprint = status["fingerprint"]
assert fingerprint["run_prefix"] == source_prefix, status_path
assert fingerprint["topk"] == 40, status_path
assert fingerprint["candidate_type"] == candidate, status_path
assert fingerprint["model_type"] == "transformer", status_path
assert fingerprint["seed"] == 42, status_path
assert fingerprint["exclude_val_query_indices"] == indices, status_path
for cache_records in (fingerprint["cache_files"], status["cache_files"]):
    test_record = cache_records["test"]
    assert resolve_recorded(test_record["path"]) == cache, status_path
    assert test_record["size_bytes"] == cache.stat().st_size, status_path
    assert test_record["mtime_ns"] == cache.stat().st_mtime_ns, status_path

eval_command = status["eval_command"]
def command_value(option: str) -> str:
    index = eval_command.index(option)
    return eval_command[index + 1]
assert resolve_recorded(command_value("--cache")) == cache, status_path
assert resolve_recorded(command_value("--checkpoint")) == checkpoint, status_path
assert "--no-mces" in eval_command, status_path

cache_payload = torch.load(cache, map_location="cpu", weights_only=False)
meta = cache_payload["meta"]
assert meta["dataset_type"] == "massspecgym"
assert meta["split"] == "test"
assert meta["candidate_type"] == candidate
assert meta["candidate_sha256"] == candidate_sha256
assert meta["pre_top_k"] == 40
assert meta["force_include_positive"] is False
assert meta["num_queries"] == 17556
assert meta["checkpoint_sha256"] == alignment_sha256

checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert checkpoint_payload["seed"] == 42
assert checkpoint_payload["model_config"]["model_type"] == "transformer"
training_config = checkpoint_payload["training_config"]
assert training_config["model_type"] == "transformer"
assert training_config["seed"] == 42
assert training_config["train_k"] == 40
assert training_config["exclude_val_query_indices"] == indices
assert candidate in Path(training_config["train_cache"]).name
assert candidate in Path(training_config["val_cache"]).name
print(f"Validated {candidate} source cache, checkpoint, and completed status.")
PY
}

completed_attempt() {
  local candidate="$1"
  local candidate_root="$OUTPUT_ROOT/$candidate"
  local status_path
  shopt -s nullglob
  for status_path in "$candidate_root"/attempt_*/status.json; do
    if python3 - "$status_path" "$REPO_ROOT" "$candidate" \
      "${CACHE_PATHS[$candidate]}" "${CACHE_SHA256[$candidate]}" \
      "${CHECKPOINT_PATHS[$candidate]}" "${CHECKPOINT_SHA256[$candidate]}" \
      "$SOURCE_COMMIT" "${EXPECTED_BASE_R1[$candidate]}" \
      "${EXPECTED_RERANK_R1[$candidate]}" <<'PY' >/dev/null 2>&1
import json
import re
import sys
from pathlib import Path

(
    status_raw,
    repo_root_raw,
    candidate,
    cache_raw,
    cache_sha256,
    checkpoint_raw,
    checkpoint_sha256,
    source_commit,
    expected_base_r1,
    expected_rerank_r1,
) = sys.argv[1:]
status_path = Path(status_raw).resolve()
repo_root = Path(repo_root_raw).resolve()
attempt = status_path.parent
payload = json.loads(status_path.read_text(encoding="utf-8"))
assert payload["state"] == "complete"
assert payload["candidate_type"] == candidate
assert payload["model_type"] == "transformer"
assert payload["seed"] == 42
assert payload["mces_enabled"] is True
assert payload["source_commit"] == source_commit
assert Path(payload["cache"]).resolve() == (repo_root / cache_raw).resolve()
assert Path(payload["checkpoint"]).resolve() == (repo_root / checkpoint_raw).resolve()
assert payload["cache_sha256"] == cache_sha256
assert payload["checkpoint_sha256"] == checkpoint_sha256
assert payload["metrics"]["base_top1_accuracy_pct"] == float(expected_base_r1)
assert payload["metrics"]["rerank_top1_accuracy_pct"] == float(expected_rerank_r1)
log_path = attempt / "eval_rerank.log"
assert log_path.is_file() and log_path.stat().st_size > 0
text = log_path.read_text(encoding="utf-8")
base_values = re.findall(r"Base\s+MCES@1\s*:\s*([0-9.]+)", text)
rerank_values = re.findall(r"Rerank MCES@1\s*:\s*([0-9.]+)", text)
assert len(base_values) == len(rerank_values) == 1
assert float(base_values[0]) == payload["metrics"]["base_mces_at_1"]
assert float(rerank_values[0]) == payload["metrics"]["rerank_mces_at_1"]
assert "MCES calculation skipped." not in text
PY
    then
      printf '%s\n' "$(dirname -- "$status_path")"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

next_attempt_dir() {
  local candidate="$1"
  local candidate_root="$OUTPUT_ROOT/$candidate"
  local maximum=0
  local path basename number
  shopt -s nullglob
  for path in "$candidate_root"/attempt_*; do
    basename="${path##*/}"
    if [[ "$basename" =~ ^attempt_([0-9]{3})$ ]]; then
      number=$((10#${BASH_REMATCH[1]}))
      if (( number > maximum )); then
        maximum="$number"
      fi
    fi
  done
  shopt -u nullglob
  printf '%s/attempt_%03d\n' "$candidate_root" "$((maximum + 1))"
}

initialize_attempt_status() {
  local attempt="$1"
  local candidate="$2"
  local cache="$3"
  local checkpoint="$4"
  shift 4
  python3 - "$attempt/status.json" "$candidate" "$cache" "$checkpoint" \
    "${CACHE_SHA256[$candidate]}" "${CHECKPOINT_SHA256[$candidate]}" \
    "$SOURCE_COMMIT" "$DEVICE" "$CONDA_ENV" "$@" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

(
    status_raw,
    candidate,
    cache_raw,
    checkpoint_raw,
    cache_sha256,
    checkpoint_sha256,
    source_commit,
    device,
    conda_env,
    *command,
) = sys.argv[1:]
path = Path(status_raw)
payload = {
    "schema_version": 1,
    "task": "overlap_clean_seed42_transformer_mces",
    "state": "pending_gpu_check",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "candidate_type": candidate,
    "model_type": "transformer",
    "seed": 42,
    "mces_enabled": True,
    "source_commit": source_commit,
    "cache": str(Path(cache_raw).resolve()),
    "cache_sha256": cache_sha256,
    "checkpoint": str(Path(checkpoint_raw).resolve()),
    "checkpoint_sha256": checkpoint_sha256,
    "device": device,
    "conda_env": conda_env,
    "command": command,
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

record_gpu_state() {
  local status_path="$1"
  local snapshot_path="$2"
  "${PYTHON[@]}" - "$DEVICE" "$MIN_FREE_MIB" "$MAX_UTILIZATION" \
    "$snapshot_path" "$status_path" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from SpecEmbedding.utils.gpu import require_available_gpu

device, min_free_mib, max_utilization, snapshot_raw, status_raw = sys.argv[1:]
state = require_available_gpu(
    device,
    int(min_free_mib),
    int(max_utilization),
    snapshot_path=Path(snapshot_raw),
)
path = Path(status_raw)
payload = json.loads(path.read_text(encoding="utf-8"))
payload["gpu_before_eval"] = state
payload["gpu_checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

finalize_attempt() {
  local status_path="$1"
  local candidate="$2"
  local log_path="$3"
  python3 - "$status_path" "$candidate" "$log_path" \
    "${EXPECTED_BASE_R1[$candidate]}" "${EXPECTED_RERANK_R1[$candidate]}" <<'PY'
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

status_raw, candidate, log_raw, expected_base_r1, expected_rerank_r1 = sys.argv[1:]
status_path = Path(status_raw)
log_path = Path(log_raw)
text = log_path.read_text(encoding="utf-8")
if "MCES calculation skipped." in text:
    raise RuntimeError(f"MCES was skipped according to {log_path}")

def exactly_one(pattern: str, label: str) -> float:
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} in {log_path}, found {len(matches)}")
    return float(matches[0])

total_queries = int(exactly_one(r"Total queries:\s*(\d+)", "query count"))
base_mces = exactly_one(r"Base\s+MCES@1\s*:\s*([0-9.]+)", "Base MCES@1")
rerank_mces = exactly_one(r"Rerank MCES@1\s*:\s*([0-9.]+)", "Rerank MCES@1")

sections = re.search(
    r"BASE RESULTS.*?Top-1\s+Accuracy\s*:\s*([0-9.]+)%.*?"
    r"RERANK RESULTS.*?Top-1\s+Accuracy\s*:\s*([0-9.]+)%",
    text,
    flags=re.DOTALL,
)
if sections is None:
    raise RuntimeError(f"Could not parse ranking sections in {log_path}")
base_r1, rerank_r1 = map(float, sections.groups())
if total_queries != 17556:
    raise RuntimeError(f"Unexpected query count for {candidate}: {total_queries}")
if base_r1 != float(expected_base_r1) or rerank_r1 != float(expected_rerank_r1):
    raise RuntimeError(
        f"Ranking fingerprint mismatch for {candidate}: "
        f"base={base_r1}, rerank={rerank_r1}"
    )

payload = json.loads(status_path.read_text(encoding="utf-8"))
payload["state"] = "complete"
payload["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
payload["metrics"] = {
    "total_queries": total_queries,
    "base_top1_accuracy_pct": base_r1,
    "rerank_top1_accuracy_pct": rerank_r1,
    "base_mces_at_1": base_mces,
    "rerank_mces_at_1": rerank_mces,
}
temporary = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(status_path)
print(
    f"{candidate}: Base MCES@1={base_mces:.4f}, "
    f"Rerank MCES@1={rerank_mces:.4f}"
)
PY
}

run_candidate() {
  local candidate="$1"
  local completed=""
  if [[ "$FORCE" == false ]] && completed="$(completed_attempt "$candidate")"; then
    echo "Skipping $candidate; validated completed MCES attempt: $completed"
    RESULT_ATTEMPT="$completed"
    return 0
  fi

  local attempt
  attempt="$(next_attempt_dir "$candidate")"
  local cache="${CACHE_PATHS[$candidate]}"
  local checkpoint="${CHECKPOINT_PATHS[$candidate]}"
  local eval_log="$attempt/eval_rerank.log"
  local driver_log="$attempt/driver.log"
  local gpu_snapshot="$attempt/gpu_before_eval.txt"
  local -a eval_command=(
    "${PYTHON[@]}" "$REPO_ROOT/eval_rerank.py"
    --cache "$cache"
    --checkpoint "$checkpoint"
    --save_dir "$attempt"
    --device "$DEVICE"
    --mces
  )

  if [[ "$DRY_RUN" == true ]]; then
    echo "Planned $candidate attempt: $attempt"
    printf '  GPU inventory:'
    printf ' %q' nvidia-smi -i "$GPU_INDEX"
    printf '\n'
    printf '  GPU threshold check: require_available_gpu %q %q %q\n' \
      "$DEVICE" "$MIN_FREE_MIB" "$MAX_UTILIZATION"
    printf '  Evaluation:'
    printf ' %q' "${eval_command[@]}"
    printf '\n'
    RESULT_ATTEMPT="$attempt"
    return 0
  fi

  mkdir -p "$attempt"
  : > "$driver_log"
  initialize_attempt_status "$attempt" "$candidate" "$cache" "$checkpoint" \
    "${eval_command[@]}"
  CURRENT_STATUS="$attempt/status.json"
  CURRENT_CANDIDATE="$candidate"

  echo "[$candidate] nvidia-smi immediately before eval_rerank.py" | tee -a "$driver_log"
  if ! nvidia-smi -i "$GPU_INDEX" 2>&1 | tee -a "$driver_log"; then
    atomic_status_update "$CURRENT_STATUS" failed "nvidia-smi failed before evaluation"
    return 1
  fi
  if ! record_gpu_state "$CURRENT_STATUS" "$gpu_snapshot" 2>&1 | tee -a "$driver_log"; then
    atomic_status_update "$CURRENT_STATUS" failed "require_available_gpu rejected $DEVICE"
    return 1
  fi
  atomic_status_update "$CURRENT_STATUS" evaluating

  printf 'Running:' | tee -a "$driver_log"
  printf ' %q' "${eval_command[@]}" | tee -a "$driver_log"
  printf '\n' | tee -a "$driver_log"
  if ! "${eval_command[@]}" 2>&1 | tee -a "$driver_log"; then
    atomic_status_update "$CURRENT_STATUS" failed "eval_rerank.py failed"
    return 1
  fi
  if ! finalize_attempt "$CURRENT_STATUS" "$candidate" "$eval_log" | tee -a "$driver_log"; then
    atomic_status_update "$CURRENT_STATUS" failed "Evaluation log validation failed"
    return 1
  fi

  CURRENT_STATUS=""
  CURRENT_CANDIDATE=""
  RESULT_ATTEMPT="$attempt"
}

validate_runtime_source
for candidate in "${CANDIDATES[@]}"; do
  validate_source_artifact "$candidate"
done

if [[ "$DRY_RUN" == false ]] && [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run paper evaluation from a dirty worktree." >&2
  echo "Commit the MCES script and documentation changes first." >&2
  exit 1
fi

if [[ "$DRY_RUN" == true ]]; then
  for candidate in "${CANDIDATES[@]}"; do
    run_candidate "$candidate"
  done
  echo "Dry-run complete; no GPU query or evaluation was executed."
  echo "MCES output root: $OUTPUT_ROOT"
  exit 0
fi

write_batch_status running
MASS_ATTEMPT=""
FORMULA_ATTEMPT=""
if ! run_candidate mass; then
  write_batch_status failed "Mass MCES evaluation failed. Inspect the latest attempt status and driver.log."
  exit 1
fi
MASS_ATTEMPT="$RESULT_ATTEMPT"
write_batch_status running "" "$MASS_ATTEMPT"
if ! run_candidate formula; then
  write_batch_status failed "Formula MCES evaluation failed. Inspect the latest attempt status and driver.log." \
    "$MASS_ATTEMPT"
  exit 1
fi
FORMULA_ATTEMPT="$RESULT_ATTEMPT"
write_batch_status complete "" "$MASS_ATTEMPT" "$FORMULA_ATTEMPT"

echo "Completed overlap-clean seed-42 Transformer MCES@1 evaluation."
echo "Batch status: $OUTPUT_ROOT/batch_status.json"
echo "Mass result: $MASS_ATTEMPT/status.json"
echo "Formula result: $FORMULA_ATTEMPT/status.json"
