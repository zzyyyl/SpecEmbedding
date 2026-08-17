#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-specembedding}"
DATA_ROOT="${SPECEMBEDDING_DATA_ROOT:-${REPO_ROOT}/data}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/processed}"
TOKENSET_CACHE="${TOKENSET_CACHE:-${DATA_ROOT}/train_cache/tokenset_massspecgym.pkl}"
CURRENT_COMMIT="$(git rev-parse --short HEAD)"
ALIGNMENT_SEED="${ALIGNMENT_SEED:-42}"
DEFAULT_RUN_PREFIX="${CURRENT_COMMIT}_massspecgym_nopretrain_valoverlapclean"
if [[ -z "${RUN_PREFIX:-}" ]]; then
  if [[ "$ALIGNMENT_SEED" == 42 ]]; then
    RUN_PREFIX="$DEFAULT_RUN_PREFIX"
  else
    RUN_PREFIX="${DEFAULT_RUN_PREFIX}_alignseed${ALIGNMENT_SEED}"
  fi
fi
ALIGN_DEVICE="${ALIGN_DEVICE:-cuda:1}"
RERANK_DEVICE_LIST="${RERANK_DEVICE_LIST:-cuda:1}"
read -r -a RERANK_DEVICES <<< "$RERANK_DEVICE_LIST"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
TOPK=40
VAL_INDICES=(7686 7687 7688 8464 8465 8466)
TRAIN_INDICES=(127216 127217 127218 129178 129179 129180)
EXPECTED_TRAIN_SHA256="071b6c28ade639261118fdcb0da2510efcea764b2978e2a5d0b61a95a1960c48"
EXPECTED_VAL_SHA256="bd13a336e86fb35e77a534e9415cd40cc1a9f15b4a33174c85573b8639c9ca6e"
EXPECTED_TEST_SHA256="2309cd689b06a493c66e22f4aa70613b0cfbe2f8fd1d82e605d657f27015d660"
EXPECTED_TOKENSET_SHA256="efb9fec12afdd1d66aa6a18492deb83d6f7fc22128d044598fb8cf25619b377c"
EXPECTED_MASS_CANDIDATES_SHA256="b4ddf39783ea2dceb32217eab68b71133572f63cc482ba9d17ed9a530c906a8f"
EXPECTED_FORMULA_CANDIDATES_SHA256="0b0d8ffff166eeda9c9dc4060618f823ae4984a71d045119b13b01b5dd5f8194"

ALIGN_DIR="checkpoints_align/${RUN_PREFIX}"
ALIGN_CHECKPOINT="${ALIGN_DIR}/best_model_stage2.pth"
ALIGN_SELECTION="${ALIGN_DIR}/alignment_selection.json"
CACHE_ROOT="rerank_cache"
RERANK_OUTPUT="checkpoints_rerank/${RUN_PREFIX}_topk${TOPK}_multiseed"
PYTHON=(conda run -n "$CONDA_ENV" python)
DRY_RUN=false

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ "${#RERANK_DEVICES[@]}" -eq 0 ]]; then
  echo "RERANK_DEVICE_LIST must contain at least one explicit cuda:N device." >&2
  exit 2
fi
if ! [[ "$ALIGNMENT_SEED" =~ ^[0-9]+$ ]]; then
  echo "ALIGNMENT_SEED must be a non-negative integer." >&2
  exit 2
fi

if [[ "$DRY_RUN" == false ]] && [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run a paper experiment from a dirty worktree." >&2
  echo "Commit the train--validation sensitivity implementation first." >&2
  exit 1
fi

gpu_index() {
  local device="$1"
  if [[ ! "$device" =~ ^cuda:([0-9]+)$ ]]; then
    echo "Expected an explicit cuda:N device, got: $device" >&2
    return 1
  fi
  printf '%s\n' "${BASH_REMATCH[1]}"
}

if ! [[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_MIB must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$MAX_UTILIZATION" =~ ^[0-9]+$ ]] \
  || (( MAX_UTILIZATION > 100 )); then
  echo "MAX_UTILIZATION must be an integer between 0 and 100." >&2
  exit 2
fi
gpu_index "$ALIGN_DEVICE" >/dev/null
declare -A SEEN_RERANK_DEVICES=()
for device in "${RERANK_DEVICES[@]}"; do
  gpu_index "$device" >/dev/null
  if [[ -n "${SEEN_RERANK_DEVICES[$device]:-}" ]]; then
    echo "RERANK_DEVICE_LIST must not contain duplicate devices: $device" >&2
    exit 2
  fi
  SEEN_RERANK_DEVICES[$device]=1
done

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum -- "$path")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Data fingerprint mismatch for $path" >&2
    echo "Expected $expected but found $actual" >&2
    echo "Re-audit the overlap indices before running this experiment." >&2
    return 1
  fi
}

check_sha256 "$DATA_PATH/MassSpecGym/train.pkl" "$EXPECTED_TRAIN_SHA256"
check_sha256 "$DATA_PATH/MassSpecGym/val.pkl" "$EXPECTED_VAL_SHA256"
check_sha256 "$DATA_PATH/MassSpecGym/test.pkl" "$EXPECTED_TEST_SHA256"
check_sha256 "$TOKENSET_CACHE" "$EXPECTED_TOKENSET_SHA256"
check_sha256 "$DATA_PATH/MassSpecGym/candidates_mass.pkl" "$EXPECTED_MASS_CANDIDATES_SHA256"
check_sha256 "$DATA_PATH/MassSpecGym/candidates_formula.pkl" "$EXPECTED_FORMULA_CANDIDATES_SHA256"
echo "Validated audited data fingerprints."
echo "Validation indices: ${VAL_INDICES[*]}"
echo "Matching train indices: ${TRAIN_INDICES[*]}"
echo "Alignment seed: $ALIGNMENT_SEED"
echo "Run prefix: $RUN_PREFIX"

run_on_gpu() {
  local device="$1"
  shift
  nvidia-smi -i "$(gpu_index "$device")"
  "${PYTHON[@]}" -c '
import sys

from SpecEmbedding.utils.gpu import require_available_gpu

require_available_gpu(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
' "$device" "$MIN_FREE_MIB" "$MAX_UTILIZATION"
  if [[ "$DRY_RUN" == true ]]; then
    printf 'Planned:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

run_after_full_gpu_check() {
  nvidia-smi
  if [[ "$DRY_RUN" == true ]]; then
    printf 'Planned:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

alignment_complete() {
  [[ -s "$ALIGN_CHECKPOINT" ]] \
    && [[ -s "${ALIGN_DIR}/final_aligned_model.pth" ]] \
    && [[ -s "$ALIGN_SELECTION" ]] \
    && [[ -s "${ALIGN_DIR}/align_train.log" ]] \
    && grep -q "Two-stage training completed successfully" "${ALIGN_DIR}/align_train.log" \
    && "${PYTHON[@]}" -c '
import hashlib
import json
import sys
from pathlib import Path

summary_path, checkpoint_path, tokenset_cache, expected_seed, *raw_indices = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
indices = [int(index) for index in raw_indices]
report = summary["validation_exclusion_report"]
stage2 = summary["stages"]["stage2"]
assert summary["dataset_type"] == "massspecgym"
assert summary["seed"] == int(expected_seed)
assert Path(summary["tokenset_cache"]).resolve() == Path(tokenset_cache).resolve()
assert summary["exclude_val_query_indices"] == indices
assert report["query_indices"] == indices
assert report["keys"] == [9417, 19728]
assert report["original_num_keys"] == 3386
assert report["filtered_num_keys"] == 3384
assert Path(summary["checkpoint"]).resolve() == Path(checkpoint_path).resolve()
digest = hashlib.sha256()
with Path(checkpoint_path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
assert summary["checkpoint_sha256"] == digest.hexdigest()
assert stage2["metric_for_best"] == "validation_contrastive_loss"
assert isinstance(stage2["best_epoch"], int)
assert isinstance(stage2["best_val_loss"], float)
assert stage2["stop_epoch"] >= stage2["best_epoch"]
' \
      "$ALIGN_SELECTION" "$ALIGN_CHECKPOINT" "$TOKENSET_CACHE" "$ALIGNMENT_SEED" \
      "${VAL_INDICES[@]}" >/dev/null
}

cache_valid() {
  local path="$1"
  local split="$2"
  local candidate="$3"
  local expected_queries="$4"
  local candidate_path="$DATA_PATH/MassSpecGym/candidates_${candidate}.pkl"
  local candidate_sha256
  if [[ "$candidate" == mass ]]; then
    candidate_sha256="$EXPECTED_MASS_CANDIDATES_SHA256"
  else
    candidate_sha256="$EXPECTED_FORMULA_CANDIDATES_SHA256"
  fi
  [[ -s "$path" ]] || return 1
  "${PYTHON[@]}" -c '
import hashlib
import pickle
import sys
from pathlib import Path
import torch

(
    path,
    split,
    candidate,
    topk,
    expected_queries,
    checkpoint,
    val_path,
    candidate_path,
    candidate_sha256,
    *raw_indices,
) = sys.argv[1:]
cache = torch.load(path, map_location="cpu", weights_only=False)
meta = cache.get("meta", {})
expected_force = split == "train"
assert meta.get("dataset_type") == "massspecgym"
assert meta.get("split") == split
assert meta.get("candidate_type") == candidate
assert meta.get("pre_top_k") == int(topk)
assert meta.get("force_include_positive") is expected_force
assert meta.get("num_queries") == int(expected_queries)
assert Path(meta.get("checkpoint", "")).resolve() == Path(checkpoint).resolve()
assert Path(meta.get("candidate_source_path", "")).resolve() == Path(candidate_path).resolve()
assert meta.get("candidate_sha256") == candidate_sha256
digest = hashlib.sha256()
with Path(checkpoint).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
assert meta.get("checkpoint_sha256") == digest.hexdigest()
assert len(cache["queries"]) == int(expected_queries)
assert int(cache["spec_embs"].shape[0]) == int(expected_queries)
if split == "val":
    excluded = [int(index) for index in raw_indices]
    queries = cache["queries"]
    val_raw = pickle.load(open(val_path, "rb"))
    assert len(val_raw) == int(expected_queries)
    assert all(
        query["spec_index"] == index
        and query["true_smiles"] == val_raw[index].get("smiles")
        for index, query in enumerate(queries)
    )
    assert len(queries) - len(set(excluded)) == 19423
' \
    "$path" "$split" "$candidate" "$TOPK" "$expected_queries" "$ALIGN_CHECKPOINT" \
    "$DATA_PATH/MassSpecGym/val.pkl" \
    "$candidate_path" "$candidate_sha256" \
    "${VAL_INDICES[@]}" >/dev/null
}

"${PYTHON[@]}" -c '
import pickle
import sys

from SpecEmbedding.data.overlap import filter_classified_validation

token_cache, val_path, *raw_indices = sys.argv[1:]
classified = pickle.load(open(token_cache, "rb"))
val_raw = pickle.load(open(val_path, "rb"))
filtered, report = filter_classified_validation(
    classified,
    val_raw,
    [int(index) for index in raw_indices],
)
assert report["keys"] == [9417, 19728], report
assert report["original_num_keys"] == 3386, report
assert report["filtered_num_keys"] == 3384, report
assert len(classified["train_keys"]) == len(filtered["train_keys"]) == 25046
print("Validated train--validation exclusions:", report)
' \
  "$TOKENSET_CACHE" \
  "$DATA_PATH/MassSpecGym/val.pkl" \
  "${VAL_INDICES[@]}"

if alignment_complete; then
  echo "Skipping completed filtered-validation alignment: $ALIGN_DIR"
elif [[ "$DRY_RUN" == false ]] && [[ -e "$ALIGN_DIR" ]]; then
  echo "Alignment directory exists but is incomplete: $ALIGN_DIR" >&2
  echo "Move it aside or inspect its logs before rerunning." >&2
  exit 1
else
  run_on_gpu "$ALIGN_DEVICE" \
    "${PYTHON[@]}" train_align.py \
      --dataset_type massspecgym \
      --data_path "$DATA_PATH" \
      --tokenset-cache "$TOKENSET_CACHE" \
      --save_dir "$ALIGN_DIR" \
      --device "$ALIGN_DEVICE" \
      --seed "$ALIGNMENT_SEED" \
      --exclude-val-query-indices "${VAL_INDICES[@]}"
  if [[ "$DRY_RUN" == false ]] && ! alignment_complete; then
    echo "Filtered-validation alignment artifacts failed validation: $ALIGN_DIR" >&2
    exit 1
  fi
fi

declare -A SPLIT_QUERIES=(
  [train]=194119
  [val]=19429
  [test]=17556
)

for candidate in mass formula; do
  cache_dir="${CACHE_ROOT}/${RUN_PREFIX}_${candidate}_topk${TOPK}"
  if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$cache_dir"
  fi
  for split in train val test; do
    cache_path="${cache_dir}/massspecgym_${candidate}_${split}.pt"
    if cache_valid "$cache_path" "$split" "$candidate" "${SPLIT_QUERIES[$split]}"; then
      echo "Skipping validated cache: $cache_path"
      continue
    fi

    force_flag=--no-force_include_positive
    if [[ "$split" == train ]]; then
      force_flag=--force_include_positive
    fi
    run_on_gpu "$ALIGN_DEVICE" \
      "${PYTHON[@]}" prepare_rerank_cache.py \
        --checkpoint "$ALIGN_CHECKPOINT" \
        --dataset_type massspecgym \
        --data_path "$DATA_PATH" \
        --split "$split" \
        --candidate_type "$candidate" \
        --pre_top_k "$TOPK" \
        --save_path "$cache_path" \
        --device "$ALIGN_DEVICE" \
        "$force_flag"
    if [[ "$DRY_RUN" == false ]] \
      && ! cache_valid "$cache_path" "$split" "$candidate" "${SPLIT_QUERIES[$split]}"; then
      echo "Generated cache failed identity and metadata validation: $cache_path" >&2
      exit 1
    fi
  done
done

run_after_full_gpu_check \
  "${PYTHON[@]}" run_rerank_multiseed.py \
  --run-prefix "$RUN_PREFIX" \
  --dataset-type massspecgym \
  --topk "$TOPK" \
  --candidate-types mass formula \
  --model-types pointwise transformer \
  --ablations full \
  --seeds 42 43 44 \
  --devices "${RERANK_DEVICES[@]}" \
  --min-free-mib "$MIN_FREE_MIB" \
  --max-utilization "$MAX_UTILIZATION" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$RERANK_OUTPUT" \
  --exclude-val-query-indices "${VAL_INDICES[@]}" \
  --no-mces

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run completed; no training, cache generation, or evaluation was started."
else
  "${PYTHON[@]}" -c '
import csv
import json
import sys
from pathlib import Path

output_root, *raw_indices = sys.argv[1:]
output_root = Path(output_root)
indices = [int(index) for index in raw_indices]
status = json.loads((output_root / "batch_status.json").read_text(encoding="utf-8"))
with (output_root / "summary.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
with (output_root / "summary_aggregate.csv").open(encoding="utf-8") as handle:
    aggregate_rows = list(csv.DictReader(handle))

assert status["state"] == "complete", status
assert status["errors"] == [], status
assert status["exclude_val_query_indices"] == indices, status
assert len(status["experiments"]) == len(status["results"]) == 12, status
assert all(result["state"] in {"complete", "skipped"} for result in status["results"])
assert len(rows) == 12
assert len({(row["candidate_type"], row["model_type"], row["seed"]) for row in rows}) == 12
assert all(row["best_epoch"] and row["best_val_metric"] and row["stop_epoch"] for row in rows)
assert len(aggregate_rows) == 4
assert all(int(row["num_seeds"]) == 3 for row in aggregate_rows)
' "$RERANK_OUTPUT" "${VAL_INDICES[@]}"
  echo "Completed train--validation overlap sensitivity pipeline."
fi
echo "Alignment: $ALIGN_DIR"
echo "Alignment selection: $ALIGN_SELECTION"
echo "Reranker per-seed summary: ${RERANK_OUTPUT}/summary.csv"
echo "Reranker summary: ${RERANK_OUTPUT}/summary_aggregate.csv"
