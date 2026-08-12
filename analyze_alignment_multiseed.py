import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

EXPECTED_ALIGNMENT_SEEDS = (42, 43, 44)
CANDIDATE_TYPES = ("mass", "formula")
MODEL_TYPES = ("pointwise", "transformer")
RERANKER_SEEDS = (42, 43, 44)
VALIDATION_EXCLUSIONS = (7686, 7687, 7688, 8464, 8465, 8466)
SPLITS = ("train", "val", "test")
TOPK = 40
FULL_RERANKER_OVERRIDES = {
    "use_base_score_feature": True,
    "use_residual_score": True,
    "use_rank_embedding": True,
    "use_product_feature": True,
    "use_abs_diff_feature": True,
    "lambda_pair": 0.2,
    "shuffle_candidates": True,
}

BASE_METRICS = (
    "upper_bound_pct",
    "base_top1_pct",
    "base_top5_pct",
    "base_top10_pct",
    "base_top20_pct",
    "base_mrr_raw",
)
RERANK_METRICS = (
    "rerank_top1_pct",
    "rerank_top5_pct",
    "rerank_top10_pct",
    "rerank_top20_pct",
    "rerank_mrr_raw",
)
AGGREGATE_METRICS = (
    *RERANK_METRICS,
    "best_val_metric",
    "best_epoch",
    "stop_epoch",
    "early_stopped",
)
SUMMARY_REQUIRED_COLUMNS = (
    "candidate_type",
    "model_type",
    "ablation",
    "seed",
    "metric_for_best",
    "best_epoch",
    "best_val_metric",
    "stop_epoch",
    "early_stopped",
    *BASE_METRICS,
    *RERANK_METRICS,
    "attempt_dir",
)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSpec:
    alignment_seed: int
    selection_path: Path
    rerank_root: Path


@dataclass
class LoadedRun:
    spec: RunSpec
    selection: dict[str, Any]
    batch: dict[str, Any]
    rows: dict[tuple[str, str, int], dict[str, Any]]
    input_files: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"Missing or empty {label}: {path}")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object for {label}: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AnalysisError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def require_close(actual: Any, expected: Any, label: str) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"Expected numeric values for {label}") from error
    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise AnalysisError(f"Expected finite numeric values for {label}")
    if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
        raise AnalysisError(
            f"{label} mismatch: expected {expected_value!r}, found {actual_value!r}"
        )


def parse_bool(raw: Any, label: str) -> bool:
    if raw is True or raw == "True":
        return True
    if raw is False or raw == "False":
        return False
    raise AnalysisError(f"Expected True/False for {label}, found {raw!r}")


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    require_file(path, "input artifact")
    return {
        "path": display_path(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_recorded_path(raw: Any, expected: Path, repo_root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AnalysisError(f"Missing recorded path for {label}")
    expected = expected.resolve()
    try:
        expected_relative = expected.relative_to(repo_root.resolve())
    except ValueError as error:
        raise AnalysisError(
            f"Expected repository path escapes root for {label}: {expected}"
        ) from error
    recorded = Path(raw).expanduser()
    if recorded.is_absolute():
        expected_parts = expected_relative.parts
        if tuple(recorded.parts[-len(expected_parts) :]) != expected_parts:
            raise AnalysisError(
                f"{label} mismatch: expected repository suffix "
                f"{expected_relative.as_posix()!r}, found {raw!r}"
            )
    else:
        normalized = Path(os.path.normpath(raw))
        if ".." in normalized.parts or normalized != expected_relative:
            raise AnalysisError(
                f"{label} mismatch: expected {expected_relative.as_posix()!r}, "
                f"found {raw!r}"
            )
    return expected


def finite_float(raw: Any, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"Invalid numeric value for {label}: {raw!r}") from error
    if not math.isfinite(value):
        raise AnalysisError(f"Expected a finite numeric value for {label}: {raw!r}")
    return value


def finite_int(raw: Any, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"Invalid integer value for {label}: {raw!r}") from error
    return value


def parse_summary(path: Path) -> list[dict[str, Any]]:
    require_file(path, "summary.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [field for field in SUMMARY_REQUIRED_COLUMNS if field not in fieldnames]
        if missing:
            raise AnalysisError(f"summary.csv is missing columns {missing}: {path}")
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        row: dict[str, Any] = dict(raw)
        try:
            row["seed"] = int(raw["seed"])
            row["best_epoch"] = int(raw["best_epoch"])
            row["stop_epoch"] = int(raw["stop_epoch"])
        except (TypeError, ValueError) as error:
            raise AnalysisError(f"Invalid numeric value in {path}: {raw}") from error
        row["best_val_metric"] = finite_float(
            raw["best_val_metric"], f"{path}: row {row_index} best_val_metric"
        )
        for metric in (*BASE_METRICS, *RERANK_METRICS):
            row[metric] = finite_float(
                raw[metric], f"{path}: row {row_index} {metric}"
            )
        row["early_stopped"] = parse_bool(
            raw["early_stopped"], "summary early_stopped"
        )
        rows.append(row)
    return rows


def expected_experiment_keys() -> set[tuple[str, str, int]]:
    return {
        (candidate, model, seed)
        for candidate in CANDIDATE_TYPES
        for model in MODEL_TYPES
        for seed in RERANKER_SEEDS
    }


def resolve_attempt(
    raw: Any,
    rerank_root: Path,
    key: tuple[str, str, int],
    repo_root: Path,
) -> Path:
    candidate, model, seed = key
    if not isinstance(raw, str) or not raw:
        raise AnalysisError(f"Missing attempt path for {key}")
    recorded = Path(raw).expanduser()
    if not re.fullmatch(r"attempt_[0-9]{3}", recorded.name):
        raise AnalysisError(f"Unexpected attempt name for {key}: {raw}")
    expected = rerank_root / candidate / model / f"seed{seed}" / recorded.name
    resolve_recorded_path(raw, expected, repo_root, f"attempt path {key}")
    if not expected.is_dir():
        raise AnalysisError(f"Missing attempt directory for {key}: {expected}")
    return expected.resolve()


def parse_eval_metrics(path: Path, label: str) -> dict[str, float]:
    text = require_file(path, label).read_text(encoding="utf-8", errors="replace")
    total_query_matches = re.findall(r"Total queries:\s*([0-9]+)", text)
    require_equal(len(total_query_matches), 1, f"{label} total-query count fields")
    require_equal(int(total_query_matches[0]), 17_556, f"{label} total queries")
    if "MCES calculation skipped." not in text:
        raise AnalysisError(f"Expected MCES-skipped marker in {path}")

    def extract_one(pattern: str, metric_label: str, source: str = text) -> float:
        matches = re.findall(pattern, source, flags=re.DOTALL)
        if len(matches) != 1:
            raise AnalysisError(
                f"Expected exactly one {metric_label} value in {path}, "
                f"found {len(matches)}"
            )
        return finite_float(matches[0], f"{label} {metric_label}")

    metrics = {
        "upper_bound_pct": extract_one(
            r"Pre-retrieval upper bound:\s*([0-9.]+)%", "upper bound"
        )
    }
    for section, prefix in (("BASE", "base"), ("RERANK", "rerank")):
        section_match = re.search(
            rf"{section} RESULTS(?P<body>.*?)(?:RERANK RESULTS|MCES calculation|=+\s*$)",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
        if section_match is None:
            raise AnalysisError(f"Missing {section} RESULTS block in {path}")
        body = section_match.group("body")
        for top_k in (1, 5, 10, 20):
            metrics[f"{prefix}_top{top_k}_pct"] = extract_one(
                rf"Top-{top_k}\s+Accuracy\s*:\s*([0-9.]+)%",
                f"{section} Top-{top_k}",
                body,
            )
        metrics[f"{prefix}_mrr_raw"] = extract_one(
            r"MRR\s*:\s*([0-9.]+)", f"{section} MRR", body
        )
    if re.search(r"Traceback|\bERROR\b|interrupted|killed", text, re.IGNORECASE):
        raise AnalysisError(f"Evaluation log contains an error marker: {path}")
    return metrics


def validate_train_log(path: Path, expected_stop_epoch: int, label: str) -> None:
    text = require_file(path, label).read_text(encoding="utf-8", errors="replace")
    if "Training finished." not in text:
        raise AnalysisError(f"Missing training completion marker in {path}")
    if re.search(r"Traceback|\bERROR\b|interrupted|killed", text, re.IGNORECASE):
        raise AnalysisError(f"Training log contains an error marker: {path}")
    epochs = [
        int(match.group(1))
        for match in re.finditer(r"Epoch\s+(\d+):\s+train_loss=", text)
    ]
    if not epochs:
        raise AnalysisError(f"Missing epoch records in {path}")
    require_equal(max(epochs), expected_stop_epoch, f"{label} stop epoch")


def canonical_cache_path(
    repo_root: Path,
    run_prefix: str,
    dataset_type: str,
    candidate: str,
    split: str,
) -> Path:
    return (
        repo_root
        / "rerank_cache"
        / f"{run_prefix}_{candidate}_topk{TOPK}"
        / f"{dataset_type}_{candidate}_{split}.pt"
    )


def validate_cache_record(
    record: Any,
    expected_path: Path,
    repo_root: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise AnalysisError(f"Invalid cache record for {label}: {record!r}")
    resolve_recorded_path(record.get("path"), expected_path, repo_root, label)
    require_file(expected_path, label)
    stat = expected_path.stat()
    require_equal(record.get("size_bytes"), stat.st_size, f"{label} size")
    require_equal(record.get("mtime_ns"), stat.st_mtime_ns, f"{label} mtime")


def validate_aggregate(
    path: Path,
    rows: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    require_file(path, "summary_aggregate.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    require_equal(len(aggregate_rows), 4, "aggregate row count")
    observed: set[tuple[str, str]] = set()
    for aggregate in aggregate_rows:
        key = (aggregate.get("candidate_type"), aggregate.get("model_type"))
        if key in observed:
            raise AnalysisError(f"Duplicate aggregate row: {key}")
        if key not in {
            (candidate, model)
            for candidate in CANDIDATE_TYPES
            for model in MODEL_TYPES
        }:
            raise AnalysisError(f"Unexpected aggregate row: {key}")
        observed.add(key)
        require_equal(aggregate.get("ablation"), "full", f"aggregate ablation {key}")
        require_equal(
            finite_int(aggregate.get("num_seeds"), f"aggregate seed count {key}"),
            3,
            f"aggregate seed count {key}",
        )
        for metric in AGGREGATE_METRICS:
            values = [rows[(key[0], key[1], seed)][metric] for seed in RERANKER_SEEDS]
            require_equal(
                finite_int(
                    aggregate.get(f"{metric}_num_seeds"),
                    f"aggregate metric count {key}/{metric}",
                ),
                3,
                f"aggregate metric count {key}/{metric}",
            )
            require_close(
                aggregate.get(f"{metric}_mean"),
                mean(values),
                f"aggregate mean {key}/{metric}",
            )
            require_close(
                aggregate.get(f"{metric}_std"),
                stdev(values),
                f"aggregate std {key}/{metric}",
            )


def load_run(spec: RunSpec, repo_root: Path) -> LoadedRun:
    selection_path = spec.selection_path.resolve()
    rerank_root = spec.rerank_root.resolve()
    selection = load_json(selection_path, f"alignment {spec.alignment_seed} selection")
    require_equal(
        selection.get("seed"), spec.alignment_seed, "alignment selection seed"
    )
    require_equal(selection.get("dataset_type"), "massspecgym", "alignment dataset")
    exclusions = tuple(selection.get("exclude_val_query_indices", []))
    require_equal(exclusions, VALIDATION_EXCLUSIONS, "alignment validation exclusions")
    checkpoint_path = selection_path.parent / "best_model_stage2.pth"
    resolve_recorded_path(
        selection.get("checkpoint"),
        checkpoint_path,
        repo_root,
        "alignment checkpoint path",
    )
    require_file(checkpoint_path, "alignment checkpoint")
    require_equal(
        sha256_file(checkpoint_path),
        selection.get("checkpoint_sha256"),
        "alignment checkpoint SHA256",
    )

    batch_path = rerank_root / "batch_status.json"
    summary_path = rerank_root / "summary.csv"
    aggregate_path = rerank_root / "summary_aggregate.csv"
    batch = load_json(batch_path, f"alignment {spec.alignment_seed} batch status")
    require_equal(batch.get("state"), "complete", "reranker batch state")
    require_equal(batch.get("errors"), [], "reranker batch errors")
    require_equal(batch.get("mces"), False, "reranker batch MCES mode")
    require_equal(batch.get("git_worktree_changes"), [], "reranker dirty-worktree record")
    require_equal(
        tuple(batch.get("exclude_val_query_indices", [])),
        VALIDATION_EXCLUSIONS,
        "reranker validation exclusions",
    )
    if not isinstance(batch.get("git_commit"), str) or not batch["git_commit"]:
        raise AnalysisError("Reranker batch is missing git_commit")
    if not isinstance(batch.get("params_sha256"), str) or not batch["params_sha256"]:
        raise AnalysisError("Reranker batch is missing params_sha256")

    expected = expected_experiment_keys()
    result_attempts: dict[tuple[str, str, int], Path] = {}
    result_statuses: dict[tuple[str, str, int], dict[str, Any]] = {}
    status_paths: dict[tuple[str, str, int], Path] = {}
    results = batch.get("results")
    if not isinstance(results, list):
        raise AnalysisError("Reranker batch results must be a list")
    experiments = batch.get("experiments")
    if not isinstance(experiments, list):
        raise AnalysisError("Reranker batch experiments must be a list")
    experiment_keys = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise AnalysisError(f"Invalid reranker experiment: {experiment!r}")
        key = (
            experiment.get("candidate_type"),
            experiment.get("model_type"),
            experiment.get("seed"),
        )
        require_equal(experiment.get("ablation"), "full", f"experiment ablation {key}")
        if key not in expected or key in experiment_keys:
            raise AnalysisError(f"Unexpected or duplicate reranker experiment: {key}")
        experiment_keys.add(key)
    require_equal(experiment_keys, expected, "planned experiment matrix")
    for result in results:
        if not isinstance(result, dict):
            raise AnalysisError(f"Invalid reranker result: {result!r}")
        key = (
            result.get("candidate_type"),
            result.get("model_type"),
            result.get("seed"),
        )
        require_equal(result.get("ablation"), "full", f"batch ablation {key}")
        require_equal(result.get("state"), "complete", f"batch result state {key}")
        if key not in expected:
            raise AnalysisError(f"Unexpected reranker result: {key}")
        if key in result_attempts:
            raise AnalysisError(f"Duplicate reranker result: {key}")
        attempt = resolve_attempt(result.get("attempt_dir"), rerank_root, key, repo_root)
        status_path = attempt / "status.json"
        status = load_json(status_path, f"attempt status {key}")
        for field, value in (
            ("state", "complete"),
            ("candidate_type", key[0]),
            ("model_type", key[1]),
            ("ablation", "full"),
            ("seed", key[2]),
            ("git_commit", batch["git_commit"]),
            ("params_sha256", batch["params_sha256"]),
        ):
            require_equal(status.get(field), value, f"attempt {key} {field}")
        require_equal(
            status.get("git_worktree_changes"), [], f"attempt {key} dirty worktree"
        )
        require_equal(
            status.get("ablation_overrides"),
            FULL_RERANKER_OVERRIDES,
            f"attempt {key} ablation overrides",
        )
        require_equal(
            tuple(status.get("exclude_val_query_indices", [])),
            VALIDATION_EXCLUSIONS,
            f"attempt {key} validation exclusions",
        )
        resolve_recorded_path(
            status.get("attempt_dir"), attempt, repo_root, f"status attempt {key}"
        )
        result_attempts[key] = attempt
        result_statuses[key] = status
        status_paths[key] = status_path
    require_equal(set(result_attempts), expected, "completed experiment matrix")

    parsed_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in parse_summary(summary_path):
        key = (row["candidate_type"], row["model_type"], row["seed"])
        require_equal(row["ablation"], "full", f"summary ablation {key}")
        require_equal(row["metric_for_best"], "mrr", f"summary selection metric {key}")
        if key not in expected:
            raise AnalysisError(f"Unexpected summary row: {key}")
        if key in parsed_rows:
            raise AnalysisError(f"Duplicate summary row: {key}")
        attempt = resolve_attempt(row["attempt_dir"], rerank_root, key, repo_root)
        require_equal(attempt, result_attempts[key], f"summary attempt {key}")
        parsed_rows[key] = row
    require_equal(set(parsed_rows), expected, "summary experiment matrix")

    for candidate in CANDIDATE_TYPES:
        candidate_rows = [
            parsed_rows[(candidate, model, seed)]
            for model in MODEL_TYPES
            for seed in RERANKER_SEEDS
        ]
        for metric in BASE_METRICS:
            reference = candidate_rows[0][metric]
            for row in candidate_rows[1:]:
                require_close(
                    row[metric],
                    reference,
                    f"alignment {spec.alignment_seed}/{candidate} invariant {metric}",
                )

    validate_aggregate(aggregate_path, parsed_rows)
    cache_inputs: dict[tuple[str, str], dict[str, Any]] = {}
    attempt_inputs = []
    expected_run_prefix = selection_path.parent.name
    for key in sorted(expected):
        candidate, model, seed = key
        attempt = result_attempts[key]
        status = result_statuses[key]
        summary_row = parsed_rows[key]

        training_selection = status.get("training_selection")
        if not isinstance(training_selection, dict):
            raise AnalysisError(f"Invalid training selection for {key}")
        for field, expected_value in (
            ("metric_for_best", "mrr"),
            ("best_epoch", summary_row["best_epoch"]),
            ("stop_epoch", summary_row["stop_epoch"]),
            ("early_stopped", summary_row["early_stopped"]),
        ):
            require_equal(
                training_selection.get(field),
                expected_value,
                f"training selection {key} {field}",
            )
        require_close(
            training_selection.get("best_val_metric"),
            summary_row["best_val_metric"],
            f"training selection {key} best_val_metric",
        )

        eval_log = attempt / "eval_rerank.log"
        for metric, actual in parse_eval_metrics(
            eval_log, f"reranker evaluation log {key}"
        ).items():
            require_close(actual, summary_row[metric], f"evaluation {metric} {key}")
        checkpoint = require_file(attempt / "best_reranker.pth", f"checkpoint {key}")
        last_checkpoint = require_file(
            attempt / "last_reranker.pth", f"last checkpoint {key}"
        )
        train_log = attempt / "train_rerank.log"
        validate_train_log(
            train_log,
            summary_row["stop_epoch"],
            f"reranker training log {key}",
        )

        fingerprint = status.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise AnalysisError(f"Invalid fingerprint for {key}")
        for field, expected_value in (
            ("git_commit", batch["git_commit"]),
            ("params_sha256", batch["params_sha256"]),
            ("dataset_type", "massspecgym"),
            ("run_prefix", expected_run_prefix),
            ("topk", TOPK),
            ("candidate_type", candidate),
            ("model_type", model),
            ("ablation", "full"),
            ("ablation_overrides", FULL_RERANKER_OVERRIDES),
            ("seed", seed),
            ("exclude_val_query_indices", list(VALIDATION_EXCLUSIONS)),
        ):
            require_equal(fingerprint.get(field), expected_value, f"fingerprint {key} {field}")
        fingerprint_caches = fingerprint.get("cache_files")
        status_caches = status.get("cache_files")
        if not isinstance(fingerprint_caches, dict) or not isinstance(status_caches, dict):
            raise AnalysisError(f"Invalid cache metadata for {key}")
        require_equal(status_caches, fingerprint_caches, f"status/fingerprint caches {key}")
        for split in SPLITS:
            expected_cache = canonical_cache_path(
                repo_root,
                expected_run_prefix,
                "massspecgym",
                candidate,
                split,
            )
            record = fingerprint_caches.get(split)
            validate_cache_record(
                record,
                expected_cache,
                repo_root,
                f"alignment {spec.alignment_seed}/{candidate}/{split} cache",
            )
            cache_key = (candidate, split)
            if cache_key not in cache_inputs:
                artifact = file_record(expected_cache, repo_root)
                artifact.update(
                    {
                        "id": (
                            f"alignment{spec.alignment_seed}.cache."
                            f"{candidate}.{split}"
                        ),
                        "candidate_type": candidate,
                        "split": split,
                    }
                )
                cache_inputs[cache_key] = artifact

        attempt_inputs.append(
            {
                "candidate_type": candidate,
                "model_type": model,
                "reranker_seed": seed,
                "status": file_record(status_paths[key], repo_root),
                "training_log": file_record(train_log, repo_root),
                "evaluation_log": file_record(eval_log, repo_root),
                "best_checkpoint": file_record(checkpoint, repo_root),
                "last_checkpoint": file_record(last_checkpoint, repo_root),
            }
        )

    inputs = {
        "selection": file_record(selection_path, repo_root),
        "checkpoint": file_record(checkpoint_path, repo_root),
        "batch_status": file_record(batch_path, repo_root),
        "summary": file_record(summary_path, repo_root),
        "summary_aggregate": file_record(aggregate_path, repo_root),
        "cache_files": [cache_inputs[key] for key in sorted(cache_inputs)],
        "attempts": attempt_inputs,
    }
    return LoadedRun(spec, selection, batch, parsed_rows, inputs)


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def build_long_rows(runs: list[LoadedRun], repo_root: Path) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        for key in sorted(run.rows):
            row = run.rows[key]
            item = {
                "alignment_seed": run.spec.alignment_seed,
                "source_git_commit": run.batch["git_commit"],
                "params_sha256": run.batch["params_sha256"],
                "selection_path": display_path(run.spec.selection_path, repo_root),
                "summary_path": display_path(run.spec.rerank_root / "summary.csv", repo_root),
                "candidate_type": row["candidate_type"],
                "model_type": row["model_type"],
                "ablation": row["ablation"],
                "reranker_seed": row["seed"],
                "metric_for_best": row["metric_for_best"],
                "best_epoch": row["best_epoch"],
                "best_val_metric": row["best_val_metric"],
                "stop_epoch": row["stop_epoch"],
                "early_stopped": row["early_stopped"],
            }
            item.update({metric: row[metric] for metric in (*BASE_METRICS, *RERANK_METRICS)})
            output.append(item)
    return output


def build_paired_rows(runs: list[LoadedRun]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        for candidate in CANDIDATE_TYPES:
            for seed in RERANKER_SEEDS:
                pointwise = run.rows[(candidate, "pointwise", seed)]
                transformer = run.rows[(candidate, "transformer", seed)]
                item: dict[str, Any] = {
                    "alignment_seed": run.spec.alignment_seed,
                    "candidate_type": candidate,
                    "reranker_seed": seed,
                }
                for metric in RERANK_METRICS:
                    short = metric.removeprefix("rerank_")
                    item[f"pointwise_{short}"] = pointwise[metric]
                    item[f"transformer_{short}"] = transformer[metric]
                    item[f"delta_{short}"] = transformer[metric] - pointwise[metric]
                output.append(item)
    return output


def build_alignment_rows(
    runs: list[LoadedRun], paired_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        for candidate in CANDIDATE_TYPES:
            base = run.rows[(candidate, "pointwise", RERANKER_SEEDS[0])]
            item: dict[str, Any] = {
                "alignment_seed": run.spec.alignment_seed,
                "candidate_type": candidate,
                "n_reranker_seeds": len(RERANKER_SEEDS),
            }
            item.update({metric: base[metric] for metric in BASE_METRICS})
            for model in MODEL_TYPES:
                for metric in RERANK_METRICS:
                    values = [run.rows[(candidate, model, seed)][metric] for seed in RERANKER_SEEDS]
                    mean, std = mean_std(values)
                    short = metric.removeprefix("rerank_")
                    item[f"{model}_{short}_mean"] = mean
                    item[f"{model}_{short}_std"] = std
            deltas = [
                row
                for row in paired_rows
                if row["alignment_seed"] == run.spec.alignment_seed
                and row["candidate_type"] == candidate
            ]
            for metric in RERANK_METRICS:
                short = metric.removeprefix("rerank_")
                values = [row[f"delta_{short}"] for row in deltas]
                mean, std = mean_std(values)
                item[f"delta_{short}_mean"] = mean
                item[f"delta_{short}_std"] = std
                item[f"delta_{short}_positive"] = sum(value > 0 for value in values)
                item[f"delta_{short}_zero"] = sum(value == 0 for value in values)
                item[f"delta_{short}_negative"] = sum(value < 0 for value in values)
            output.append(item)
    return output


def build_cross_alignment_rows(alignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for candidate in CANDIDATE_TYPES:
        group = [row for row in alignment_rows if row["candidate_type"] == candidate]
        require_equal(len(group), len(EXPECTED_ALIGNMENT_SEEDS), f"alignment count {candidate}")
        item: dict[str, Any] = {
            "candidate_type": candidate,
            "n_alignment_seeds": len(group),
        }
        for metric in BASE_METRICS:
            values = [row[metric] for row in group]
            mean, std = mean_std(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        for model in MODEL_TYPES:
            for metric in RERANK_METRICS:
                short = metric.removeprefix("rerank_")
                values = [row[f"{model}_{short}_mean"] for row in group]
                mean, std = mean_std(values)
                item[f"{model}_{short}_alignment_mean"] = mean
                item[f"{model}_{short}_alignment_std"] = std
                item[f"{model}_{short}_alignment_min"] = min(values)
                item[f"{model}_{short}_alignment_max"] = max(values)
        for metric in RERANK_METRICS:
            short = metric.removeprefix("rerank_")
            values = [row[f"delta_{short}_mean"] for row in group]
            mean, std = mean_std(values)
            item[f"delta_{short}_alignment_mean"] = mean
            item[f"delta_{short}_alignment_std"] = std
            item[f"delta_{short}_alignment_min"] = min(values)
            item[f"delta_{short}_alignment_max"] = max(values)
            item[f"delta_{short}_positive_alignments"] = sum(value > 0 for value in values)
            item[f"delta_{short}_zero_alignments"] = sum(value == 0 for value in values)
            item[f"delta_{short}_negative_alignments"] = sum(value < 0 for value in values)
        output.append(item)
    return output


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_gate(alignment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate = {}
    all_positive = True
    for candidate in CANDIDATE_TYPES:
        rows = [row for row in alignment_rows if row["candidate_type"] == candidate]
        values = [row["delta_mrr_raw_mean"] for row in rows]
        directions = ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in values]
        by_candidate[candidate] = {
            "alignment_seeds": [row["alignment_seed"] for row in rows],
            "alignment_mean_deltas": values,
            "directions": directions,
        }
        all_positive = all_positive and all(value > 0 for value in values)
    if all_positive:
        return {
            "status": "manual_review_magnitude",
            "primary_metric": "rerank_mrr_raw",
            "criterion": "Transformer - Pointwise alignment-level paired means must be positive for every candidate and alignment seed; magnitude still requires review.",
            "by_candidate": by_candidate,
        }
    return {
        "status": "fail",
        "primary_metric": "rerank_mrr_raw",
        "criterion": "Transformer - Pointwise alignment-level paired means must be positive for every candidate and alignment seed; magnitude still requires review.",
        "reason": "The direction of the alignment-level paired MRR difference is not uniformly positive.",
        "by_candidate": by_candidate,
    }


def residual_gate(alignment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in alignment_rows:
        for model in MODEL_TYPES:
            top1_delta = row[f"{model}_top1_pct_mean"] - row["base_top1_pct"]
            mrr_delta = row[f"{model}_mrr_raw_mean"] - row["base_mrr_raw"]
            checks.append(
                {
                    "alignment_seed": row["alignment_seed"],
                    "candidate_type": row["candidate_type"],
                    "model_type": model,
                    "delta_top1_pct": top1_delta,
                    "delta_mrr_raw": mrr_delta,
                    "both_positive": top1_delta > 0 and mrr_delta > 0,
                }
            )
    passed = all(check["both_positive"] for check in checks)
    return {
        "status": "pass_descriptive" if passed else "fail",
        "criterion": (
            "Every alignment/candidate/model aggregate must improve both test "
            "Top-1 and MRR over its fixed base ranking. This is descriptive and "
            "does not establish statistical significance or generalization."
        ),
        "positive_cells": sum(check["both_positive"] for check in checks),
        "total_cells": len(checks),
        "checks": checks,
    }


def format_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    runs: list[LoadedRun],
    alignment_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    residual: dict[str, Any],
) -> str:
    lines = [
        "# Cross-alignment multi-seed analysis",
        "",
        "This report is generated from the three original per-run `summary.csv` files.",
        "Reranker-seed combinations are paired within each alignment checkpoint; cross-alignment",
        "statistics use the three alignment-level estimates rather than flattening nine runs.",
        "",
        "## Provenance",
        "",
        "| Alignment seed | Source commit | Params SHA256 | Checkpoint SHA256 |",
        "| ---: | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            "| {seed} | `{commit}` | `{params}` | `{checkpoint}` |".format(
                seed=run.spec.alignment_seed,
                commit=run.batch["git_commit"],
                params=run.batch["params_sha256"],
                checkpoint=run.selection["checkpoint_sha256"],
            )
        )
    lines.extend(
        [
            "",
            "## Alignment-level results",
            "",
            "Top-1 differences are percentage points; MRR differences use the raw [0, 1] scale.",
            "",
            "| Alignment | Candidate | Base Top-1 | Pointwise Top-1 | Transformer Top-1 | ΔTop-1 | Pointwise MRR | Transformer MRR | ΔMRR |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in alignment_rows:
        lines.append(
            "| {alignment_seed} | {candidate_type} | {base} | {pointwise_top1} | {transformer_top1} | {delta_top1} | {pointwise_mrr} | {transformer_mrr} | {delta_mrr} |".format(
                alignment_seed=row["alignment_seed"],
                candidate_type=row["candidate_type"],
                base=format_number(row["base_top1_pct"], 4),
                pointwise_top1=format_number(row["pointwise_top1_pct_mean"], 4),
                transformer_top1=format_number(row["transformer_top1_pct_mean"], 4),
                delta_top1=format_number(row["delta_top1_pct_mean"], 4),
                pointwise_mrr=format_number(row["pointwise_mrr_raw_mean"]),
                transformer_mrr=format_number(row["transformer_mrr_raw_mean"]),
                delta_mrr=format_number(row["delta_mrr_raw_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cross-alignment descriptive summary",
            "",
            "The mean and sample standard deviation below use the three alignment-level",
            "paired estimates. They are descriptive statistics, not confidence intervals",
            "or statistical-significance tests.",
            "",
            "| Candidate | Mean ΔTop-1 | SD ΔTop-1 | Mean ΔMRR | SD ΔMRR | Positive alignments (Top-1 / MRR) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cross_rows:
        lines.append(
            "| {candidate} | {top1_mean} | {top1_std} | {mrr_mean} | {mrr_std} | {top1_positive}/3 / {mrr_positive}/3 |".format(
                candidate=row["candidate_type"],
                top1_mean=format_number(row["delta_top1_pct_alignment_mean"], 6),
                top1_std=format_number(row["delta_top1_pct_alignment_std"], 6),
                mrr_mean=format_number(row["delta_mrr_raw_alignment_mean"], 6),
                mrr_std=format_number(row["delta_mrr_raw_alignment_std"], 6),
                top1_positive=row["delta_top1_pct_positive_alignments"],
                mrr_positive=row["delta_mrr_raw_positive_alignments"],
            )
        )
    top1_positive = sum(row["delta_top1_pct"] > 0 for row in paired_rows)
    top1_zero = sum(row["delta_top1_pct"] == 0 for row in paired_rows)
    top1_negative = len(paired_rows) - top1_positive - top1_zero
    mrr_positive = sum(row["delta_mrr_raw"] > 0 for row in paired_rows)
    mrr_zero = sum(row["delta_mrr_raw"] == 0 for row in paired_rows)
    mrr_negative = len(paired_rows) - mrr_positive - mrr_zero
    commits = sorted({run.batch["git_commit"] for run in runs})
    lines.extend(
        [
            "",
            "## Claim gate",
            "",
            f"- Status: `{gate['status']}`.",
            f"- Fine-grained pair directions (positive / zero / negative): Top-1 {top1_positive}/{top1_zero}/{top1_negative}; MRR {mrr_positive}/{mrr_zero}/{mrr_negative}.",
            "- Decision: do not claim that Set Transformer consistently outperforms Pointwise across alignment checkpoints.",
            f"- Residual-vs-base descriptive gate: `{residual['status']}` ({residual['positive_cells']}/{residual['total_cells']} aggregate cells improve both Top-1 and MRR).",
            "- Supported boundary: supervised residual reranking improves over the base retriever in these internal descriptive results, while the independent benefit of candidate interaction remains unestablished.",
            f"- The three alignments were produced by {len(commits)} source commit(s); results retain per-alignment provenance.",
            "- No confidence intervals, hypothesis tests, cross-dataset evidence, or causal claims are provided by this analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(specs: list[RunSpec], output_root: Path, repo_root: Path) -> dict[str, Any]:
    seeds = tuple(sorted(spec.alignment_seed for spec in specs))
    require_equal(seeds, EXPECTED_ALIGNMENT_SEEDS, "alignment seed set")
    if len({spec.alignment_seed for spec in specs}) != len(specs):
        raise AnalysisError("Alignment seeds must not contain duplicates")
    runs = [load_run(spec, repo_root) for spec in sorted(specs, key=lambda item: item.alignment_seed)]
    canonical_manifest_path = repo_root / "paper" / "adma2026_artifact_manifest.json"
    canonical_manifest = load_json(canonical_manifest_path, "canonical artifact manifest")
    canonical_cache_hashes = {}
    artifacts = canonical_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AnalysisError("Canonical artifact manifest artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_id = artifact.get("id")
        if isinstance(artifact_id, str) and artifact_id.startswith("cache."):
            canonical_cache_hashes[artifact_id.removeprefix("cache.")] = artifact
    require_equal(len(canonical_cache_hashes), 6, "canonical cache artifact count")
    seed42_run = next(run for run in runs if run.spec.alignment_seed == 42)
    for cache in seed42_run.input_files["cache_files"]:
        key = f"{cache['candidate_type']}.{cache['split']}"
        canonical = canonical_cache_hashes.get(key)
        if not isinstance(canonical, dict):
            raise AnalysisError(f"Missing canonical cache artifact {key}")
        require_equal(cache["sha256"], canonical.get("sha256"), f"canonical cache hash {key}")
        require_equal(cache["size_bytes"], canonical.get("size_bytes"), f"canonical cache size {key}")
        require_equal(
            f"repo:{cache['path']}", canonical.get("location"), f"canonical cache path {key}"
        )
    params_hashes = {run.batch["params_sha256"] for run in runs}
    require_equal(len(params_hashes), 1, "params.yaml hash count")
    exclusions = {
        tuple(run.selection.get("exclude_val_query_indices", [])) for run in runs
    }
    require_equal(len(exclusions), 1, "validation exclusion protocol count")

    long_rows = build_long_rows(runs, repo_root)
    paired_rows = build_paired_rows(runs)
    alignment_rows = build_alignment_rows(runs, paired_rows)
    cross_rows = build_cross_alignment_rows(alignment_rows)
    gate = claim_gate(alignment_rows)
    residual = residual_gate(alignment_rows)

    output_root = output_root.resolve()
    paths = {
        "summary_long": output_root / "summary_long.csv",
        "paired_deltas": output_root / "paired_deltas.csv",
        "alignment_summary": output_root / "alignment_summary.csv",
        "cross_alignment_summary": output_root / "cross_alignment_summary.csv",
        "report": output_root / "report.md",
        "manifest": output_root / "analysis_manifest.json",
    }
    atomic_write_csv(paths["summary_long"], long_rows)
    atomic_write_csv(paths["paired_deltas"], paired_rows)
    atomic_write_csv(paths["alignment_summary"], alignment_rows)
    atomic_write_csv(paths["cross_alignment_summary"], cross_rows)
    report = build_report(
        runs, alignment_rows, cross_rows, paired_rows, gate, residual
    )
    atomic_write_text(paths["report"], report)

    manifest = {
        "schema_version": 2,
        "analysis": "cross_alignment_multiseed_paired_differences",
        "design": {
            "alignment_seeds": list(EXPECTED_ALIGNMENT_SEEDS),
            "candidate_types": list(CANDIDATE_TYPES),
            "model_types": list(MODEL_TYPES),
            "reranker_seeds": list(RERANKER_SEEDS),
            "pairing_key": ["alignment_seed", "candidate_type", "reranker_seed"],
            "cross_alignment_unit": "alignment_seed",
            "n_alignment_seeds": len(EXPECTED_ALIGNMENT_SEEDS),
            "validation_query_exclusions": list(VALIDATION_EXCLUSIONS),
        },
        "units": {
            "topk_metrics": "percentage_points",
            "mrr": "raw_[0,1]",
            "delta": "transformer_minus_pointwise",
        },
        "compatibility": {
            "params_sha256": next(iter(params_hashes)),
            "source_commits": sorted({run.batch["git_commit"] for run in runs}),
            "same_source_commit": len({run.batch["git_commit"] for run in runs}) == 1,
        },
        "inputs": [
            {
                "alignment_seed": run.spec.alignment_seed,
                "source_git_commit": run.batch["git_commit"],
                "params_sha256": run.batch["params_sha256"],
                "checkpoint_sha256": run.selection["checkpoint_sha256"],
                "files": run.input_files,
            }
            for run in runs
        ],
        "canonical_cache_crosscheck": file_record(
            canonical_manifest_path, repo_root
        ),
        "counts": {
            "summary_rows": len(long_rows),
            "paired_rows": len(paired_rows),
            "alignment_candidate_rows": len(alignment_rows),
            "cross_alignment_rows": len(cross_rows),
            "cache_artifacts": sum(
                len(run.input_files["cache_files"]) for run in runs
            ),
            "cache_total_size_bytes": sum(
                artifact["size_bytes"]
                for run in runs
                for artifact in run.input_files["cache_files"]
            ),
            "attempts": sum(len(run.input_files["attempts"]) for run in runs),
        },
        "claim_gate": gate,
        "residual_vs_base_gate": residual,
        "outputs": {
            key: file_record(path, repo_root)
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    atomic_write_text(
        paths["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze paired Pointwise/Transformer results across "
            "alignment seeds 42, 43, and 44."
        )
    )
    parser.add_argument(
        "--alignment-run",
        action="append",
        nargs=3,
        metavar=("SEED", "SELECTION_JSON", "RERANK_ROOT"),
        required=True,
        help="Repeat once per alignment seed.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = []
    for raw_seed, raw_selection, raw_root in args.alignment_run:
        try:
            seed = int(raw_seed)
        except ValueError as error:
            parser.error(f"Invalid alignment seed: {raw_seed}")
            raise AssertionError from error
        specs.append(RunSpec(seed, Path(raw_selection), Path(raw_root)))
    args.alignment_runs = specs
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    try:
        manifest = run_analysis(args.alignment_runs, args.output_root, repo_root)
    except AnalysisError as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 2
    print(f"Wrote cross-alignment analysis to {args.output_root.resolve()}")
    print(f"Claim gate: {manifest['claim_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
