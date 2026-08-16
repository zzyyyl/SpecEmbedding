#!/usr/bin/env python3
"""Validate and freeze the canonical ADMA 2026 experiment artifacts.

The generated manifest is deterministic and contains only repository-relative
or logical data locations.  It deliberately omits absolute paths, usernames,
hostnames, process IDs, GPU UUIDs, and timestamps copied from runtime status
files.  It is an internal reproducibility record: because it includes the
source Git commit, it must still undergo anonymity review before release.  Its
artifact inventory is not an anonymous packaging allowlist; referenced status
and log files can contain machine identity and must be redacted before release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation_protocol import EVALUATION_IDENTITY_PROTOCOL

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "a2280d2828ce872da1f69319b49e0ef7f1bed572"
RUN_PREFIX = "a2280d2_massspecgym_nopretrain_valoverlapclean"
PARAMS_SHA256 = "b6260dad043f9c0f1cb4ff36dc7b7bf8bac4481998aa36b4d55f44392e1b0bc6"
ALIGNMENT_SHA256 = "ad5d1eb76805c51563349f259a4b4c935336064b6171a4e650472b78aeeaa06f"
VAL_EXCLUSIONS = [7686, 7687, 7688, 8464, 8465, 8466]
CANDIDATES = ("mass", "formula")
MODELS = ("pointwise", "transformer")
SEEDS = (42, 43, 44)
SPLITS = ("train", "val", "test")

EXPECTED_DATA_SHA256 = {
    "train": "071b6c28ade639261118fdcb0da2510efcea764b2978e2a5d0b61a95a1960c48",
    "val": "bd13a336e86fb35e77a534e9415cd40cc1a9f15b4a33174c85573b8639c9ca6e",
    "test": "2309cd689b06a493c66e22f4aa70613b0cfbe2f8fd1d82e605d657f27015d660",
}
EXPECTED_CANDIDATE_SHA256 = {
    "mass": "b4ddf39783ea2dceb32217eab68b71133572f63cc482ba9d17ed9a530c906a8f",
    "formula": "0b0d8ffff166eeda9c9dc4060618f823ae4984a71d045119b13b01b5dd5f8194",
}
EXPECTED_TEST_CACHE_SHA256 = {
    "mass": "6842fa615b7cf07fbf3c82825d75e72cddcd2e0da86dcdb3c5d8e3679f931ea6",
    "formula": "f2459430637f3495d22f40d242fe7c6210d35993aeca5c9667229a917372b0d0",
}
EXPECTED_MCES_CHECKPOINT_SHA256 = {
    "mass": "aedf5e82089ec8ec350da2ab04718a879f5019756f701c1a1e1da3559e541c57",
    "formula": "ca13cd793ba87b6eceefb6aa3c1dc46e95eb8d36a92ddeb31cca7d04cf552ba8",
}
EXPECTED_RESULT_SHA256 = {
    "summary": "27cebd51c9e667ca14596809e716d7133d08f8a7aef1899140c8aac0659911ef",
    "summary_aggregate": "46d66854a0ec8f94a87ed0855a6dc94b565f1d32f423063d937bcd2b5a732df7",
}
SUMMARY_COLUMNS = (
    "candidate_type",
    "model_type",
    "ablation",
    "seed",
    "metric_for_best",
    "best_epoch",
    "best_val_metric",
    "stop_epoch",
    "early_stopped",
    "upper_bound_pct",
    "base_top1_pct",
    "base_top5_pct",
    "base_top10_pct",
    "base_top20_pct",
    "base_mrr_raw",
    "rerank_top1_pct",
    "rerank_top5_pct",
    "rerank_top10_pct",
    "rerank_top20_pct",
    "rerank_mrr_raw",
    "attempt_dir",
)
SUMMARY_FLOAT_COLUMNS = (
    "best_val_metric",
    "upper_bound_pct",
    "base_top1_pct",
    "base_top5_pct",
    "base_top10_pct",
    "base_top20_pct",
    "base_mrr_raw",
    "rerank_top1_pct",
    "rerank_top5_pct",
    "rerank_top10_pct",
    "rerank_top20_pct",
    "rerank_mrr_raw",
)
AGGREGATE_METRICS = (
    "rerank_top1_pct",
    "rerank_top5_pct",
    "rerank_top10_pct",
    "rerank_top20_pct",
    "rerank_mrr_raw",
    "best_val_metric",
    "best_epoch",
    "stop_epoch",
    "early_stopped",
)
AGGREGATE_COLUMNS = (
    "candidate_type",
    "model_type",
    "ablation",
    "num_seeds",
    *(
        field
        for metric in AGGREGATE_METRICS
        for field in (f"{metric}_num_seeds", f"{metric}_mean", f"{metric}_std")
    ),
)
EXPECTED_MCES = {
    "mass": {
        "total_queries": 17556,
        "base_top1_accuracy_pct": 47.4596,
        "rerank_top1_accuracy_pct": 68.74,
        "base_mces_at_1": 15.3681,
        "rerank_mces_at_1": 7.7057,
    },
    "formula": {
        "total_queries": 17556,
        "base_top1_accuracy_pct": 63.1009,
        "rerank_top1_accuracy_pct": 74.7494,
        "base_mces_at_1": 5.4389,
        "rerank_mces_at_1": 3.0913,
    },
}


class AuditError(RuntimeError):
    """Raised when a canonical artifact fails validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the canonical ADMA 2026 artifacts."
    )
    parser.add_argument(
        "--output",
        default="paper/adma2026_artifact_manifest.json",
        help="Repository-relative manifest path.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and hash everything without writing a manifest.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify that an existing manifest exactly matches current artifacts.",
    )
    return parser.parse_args()


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def require_clean_worktree() -> None:
    status = run_git("status", "--porcelain", "--untracked-files=all").stdout.strip()
    if status:
        raise AuditError(
            "Refusing to freeze artifacts from a dirty worktree. Commit or stash changes first:\n"
            + status
        )


def require_source_commit() -> None:
    run_git("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
    runtime_paths = [
        "params.yaml",
        "SpecEmbedding",
        "eval_rerank.py",
        "prepare_rerank_cache.py",
        "run_pipeline.py",
        "run_rerank_multiseed.py",
        "train_rerank.py",
    ]
    for runtime_path in runtime_paths:
        run_git("cat-file", "-e", f"{SOURCE_COMMIT}:{runtime_path}")


def git_blob_artifact(
    artifact_id: str, commit: str, repository_path: str
) -> dict[str, Any]:
    path = Path(repository_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != repository_path:
        raise AuditError(f"Invalid repository path for {artifact_id}: {repository_path!r}")
    result = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    payload = result.stdout
    if not payload:
        raise AuditError(
            f"Missing or empty Git blob for {artifact_id}: {commit}:{repository_path}"
        )
    return {
        "id": artifact_id,
        "location": f"git:{commit}:{repository_path}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AuditError(f"Missing or empty {label}: {path}")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object for {label}: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_location(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise AuditError(f"Repository artifact escapes repository root: {path}") from exc
    return f"repo:{relative.as_posix()}"


def artifact(artifact_id: str, path: Path, location: str | None = None) -> dict[str, Any]:
    require_file(path, artifact_id)
    return {
        "id": artifact_id,
        "location": location or repo_location(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def require_recorded_repo_path(raw: Any, expected: Path, label: str) -> Path:
    """Match a recorded path by repository-relative suffix.

    Historical status files contain absolute paths from the original clone.
    Comparing the repository-relative suffix keeps the audit portable without
    weakening the expected canonical location inside the repository.
    """

    if not isinstance(raw, str) or not raw:
        raise AuditError(f"Missing recorded path for {label}")
    try:
        expected_relative = expected.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise AuditError(f"Expected repository path escapes root for {label}: {expected}") from exc

    recorded = Path(raw).expanduser()
    if recorded.is_absolute():
        expected_parts = expected_relative.parts
        if tuple(recorded.parts[-len(expected_parts) :]) != expected_parts:
            raise AuditError(
                f"{label} mismatch: expected repository suffix "
                f"{expected_relative.as_posix()!r}, found {raw!r}"
            )
    else:
        normalized = Path(os.path.normpath(raw))
        if ".." in normalized.parts or normalized != expected_relative:
            raise AuditError(
                f"{label} mismatch: expected {expected_relative.as_posix()!r}, found {raw!r}"
            )
    return expected.resolve()


def canonical_cache_path(candidate: str, split: str) -> Path:
    return (
        REPO_ROOT
        / "rerank_cache"
        / f"{RUN_PREFIX}_{candidate}_topk40"
        / f"massspecgym_{candidate}_{split}.pt"
    )


def validate_alignment(artifacts: list[dict[str, Any]]) -> Path:
    align_root = REPO_ROOT / "checkpoints_align" / RUN_PREFIX
    selection_path = align_root / "alignment_selection.json"
    checkpoint_path = align_root / "best_model_stage2.pth"
    selection = load_json(selection_path, "alignment selection")

    require_equal(selection.get("dataset_type"), "massspecgym", "alignment dataset")
    require_equal(selection.get("seed"), 42, "alignment seed")
    require_equal(
        selection.get("exclude_val_query_indices"), VAL_EXCLUSIONS, "alignment exclusions"
    )
    require_equal(
        selection.get("checkpoint_sha256"), ALIGNMENT_SHA256, "recorded alignment hash"
    )
    require_equal(
        require_recorded_repo_path(
            selection.get("checkpoint"), checkpoint_path, "alignment checkpoint path"
        ),
        checkpoint_path.resolve(),
        "alignment checkpoint path",
    )
    stage2 = selection.get("stages", {}).get("stage2", {})
    require_equal(stage2.get("best_epoch"), 16, "alignment best epoch")
    require_equal(stage2.get("stop_epoch"), 21, "alignment stop epoch")
    require_equal(stage2.get("early_stopped"), True, "alignment early-stop state")

    checkpoint_artifact = artifact("alignment.best_checkpoint", checkpoint_path)
    require_equal(checkpoint_artifact["sha256"], ALIGNMENT_SHA256, "alignment hash")
    artifacts.append(artifact("alignment.selection", selection_path))
    artifacts.append(checkpoint_artifact)

    data_path = selection.get("data_path")
    if not isinstance(data_path, str) or not data_path:
        raise AuditError("alignment selection does not contain a usable data_path")
    return Path(data_path).expanduser().resolve() / "MassSpecGym"


def validate_data(data_root: Path, artifacts: list[dict[str, Any]]) -> None:
    for split in SPLITS:
        item = artifact(
            f"data.{split}",
            data_root / f"{split}.pkl",
            f"data:MassSpecGym/{split}.pkl",
        )
        require_equal(item["sha256"], EXPECTED_DATA_SHA256[split], f"{split} data hash")
        artifacts.append(item)
    for candidate in CANDIDATES:
        item = artifact(
            f"data.candidates_{candidate}",
            data_root / f"candidates_{candidate}.pkl",
            f"data:MassSpecGym/candidates_{candidate}.pkl",
        )
        require_equal(
            item["sha256"], EXPECTED_CANDIDATE_SHA256[candidate], f"{candidate} candidates hash"
        )
        artifacts.append(item)


def require_close(actual: Any, expected: Any, label: str) -> None:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"Expected numeric values for {label}: {actual!r}, {expected!r}") from exc
    if not math.isclose(actual_float, expected_float, rel_tol=1e-12, abs_tol=1e-12):
        raise AuditError(f"{label} mismatch: expected {expected_float!r}, found {actual_float!r}")


def parse_bool(raw: Any, label: str) -> bool:
    if raw == "True" or raw is True:
        return True
    if raw == "False" or raw is False:
        return False
    raise AuditError(f"Expected True/False for {label}, found {raw!r}")


def load_csv_rows(path: Path, label: str, columns: tuple[str, ...]) -> list[dict[str, str]]:
    require_file(path, label)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_equal(tuple(reader.fieldnames or ()), columns, f"{label} columns")
        return list(reader)


def parse_eval_metrics(path: Path, label: str) -> dict[str, float]:
    text = require_file(path, label).read_text(encoding="utf-8", errors="replace")

    def extract_one(pattern: str, metric_label: str, source: str = text) -> float:
        matches = re.findall(pattern, source, flags=re.DOTALL)
        if len(matches) != 1:
            raise AuditError(
                f"Expected exactly one {metric_label} value in {path}, found {len(matches)}"
            )
        return float(matches[0])

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
            raise AuditError(f"Missing {section} RESULTS block in {path}")
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
        raise AuditError(f"Evaluation log contains an error marker: {path}")
    return metrics


def validate_summary_csv(
    path: Path, result_map: dict[tuple[str, str, int], Path]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = load_csv_rows(path, "multi-seed summary", SUMMARY_COLUMNS)
    expected = {(candidate, model, seed) for candidate in CANDIDATES for model in MODELS for seed in SEEDS}
    parsed_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        key = (row["candidate_type"], row["model_type"], seed)
        if key in parsed_rows:
            raise AuditError(f"Duplicate multi-seed summary row: {key}")
        if key not in result_map:
            raise AuditError(f"Unexpected multi-seed summary row: {key}")
        require_equal(row["ablation"], "full", f"summary ablation {key}")
        require_equal(row["metric_for_best"], "mrr", f"summary selection metric {key}")
        require_recorded_repo_path(row["attempt_dir"], result_map[key], f"summary attempt {key}")
        parsed: dict[str, Any] = dict(row)
        parsed["seed"] = seed
        parsed["best_epoch"] = int(row["best_epoch"])
        parsed["stop_epoch"] = int(row["stop_epoch"])
        parsed["early_stopped"] = parse_bool(row["early_stopped"], f"summary early stop {key}")
        for field in SUMMARY_FLOAT_COLUMNS:
            parsed[field] = float(row[field])
        parsed_rows[key] = parsed
    require_equal(len(rows), 12, "multi-seed summary row count")
    require_equal(set(parsed_rows), expected, "multi-seed summary experiments")
    return parsed_rows


def validate_aggregate_csv(
    path: Path, summary_rows: dict[tuple[str, str, int], dict[str, Any]]
) -> None:
    rows = load_csv_rows(path, "aggregate summary", AGGREGATE_COLUMNS)
    observed: set[tuple[str, str]] = set()
    expected = {(candidate, model) for candidate in CANDIDATES for model in MODELS}
    for row in rows:
        key = (row["candidate_type"], row["model_type"])
        if key in observed:
            raise AuditError(f"Duplicate aggregate summary row: {key}")
        observed.add(key)
        require_equal(row["ablation"], "full", f"aggregate ablation {key}")
        group = [
            summary_rows[(key[0], key[1], seed)]
            for seed in SEEDS
        ]
        require_equal(int(row["num_seeds"]), len(group), f"aggregate seed count {key}")
        for metric in AGGREGATE_METRICS:
            values = [float(item[metric]) for item in group]
            require_equal(
                int(row[f"{metric}_num_seeds"]), len(values), f"aggregate {metric} count {key}"
            )
            require_close(
                row[f"{metric}_mean"], statistics.mean(values), f"aggregate {metric} mean {key}"
            )
            require_close(
                row[f"{metric}_std"], statistics.stdev(values), f"aggregate {metric} std {key}"
            )
    require_equal(len(rows), 4, "aggregate summary row count")
    require_equal(observed, expected, "aggregate summary experiments")


def validate_multiseed(artifacts: list[dict[str, Any]]) -> dict[tuple[str, str, int], Path]:
    root = REPO_ROOT / "checkpoints_rerank" / f"{RUN_PREFIX}_topk40_multiseed"
    batch_path = root / "batch_status.json"
    summary_path = root / "summary.csv"
    aggregate_path = root / "summary_aggregate.csv"
    batch = load_json(batch_path, "multi-seed batch status")
    require_equal(batch.get("state"), "complete", "multi-seed batch state")
    require_equal(batch.get("git_commit"), SOURCE_COMMIT, "multi-seed source commit")
    require_equal(batch.get("params_sha256"), PARAMS_SHA256, "multi-seed params hash")
    require_equal(batch.get("exclude_val_query_indices"), VAL_EXCLUSIONS, "multi-seed exclusions")
    require_equal(batch.get("errors"), [], "multi-seed errors")
    require_equal(batch.get("mces"), False, "multi-seed MCES mode")

    expected_keys = {
        (candidate, model, seed)
        for candidate in CANDIDATES
        for model in MODELS
        for seed in SEEDS
    }
    result_map: dict[tuple[str, str, int], Path] = {}
    batch_results = batch.get("results")
    if not isinstance(batch_results, list):
        raise AuditError("multi-seed batch results must be a list")
    for row in batch_results:
        if not isinstance(row, dict):
            raise AuditError(f"Invalid multi-seed batch result: {row!r}")
        key = (row.get("candidate_type"), row.get("model_type"), row.get("seed"))
        if key not in expected_keys:
            raise AuditError(f"Unexpected multi-seed result: {key}")
        require_equal(row.get("state"), "complete", f"multi-seed result {key}")
        if key in result_map:
            raise AuditError(f"Duplicate multi-seed result: {key}")
        candidate, model, seed = key
        attempt = root / str(candidate) / str(model) / f"seed{seed}" / "attempt_001"
        require_recorded_repo_path(row.get("attempt_dir"), attempt, f"batch attempt {key}")
        result_map[key] = attempt
    require_equal(set(result_map), expected_keys, "multi-seed completed result set")

    summary_rows = validate_summary_csv(summary_path, result_map)
    validate_aggregate_csv(aggregate_path, summary_rows)
    batch_item = artifact("reranker.batch_status", batch_path)
    summary_item = artifact("reranker.summary", summary_path)
    aggregate_item = artifact("reranker.summary_aggregate", aggregate_path)
    require_equal(summary_item["sha256"], EXPECTED_RESULT_SHA256["summary"], "summary hash")
    require_equal(
        aggregate_item["sha256"],
        EXPECTED_RESULT_SHA256["summary_aggregate"],
        "aggregate summary hash",
    )
    artifacts.extend([batch_item, summary_item, aggregate_item])

    cache_artifacts_added: set[tuple[str, str]] = set()
    for key in sorted(result_map):
        candidate, model, seed = key
        attempt = result_map[key]
        status_path = attempt / "status.json"
        checkpoint_path = attempt / "best_reranker.pth"
        eval_log_path = attempt / "eval_rerank.log"
        status = load_json(status_path, f"reranker status {key}")
        require_equal(status.get("state"), "complete", f"reranker state {key}")
        require_equal(status.get("candidate_type"), candidate, f"candidate {key}")
        require_equal(status.get("model_type"), model, f"model {key}")
        require_equal(status.get("ablation"), "full", f"ablation {key}")
        require_equal(status.get("seed"), seed, f"seed {key}")
        require_equal(status.get("git_commit"), SOURCE_COMMIT, f"source commit {key}")
        require_equal(status.get("params_sha256"), PARAMS_SHA256, f"params hash {key}")
        require_equal(status.get("exclude_val_query_indices"), VAL_EXCLUSIONS, f"exclusions {key}")
        require_recorded_repo_path(status.get("attempt_dir"), attempt, f"status attempt {key}")
        selection = status.get("training_selection", {})
        if not isinstance(selection, dict):
            raise AuditError(f"Invalid training selection {key}: {selection!r}")
        summary_row = summary_rows[key]
        require_equal(selection.get("metric_for_best"), "mrr", f"selection metric {key}")
        require_equal(selection.get("best_epoch"), summary_row["best_epoch"], f"best epoch {key}")
        require_close(
            selection.get("best_val_metric"), summary_row["best_val_metric"], f"best val metric {key}"
        )
        require_equal(selection.get("stop_epoch"), summary_row["stop_epoch"], f"stop epoch {key}")
        require_equal(selection.get("early_stopped"), True, f"early-stop state {key}")
        require_equal(
            selection.get("early_stopped"), summary_row["early_stopped"], f"summary early stop {key}"
        )

        eval_metrics = parse_eval_metrics(eval_log_path, f"reranker evaluation log {key}")
        for metric, actual in eval_metrics.items():
            require_close(actual, summary_row[metric], f"evaluation {metric} {key}")

        cache_records = status.get("cache_files", {})
        if not isinstance(cache_records, dict):
            raise AuditError(f"Invalid cache records {key}: {cache_records!r}")
        for split in SPLITS:
            expected_path = canonical_cache_path(candidate, split)
            record = cache_records.get(split, {})
            if not isinstance(record, dict):
                raise AuditError(f"Invalid cache record {candidate}/{split}: {record!r}")
            require_equal(
                require_recorded_repo_path(
                    record.get("path"), expected_path, f"cache path {candidate}/{split}"
                ),
                expected_path.resolve(),
                f"cache path {candidate}/{split}",
            )
            require_file(expected_path, f"{candidate} {split} cache")
            require_equal(
                record.get("size_bytes"), expected_path.stat().st_size, f"cache size {candidate}/{split}"
            )
            cache_key = (candidate, split)
            if cache_key not in cache_artifacts_added:
                item = artifact(f"cache.{candidate}.{split}", expected_path)
                if split == "test":
                    require_equal(
                        item["sha256"],
                        EXPECTED_TEST_CACHE_SHA256[candidate],
                        f"test cache hash {candidate}",
                    )
                artifacts.append(item)
                cache_artifacts_added.add(cache_key)

        artifacts.append(artifact(f"reranker.{candidate}.{model}.seed{seed}.status", status_path))
        artifacts.append(
            artifact(f"reranker.{candidate}.{model}.seed{seed}.eval_log", eval_log_path)
        )
        artifacts.append(
            artifact(f"reranker.{candidate}.{model}.seed{seed}.best_checkpoint", checkpoint_path)
        )
    return result_map


def validate_mces(
    result_map: dict[tuple[str, str, int], Path], artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    root = REPO_ROOT / "checkpoints_rerank" / f"{RUN_PREFIX}_topk40_mces_seed42_transformer"
    batch_path = root / "batch_status.json"
    batch = load_json(batch_path, "MCES batch status")
    require_equal(batch.get("state"), "complete", "MCES batch state")
    require_equal(batch.get("source_commit"), SOURCE_COMMIT, "MCES source commit")
    require_equal(set(batch.get("results", {})), set(CANDIDATES), "MCES result candidates")
    artifacts.append(artifact("mces.batch_status", batch_path))

    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        attempt = root / candidate / "attempt_001"
        require_recorded_repo_path(
            batch["results"][candidate], attempt, f"MCES batch attempt {candidate}"
        )
        status_path = attempt / "status.json"
        eval_log = attempt / "eval_rerank.log"
        driver_log = attempt / "driver.log"
        status = load_json(status_path, f"MCES status {candidate}")
        require_equal(status.get("state"), "complete", f"MCES state {candidate}")
        require_equal(status.get("candidate_type"), candidate, f"MCES candidate {candidate}")
        require_equal(status.get("model_type"), "transformer", f"MCES model {candidate}")
        require_equal(status.get("seed"), 42, f"MCES seed {candidate}")
        require_equal(status.get("mces_enabled"), True, f"MCES enabled {candidate}")
        require_equal(status.get("source_commit"), SOURCE_COMMIT, f"MCES source {candidate}")
        require_equal(status.get("metrics"), EXPECTED_MCES[candidate], f"MCES metrics {candidate}")

        cache_path = canonical_cache_path(candidate, "test")
        checkpoint_path = result_map[(candidate, "transformer", 42)] / "best_reranker.pth"
        require_equal(
            require_recorded_repo_path(status.get("cache"), cache_path, f"MCES cache {candidate}"),
            cache_path.resolve(),
            f"MCES cache {candidate}",
        )
        require_equal(
            require_recorded_repo_path(
                status.get("checkpoint"), checkpoint_path, f"MCES checkpoint {candidate}"
            ),
            checkpoint_path.resolve(),
            f"MCES checkpoint {candidate}",
        )
        require_equal(
            status.get("cache_sha256"), EXPECTED_TEST_CACHE_SHA256[candidate], f"MCES cache hash {candidate}"
        )
        require_equal(
            status.get("checkpoint_sha256"),
            EXPECTED_MCES_CHECKPOINT_SHA256[candidate],
            f"MCES checkpoint hash {candidate}",
        )
        checkpoint_digest = sha256(checkpoint_path)
        require_equal(
            checkpoint_digest, EXPECTED_MCES_CHECKPOINT_SHA256[candidate], f"checkpoint digest {candidate}"
        )

        log_text = require_file(eval_log, f"MCES eval log {candidate}").read_text(
            encoding="utf-8", errors="replace"
        )
        required_tokens = [
            "Total queries: 17556",
            f"Base   MCES@1 : {EXPECTED_MCES[candidate]['base_mces_at_1']:.4f}",
            f"Rerank MCES@1 : {EXPECTED_MCES[candidate]['rerank_mces_at_1']:.4f}",
        ]
        for token in required_tokens:
            if log_text.count(token) != 1:
                raise AuditError(f"Expected exactly one {token!r} in {eval_log}")
        if re.search(r"Traceback|\bERROR\b|MCES calculation skipped|interrupted|killed", log_text, re.I):
            raise AuditError(f"MCES log contains an error marker: {eval_log}")

        artifacts.extend(
            [
                artifact(f"mces.{candidate}.status", status_path),
                artifact(f"mces.{candidate}.eval_log", eval_log),
                artifact(f"mces.{candidate}.driver_log", driver_log),
            ]
        )
        metrics[candidate] = dict(EXPECTED_MCES[candidate])
    return metrics


def serialize_internal_manifest(manifest: dict[str, Any]) -> str:
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    forbidden_patterns = [
        r"/(?:home|data1|Users)/",
        re.escape(str(REPO_ROOT)),
        r"(?:^|[\\/])\.git(?:[\\/]|$)",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            raise AuditError(f"Manifest contains a forbidden identity/path pattern: {pattern}")
    return text


def build_manifest() -> dict[str, Any]:
    require_source_commit()
    artifacts: list[dict[str, Any]] = []
    params_item = git_blob_artifact("config.params", SOURCE_COMMIT, "params.yaml")
    require_equal(params_item["sha256"], PARAMS_SHA256, "params.yaml hash")
    artifacts.append(params_item)

    data_root = validate_alignment(artifacts)
    validate_data(data_root, artifacts)
    result_map = validate_multiseed(artifacts)
    mces_metrics = validate_mces(result_map, artifacts)
    artifacts.sort(key=lambda item: item["id"])

    return {
        "schema_version": 2,
        "purpose": "Internal ADMA 2026 canonical experiment artifact freeze",
        "release_note": "Not for direct submission; review the recorded Git commit for anonymity.",
        "packaging_policy": (
            "Inventory only, not an anonymous bundle allowlist. Redact referenced status and log "
            "files before release because they may contain host, user, process, GPU, or timestamp data."
        ),
        "provenance_policy": (
            "Experiment runtime and configuration remain bound to source_commit; "
            "config.params is hashed from that Git tree rather than the current worktree."
        ),
        "experiment": {
            "dataset": "MassSpecGym",
            "run_prefix": RUN_PREFIX,
            "source_commit": SOURCE_COMMIT,
            "params_sha256": PARAMS_SHA256,
            "alignment_checkpoint_sha256": ALIGNMENT_SHA256,
            "alignment_seed": 42,
            "reranker_seeds": list(SEEDS),
            "candidate_types": list(CANDIDATES),
            "top_k": 40,
            "validation_query_exclusions": VAL_EXCLUSIONS,
            "test_queries": 17556,
        },
        "evaluation_identity_protocol": dict(EVALUATION_IDENTITY_PROTOCOL),
        "mces_at_1": mces_metrics,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def resolve_output(raw: str) -> Path:
    path = Path(raw)
    path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise AuditError("--output must stay inside the repository") from exc
    return path


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    output = resolve_output(args.output)
    if not args.dry_run:
        require_clean_worktree()

    manifest = build_manifest()
    text = serialize_internal_manifest(manifest)
    print(f"Validated {manifest['artifact_count']} canonical artifacts.")
    print(
        "MCES@1: mass "
        f"{manifest['mces_at_1']['mass']['base_mces_at_1']:.4f} -> "
        f"{manifest['mces_at_1']['mass']['rerank_mces_at_1']:.4f}; formula "
        f"{manifest['mces_at_1']['formula']['base_mces_at_1']:.4f} -> "
        f"{manifest['mces_at_1']['formula']['rerank_mces_at_1']:.4f}."
    )
    print(
        "Internal manifest only: review the Git commit and redact referenced status/log files "
        "before any anonymous release."
    )

    if args.dry_run:
        print(f"Dry-run complete; no manifest written. Planned output: {output.relative_to(REPO_ROOT)}")
        return 0
    if args.check:
        if not output.is_file():
            raise AuditError(f"Manifest does not exist: {output}")
        existing = output.read_text(encoding="utf-8")
        if existing != text:
            raise AuditError(f"Manifest is stale or does not match canonical artifacts: {output}")
        print(f"Manifest matches canonical artifacts: {output.relative_to(REPO_ROOT)}")
        return 0

    write_atomic(output, text)
    print(f"Wrote deterministic artifact manifest: {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AuditError,
        AttributeError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
