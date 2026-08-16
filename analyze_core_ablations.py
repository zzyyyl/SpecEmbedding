import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import analyze_alignment_multiseed as common
from evaluation_protocol import (
    EVALUATION_IDENTITY_PROTOCOL,
    EVALUATION_IDENTITY_PROTOCOL_NOTE,
)

ALIGNMENT_SEED = 42
CANDIDATE_TYPES = common.CANDIDATE_TYPES
RERANKER_SEEDS = common.RERANKER_SEEDS
ABLATIONS = (
    "no_residual",
    "no_base_score",
    "no_rank_embedding",
    "no_interaction_features",
    "listwise_only",
)
EXPECTED_OVERRIDES = {
    "no_residual": {
        **common.FULL_RERANKER_OVERRIDES,
        "use_residual_score": False,
    },
    "no_base_score": {
        **common.FULL_RERANKER_OVERRIDES,
        "use_base_score_feature": False,
        "use_residual_score": False,
    },
    "no_rank_embedding": {
        **common.FULL_RERANKER_OVERRIDES,
        "use_rank_embedding": False,
    },
    "no_interaction_features": {
        **common.FULL_RERANKER_OVERRIDES,
        "use_product_feature": False,
        "use_abs_diff_feature": False,
    },
    "listwise_only": {
        **common.FULL_RERANKER_OVERRIDES,
        "lambda_pair": 0.0,
    },
}
EXECUTION_RELEVANT_PATHS = (
    "SpecEmbedding",
    "train_rerank.py",
    "eval_rerank.py",
    "run_rerank_multiseed.py",
    "params.yaml",
)
COMPONENT_LABELS = {
    "no_residual": "residual base-score shortcut",
    "no_base_score": "joint base-score feature and residual pathway",
    "no_rank_embedding": "rank embedding",
    "no_interaction_features": "product and absolute-difference features",
    "listwise_only": "pairwise loss term",
}
COMPONENT_NOTES = {
    "no_residual": (
        "Only the residual base-score addition is removed; the base score remains "
        "an MLP input."
    ),
    "no_base_score": (
        "Both the base-score MLP feature and residual pathway are removed, so this "
        "is not a single-factor feature ablation."
    ),
    "no_rank_embedding": "Only the rank embedding is zeroed at the fixed MLP width.",
    "no_interaction_features": (
        "Product and absolute-difference features are removed; candidate "
        "self-attention remains enabled."
    ),
    "listwise_only": (
        "The pairwise loss weight is set to zero; the model architecture is unchanged."
    ),
}


@dataclass
class LoadedAblations:
    root: Path
    batch: dict[str, Any]
    retry: dict[str, Any]
    rows: dict[tuple[str, str, int], dict[str, Any]]
    attempts: dict[tuple[str, str, int], Path]
    input_files: dict[str, Any]


def expected_ablation_keys() -> set[tuple[str, str, int]]:
    return {
        (candidate, ablation, seed)
        for candidate in CANDIDATE_TYPES
        for ablation in ABLATIONS
        for seed in RERANKER_SEEDS
    }


def resolve_ablation_attempt(
    raw: Any,
    root: Path,
    key: tuple[str, str, int],
    repo_root: Path,
) -> Path:
    candidate, ablation, seed = key
    if not isinstance(raw, str) or not raw:
        raise common.AnalysisError(f"Missing attempt path for {key}")
    recorded = Path(raw).expanduser()
    if not re.fullmatch(r"attempt_[0-9]{3}", recorded.name):
        raise common.AnalysisError(f"Unexpected attempt name for {key}: {raw}")
    expected = root / candidate / "transformer" / ablation / f"seed{seed}" / recorded.name
    common.resolve_recorded_path(raw, expected, repo_root, f"attempt path {key}")
    if not expected.is_dir():
        raise common.AnalysisError(f"Missing attempt directory for {key}: {expected}")
    return expected.resolve()


def infer_full_selection(full_root: Path, repo_root: Path) -> Path:
    batch = common.load_json(full_root / "batch_status.json", "full batch status")
    results = batch.get("results")
    if not isinstance(results, list):
        raise common.AnalysisError("Full batch results must be a list")
    transformer = next(
        (
            result
            for result in results
            if isinstance(result, dict) and result.get("model_type") == "transformer"
        ),
        None,
    )
    if transformer is None:
        raise common.AnalysisError("Full batch has no Transformer result")
    key = (
        transformer.get("candidate_type"),
        transformer.get("model_type"),
        transformer.get("seed"),
    )
    attempt = common.resolve_attempt(
        transformer.get("attempt_dir"), full_root, key, repo_root
    )
    status = common.load_json(attempt / "status.json", "full attempt status")
    fingerprint = status.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise common.AnalysisError("Full attempt has no valid fingerprint")
    run_prefix = fingerprint.get("run_prefix")
    if not isinstance(run_prefix, str) or not run_prefix:
        raise common.AnalysisError("Full attempt fingerprint has no run_prefix")
    selection = repo_root / "checkpoints_align" / run_prefix / "alignment_selection.json"
    common.require_file(selection, "canonical alignment selection")
    return selection


def load_full_run(full_root: Path, repo_root: Path) -> common.LoadedRun:
    selection = infer_full_selection(full_root, repo_root)
    return common.load_run(
        common.RunSpec(ALIGNMENT_SEED, selection, full_root), repo_root
    )


def validate_canonical_caches(
    full: common.LoadedRun, repo_root: Path
) -> dict[str, Any]:
    manifest_path = repo_root / "paper" / "adma2026_artifact_manifest.json"
    manifest = common.load_json(manifest_path, "canonical artifact manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise common.AnalysisError("Canonical manifest artifacts must be a list")
    canonical = {
        artifact["id"].removeprefix("cache."): artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and isinstance(artifact.get("id"), str)
        and artifact["id"].startswith("cache.")
    }
    common.require_equal(len(canonical), 6, "canonical cache artifact count")
    for cache in full.input_files["cache_files"]:
        key = f"{cache['candidate_type']}.{cache['split']}"
        record = canonical.get(key)
        if not isinstance(record, dict):
            raise common.AnalysisError(f"Missing canonical cache record {key}")
        common.require_equal(cache["sha256"], record.get("sha256"), f"cache hash {key}")
        common.require_equal(
            cache["size_bytes"], record.get("size_bytes"), f"cache size {key}"
        )
        common.require_equal(
            f"repo:{cache['path']}", record.get("location"), f"cache path {key}"
        )
    return common.file_record(manifest_path, repo_root)


def validate_aggregate(
    path: Path,
    rows: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    common.require_file(path, "ablation summary_aggregate.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        aggregates = list(csv.DictReader(handle))
    common.require_equal(len(aggregates), 10, "ablation aggregate row count")
    observed: set[tuple[str, str]] = set()
    expected = {
        (candidate, ablation)
        for candidate in CANDIDATE_TYPES
        for ablation in ABLATIONS
    }
    for aggregate in aggregates:
        key = (aggregate.get("candidate_type"), aggregate.get("ablation"))
        common.require_equal(aggregate.get("model_type"), "transformer", f"model {key}")
        if key not in expected or key in observed:
            raise common.AnalysisError(f"Unexpected or duplicate aggregate row: {key}")
        observed.add(key)
        common.require_equal(
            common.finite_int(aggregate.get("num_seeds"), f"seed count {key}"),
            3,
            f"seed count {key}",
        )
        for metric in common.AGGREGATE_METRICS:
            values = [rows[(key[0], key[1], seed)][metric] for seed in RERANKER_SEEDS]
            common.require_equal(
                common.finite_int(
                    aggregate.get(f"{metric}_num_seeds"),
                    f"aggregate metric count {key}/{metric}",
                ),
                3,
                f"aggregate metric count {key}/{metric}",
            )
            expected_mean, expected_std = common.mean_std(values)
            common.require_close(
                aggregate.get(f"{metric}_mean"),
                expected_mean,
                f"aggregate mean {key}/{metric}",
            )
            common.require_close(
                aggregate.get(f"{metric}_std"),
                expected_std,
                f"aggregate std {key}/{metric}",
            )
    common.require_equal(observed, expected, "ablation aggregate matrix")


def validate_retry_status(
    retry: dict[str, Any], batch: dict[str, Any], history: list[dict[str, Any]]
) -> None:
    common.require_equal(retry.get("state"), "complete", "retry supervisor state")
    common.require_equal(retry.get("child_pid"), None, "retry child PID")
    progress = retry.get("progress")
    if not isinstance(progress, dict):
        raise common.AnalysisError("Retry status has no valid progress object")
    common.require_equal(progress.get("total_experiments"), 30, "retry total experiments")
    common.require_equal(progress.get("experiments_seen"), 30, "retry seen experiments")
    common.require_equal(progress.get("not_started"), 0, "retry not-started count")
    common.require_equal(
        progress.get("latest_experiment_states"),
        {"complete": 30},
        "retry latest experiment states",
    )
    observed_states: dict[str, int] = {}
    for record in history:
        state = record["state"]
        observed_states[state] = observed_states.get(state, 0) + 1
    common.require_equal(
        progress.get("all_attempt_states"), observed_states, "retry attempt-state counts"
    )
    common.require_equal(
        progress.get("attempt_status_files"), len(history), "retry status-file count"
    )
    last = retry.get("last_invocation")
    if not isinstance(last, dict):
        raise common.AnalysisError("Retry status has no last invocation")
    common.require_equal(last.get("return_code"), 0, "last invocation return code")
    provenance = retry.get("provenance")
    if not isinstance(provenance, dict):
        raise common.AnalysisError("Retry status has no provenance")
    common.require_equal(
        provenance.get("git_commit"), batch.get("git_commit"), "retry git commit"
    )
    common.require_equal(
        provenance.get("params_sha256"),
        batch.get("params_sha256"),
        "retry params SHA256",
    )


def load_ablation_run(
    root: Path,
    full: common.LoadedRun,
    repo_root: Path,
) -> LoadedAblations:
    root = root.resolve()
    batch_path = root / "batch_status.json"
    retry_path = root / "retry_status.json"
    summary_path = root / "summary.csv"
    aggregate_path = root / "summary_aggregate.csv"
    batch = common.load_json(batch_path, "ablation batch status")
    retry = common.load_json(retry_path, "retry supervisor status")
    common.require_equal(batch.get("state"), "complete", "ablation batch state")
    common.require_equal(batch.get("errors"), [], "ablation batch errors")
    common.require_equal(batch.get("mces"), False, "ablation MCES mode")
    common.require_equal(
        batch.get("git_worktree_changes"), [], "ablation dirty-worktree record"
    )
    common.require_equal(
        tuple(batch.get("exclude_val_query_indices", [])),
        common.VALIDATION_EXCLUSIONS,
        "ablation validation exclusions",
    )
    if not isinstance(batch.get("git_commit"), str) or not batch["git_commit"]:
        raise common.AnalysisError("Ablation batch is missing git_commit")
    common.require_equal(
        batch.get("params_sha256"),
        full.batch["params_sha256"],
        "full/ablation params SHA256",
    )

    expected = expected_ablation_keys()
    experiments = batch.get("experiments")
    results = batch.get("results")
    if not isinstance(experiments, list) or not isinstance(results, list):
        raise common.AnalysisError("Ablation batch experiments/results must be lists")
    experiment_keys: set[tuple[str, str, int]] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise common.AnalysisError(f"Invalid ablation experiment: {experiment!r}")
        common.require_equal(
            experiment.get("model_type"), "transformer", "ablation experiment model"
        )
        key = (
            experiment.get("candidate_type"),
            experiment.get("ablation"),
            experiment.get("seed"),
        )
        if key not in expected or key in experiment_keys:
            raise common.AnalysisError(f"Unexpected or duplicate experiment: {key}")
        experiment_keys.add(key)
    common.require_equal(experiment_keys, expected, "ablation planned matrix")

    attempts: dict[tuple[str, str, int], Path] = {}
    for result in results:
        if not isinstance(result, dict):
            raise common.AnalysisError(f"Invalid ablation result: {result!r}")
        common.require_equal(result.get("model_type"), "transformer", "result model")
        key = (result.get("candidate_type"), result.get("ablation"), result.get("seed"))
        if key not in expected or key in attempts:
            raise common.AnalysisError(f"Unexpected or duplicate result: {key}")
        if result.get("state") not in {"complete", "skipped"}:
            raise common.AnalysisError(f"Invalid final batch result state for {key}")
        attempts[key] = resolve_ablation_attempt(
            result.get("attempt_dir"), root, key, repo_root
        )
    common.require_equal(set(attempts), expected, "ablation result matrix")

    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in common.parse_summary(summary_path):
        common.require_equal(row["model_type"], "transformer", "summary model type")
        key = (row["candidate_type"], row["ablation"], row["seed"])
        common.require_equal(row["metric_for_best"], "mrr", f"selection metric {key}")
        if key not in expected or key in rows:
            raise common.AnalysisError(f"Unexpected or duplicate summary row: {key}")
        attempt = resolve_ablation_attempt(row["attempt_dir"], root, key, repo_root)
        common.require_equal(attempt, attempts[key], f"summary attempt {key}")
        rows[key] = row
    common.require_equal(set(rows), expected, "ablation summary matrix")
    validate_aggregate(aggregate_path, rows)

    run_prefix = full.spec.selection_path.parent.name
    full_caches = {
        (item["candidate_type"], item["split"]): item
        for item in full.input_files["cache_files"]
    }
    attempt_inputs = []
    for key in sorted(expected):
        candidate, ablation, seed = key
        attempt = attempts[key]
        row = rows[key]
        status_path = attempt / "status.json"
        status = common.load_json(status_path, f"ablation status {key}")
        for field, expected_value in (
            ("state", "complete"),
            ("candidate_type", candidate),
            ("model_type", "transformer"),
            ("ablation", ablation),
            ("seed", seed),
            ("git_commit", batch["git_commit"]),
            ("params_sha256", batch["params_sha256"]),
            ("ablation_overrides", EXPECTED_OVERRIDES[ablation]),
            ("git_worktree_changes", []),
        ):
            common.require_equal(status.get(field), expected_value, f"status {key} {field}")
        common.require_equal(
            tuple(status.get("exclude_val_query_indices", [])),
            common.VALIDATION_EXCLUSIONS,
            f"status {key} validation exclusions",
        )
        common.resolve_recorded_path(
            status.get("attempt_dir"), attempt, repo_root, f"status attempt {key}"
        )
        selection = status.get("training_selection")
        if not isinstance(selection, dict):
            raise common.AnalysisError(f"Invalid training selection for {key}")
        for field, expected_value in (
            ("metric_for_best", "mrr"),
            ("best_epoch", row["best_epoch"]),
            ("stop_epoch", row["stop_epoch"]),
            ("early_stopped", row["early_stopped"]),
        ):
            common.require_equal(selection.get(field), expected_value, f"selection {key} {field}")
        common.require_close(
            selection.get("best_val_metric"), row["best_val_metric"], f"selection {key} value"
        )

        fingerprint = status.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise common.AnalysisError(f"Invalid fingerprint for {key}")
        for field, expected_value in (
            ("git_commit", batch["git_commit"]),
            ("params_sha256", batch["params_sha256"]),
            ("dataset_type", "massspecgym"),
            ("run_prefix", run_prefix),
            ("topk", common.TOPK),
            ("candidate_type", candidate),
            ("model_type", "transformer"),
            ("ablation", ablation),
            ("ablation_overrides", EXPECTED_OVERRIDES[ablation]),
            ("seed", seed),
            ("exclude_val_query_indices", list(common.VALIDATION_EXCLUSIONS)),
        ):
            common.require_equal(
                fingerprint.get(field), expected_value, f"fingerprint {key} {field}"
            )
        common.require_equal(
            status.get("cache_files"),
            fingerprint.get("cache_files"),
            f"status/fingerprint caches {key}",
        )
        for split in common.SPLITS:
            expected_cache = common.canonical_cache_path(
                repo_root, run_prefix, "massspecgym", candidate, split
            )
            record = fingerprint["cache_files"].get(split)
            common.validate_cache_record(
                record, expected_cache, repo_root, f"ablation {candidate}/{split} cache"
            )
            canonical = full_caches[(candidate, split)]
            common.require_equal(
                common.display_path(expected_cache, repo_root),
                canonical["path"],
                f"full/ablation cache path {candidate}/{split}",
            )
            common.require_equal(
                expected_cache.stat().st_size,
                canonical["size_bytes"],
                f"full/ablation cache size {candidate}/{split}",
            )

        eval_log = attempt / "eval_rerank.log"
        parsed_metrics = common.parse_eval_metrics(eval_log, f"evaluation log {key}")
        for metric, actual in parsed_metrics.items():
            common.require_close(actual, row[metric], f"evaluation {key}/{metric}")
        train_log = attempt / "train_rerank.log"
        common.validate_train_log(train_log, row["stop_epoch"], f"training log {key}")
        best = common.require_file(attempt / "best_reranker.pth", f"best checkpoint {key}")
        last = common.require_file(attempt / "last_reranker.pth", f"last checkpoint {key}")
        attempt_inputs.append(
            {
                "candidate_type": candidate,
                "ablation": ablation,
                "reranker_seed": seed,
                "status": common.file_record(status_path, repo_root),
                "training_log": common.file_record(train_log, repo_root),
                "evaluation_log": common.file_record(eval_log, repo_root),
                "best_checkpoint": common.file_record(best, repo_root),
                "last_checkpoint": common.file_record(last, repo_root),
            }
        )

    for candidate in CANDIDATE_TYPES:
        reference = full.rows[(candidate, "transformer", RERANKER_SEEDS[0])]
        candidate_rows = [
            rows[(candidate, ablation, seed)]
            for ablation in ABLATIONS
            for seed in RERANKER_SEEDS
        ]
        for metric in common.BASE_METRICS:
            for row in candidate_rows:
                common.require_close(
                    row[metric], reference[metric], f"full/ablation invariant {candidate}/{metric}"
                )

    history = []
    for status_path in sorted(root.glob("**/status.json")):
        payload = common.load_json(status_path, "attempt history status")
        history.append(
            {
                "state": payload.get("state"),
                "file": common.file_record(status_path, repo_root),
            }
        )
    validate_retry_status(retry, batch, history)
    supervisor_log = root / "retry_supervisor.log"
    inputs = {
        "batch_status": common.file_record(batch_path, repo_root),
        "retry_status": common.file_record(retry_path, repo_root),
        "retry_supervisor_log": common.file_record(supervisor_log, repo_root),
        "summary": common.file_record(summary_path, repo_root),
        "summary_aggregate": common.file_record(aggregate_path, repo_root),
        "attempts": attempt_inputs,
        "attempt_history": history,
    }
    return LoadedAblations(root, batch, retry, rows, attempts, inputs)


def runtime_changed_paths(
    repo_root: Path, full_commit: str, ablation_commit: str
) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            full_commit,
            ablation_commit,
            "--",
            *EXECUTION_RELEVANT_PATHS,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise common.AnalysisError(
            "Unable to compare execution-relevant source trees: "
            + (result.stderr.strip() or f"git exited {result.returncode}")
        )
    return [line for line in result.stdout.splitlines() if line]


def validate_output_root(
    output_root: Path,
    repo_root: Path,
    protected_roots: tuple[Path, ...],
) -> Path:
    resolved = output_root.resolve()
    analysis_root = (repo_root / "analysis").resolve()
    try:
        relative = resolved.relative_to(analysis_root)
    except ValueError as error:
        raise common.AnalysisError(
            f"Output root must be inside {analysis_root}: {resolved}"
        ) from error
    if not relative.parts:
        raise common.AnalysisError("Output root must be a subdirectory of analysis/")
    for protected in protected_roots:
        protected = protected.resolve()
        if resolved == protected or protected in resolved.parents:
            raise common.AnalysisError(
                f"Output root overlaps protected experiment data: {resolved}"
            )
    return resolved


def build_long_rows(
    full: common.LoadedRun, ablations: LoadedAblations
) -> list[dict[str, Any]]:
    output = []
    for candidate in CANDIDATE_TYPES:
        for seed in RERANKER_SEEDS:
            row = full.rows[(candidate, "transformer", seed)]
            output.append(
                long_row(row, full.batch["git_commit"], full.batch["params_sha256"])
            )
        for ablation in ABLATIONS:
            for seed in RERANKER_SEEDS:
                row = ablations.rows[(candidate, ablation, seed)]
                output.append(
                    long_row(
                        row,
                        ablations.batch["git_commit"],
                        ablations.batch["params_sha256"],
                    )
                )
    return output


def long_row(row: dict[str, Any], commit: str, params_sha256: str) -> dict[str, Any]:
    output = {
        "alignment_seed": ALIGNMENT_SEED,
        "candidate_type": row["candidate_type"],
        "model_type": row["model_type"],
        "ablation": row["ablation"],
        "reranker_seed": row["seed"],
        "source_git_commit": commit,
        "params_sha256": params_sha256,
        "metric_for_best": row["metric_for_best"],
        "best_epoch": row["best_epoch"],
        "best_val_metric": row["best_val_metric"],
        "stop_epoch": row["stop_epoch"],
        "early_stopped": row["early_stopped"],
    }
    output.update({metric: row[metric] for metric in (*common.BASE_METRICS, *common.RERANK_METRICS)})
    return output


def build_paired_rows(
    full: common.LoadedRun, ablations: LoadedAblations
) -> list[dict[str, Any]]:
    output = []
    for candidate in CANDIDATE_TYPES:
        for ablation in ABLATIONS:
            for seed in RERANKER_SEEDS:
                full_row = full.rows[(candidate, "transformer", seed)]
                ablated = ablations.rows[(candidate, ablation, seed)]
                item: dict[str, Any] = {
                    "alignment_seed": ALIGNMENT_SEED,
                    "candidate_type": candidate,
                    "ablation": ablation,
                    "component": COMPONENT_LABELS[ablation],
                    "reranker_seed": seed,
                }
                for metric in common.RERANK_METRICS:
                    short = metric.removeprefix("rerank_")
                    full_value = common.finite_float(
                        full_row[metric], f"full {candidate}/{seed}/{metric}"
                    )
                    ablation_value = common.finite_float(
                        ablated[metric],
                        f"ablation {candidate}/{ablation}/{seed}/{metric}",
                    )
                    item[f"full_{short}"] = full_value
                    item[f"ablation_{short}"] = ablation_value
                    item[f"delta_{short}"] = full_value - ablation_value
                output.append(item)
    return output


def build_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for candidate in CANDIDATE_TYPES:
        for ablation in ABLATIONS:
            group = [
                row
                for row in paired
                if row["candidate_type"] == candidate and row["ablation"] == ablation
            ]
            common.require_equal(len(group), 3, f"paired group size {candidate}/{ablation}")
            item: dict[str, Any] = {
                "candidate_type": candidate,
                "ablation": ablation,
                "component": COMPONENT_LABELS[ablation],
                "num_paired_seeds": len(group),
            }
            for metric in common.RERANK_METRICS:
                short = metric.removeprefix("rerank_")
                full_values = [row[f"full_{short}"] for row in group]
                ablation_values = [row[f"ablation_{short}"] for row in group]
                deltas = [row[f"delta_{short}"] for row in group]
                item[f"full_{short}_mean"], item[f"full_{short}_std"] = common.mean_std(
                    full_values
                )
                item[f"ablation_{short}_mean"], item[f"ablation_{short}_std"] = common.mean_std(
                    ablation_values
                )
                item[f"delta_{short}_mean"], item[f"delta_{short}_std"] = common.mean_std(
                    deltas
                )
                item[f"delta_{short}_positive"] = sum(value > 0 for value in deltas)
                item[f"delta_{short}_zero"] = sum(value == 0 for value in deltas)
                item[f"delta_{short}_negative"] = sum(value < 0 for value in deltas)
            output.append(item)
    return output


def build_component_gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    for ablation in ABLATIONS:
        rows = [row for row in summary_rows if row["ablation"] == ablation]
        common.require_equal(len(rows), 2, f"component candidate count {ablation}")
        mrr_means = [row["delta_mrr_raw_mean"] for row in rows]
        strict = all(
            row["delta_mrr_raw_mean"] > 0 and row["delta_mrr_raw_positive"] == 3
            for row in rows
        )
        removal_better = all(
            row["delta_mrr_raw_negative"] == 3
            and row["delta_top1_pct_negative"] == 3
            for row in rows
        )
        if strict:
            status = "pass_descriptive"
        elif removal_better:
            status = "removal_consistently_better"
        elif all(value > 0 for value in mrr_means):
            status = "mean_favors_full_seed_mixed"
        elif any(value > 0 for value in mrr_means) and any(value < 0 for value in mrr_means):
            status = "candidate_dependent"
        else:
            status = "not_supported"
        decisions.append(
            {
                "ablation": ablation,
                "component": COMPONENT_LABELS[ablation],
                "status": status,
                "criterion": (
                    "Full-minus-ablation MRR must be positive for all three paired "
                    "reranker seeds in both candidate protocols."
                ),
                "candidate_results": [
                    {
                        "candidate_type": row["candidate_type"],
                        "delta_top1_pct_mean": row["delta_top1_pct_mean"],
                        "delta_top1_pct_directions": {
                            "positive": row["delta_top1_pct_positive"],
                            "zero": row["delta_top1_pct_zero"],
                            "negative": row["delta_top1_pct_negative"],
                        },
                        "delta_mrr_raw_mean": row["delta_mrr_raw_mean"],
                        "delta_mrr_raw_directions": {
                            "positive": row["delta_mrr_raw_positive"],
                            "zero": row["delta_mrr_raw_zero"],
                            "negative": row["delta_mrr_raw_negative"],
                        },
                    }
                    for row in rows
                ],
                "interpretation_note": COMPONENT_NOTES[ablation],
            }
        )
    passed = sum(item["status"] == "pass_descriptive" for item in decisions)
    return {
        "status": "pass" if passed == len(ABLATIONS) else "component_claims_require_narrowing",
        "primary_metric": "rerank_mrr_raw",
        "delta_definition": "full_minus_ablation",
        "passed_components": passed,
        "total_components": len(ABLATIONS),
        "decisions": decisions,
        "evidence_boundary": (
            "Descriptive paired-seed decisions at one fixed alignment checkpoint; "
            "not a significance test, causal estimate, or cross-alignment claim."
        ),
    }


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def plus_minus(mean_value: float, std_value: float, digits: int) -> str:
    return f"{fmt(mean_value, digits)} ± {fmt(std_value, digits)}"


def build_report(
    full: common.LoadedRun,
    ablations: LoadedAblations,
    summary_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    changed_paths: list[str],
) -> str:
    result_states: dict[str, int] = {}
    for result in ablations.batch["results"]:
        state = result["state"]
        result_states[state] = result_states.get(state, 0) + 1
    lines = [
        "# Core component ablation analysis",
        "",
        "This report compares the frozen full Set Transformer with five pre-registered",
        "component removals at canonical alignment seed 42. Reranker seeds are paired",
        "within candidate protocol; statistics are descriptive sample means/SDs (n=3).",
        "",
        "## Evaluation identity protocol",
        "",
        EVALUATION_IDENTITY_PROTOCOL_NOTE,
        "",
        "Delta is defined as `full - ablation`: positive values favor the full model.",
        "Top-1 differences are percentage points; MRR uses the raw [0, 1] scale.",
        "",
        "## Provenance and compatibility",
        "",
        f"- Full source commit: `{full.batch['git_commit']}`.",
        f"- Ablation source commit: `{ablations.batch['git_commit']}`.",
        f"- Shared params SHA-256: `{full.batch['params_sha256']}`.",
        f"- Execution-relevant changed paths between source commits: `{changed_paths}`.",
        "- Both runs use the same canonical top-40 caches, validation exclusions,",
        "  selection metric, training budget, and 17,556-query test evaluation.",
        "- The final supervisor batch reports "
        f"{result_states.get('skipped', 0)} reused completed attempts (`skipped`) and "
        f"{result_states.get('complete', 0)} completed in that invocation; every selected "
        "attempt status is `complete`.",
        "",
        "## Paired component results",
        "",
        "| Candidate | Removal | Full Top-1 | Removal Top-1 | ΔTop-1 (P/Z/N) | Full MRR | Removal MRR | ΔMRR (P/Z/N) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {candidate} | {ablation} | {full_top1} | {ablation_top1} | "
            "{delta_top1} ({top1_p}/{top1_z}/{top1_n}) | {full_mrr} | "
            "{ablation_mrr} | {delta_mrr} ({mrr_p}/{mrr_z}/{mrr_n}) |".format(
                candidate=row["candidate_type"],
                ablation=row["ablation"],
                full_top1=plus_minus(row["full_top1_pct_mean"], row["full_top1_pct_std"], 4),
                ablation_top1=plus_minus(
                    row["ablation_top1_pct_mean"], row["ablation_top1_pct_std"], 4
                ),
                delta_top1=plus_minus(
                    row["delta_top1_pct_mean"], row["delta_top1_pct_std"], 4
                ),
                top1_p=row["delta_top1_pct_positive"],
                top1_z=row["delta_top1_pct_zero"],
                top1_n=row["delta_top1_pct_negative"],
                full_mrr=plus_minus(row["full_mrr_raw_mean"], row["full_mrr_raw_std"], 6),
                ablation_mrr=plus_minus(
                    row["ablation_mrr_raw_mean"], row["ablation_mrr_raw_std"], 6
                ),
                delta_mrr=plus_minus(
                    row["delta_mrr_raw_mean"], row["delta_mrr_raw_std"], 6
                ),
                mrr_p=row["delta_mrr_raw_positive"],
                mrr_z=row["delta_mrr_raw_zero"],
                mrr_n=row["delta_mrr_raw_negative"],
            )
        )
    lines.extend(
        [
            "",
            "`P/Z/N` counts positive/zero/negative full-minus-ablation deltas across",
            "reranker seeds 42/43/44.",
            "",
            "## Component claim gate",
            "",
            f"- Overall status: `{gate['status']}`; {gate['passed_components']}/"
            f"{gate['total_components']} components pass the strict descriptive gate.",
        ]
    )
    for decision in gate["decisions"]:
        lines.append(
            f"- `{decision['ablation']}` ({decision['component']}): "
            f"`{decision['status']}`. {decision['interpretation_note']}"
        )
    decisions = {item["ablation"]: item for item in gate["decisions"]}
    dependent = [
        item["component"]
        for item in gate["decisions"]
        if item["status"] == "candidate_dependent"
    ]
    lines.extend(
        [
            "",
            (
                "The rank embedding is not supported under this frozen protocol: removing "
                "it improves both Top-1 and MRR in all six candidate/seed cells."
                if decisions["no_rank_embedding"]["status"]
                == "removal_consistently_better"
                else "The rank-embedding result does not meet a uniform descriptive gate."
            ),
            (
                "Candidate-dependent components under the frozen gate: "
                + ", ".join(dependent)
                + "."
                if dependent
                else "No component has candidate-dependent directions under the frozen gate."
            ),
            (
                "Removing the joint base-score pathway lowers the mean MRR in both "
                "protocols, but paired-seed directions are mixed."
                if decisions["no_base_score"]["status"]
                == "mean_favors_full_seed_mixed"
                else "The joint base-score-pathway result does not favor the full model "
                "in both protocol means."
            ),
            "",
            "These results do not justify selecting a new post-hoc main model or claiming",
            "statistical significance, causal component effects, cross-dataset generalization,",
            "or cross-alignment component stability. The supported paper-level boundary remains",
            "supervised non-generative second-stage reranking; individual component claims must",
            "be narrowed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    full_root: Path,
    ablation_root: Path,
    output_root: Path,
    repo_root: Path,
    *,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    full_root = full_root.resolve()
    ablation_root = ablation_root.resolve()
    output_root = validate_output_root(
        output_root, repo_root, (full_root, ablation_root)
    )
    full = load_full_run(full_root, repo_root)
    canonical_manifest = validate_canonical_caches(full, repo_root)
    ablations = load_ablation_run(ablation_root, full, repo_root)
    changed_paths = (
        runtime_changed_paths(
            repo_root, full.batch["git_commit"], ablations.batch["git_commit"]
        )
        if verify_runtime
        else []
    )
    common.require_equal(changed_paths, [], "execution-relevant changed paths")

    long_rows = build_long_rows(full, ablations)
    paired_rows = build_paired_rows(full, ablations)
    summary_rows = build_summary_rows(paired_rows)
    gate = build_component_gate(summary_rows)
    common.require_equal(len(long_rows), 36, "long summary row count")
    common.require_equal(len(paired_rows), 30, "paired row count")
    common.require_equal(len(summary_rows), 10, "ablation summary row count")

    paths = {
        "summary_long": output_root / "summary_long.csv",
        "paired_deltas": output_root / "paired_deltas.csv",
        "ablation_summary": output_root / "ablation_summary.csv",
        "report": output_root / "report.md",
        "manifest": output_root / "analysis_manifest.json",
    }
    common.atomic_write_csv(paths["summary_long"], long_rows)
    common.atomic_write_csv(paths["paired_deltas"], paired_rows)
    common.atomic_write_csv(paths["ablation_summary"], summary_rows)
    common.atomic_write_text(
        paths["report"],
        build_report(full, ablations, summary_rows, gate, changed_paths),
    )
    manifest = {
        "schema_version": 2,
        "analysis": "canonical_alignment_seed42_core_component_ablations",
        "evaluation_identity_protocol": dict(EVALUATION_IDENTITY_PROTOCOL),
        "design": {
            "alignment_seed": ALIGNMENT_SEED,
            "candidate_types": list(CANDIDATE_TYPES),
            "model_type": "transformer",
            "ablations": list(ABLATIONS),
            "reranker_seeds": list(RERANKER_SEEDS),
            "pairing_key": ["candidate_type", "reranker_seed"],
            "validation_query_exclusions": list(common.VALIDATION_EXCLUSIONS),
            "topk": common.TOPK,
        },
        "units": {
            "topk_metrics": "percentage_points",
            "mrr": "raw_[0,1]",
            "delta": "full_minus_ablation",
            "std": "sample_standard_deviation_ddof_1",
        },
        "compatibility": {
            "full_source_commit": full.batch["git_commit"],
            "ablation_source_commit": ablations.batch["git_commit"],
            "params_sha256": full.batch["params_sha256"],
            "execution_relevant_paths": list(EXECUTION_RELEVANT_PATHS),
            "changed_paths": changed_paths,
            "same_params": full.batch["params_sha256"]
            == ablations.batch["params_sha256"],
            "same_canonical_caches": True,
        },
        "inputs": {
            "analysis_runtime": {
                "core_ablation_script": common.file_record(
                    repo_root / "analyze_core_ablations.py", repo_root
                ),
                "shared_validation_script": common.file_record(
                    repo_root / "analyze_alignment_multiseed.py", repo_root
                ),
            },
            "full": {
                "source_git_commit": full.batch["git_commit"],
                "params_sha256": full.batch["params_sha256"],
                "files": full.input_files,
            },
            "ablations": {
                "source_git_commit": ablations.batch["git_commit"],
                "params_sha256": ablations.batch["params_sha256"],
                "files": ablations.input_files,
            },
            "canonical_artifact_manifest": canonical_manifest,
        },
        "counts": {
            "summary_long_rows": len(long_rows),
            "paired_rows": len(paired_rows),
            "ablation_summary_rows": len(summary_rows),
            "full_transformer_attempts": 6,
            "selected_ablation_attempts": len(ablations.attempts),
            "historical_attempt_statuses": len(
                ablations.input_files["attempt_history"]
            ),
        },
        "component_claim_gate": gate,
        "outputs": {
            key: common.file_record(path, repo_root)
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    common.atomic_write_text(
        paths["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze the five core Set Transformer ablations against "
            "the canonical alignment-seed42 full model."
        )
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    try:
        manifest = run_analysis(
            args.full_root, args.ablation_root, args.output_root, repo_root
        )
    except common.AnalysisError as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 2
    print(f"Wrote core ablation analysis to {args.output_root.resolve()}")
    print(f"Component claim gate: {manifest['component_claim_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
