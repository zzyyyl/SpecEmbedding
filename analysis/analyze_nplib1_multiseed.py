"""Validate and aggregate the NPLIB1 alignment/reranker seed matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

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
DELTA_METRICS = {
    "top1": ("rerank_top1_pct", "base_top1_pct"),
    "top5": ("rerank_top5_pct", "base_top5_pct"),
    "top10": ("rerank_top10_pct", "base_top10_pct"),
    "top20": ("rerank_top20_pct", "base_top20_pct"),
    "mrr": ("rerank_mrr_raw", "base_mrr_raw"),
}


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"Missing or empty artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finite_float(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"Invalid number for {label}: {raw!r}") from error
    if not math.isfinite(value):
        raise AnalysisError(f"Non-finite number for {label}: {raw!r}")
    return value


def aggregate(values: list[float]) -> dict:
    if not values:
        raise AnalysisError("Cannot aggregate an empty value list")
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def load_json(path: Path) -> dict:
    file_record(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object: {path}")
    return value


def parse_total_queries(path: Path) -> int:
    file_record(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Total queries:\s*([0-9]+)", text)
    if len(matches) != 1:
        raise AnalysisError(f"Expected one total-query field in {path}")
    if re.search(r"Traceback|\bERROR\b|interrupted|killed", text, re.IGNORECASE):
        raise AnalysisError(f"Evaluation log contains an error marker: {path}")
    return int(matches[0])


def resolve_attempt(raw: str, repo_root: Path, rerank_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    try:
        path.relative_to(rerank_root.resolve())
    except ValueError as error:
        raise AnalysisError(f"Attempt escapes rerank root: {path}") from error
    if not re.fullmatch(r"attempt_[0-9]{3}", path.name) or not path.is_dir():
        raise AnalysisError(f"Invalid attempt directory: {path}")
    return path


def parse_summary(
    path: Path,
    *,
    repo_root: Path,
    rerank_root: Path,
    candidate_type: str,
    model_types: tuple[str, ...],
    reranker_seeds: tuple[int, ...],
) -> list[dict]:
    file_record(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = {(model, seed) for model in model_types for seed in reranker_seeds}
    parsed = []
    observed = set()
    for raw in rows:
        if raw.get("candidate_type") != candidate_type or raw.get("ablation") != "full":
            continue
        model = raw.get("model_type")
        try:
            seed = int(raw.get("seed", ""))
        except ValueError as error:
            raise AnalysisError(f"Invalid reranker seed in {path}: {raw}") from error
        key = (model, seed)
        if key not in expected:
            continue
        if key in observed:
            raise AnalysisError(f"Duplicate summary row for {key}: {path}")
        observed.add(key)
        row = {
            "model_type": model,
            "reranker_seed": seed,
            "best_epoch": int(raw["best_epoch"]),
            "best_val_metric": finite_float(raw["best_val_metric"], "best_val_metric"),
            "stop_epoch": int(raw["stop_epoch"]),
            "early_stopped": raw["early_stopped"] == "True",
        }
        for metric in (*BASE_METRICS, *RERANK_METRICS):
            row[metric] = finite_float(raw[metric], metric)
        attempt = resolve_attempt(raw["attempt_dir"], repo_root, rerank_root)
        status = load_json(attempt / "status.json")
        if status.get("state") != "complete":
            raise AnalysisError(f"Attempt is not complete: {attempt}")
        for field, expected_value in (
            ("candidate_type", candidate_type),
            ("model_type", model),
            ("ablation", "full"),
            ("seed", seed),
        ):
            if status.get(field) != expected_value:
                raise AnalysisError(
                    f"Attempt {field} mismatch in {attempt}: {status.get(field)!r}"
                )
        row["total_queries"] = parse_total_queries(attempt / "eval_rerank.log")
        row["attempt"] = str(attempt)
        parsed.append(row)

    if observed != expected:
        raise AnalysisError(
            f"Incomplete experiment matrix in {path}: missing={sorted(expected - observed)}"
        )
    return parsed


def validate_batch(
    path: Path,
    *,
    candidate_type: str,
    model_types: tuple[str, ...],
    reranker_seeds: tuple[int, ...],
    expected_git_commit: str | None,
) -> dict:
    batch = load_json(path)
    if batch.get("state") != "complete" or batch.get("errors"):
        raise AnalysisError(f"Batch is incomplete or has errors: {path}")
    if batch.get("git_worktree_changes"):
        raise AnalysisError(f"Batch used a dirty worktree: {path}")
    if expected_git_commit and batch.get("git_commit") != expected_git_commit:
        raise AnalysisError(
            f"Batch commit mismatch in {path}: {batch.get('git_commit')!r}"
        )
    expected = {
        (candidate_type, model, "full", seed)
        for model in model_types
        for seed in reranker_seeds
    }
    observed = {
        (
            item.get("candidate_type"),
            item.get("model_type"),
            item.get("ablation"),
            item.get("seed"),
        )
        for item in batch.get("experiments", [])
    }
    if observed != expected:
        raise AnalysisError(f"Batch matrix mismatch in {path}")
    return batch


def analyze(args: argparse.Namespace) -> dict:
    repo_root = args.repo_root.resolve()
    per_alignment = {}
    raw_runs = []
    input_files = {}

    for alignment_seed in args.alignment_seeds:
        run_prefix = f"{args.run_prefix_base}_alignseed{alignment_seed}"
        rerank_root = (
            repo_root
            / "checkpoints_rerank"
            / f"{run_prefix}_topk{args.topk}_multiseed"
        )
        batch_path = rerank_root / "batch_status.json"
        summary_path = rerank_root / "summary.csv"
        batch = validate_batch(
            batch_path,
            candidate_type=args.candidate_type,
            model_types=args.model_types,
            reranker_seeds=args.reranker_seeds,
            expected_git_commit=args.expected_git_commit,
        )
        rows = parse_summary(
            summary_path,
            repo_root=repo_root,
            rerank_root=rerank_root,
            candidate_type=args.candidate_type,
            model_types=args.model_types,
            reranker_seeds=args.reranker_seeds,
        )
        query_counts = {row["total_queries"] for row in rows}
        if len(query_counts) != 1:
            raise AnalysisError(f"Query count mismatch for alignment seed {alignment_seed}")
        for metric in BASE_METRICS:
            values = {row[metric] for row in rows}
            if len(values) != 1:
                raise AnalysisError(
                    f"Base metric {metric} varies for alignment seed {alignment_seed}"
                )

        alignment_result = {
            "total_queries": query_counts.pop(),
            "git_commit": batch.get("git_commit"),
            "params_sha256": batch.get("params_sha256"),
            "base": {metric: rows[0][metric] for metric in BASE_METRICS},
            "models": {},
        }
        for model in args.model_types:
            model_rows = sorted(
                (row for row in rows if row["model_type"] == model),
                key=lambda row: row["reranker_seed"],
            )
            alignment_result["models"][model] = {
                "rerank": {
                    metric: aggregate([row[metric] for row in model_rows])
                    for metric in RERANK_METRICS
                },
                "delta_vs_base": {
                    name: aggregate(
                        [row[rerank] - row[base] for row in model_rows]
                    )
                    for name, (rerank, base) in DELTA_METRICS.items()
                },
            }
        if {"pointwise", "relative"}.issubset(args.model_types):
            indexed = {
                (row["model_type"], row["reranker_seed"]): row for row in rows
            }
            alignment_result["relative_minus_pointwise"] = {
                metric.removeprefix("rerank_"): aggregate(
                    [
                        indexed[("relative", seed)][metric]
                        - indexed[("pointwise", seed)][metric]
                        for seed in args.reranker_seeds
                    ]
                )
                for metric in RERANK_METRICS
            }
        per_alignment[str(alignment_seed)] = alignment_result
        raw_runs.extend({"alignment_seed": alignment_seed, **row} for row in rows)
        input_files[str(alignment_seed)] = {
            "batch_status": file_record(batch_path),
            "summary": file_record(summary_path),
        }

    across_alignment = {"models": {}}
    for model in args.model_types:
        across_alignment["models"][model] = {
            "rerank_alignment_means": {
                metric: aggregate(
                    [
                        per_alignment[str(seed)]["models"][model]["rerank"][metric][
                            "mean"
                        ]
                        for seed in args.alignment_seeds
                    ]
                )
                for metric in RERANK_METRICS
            },
            "delta_vs_base_alignment_means": {
                name: aggregate(
                    [
                        per_alignment[str(seed)]["models"][model]["delta_vs_base"][
                            name
                        ]["mean"]
                        for seed in args.alignment_seeds
                    ]
                )
                for name in DELTA_METRICS
            },
        }
    if {"pointwise", "relative"}.issubset(args.model_types):
        across_alignment["relative_minus_pointwise_alignment_means"] = {
            metric.removeprefix("rerank_"): aggregate(
                [
                    per_alignment[str(seed)]["relative_minus_pointwise"][
                        metric.removeprefix("rerank_")
                    ]["mean"]
                    for seed in args.alignment_seeds
                ]
            )
            for metric in RERANK_METRICS
        }

    return {
        "schema_version": 1,
        "dataset_type": args.dataset_type,
        "candidate_type": args.candidate_type,
        "topk": args.topk,
        "alignment_seeds": list(args.alignment_seeds),
        "reranker_seeds": list(args.reranker_seeds),
        "model_types": list(args.model_types),
        "statistics_policy": (
            "reranker seeds summarized within alignment; alignment-seed means then "
            "summarized with sample standard deviation; no significance test"
        ),
        "input_files": input_files,
        "per_alignment": per_alignment,
        "across_alignment": across_alignment,
        "raw_runs": raw_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix-base", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-type", default="nplib1")
    parser.add_argument("--candidate-type", default="formula")
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--alignment-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--reranker-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["pointwise", "relative"],
        default=["pointwise", "relative"],
    )
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.alignment_seeds = tuple(args.alignment_seeds)
    args.reranker_seeds = tuple(args.reranker_seeds)
    args.model_types = tuple(args.model_types)
    if args.topk <= 0:
        parser.error("--topk must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["across_alignment"], indent=2))


if __name__ == "__main__":
    main()
