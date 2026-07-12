import argparse
import csv
import gc
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from SpecEmbedding.data.datasets_rerank import load_rerank_cache
from SpecEmbedding.utils.gpu import parse_cuda_device, require_available_gpu
from SpecEmbedding.utils.rerank import load_reranker, parse_rerank_eval_metrics

DEFAULT_RUN_PREFIX = "d4c1f70_massspecgym_nopretrain"
DEFAULT_EXCLUDED_INDICES = [5908, 5909, 5910]
DEFAULT_SOURCE_TOTAL = 17_556
DEFAULT_FILTERED_TOTAL = 17_553
DEFAULT_FILTERED_UPPER_BOUND = {
    "mass": 82.1512,
    "formula": 87.3013,
}
DEFAULT_ORIGINAL_LABELED = {
    "mass": 14_423,
    "formula": 15_327,
}
DEFAULT_FILTERED_LABELED = {
    "mass": 14_420,
    "formula": 15_324,
}
DEFAULT_EXCLUDED_LABELS = {
    "mass": [0, 1, 1],
    "formula": [0, 0, 0],
}
AUDITED_TRUE_SMILES = (
    "CC(C)(C)C1=CC(=C(C=C1NC(=O)C2=CNC3=CC=CC=C3C2=O)O)C(C)(C)C"
)

METRIC_FIELDS = [
    "total_queries",
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
]


@dataclass(frozen=True)
class SensitivityExperiment:
    candidate_type: str
    model_type: str
    seed: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def git_worktree_changes(repo_root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, include_sha256: bool = False) -> dict:
    stat = path.stat()
    metadata = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        metadata["sha256"] = sha256_file(path)
    return metadata


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def atomic_write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def experiment_key(experiment: SensitivityExperiment) -> tuple[str, str, int]:
    return experiment.candidate_type, experiment.model_type, experiment.seed


def resolve_runtime_paths(args, repo_root: Path) -> None:
    for attribute in ("source_root", "cache_root", "output_root"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            path = repo_root / path
        setattr(args, attribute, path.resolve())


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def validate_path_isolation(args) -> None:
    if paths_overlap(args.source_root, args.output_root):
        raise RuntimeError(
            "--source-root and --output-root must be separate, non-nested directories: "
            f"{args.source_root} vs {args.output_root}"
        )


def is_default_protocol(args) -> bool:
    return (
        args.run_prefix == DEFAULT_RUN_PREFIX
        and args.dataset_type == "massspecgym"
        and args.topk == 40
        and args.exclude_query_indices == DEFAULT_EXCLUDED_INDICES
    )


def load_source_attempts(args, repo_root: Path) -> dict[tuple[str, str, int], Path]:
    summary_path = args.source_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Source summary not found: {summary_path}")

    requested = {
        experiment_key(SensitivityExperiment(candidate_type, model_type, seed))
        for candidate_type in args.candidate_types
        for model_type in args.model_types
        for seed in args.seeds
    }
    source_attempts: dict[tuple[str, str, int], Path] = {}
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("ablation") or "full").strip() != "full":
                continue
            try:
                key = (row["candidate_type"], row["model_type"], int(row["seed"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in source_attempts:
                raise RuntimeError(f"Duplicate source summary row for {key}")
            attempt = Path(row["attempt_dir"])
            if not attempt.is_absolute():
                attempt = repo_root / attempt
            source_attempts[key] = attempt.resolve()

    missing = sorted(requested - set(source_attempts))
    if missing:
        raise RuntimeError(f"Missing source experiments in {summary_path}: {missing}")
    return source_attempts


def validate_source_attempt(
    experiment: SensitivityExperiment,
    attempt: Path,
    cache_metadata: dict,
) -> dict:
    checkpoint_path = attempt / "best_reranker.pth"
    eval_log = attempt / "eval_rerank.log"
    status_path = attempt / "status.json"
    for path, label in (
        (checkpoint_path, "checkpoint"),
        (eval_log, "canonical evaluation log"),
        (status_path, "source status"),
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Source {label} is missing or empty: {path}")

    status = read_json(status_path)
    if status.get("state") != "complete":
        raise RuntimeError(f"Source attempt is not complete: {status_path}")
    if status.get("candidate_type") != experiment.candidate_type:
        raise RuntimeError(f"Source candidate type mismatch: {status_path}")
    if status.get("model_type") != experiment.model_type or status.get("seed") != experiment.seed:
        raise RuntimeError(f"Source model/seed mismatch: {status_path}")

    expected_test_cache = {
        key: cache_metadata[key]
        for key in ("path", "size_bytes", "mtime_ns")
    }
    recorded_test_cache = (
        status.get("fingerprint", {}).get("cache_files", {}).get("test")
    )
    if recorded_test_cache != expected_test_cache:
        raise RuntimeError(
            f"Source fingerprint test cache mismatch in {status_path}: "
            f"{recorded_test_cache} vs {expected_test_cache}"
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("seed") != experiment.seed:
        raise RuntimeError(f"Checkpoint seed mismatch: {checkpoint_path}")
    if checkpoint.get("model_config", {}).get("model_type") != experiment.model_type:
        raise RuntimeError(f"Checkpoint model type mismatch: {checkpoint_path}")
    model = load_reranker(checkpoint_path, torch.device("cpu"))
    del model, checkpoint
    gc.collect()

    metrics = parse_rerank_eval_metrics(eval_log)
    required = [
        "total_queries",
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
    ]
    missing = [field for field in required if field not in metrics]
    if missing:
        raise RuntimeError(f"Source evaluation log is incomplete ({missing}): {eval_log}")
    if metrics["total_queries"] != DEFAULT_SOURCE_TOTAL:
        raise RuntimeError(
            f"Expected {DEFAULT_SOURCE_TOTAL} source queries, got {metrics['total_queries']}: {eval_log}"
        )
    if round(float(metrics["upper_bound_pct"]), 4) != round(
        float(cache_metadata["expected_original_upper_bound_pct"]),
        4,
    ):
        raise RuntimeError(
            f"Source upper bound does not match the current cache: {eval_log}"
        )

    return {
        "attempt": str(attempt),
        "checkpoint": file_metadata(checkpoint_path, include_sha256=True),
        "eval_log": file_metadata(eval_log, include_sha256=True),
        "status": file_metadata(status_path),
        "source_git_commit": status.get("git_commit"),
        "metrics": metrics,
    }


def cache_path(args, experiment: SensitivityExperiment) -> Path:
    cache_dir = args.cache_root / f"{args.run_prefix}_{experiment.candidate_type}_topk{args.topk}"
    return cache_dir / f"{args.dataset_type}_{experiment.candidate_type}_test.pt"


def validate_test_cache(args, candidate_type: str) -> dict:
    experiment = SensitivityExperiment(candidate_type, args.model_types[0], args.seeds[0])
    path = cache_path(args, experiment)
    if not path.exists():
        raise FileNotFoundError(f"Test cache not found: {path}")

    cache = load_rerank_cache(path)
    meta = cache.get("meta", {})
    expected_meta = {
        "dataset_type": args.dataset_type,
        "split": "test",
        "candidate_type": candidate_type,
        "pre_top_k": args.topk,
        "force_include_positive": False,
        "num_queries": DEFAULT_SOURCE_TOTAL,
    }
    mismatches = {
        key: (meta.get(key), expected)
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected cache metadata in {path}: {mismatches}")

    queries = cache["queries"]
    spec_embs = cache["spec_embs"]
    if len(queries) != DEFAULT_SOURCE_TOTAL or int(spec_embs.shape[0]) != DEFAULT_SOURCE_TOTAL:
        raise RuntimeError(
            f"Cache dimensions must both equal {DEFAULT_SOURCE_TOTAL} in {path}: "
            f"queries={len(queries)}, spec_embs={int(spec_embs.shape[0])}"
        )

    mol_smiles = cache["mol_smiles"]
    excluded_queries = []
    for index in args.exclude_query_indices:
        if not 0 <= index < len(queries):
            raise IndexError(f"Excluded query index {index} is out of range for {path}")
        query = queries[index]
        if int(query["spec_index"]) != index:
            raise RuntimeError(f"query_index/spec_index mismatch at {index} in {path}")
        if query.get("label") is None or not query.get("positive_in_base_topk"):
            raise RuntimeError(f"Expected excluded query {index} to be labeled in {path}")

        label = int(query["label"])
        candidate_indices = query["candidate_indices"]
        if not 0 <= label < len(candidate_indices):
            raise RuntimeError(f"Invalid label {label} at query {index} in {path}")
        candidate_smiles = mol_smiles[int(candidate_indices[label])]
        if candidate_smiles != query["true_smiles"]:
            raise RuntimeError(
                f"Label candidate does not match true_smiles at query {index} in {path}"
            )
        excluded_queries.append(
            {
                "query_index": index,
                "spec_index": int(query["spec_index"]),
                "true_smiles": query["true_smiles"],
                "label": label,
                "candidate_smiles": candidate_smiles,
            }
        )

    excluded_set = set(args.exclude_query_indices)
    original_labeled = sum(query.get("label") is not None for query in queries)
    filtered_labeled = sum(
        query.get("label") is not None
        for index, query in enumerate(queries)
        if index not in excluded_set
    )
    filtered_total = len(queries) - len(excluded_set)
    filtered_upper_bound = filtered_labeled / filtered_total * 100

    if is_default_protocol(args):
        labels = [query["label"] for query in excluded_queries]
        true_smiles = [query["true_smiles"] for query in excluded_queries]
        if labels != DEFAULT_EXCLUDED_LABELS[candidate_type]:
            raise RuntimeError(
                f"Audited labels changed for {candidate_type}: "
                f"{labels} vs {DEFAULT_EXCLUDED_LABELS[candidate_type]}"
            )
        if true_smiles != [AUDITED_TRUE_SMILES] * len(DEFAULT_EXCLUDED_INDICES):
            raise RuntimeError(f"Audited true_smiles changed for {candidate_type}: {true_smiles}")
        if original_labeled != DEFAULT_ORIGINAL_LABELED[candidate_type]:
            raise RuntimeError(
                f"Audited labeled-query count changed for {candidate_type}: {original_labeled}"
            )
        if filtered_labeled != DEFAULT_FILTERED_LABELED[candidate_type]:
            raise RuntimeError(
                f"Audited filtered labeled-query count changed for {candidate_type}: "
                f"{filtered_labeled}"
            )
        if filtered_total != DEFAULT_FILTERED_TOTAL:
            raise RuntimeError(
                f"Audited filtered total changed for {candidate_type}: {filtered_total}"
            )
        if round(filtered_upper_bound, 4) != DEFAULT_FILTERED_UPPER_BOUND[candidate_type]:
            raise RuntimeError(
                f"Audited filtered upper bound changed for {candidate_type}: "
                f"{filtered_upper_bound}"
            )

    metadata = file_metadata(path)
    metadata["cache_meta"] = meta
    metadata["excluded_queries"] = excluded_queries
    metadata["original_total"] = len(queries)
    metadata["original_labeled_queries"] = original_labeled
    metadata["expected_original_upper_bound_pct"] = original_labeled / len(queries) * 100
    metadata["expected_filtered_total"] = filtered_total
    metadata["filtered_labeled_queries"] = filtered_labeled
    metadata["expected_filtered_upper_bound_pct"] = filtered_upper_bound
    del cache, queries, spec_embs, mol_smiles
    gc.collect()
    return metadata


def validate_cross_cache_exclusions(cache_metadata: dict[str, dict]) -> None:
    reference = None
    for candidate_type, metadata in sorted(cache_metadata.items()):
        identities = [
            (
                query["query_index"],
                query["spec_index"],
                query["true_smiles"],
            )
            for query in metadata["excluded_queries"]
        ]
        if reference is None:
            reference = identities
        elif identities != reference:
            raise RuntimeError(
                f"Excluded query identities differ across caches for {candidate_type}: "
                f"{identities} vs {reference}"
            )


def experiment_root(args, experiment: SensitivityExperiment) -> Path:
    return (
        args.output_root
        / experiment.candidate_type
        / experiment.model_type
        / f"seed{experiment.seed}"
    )


def attempt_directories(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("attempt_[0-9][0-9][0-9]") if path.is_dir())


def experiment_fingerprint(
    args,
    experiment: SensitivityExperiment,
    source: dict,
    cache_metadata: dict,
) -> dict:
    return {
        "git_commit": args.git_commit,
        "params_sha256": args.params_sha256,
        "dataset_type": args.dataset_type,
        "run_prefix": args.run_prefix,
        "topk": args.topk,
        "candidate_type": experiment.candidate_type,
        "model_type": experiment.model_type,
        "seed": experiment.seed,
        "exclude_query_indices": args.exclude_query_indices,
        "mces": False,
        "source_checkpoint": source["checkpoint"],
        "source_eval_log": source["eval_log"],
        "test_cache": {
            key: cache_metadata[key]
            for key in ("path", "size_bytes", "mtime_ns")
        },
        "cache_protocol": {
            key: cache_metadata[key]
            for key in (
                "excluded_queries",
                "original_total",
                "original_labeled_queries",
                "expected_filtered_total",
                "filtered_labeled_queries",
                "expected_filtered_upper_bound_pct",
            )
        },
    }


def sensitivity_evaluation_complete(attempt: Path, cache_metadata: dict) -> bool:
    eval_log = attempt / "eval_rerank.log"
    if not eval_log.exists():
        return False
    metrics = parse_rerank_eval_metrics(eval_log)
    required = [
        "total_queries",
        "excluded_queries",
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
    ]
    if any(field not in metrics for field in required):
        return False
    if metrics["total_queries"] != cache_metadata["expected_filtered_total"]:
        return False
    if metrics["excluded_queries"] != len(cache_metadata["excluded_queries"]):
        return False
    if round(float(metrics["upper_bound_pct"]), 4) != round(
        float(cache_metadata["expected_filtered_upper_bound_pct"]),
        4,
    ):
        return False
    return "MCES calculation skipped." in eval_log.read_text(encoding="utf-8", errors="replace")


def select_attempt(
    args,
    fingerprint: dict,
    experiment: SensitivityExperiment,
    cache_metadata: dict,
) -> tuple[str, Path]:
    root = experiment_root(args, experiment)
    attempts = attempt_directories(root)
    for attempt in reversed(attempts):
        status = read_json(attempt / "status.json")
        if (
            status.get("state") == "complete"
            and status.get("fingerprint") == fingerprint
            and sensitivity_evaluation_complete(attempt, cache_metadata)
        ):
            if not args.rerun_completed:
                return "skip", attempt
            break
    return "eval", root / f"attempt_{len(attempts) + 1:03d}"


def build_eval_command(
    repo_root: Path,
    args,
    experiment: SensitivityExperiment,
    source_attempt: Path,
    attempt: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "eval_rerank.py"),
        "--cache",
        str(cache_path(args, experiment)),
        "--checkpoint",
        str(source_attempt / "best_reranker.pth"),
        "--save_dir",
        str(attempt),
        "--device",
        args.device,
        "--exclude-query-indices",
        *[str(index) for index in args.exclude_query_indices],
        "--no-mces",
    ]


def run_eval_subprocess(command: list[str], repo_root: Path) -> None:
    process = subprocess.Popen(command, cwd=repo_root, start_new_session=True)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_experiment(
    args,
    repo_root: Path,
    experiment: SensitivityExperiment,
    source: dict,
    cache_metadata: dict,
) -> tuple[str, Path]:
    fingerprint = experiment_fingerprint(args, experiment, source, cache_metadata)
    action, attempt = select_attempt(args, fingerprint, experiment, cache_metadata)
    source_attempt = Path(source["attempt"])
    command = build_eval_command(repo_root, args, experiment, source_attempt, attempt)
    label = f"{experiment.candidate_type}/{experiment.model_type}/seed{experiment.seed}"

    if action == "skip":
        print(f"Skipping completed sensitivity evaluation {label}: {attempt}", flush=True)
        return "skipped", attempt
    if args.dry_run:
        print(f"Planned sensitivity evaluation {label}: {attempt}", flush=True)
        print("Running:", " ".join(command), flush=True)
        return "planned", attempt

    status_path = attempt / "status.json"
    status = {
        "state": "evaluating",
        **asdict(experiment),
        "attempt": str(attempt),
        "source_attempt": str(source_attempt),
        "exclude_query_indices": args.exclude_query_indices,
        "device": args.device,
        "git_commit": args.git_commit,
        "git_worktree_changes": args.git_worktree_changes,
        "params_sha256": args.params_sha256,
        "hostname": socket.gethostname(),
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "fingerprint": fingerprint,
        "eval_command": command,
    }

    attempt_created = False
    try:
        attempt.mkdir(parents=True, exist_ok=False)
        attempt_created = True
        atomic_write_json(status_path, status)
        status["gpu_before_eval"] = require_available_gpu(
            args.device,
            args.min_free_mib,
            args.max_utilization,
            attempt / "gpu_before_eval.txt",
        )
        atomic_write_json(status_path, status)
        print("Running:", " ".join(command), flush=True)
        run_eval_subprocess(command, repo_root)
        if not sensitivity_evaluation_complete(attempt, cache_metadata):
            raise RuntimeError(f"Sensitivity evaluation artifacts are incomplete: {attempt}")

        metrics = parse_rerank_eval_metrics(attempt / "eval_rerank.log")
        status["state"] = "complete"
        status["finished_at"] = now_iso()
        status["metrics"] = metrics
        atomic_write_json(status_path, status)
        print(f"Completed sensitivity evaluation {label}: {attempt}", flush=True)
        return "complete", attempt
    except KeyboardInterrupt:
        if attempt_created:
            status["state"] = "interrupted"
            status["interrupted_at"] = now_iso()
            status["error"] = "Interrupted by user"
            atomic_write_json(status_path, status)
        raise
    except Exception as error:
        if attempt_created:
            status["state"] = "failed"
            status["failed_at"] = now_iso()
            status["error"] = str(error)
            atomic_write_json(status_path, status)
        raise


def find_complete_attempt(
    args,
    experiment: SensitivityExperiment,
    source: dict,
    cache_metadata: dict,
) -> Path | None:
    fingerprint = experiment_fingerprint(args, experiment, source, cache_metadata)
    for attempt in reversed(attempt_directories(experiment_root(args, experiment))):
        status = read_json(attempt / "status.json")
        if (
            status.get("state") == "complete"
            and status.get("fingerprint") == fingerprint
            and sensitivity_evaluation_complete(attempt, cache_metadata)
        ):
            return attempt
    return None


def write_summaries(
    args,
    sources: dict[tuple[str, str, int], dict],
    cache_metadata: dict[str, dict],
) -> None:
    rows = []
    for key in sorted(sources):
        experiment = SensitivityExperiment(*key)
        source = sources[key]
        attempt = find_complete_attempt(
            args,
            experiment,
            source,
            cache_metadata[experiment.candidate_type],
        )
        if attempt is None:
            continue

        original = source["metrics"]
        filtered = parse_rerank_eval_metrics(attempt / "eval_rerank.log")
        row = {
            **asdict(experiment),
            "exclude_query_indices": " ".join(str(index) for index in args.exclude_query_indices),
            "source_attempt": source["attempt"],
            "sensitivity_attempt": str(attempt),
            "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        }
        for metric in METRIC_FIELDS:
            if metric not in original or metric not in filtered:
                continue
            original_value = original[metric]
            filtered_value = filtered[metric]
            row[f"original_{metric}"] = original_value
            row[f"filtered_{metric}"] = filtered_value
            row[f"delta_{metric}"] = filtered_value - original_value
        rows.append(row)

    if not rows:
        return
    rows.sort(key=lambda row: (row["candidate_type"], row["model_type"], int(row["seed"])))

    preferred = [
        "candidate_type",
        "model_type",
        "seed",
        "exclude_query_indices",
        "source_attempt",
        "sensitivity_attempt",
        "source_checkpoint_sha256",
    ]
    fields = preferred.copy()
    for metric in METRIC_FIELDS:
        for prefix in ("original", "filtered", "delta"):
            field = f"{prefix}_{metric}"
            if any(field in row for row in rows):
                fields.append(field)
    atomic_write_csv(args.output_root / "summary.csv", rows, fields)

    aggregate_rows = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["candidate_type"], row["model_type"]), []).append(row)
    aggregate_metrics = [
        field
        for field in fields
        if field.startswith("filtered_rerank_") or field.startswith("delta_rerank_")
    ]
    for (candidate_type, model_type), group in sorted(groups.items()):
        aggregate = {
            "candidate_type": candidate_type,
            "model_type": model_type,
            "num_seeds": len(group),
        }
        for metric in aggregate_metrics:
            values = [float(row[metric]) for row in group if metric in row]
            if not values:
                continue
            aggregate[f"{metric}_num_seeds"] = len(values)
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)

    aggregate_fields = []
    for row in aggregate_rows:
        for field in row:
            if field not in aggregate_fields:
                aggregate_fields.append(field)
    atomic_write_csv(args.output_root / "summary_aggregate.csv", aggregate_rows, aggregate_fields)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing full rerankers after excluding exact cross-split spectrum inputs."
    )
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--dataset-type", default="massspecgym")
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("rerank_cache"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--candidate-types",
        nargs="+",
        choices=["mass", "formula"],
        default=["mass", "formula"],
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["pointwise", "transformer"],
        default=["pointwise", "transformer"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--exclude-query-indices",
        nargs="+",
        type=int,
        default=DEFAULT_EXCLUDED_INDICES,
    )
    parser.add_argument("--device", default="cuda:1", help="Explicit CUDA device, for example cuda:1.")
    parser.add_argument("--min-free-mib", type=int, default=16_000)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be greater than 0")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique non-negative integers")
    if any(index < 0 for index in args.exclude_query_indices):
        parser.error("--exclude-query-indices must contain non-negative integers")
    args.exclude_query_indices = sorted(set(args.exclude_query_indices))
    if not args.exclude_query_indices:
        parser.error("--exclude-query-indices must not be empty")
    if len(set(args.candidate_types)) != len(args.candidate_types):
        parser.error("--candidate-types must not contain duplicates")
    if len(set(args.model_types)) != len(args.model_types):
        parser.error("--model-types must not contain duplicates")
    if args.min_free_mib < 0:
        parser.error("--min-free-mib must be non-negative")
    if not 0 <= args.max_utilization <= 100:
        parser.error("--max-utilization must be between 0 and 100")
    try:
        parse_cuda_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    if args.source_root is None:
        args.source_root = Path("checkpoints_rerank") / f"{args.run_prefix}_topk{args.topk}_multiseed"
    if args.output_root is None:
        args.output_root = (
            Path("checkpoints_rerank")
            / f"{args.run_prefix}_topk{args.topk}_input_overlap_sensitivity"
        )
    return args


def write_batch_status(args, payload: dict) -> None:
    atomic_write_json(args.output_root / "batch_status.json", payload)
    atomic_write_json(args.batch_run_status_path, payload)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    resolve_runtime_paths(args, repo_root)
    validate_path_isolation(args)
    args.git_commit = git_commit(repo_root)
    args.git_worktree_changes = git_worktree_changes(repo_root)
    args.params_sha256 = sha256_file(repo_root / "params.yaml")
    if args.git_worktree_changes and not (args.dry_run or args.allow_dirty):
        changed = "\n".join(f"  {line}" for line in args.git_worktree_changes)
        raise RuntimeError(
            "Refusing to run a paper sensitivity evaluation from a dirty worktree. "
            "Commit the changes first or pass --allow-dirty explicitly:\n"
            f"{changed}"
        )

    experiments = [
        SensitivityExperiment(candidate_type, model_type, seed)
        for candidate_type in args.candidate_types
        for model_type in args.model_types
        for seed in args.seeds
    ]
    source_attempts = load_source_attempts(args, repo_root)
    source_experiments = [SensitivityExperiment(*key) for key in sorted(source_attempts)]
    source_candidate_types = sorted({experiment.candidate_type for experiment in source_experiments})
    cache_metadata = {
        candidate_type: validate_test_cache(args, candidate_type)
        for candidate_type in source_candidate_types
    }
    validate_cross_cache_exclusions(cache_metadata)
    sources = {
        experiment_key(experiment): validate_source_attempt(
            experiment,
            source_attempts[experiment_key(experiment)],
            cache_metadata[experiment.candidate_type],
        )
        for experiment in source_experiments
    }

    print(f"Planned sensitivity evaluations: {len(experiments)}")
    print(f"Source root: {args.source_root}")
    print(f"Output root: {args.output_root}")
    print(f"Excluded query indices: {args.exclude_query_indices}")

    if args.dry_run:
        require_available_gpu(args.device, args.min_free_mib, args.max_utilization)
        for experiment in experiments:
            execute_experiment(
                args,
                repo_root,
                experiment,
                sources[experiment_key(experiment)],
                cache_metadata[experiment.candidate_type],
            )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    args.batch_run_status_path = args.output_root / "batch_runs" / f"{run_stamp}_{os.getpid()}.json"
    planned = [asdict(experiment) for experiment in experiments]
    initial_status = {
        "state": "running",
        "started_at": now_iso(),
        "git_commit": args.git_commit,
        "git_worktree_changes": args.git_worktree_changes,
        "params_sha256": args.params_sha256,
        "runner_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "device": args.device,
        "exclude_query_indices": args.exclude_query_indices,
        "experiments": planned,
        "results": [],
        "errors": [],
    }
    write_batch_status(args, initial_status)

    results = []
    errors = []
    final_state = "complete"
    try:
        for experiment in experiments:
            state, attempt = execute_experiment(
                args,
                repo_root,
                experiment,
                sources[experiment_key(experiment)],
                cache_metadata[experiment.candidate_type],
            )
            results.append({**asdict(experiment), "state": state, "attempt": str(attempt)})
            write_summaries(args, sources, cache_metadata)
    except KeyboardInterrupt:
        errors.append("Interrupted by user")
        final_state = "interrupted"
        raise
    except Exception as error:
        errors.append(str(error))
        final_state = "failed"
        raise
    finally:
        final_status = {
            "state": final_state,
            "finished_at": now_iso(),
            "git_commit": args.git_commit,
            "git_worktree_changes": args.git_worktree_changes,
            "params_sha256": args.params_sha256,
            "runner_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "device": args.device,
            "exclude_query_indices": args.exclude_query_indices,
            "experiments": planned,
            "results": results,
            "errors": errors,
        }
        write_batch_status(args, final_status)


if __name__ == "__main__":
    main()
