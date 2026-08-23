import argparse
import csv
import hashlib
import json
import os
import queue
import re
import socket
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from SpecEmbedding.utils.gpu import parse_cuda_device, require_available_gpu
from SpecEmbedding.utils.rerank import parse_rerank_eval_metrics


@dataclass(frozen=True)
class Experiment:
    candidate_type: str
    model_type: str
    ablation: str
    seed: int


FULL_RERANKER_OVERRIDES = {
    "use_base_score_feature": True,
    "use_residual_score": True,
    "use_rank_embedding": False,
    "use_product_feature": True,
    "use_abs_diff_feature": True,
    "lambda_pair": 0.2,
    "lambda_spec": 0.1,
    "pair_mode": "antisymmetric",
    "use_spectrum_conditioning": True,
    "shuffle_candidates": True,
}

ABLATION_OVERRIDES = {
    "full": {**FULL_RERANKER_OVERRIDES},
    "no_base_score": {
        **FULL_RERANKER_OVERRIDES,
        "use_base_score_feature": False,
        "use_residual_score": False,
    },
    "no_residual": {**FULL_RERANKER_OVERRIDES, "use_residual_score": False},
    "no_rank_embedding": {**FULL_RERANKER_OVERRIDES, "use_rank_embedding": False},
    "no_product_feature": {**FULL_RERANKER_OVERRIDES, "use_product_feature": False},
    "no_abs_diff_feature": {**FULL_RERANKER_OVERRIDES, "use_abs_diff_feature": False},
    "no_interaction_features": {
        **FULL_RERANKER_OVERRIDES,
        "use_product_feature": False,
        "use_abs_diff_feature": False,
    },
    "listwise_only": {**FULL_RERANKER_OVERRIDES, "lambda_pair": 0.0},
    "no_spectrum_conditioning": {
        **FULL_RERANKER_OVERRIDES,
        "use_spectrum_conditioning": False,
    },
    "no_lambda_pair": {**FULL_RERANKER_OVERRIDES, "lambda_pair": 0.0},
    "no_lambda_spec": {**FULL_RERANKER_OVERRIDES, "lambda_spec": 0.0},
    "directed_pair": {**FULL_RERANKER_OVERRIDES, "pair_mode": "directed"},
    "antisymmetric_pair": {**FULL_RERANKER_OVERRIDES, "pair_mode": "antisymmetric"},
    "no_candidate_shuffle": {**FULL_RERANKER_OVERRIDES, "shuffle_candidates": False},
}


def ablation_overrides(experiment: Experiment) -> dict:
    return ABLATION_OVERRIDES[experiment.ablation]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def git_worktree_changes(repo_root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_batch_status(args, payload: dict) -> None:
    atomic_write_json(args.output_root / "batch_status.json", payload)
    atomic_write_json(args.batch_run_status_path, payload)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def cache_paths(args, experiment: Experiment) -> dict[str, Path]:
    cache_dir = args.cache_root / f"{args.run_prefix}_{experiment.candidate_type}_topk{args.topk}"
    return {
        split: cache_dir / f"{args.dataset_type}_{experiment.candidate_type}_{split}.pt"
        for split in ("train", "val", "test")
    }


def cache_file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def experiment_fingerprint(args, experiment: Experiment, caches: dict[str, Path]) -> dict:
    return {
        "git_commit": args.git_commit,
        "params_sha256": args.params_sha256,
        "dataset_type": args.dataset_type,
        "run_prefix": args.run_prefix,
        "topk": args.topk,
        "candidate_type": experiment.candidate_type,
        "model_type": experiment.model_type,
        "ablation": experiment.ablation,
        "ablation_overrides": ablation_overrides(experiment),
        "seed": experiment.seed,
        "max_train_queries": args.max_train_queries,
        "exclude_val_query_indices": args.exclude_val_query_indices,
        "cache_files": {split: cache_file_metadata(path) for split, path in caches.items()},
    }


def experiment_dir(args, experiment: Experiment) -> Path:
    root = args.output_root / experiment.candidate_type / experiment.model_type
    if experiment.ablation != "full":
        root = root / experiment.ablation
    return root / f"seed{experiment.seed}"


def attempt_directories(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("attempt_[0-9][0-9][0-9]") if path.is_dir())


def checkpoint_matches(
    path: Path,
    experiment: Experiment,
    expected_val_exclusions: list[int] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if checkpoint.get("seed") != experiment.seed:
        return False
    model_config = checkpoint.get("model_config", {})
    if model_config.get("model_type") != experiment.model_type:
        return False
    training_config = checkpoint.get("training_config", {})
    if expected_val_exclusions is not None:
        actual_exclusions = sorted(set(training_config.get("exclude_val_query_indices", [])))
        if actual_exclusions != expected_val_exclusions:
            return False
    model_config_keys = {
        "pair_mode",
        "use_base_score_feature",
        "use_residual_score",
        "use_rank_embedding",
        "use_product_feature",
        "use_abs_diff_feature",
        "use_spectrum_conditioning",
    }
    for key, expected in ablation_overrides(experiment).items():
        source = model_config if key in model_config_keys else training_config
        if source.get(key) != expected:
            return False
    return True


def training_complete(
    attempt_dir: Path,
    experiment: Experiment,
    expected_val_exclusions: list[int] | None = None,
) -> bool:
    train_log = attempt_dir / "train_rerank.log"
    if not train_log.exists():
        return False
    log_text = train_log.read_text(encoding="utf-8", errors="replace")
    return (
        "Training finished." in log_text
        and checkpoint_matches(
            attempt_dir / "best_reranker.pth",
            experiment,
            expected_val_exclusions,
        )
        and checkpoint_matches(
            attempt_dir / "last_reranker.pth",
            experiment,
            expected_val_exclusions,
        )
    )


def evaluation_complete(attempt_dir: Path, expect_mces: bool) -> bool:
    eval_log = attempt_dir / "eval_rerank.log"
    if not eval_log.exists():
        return False
    log_text = eval_log.read_text(encoding="utf-8", errors="replace")
    required = ["Total queries:", "BASE RESULTS", "RERANK RESULTS"]
    if not all(token in log_text for token in required):
        return False
    if expect_mces:
        return "Rerank MCES@1" in log_text
    return "MCES calculation skipped." in log_text or "Rerank MCES@1" in log_text


def training_selection_metadata(attempt_dir: Path) -> dict:
    checkpoint_path = attempt_dir / "best_reranker.pth"
    train_log = attempt_dir / "train_rerank.log"
    if not checkpoint_path.exists() or not train_log.exists():
        return {}
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return {}

    training_config = checkpoint.get("training_config", {})
    log_text = train_log.read_text(encoding="utf-8", errors="replace")
    epochs = [
        int(match.group(1))
        for match in re.finditer(r"Epoch\s+(\d+):\s+train_loss=", log_text)
    ]
    stop_epoch = max(epochs) if epochs else None
    configured_epochs = training_config.get("epochs")
    return {
        "metric_for_best": training_config.get("metric_for_best"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_metric": checkpoint.get("best_metric"),
        "stop_epoch": stop_epoch,
        "early_stopped": (
            isinstance(stop_epoch, int)
            and isinstance(configured_epochs, int)
            and stop_epoch < configured_epochs
        ),
    }


def process_is_alive(status: dict) -> bool:
    if status.get("hostname") != socket.gethostname():
        return False
    pid = status.get("runner_pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    return True


def attempt_matches_current_fingerprint(args, experiment: Experiment, attempt_dir: Path) -> bool:
    caches = cache_paths(args, experiment)
    if any(not path.exists() for path in caches.values()):
        return False
    status = read_json(attempt_dir / "status.json")
    return status.get("fingerprint") == experiment_fingerprint(args, experiment, caches)


def select_attempt(args, experiment: Experiment) -> tuple[str, Path]:
    root = experiment_dir(args, experiment)
    attempts = attempt_directories(root)

    for attempt in reversed(attempts):
        if (
            attempt_matches_current_fingerprint(args, experiment, attempt)
            and training_complete(attempt, experiment, args.exclude_val_query_indices)
            and evaluation_complete(attempt, args.mces)
        ):
            if not args.rerun_completed:
                return "skip", attempt
            break

    if attempts:
        latest = attempts[-1]
        latest_status = read_json(latest / "status.json")
        if latest_status.get("state") in {"training", "evaluating"} and process_is_alive(latest_status):
            raise RuntimeError(f"Experiment is already running: {latest}")
        if (
            attempt_matches_current_fingerprint(args, experiment, latest)
            and training_complete(latest, experiment, args.exclude_val_query_indices)
        ):
            if not evaluation_complete(latest, args.mces):
                return "eval", latest

    attempt_number = len(attempts) + 1
    return "train_eval", root / f"attempt_{attempt_number:03d}"


def build_train_command(
    repo_root: Path,
    caches: dict[str, Path],
    attempt_dir: Path,
    experiment: Experiment,
    device: str,
    train_k: int,
    max_train_queries: int | None = None,
    exclude_val_query_indices: list[int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "train_rerank.py"),
        "--train_cache",
        str(caches["train"]),
        "--val_cache",
        str(caches["val"]),
        "--save_dir",
        str(attempt_dir),
        "--model_type",
        experiment.model_type,
        "--seed",
        str(experiment.seed),
        "--device",
        device,
        "--train-k",
        str(train_k),
    ]
    if max_train_queries is not None:
        command.extend(["--max-train-queries", str(max_train_queries)])
    flag_names = {
        "use_base_score_feature": "--base-score-feature",
        "use_residual_score": "--residual-score",
        "use_rank_embedding": "--rank-embedding",
        "use_product_feature": "--product-feature",
        "use_abs_diff_feature": "--abs-diff-feature",
        "use_spectrum_conditioning": "--spectrum-conditioning",
        "shuffle_candidates": "--shuffle-candidates",
    }
    for key, value in ablation_overrides(experiment).items():
        if key == "lambda_pair":
            command.extend(["--lambda-pair", str(value)])
        elif key == "lambda_spec":
            command.extend(["--lambda-spec", str(value)])
        elif key == "pair_mode":
            command.extend(["--pair-mode", value])
        elif key in flag_names:
            command.append(flag_names[key] if value else flag_names[key].replace("--", "--no-", 1))
        else:
            raise ValueError(f"Unsupported ablation override: {key}")
    if exclude_val_query_indices:
        command.extend(
            [
                "--exclude-val-query-indices",
                *[str(index) for index in exclude_val_query_indices],
            ]
        )
    return command


def build_eval_command(
    repo_root: Path,
    caches: dict[str, Path],
    attempt_dir: Path,
    device: str,
    mces: bool,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "eval_rerank.py"),
        "--cache",
        str(caches["test"]),
        "--checkpoint",
        str(attempt_dir / "best_reranker.pth"),
        "--save_dir",
        str(attempt_dir),
        "--device",
        device,
        "--mces" if mces else "--no-mces",
    ]


def print_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)


def gpu_preflight(args, device: str, output_path: Path | None = None) -> dict:
    if args.skip_gpu_preflight:
        payload = {
            "status": "skipped",
            "reason": "nvidia-smi subprocess preflight is unavailable in this execution environment",
            "device": device,
        }
        if output_path is not None:
            atomic_write_json(output_path.with_suffix(".json"), payload)
        return payload
    return require_available_gpu(
        device,
        args.min_free_mib,
        args.max_utilization,
        output_path,
    )


def base_status(
    args,
    experiment: Experiment,
    attempt_dir: Path,
    device: str,
    caches: dict[str, Path],
    train_command: list[str],
    eval_command: list[str],
) -> dict:
    return {
        "state": "pending",
        "candidate_type": experiment.candidate_type,
        "model_type": experiment.model_type,
        "ablation": experiment.ablation,
        "ablation_overrides": ablation_overrides(experiment),
        "exclude_val_query_indices": args.exclude_val_query_indices,
        "seed": experiment.seed,
        "device": device,
        "attempt_dir": str(attempt_dir),
        "git_commit": args.git_commit,
        "git_worktree_changes": args.git_worktree_changes,
        "params_sha256": args.params_sha256,
        "hostname": socket.gethostname(),
        "runner_pid": os.getpid(),
        "fingerprint": experiment_fingerprint(args, experiment, caches),
        "cache_files": {split: cache_file_metadata(path) for split, path in caches.items()},
        "train_command": train_command,
        "eval_command": eval_command,
    }


def run_subprocess(
    command: list[str],
    repo_root: Path,
    device: str,
    active_processes: dict[str, subprocess.Popen],
    active_lock: threading.Lock,
) -> None:
    print_command(command)
    process = subprocess.Popen(command, cwd=repo_root)
    with active_lock:
        active_processes[device] = process
    try:
        return_code = process.wait()
    finally:
        with active_lock:
            active_processes.pop(device, None)
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def terminate_active_processes(
    active_processes: dict[str, subprocess.Popen],
    active_lock: threading.Lock,
    failed_device: str,
) -> None:
    with active_lock:
        processes = list(active_processes.items())
    for device, process in processes:
        if device == failed_device or process.poll() is not None:
            continue
        print(f"Stopping batch-owned process on {device} (pid={process.pid}) after a peer failure.")
        process.terminate()


def execute_experiment(
    args,
    repo_root: Path,
    experiment: Experiment,
    device: str,
    active_processes: dict[str, subprocess.Popen],
    active_lock: threading.Lock,
) -> tuple[str, Path]:
    action, attempt_dir = select_attempt(args, experiment)
    caches = cache_paths(args, experiment)
    for split, path in caches.items():
        if not path.exists():
            raise FileNotFoundError(f"{split} cache not found: {path}")

    train_command = build_train_command(
        repo_root,
        caches,
        attempt_dir,
        experiment,
        device,
        args.topk,
        args.max_train_queries,
        args.exclude_val_query_indices,
    )
    eval_command = build_eval_command(repo_root, caches, attempt_dir, device, args.mces)
    label = (
        f"{experiment.candidate_type}/{experiment.model_type}/"
        f"{experiment.ablation}/seed{experiment.seed}"
    )

    if action == "skip":
        print(f"Skipping completed experiment {label}: {attempt_dir}")
        return "skipped", attempt_dir
    if args.dry_run:
        print(f"Planned experiment {label} on {device} ({action}): {attempt_dir}")
        if action == "train_eval":
            print_command(train_command)
        print_command(eval_command)
        return "planned", attempt_dir

    attempt_dir.mkdir(parents=True, exist_ok=False if action == "train_eval" else True)
    status_path = attempt_dir / "status.json"
    status = base_status(args, experiment, attempt_dir, device, caches, train_command, eval_command)
    status["started_at"] = now_iso()
    existing_selection = training_selection_metadata(attempt_dir)
    if existing_selection:
        status["training_selection"] = existing_selection

    try:
        if action == "train_eval":
            status["state"] = "training"
            atomic_write_json(status_path, status)
            status["gpu_before_train"] = gpu_preflight(
                args,
                device,
                attempt_dir / "gpu_before_train.txt",
            )
            atomic_write_json(status_path, status)
            run_subprocess(train_command, repo_root, device, active_processes, active_lock)
            if not training_complete(
                attempt_dir,
                experiment,
                args.exclude_val_query_indices,
            ):
                raise RuntimeError(f"Training command exited successfully but artifacts are incomplete: {attempt_dir}")
            status["training_selection"] = training_selection_metadata(attempt_dir)
            status["state"] = "train_complete"
            status["train_finished_at"] = now_iso()
            atomic_write_json(status_path, status)

        status["state"] = "evaluating"
        status["gpu_before_eval"] = gpu_preflight(
            args,
            device,
            attempt_dir / "gpu_before_eval.txt",
        )
        atomic_write_json(status_path, status)
        run_subprocess(eval_command, repo_root, device, active_processes, active_lock)
        if not evaluation_complete(attempt_dir, args.mces):
            raise RuntimeError(f"Evaluation command exited successfully but the log is incomplete: {attempt_dir}")

        status["state"] = "complete"
        status["finished_at"] = now_iso()
        atomic_write_json(status_path, status)
        print(f"Completed experiment {label}: {attempt_dir}")
        return "complete", attempt_dir
    except Exception as error:
        status["state"] = "failed"
        status["failed_at"] = now_iso()
        status["error"] = str(error)
        atomic_write_json(status_path, status)
        raise


def find_complete_attempt(
    args,
    experiment: Experiment,
    *,
    expect_mces: bool | None = None,
) -> Path | None:
    if expect_mces is None:
        expect_mces = args.mces
    for attempt in reversed(attempt_directories(experiment_dir(args, experiment))):
        if (
            attempt_matches_current_fingerprint(args, experiment, attempt)
            and training_complete(attempt, experiment, args.exclude_val_query_indices)
            and evaluation_complete(attempt, expect_mces)
        ):
            return attempt
    return None


def experiment_key(experiment: Experiment) -> tuple[str, str, str, int]:
    return (
        experiment.candidate_type,
        experiment.model_type,
        experiment.ablation,
        experiment.seed,
    )


def discover_experiments(args, requested: list[Experiment]) -> list[Experiment]:
    discovered = {experiment_key(experiment): experiment for experiment in requested}
    if not args.output_root.exists():
        return sorted(discovered.values(), key=experiment_key)

    for status_path in args.output_root.rglob("status.json"):
        status = read_json(status_path)
        candidate_type = status.get("candidate_type")
        model_type = status.get("model_type")
        ablation = status.get("ablation")
        seed = status.get("seed")
        if candidate_type not in {"mass", "formula"}:
            continue
        if model_type not in {"pointwise", "relative", "transformer"}:
            continue
        if ablation not in ABLATION_OVERRIDES or not isinstance(seed, int):
            continue
        experiment = Experiment(candidate_type, model_type, ablation, seed)
        discovered[experiment_key(experiment)] = experiment
    return sorted(discovered.values(), key=experiment_key)


def write_summaries(args, experiments: list[Experiment]) -> None:
    experiments = discover_experiments(args, experiments)
    rows = []
    for experiment in experiments:
        # Ranking summaries remain complete when MCES is later recomputed for
        # only a subset. MCES columns are parsed opportunistically per log.
        attempt = find_complete_attempt(args, experiment, expect_mces=False)
        if attempt is None:
            continue
        row = asdict(experiment)
        row["attempt_dir"] = str(attempt)
        row.update(training_selection_metadata(attempt))
        row.update(parse_rerank_eval_metrics(attempt / "eval_rerank.log"))
        rows.append(row)

    if not rows:
        return

    preferred_fields = [
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
        "base_mces",
        "rerank_mces",
        "attempt_dir",
    ]
    fields = [field for field in preferred_fields if any(field in row for row in rows)]
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    metric_fields = [field for field in fields if field.startswith("rerank_") and field != "rerank_mces"]
    if "rerank_mces" in fields:
        metric_fields.append("rerank_mces")
    metric_fields.extend(
        field
        for field in ("best_val_metric", "best_epoch", "stop_epoch", "early_stopped")
        if field in fields
    )
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["candidate_type"], row["model_type"], row["ablation"])
        groups.setdefault(key, []).append(row)
    for (candidate_type, model_type, ablation), group in sorted(groups.items()):
        aggregate = {
            "candidate_type": candidate_type,
            "model_type": model_type,
            "ablation": ablation,
            "num_seeds": len(group),
        }
        for metric in metric_fields:
            values = [row[metric] for row in group if metric in row]
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
    with (args.output_root / "summary_aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed reranker variants and ablations on reusable top-k caches."
    )
    parser.add_argument("--run-prefix", required=True, help="Alignment run prefix used by existing cache directories.")
    parser.add_argument("--dataset-type", default="massspecgym")
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument(
        "--candidate-types",
        nargs="+",
        choices=["mass", "formula"],
        default=["mass", "formula"],
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["pointwise", "relative", "transformer"],
        default=["pointwise", "transformer"],
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(ABLATION_OVERRIDES),
        default=["full"],
        help="Matched single-factor reranker ablations. Defaults to the full model.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=None,
        help="Limit labeled training queries per run and record the limit in the experiment fingerprint.",
    )
    parser.add_argument(
        "--exclude-val-query-indices",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based validation cache query indices excluded before model selection.",
    )
    parser.add_argument("--devices", nargs="+", required=True, help="Explicit devices, for example cuda:0 cuda:1.")
    parser.add_argument("--cache-root", type=Path, default=Path("rerank_cache"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--min-free-mib", type=int, default=16_000)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip runner nvidia-smi preflight when subprocess capture cannot access the driver; record the reason.",
    )
    parser.add_argument(
        "--mces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute MCES for every seed. Disabled by default because it is expensive.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a real run from a dirty worktree. Not recommended for paper experiments.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be greater than 0")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain only non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.max_train_queries is not None and args.max_train_queries <= 0:
        parser.error("--max-train-queries must be greater than 0")
    if len(set(args.ablations)) != len(args.ablations):
        parser.error("--ablations must not contain duplicates")
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices must not contain duplicates")
    if any(index < 0 for index in args.exclude_val_query_indices):
        parser.error("--exclude-val-query-indices must contain non-negative integers")
    args.exclude_val_query_indices = sorted(set(args.exclude_val_query_indices))
    if args.min_free_mib < 0:
        parser.error("--min-free-mib must be non-negative")
    if not 0 <= args.max_utilization <= 100:
        parser.error("--max-utilization must be between 0 and 100")
    for device in args.devices:
        try:
            parse_cuda_device(device)
        except ValueError as error:
            parser.error(str(error))

    if args.output_root is None:
        suffix = "ablations" if any(name != "full" for name in args.ablations) else "multiseed"
        args.output_root = Path("checkpoints_rerank") / f"{args.run_prefix}_topk{args.topk}_{suffix}"
    return args


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    args.git_commit = git_commit(repo_root)
    args.git_worktree_changes = git_worktree_changes(repo_root)
    args.params_sha256 = sha256_file(repo_root / "params.yaml")
    if args.git_worktree_changes and not (args.dry_run or args.allow_dirty):
        changed = "\n".join(f"  {line}" for line in args.git_worktree_changes)
        raise RuntimeError(
            "Refusing to run paper experiments from a dirty worktree. Commit the changes first "
            "or pass --allow-dirty explicitly:\n"
            f"{changed}"
        )

    experiments = [
        Experiment(
            candidate_type=candidate_type,
            model_type=model_type,
            ablation=ablation,
            seed=seed,
        )
        for candidate_type in args.candidate_types
        for model_type in args.model_types
        for ablation in args.ablations
        for seed in args.seeds
    ]

    print(f"Planned experiments: {len(experiments)}")
    print(f"Output root: {args.output_root}")
    print(f"MCES during per-seed evaluation: {args.mces}")
    print(f"Excluded validation query indices: {args.exclude_val_query_indices}")
    initial_gpu_states = {}
    for device in args.devices:
        initial_gpu_states[device] = gpu_preflight(args, device)

    if args.dry_run:
        for index, experiment in enumerate(experiments):
            device = args.devices[index % len(args.devices)]
            execute_experiment(args, repo_root, experiment, device, {}, threading.Lock())
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    args.batch_run_status_path = args.output_root / "batch_runs" / f"{run_stamp}_{os.getpid()}.json"
    write_batch_status(
        args,
        {
            "state": "running",
            "started_at": now_iso(),
            "git_commit": args.git_commit,
            "git_worktree_changes": args.git_worktree_changes,
            "params_sha256": args.params_sha256,
            "runner_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "devices": initial_gpu_states,
            "experiments": [asdict(experiment) for experiment in experiments],
            "exclude_val_query_indices": args.exclude_val_query_indices,
            "skip_gpu_preflight": args.skip_gpu_preflight,
            "mces": args.mces,
        },
    )

    experiment_queue: queue.Queue[Experiment] = queue.Queue()
    for experiment in experiments:
        experiment_queue.put(experiment)

    stop_event = threading.Event()
    active_processes: dict[str, subprocess.Popen] = {}
    active_lock = threading.Lock()
    result_lock = threading.Lock()
    results: list[dict] = []
    errors: list[str] = []

    def worker(device: str) -> None:
        while not stop_event.is_set():
            try:
                experiment = experiment_queue.get_nowait()
            except queue.Empty:
                return
            try:
                state, attempt_dir = execute_experiment(
                    args,
                    repo_root,
                    experiment,
                    device,
                    active_processes,
                    active_lock,
                )
                with result_lock:
                    results.append({**asdict(experiment), "state": state, "attempt_dir": str(attempt_dir)})
            except Exception as error:
                message = (
                    f"{experiment.candidate_type}/{experiment.model_type}/"
                    f"{experiment.ablation}/seed{experiment.seed} "
                    f"on {device}: {error}"
                )
                print(f"ERROR: {message}", file=sys.stderr, flush=True)
                with result_lock:
                    errors.append(message)
                stop_event.set()
                terminate_active_processes(active_processes, active_lock, device)
            finally:
                experiment_queue.task_done()

    workers = [threading.Thread(target=worker, args=(device,), name=device) for device in args.devices]
    for worker_thread in workers:
        worker_thread.start()
    try:
        for worker_thread in workers:
            worker_thread.join()
    except KeyboardInterrupt:
        print("Interrupted; stopping batch-owned child processes.", file=sys.stderr, flush=True)
        stop_event.set()
        terminate_active_processes(active_processes, active_lock, failed_device="")
        with result_lock:
            errors.append("Batch interrupted by user.")
        for worker_thread in workers:
            worker_thread.join()

    write_summaries(args, experiments)
    final_state = "failed" if errors else "complete"
    write_batch_status(
        args,
        {
            "state": final_state,
            "finished_at": now_iso(),
            "git_commit": args.git_commit,
            "git_worktree_changes": args.git_worktree_changes,
            "params_sha256": args.params_sha256,
            "runner_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "devices": initial_gpu_states,
            "experiments": [asdict(experiment) for experiment in experiments],
            "exclude_val_query_indices": args.exclude_val_query_indices,
            "skip_gpu_preflight": args.skip_gpu_preflight,
            "results": results,
            "errors": errors,
            "mces": args.mces,
        },
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
