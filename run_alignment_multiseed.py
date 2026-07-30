import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_ALIGNMENT_SEEDS = [43, 44]
RUN_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CUDA_DEVICE_PATTERN = re.compile(r"^cuda:[0-9]+$")


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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the overlap-clean alignment, cache, and reranker pipeline for "
            "multiple alignment seeds."
        )
    )
    parser.add_argument(
        "--run-prefix-base",
        required=True,
        help=(
            "Stable experiment prefix without an alignment-seed suffix. "
            "Each run appends _alignseedN."
        ),
    )
    parser.add_argument(
        "--alignment-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_ALIGNMENT_SEEDS,
    )
    parser.add_argument(
        "--alignment-device",
        required=True,
        help="Explicit CUDA device used for alignment and cache generation, for example cuda:1.",
    )
    parser.add_argument(
        "--rerank-devices",
        nargs="+",
        required=True,
        help="Explicit CUDA devices used by each inner reranker batch.",
    )
    parser.add_argument("--conda-env", default="specembedding")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Processed data root containing MassSpecGym train/val/test and candidate files.",
    )
    parser.add_argument(
        "--tokenset-cache",
        type=Path,
        required=True,
        help="Audited MassSpecGym TokenSet cache used for alignment training.",
    )
    parser.add_argument("--min-free-mib", type=int, default=16_000)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=Path("run_trainval_overlap_sensitivity.sh"),
        help="Inner single-alignment-seed pipeline.",
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        help=(
            "Directory for the outer batch status. Defaults to "
            "checkpoints_rerank/<run-prefix-base>_alignment_multiseed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every inner pipeline in dry-run mode without writing outer status files.",
    )
    args = parser.parse_args(argv)

    if not RUN_PREFIX_PATTERN.fullmatch(args.run_prefix_base):
        parser.error(
            "--run-prefix-base must start with an alphanumeric character and contain "
            "only alphanumerics, dots, underscores, or hyphens"
        )
    if any(seed < 0 for seed in args.alignment_seeds):
        parser.error("--alignment-seeds must contain only non-negative integers")
    if len(set(args.alignment_seeds)) != len(args.alignment_seeds):
        parser.error("--alignment-seeds must not contain duplicates")
    if not CUDA_DEVICE_PATTERN.fullmatch(args.alignment_device):
        parser.error("--alignment-device must be an explicit cuda:N device")
    if any(not CUDA_DEVICE_PATTERN.fullmatch(device) for device in args.rerank_devices):
        parser.error("--rerank-devices must contain only explicit cuda:N devices")
    if len(set(args.rerank_devices)) != len(args.rerank_devices):
        parser.error("--rerank-devices must not contain duplicates")
    if args.min_free_mib < 0:
        parser.error("--min-free-mib must be non-negative")
    if not 0 <= args.max_utilization <= 100:
        parser.error("--max-utilization must be between 0 and 100")
    if not args.conda_env.strip():
        parser.error("--conda-env must not be empty")

    if args.status_root is None:
        args.status_root = (
            Path("checkpoints_rerank")
            / f"{args.run_prefix_base}_alignment_multiseed"
        )
    return args


def resolve_path(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_seed_run(
    args: argparse.Namespace,
    repo_root: Path,
    alignment_seed: int,
) -> dict:
    run_prefix = f"{args.run_prefix_base}_alignseed{alignment_seed}"
    command = ["bash", str(args.pipeline_script)]
    if args.dry_run:
        command.append("--dry-run")
    environment = {
        "CONDA_ENV": args.conda_env,
        "DATA_PATH": str(args.data_path),
        "TOKENSET_CACHE": str(args.tokenset_cache),
        "RUN_PREFIX": run_prefix,
        "ALIGNMENT_SEED": str(alignment_seed),
        "ALIGN_DEVICE": args.alignment_device,
        "RERANK_DEVICE_LIST": " ".join(args.rerank_devices),
        "MIN_FREE_MIB": str(args.min_free_mib),
        "MAX_UTILIZATION": str(args.max_utilization),
    }
    return {
        "state": "pending",
        "alignment_seed": alignment_seed,
        "run_prefix": run_prefix,
        "alignment_dir": str(repo_root / "checkpoints_align" / run_prefix),
        "cache_dirs": {
            candidate_type: str(
                repo_root
                / "rerank_cache"
                / f"{run_prefix}_{candidate_type}_topk40"
            )
            for candidate_type in ("mass", "formula")
        },
        "rerank_output": str(
            repo_root
            / "checkpoints_rerank"
            / f"{run_prefix}_topk40_multiseed"
        ),
        "command": command,
        "environment": environment,
    }


def print_seed_run(run: dict) -> None:
    print(
        f"Alignment seed {run['alignment_seed']}: {run['run_prefix']}",
        flush=True,
    )
    print(
        "Environment:",
        " ".join(f"{key}={value}" for key, value in run["environment"].items()),
        flush=True,
    )
    print("Command:", " ".join(run["command"]), flush=True)


def execute_seed_run(run: dict, repo_root: Path) -> int:
    print_seed_run(run)
    environment = os.environ.copy()
    environment.update(run["environment"])
    result = subprocess.run(
        run["command"],
        cwd=repo_root,
        env=environment,
        check=False,
    )
    return result.returncode


def write_batch_status(
    status_root: Path,
    batch_run_status_path: Path,
    payload: dict,
) -> None:
    atomic_write_json(status_root / "batch_status.json", payload)
    atomic_write_json(batch_run_status_path, payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    args.pipeline_script = resolve_path(args.pipeline_script, repo_root)
    args.data_path = resolve_path(args.data_path, repo_root)
    args.tokenset_cache = resolve_path(args.tokenset_cache, repo_root)
    args.status_root = resolve_path(args.status_root, repo_root)

    if not args.pipeline_script.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {args.pipeline_script}")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Processed data root not found: {args.data_path}")
    if not args.tokenset_cache.is_file():
        raise FileNotFoundError(f"TokenSet cache not found: {args.tokenset_cache}")

    commit = git_commit(repo_root)
    worktree_changes = git_worktree_changes(repo_root)
    params_sha256 = sha256_file(repo_root / "params.yaml")
    pipeline_script_sha256 = sha256_file(args.pipeline_script)
    if worktree_changes and not args.dry_run:
        changed = "\n".join(f"  {line}" for line in worktree_changes)
        raise RuntimeError(
            "Refusing to run paper experiments from a dirty worktree. "
            "Commit the changes first:\n"
            f"{changed}"
        )

    runs = [
        build_seed_run(args, repo_root, alignment_seed)
        for alignment_seed in args.alignment_seeds
    ]
    print(f"Git commit: {commit}")
    print(f"Pipeline script SHA256: {pipeline_script_sha256}")
    print(f"Planned alignment seeds: {args.alignment_seeds}")
    print(f"Outer status root: {args.status_root}")

    if args.dry_run:
        for run in runs:
            returncode = execute_seed_run(run, repo_root)
            if returncode != 0:
                return returncode
        print("Alignment multi-seed dry-run completed; no outer status was written.")
        return 0

    run_stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    batch_run_status_path = (
        args.status_root
        / "batch_runs"
        / f"{run_stamp}_{os.getpid()}.json"
    )
    payload = {
        "state": "running",
        "started_at": now_iso(),
        "git_commit": commit,
        "git_worktree_changes": worktree_changes,
        "params_sha256": params_sha256,
        "pipeline_script": str(args.pipeline_script),
        "pipeline_script_sha256": pipeline_script_sha256,
        "run_prefix_base": args.run_prefix_base,
        "alignment_seeds": args.alignment_seeds,
        "alignment_device": args.alignment_device,
        "rerank_devices": args.rerank_devices,
        "data_path": str(args.data_path),
        "tokenset_cache": str(args.tokenset_cache),
        "runs": runs,
        "errors": [],
    }
    write_batch_status(args.status_root, batch_run_status_path, payload)

    for index, run in enumerate(runs):
        run["state"] = "running"
        run["started_at"] = now_iso()
        write_batch_status(args.status_root, batch_run_status_path, payload)
        try:
            returncode = execute_seed_run(run, repo_root)
        except KeyboardInterrupt:
            run["state"] = "interrupted"
            run["finished_at"] = now_iso()
            run["returncode"] = 130
            payload["state"] = "interrupted"
            payload["finished_at"] = now_iso()
            payload["errors"].append(
                {
                    "alignment_seed": run["alignment_seed"],
                    "run_prefix": run["run_prefix"],
                    "kind": "keyboard_interrupt",
                    "message": "Batch interrupted by user.",
                }
            )
            for pending in runs[index + 1 :]:
                pending["state"] = "not_started"
            write_batch_status(args.status_root, batch_run_status_path, payload)
            print(
                "Interrupted; recorded batch state before exiting.",
                file=sys.stderr,
                flush=True,
            )
            return 130
        except Exception as error:
            run["state"] = "failed"
            run["finished_at"] = now_iso()
            run["returncode"] = 1
            payload["state"] = "failed"
            payload["finished_at"] = now_iso()
            payload["errors"].append(
                {
                    "alignment_seed": run["alignment_seed"],
                    "run_prefix": run["run_prefix"],
                    "kind": "runner_exception",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            for pending in runs[index + 1 :]:
                pending["state"] = "not_started"
            write_batch_status(args.status_root, batch_run_status_path, payload)
            print(
                f"ERROR: alignment seed {run['alignment_seed']} failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        run["finished_at"] = now_iso()
        run["returncode"] = returncode
        if returncode != 0:
            run["state"] = "failed"
            payload["state"] = "failed"
            payload["finished_at"] = now_iso()
            payload["errors"].append(
                {
                    "alignment_seed": run["alignment_seed"],
                    "run_prefix": run["run_prefix"],
                    "returncode": returncode,
                }
            )
            for pending in runs[index + 1 :]:
                pending["state"] = "not_started"
            write_batch_status(args.status_root, batch_run_status_path, payload)
            return returncode
        run["state"] = "complete"
        write_batch_status(args.status_root, batch_run_status_path, payload)

    payload["state"] = "complete"
    payload["finished_at"] = now_iso()
    write_batch_status(args.status_root, batch_run_status_path, payload)
    print("Alignment multi-seed pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
