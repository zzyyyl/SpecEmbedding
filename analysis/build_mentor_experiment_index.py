#!/usr/bin/env python3
"""Build a repository-relative index for the mentor-review experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOTS = [
    REPOSITORY_ROOT / "checkpoints_rerank" / f"mentor2026_align{seed}_topk40_full_gpu_b32"
    for seed in (42, 43, 44)
]
ABLATION_ROOT = (
    REPOSITORY_ROOT
    / "checkpoints_rerank"
    / "mentor2026_align42_topk40_relative_ablations_gpu_b32"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "mentor2026_experiment_index.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def artifact(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_path(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        if include_sha256:
            record["sha256"] = sha256_file(path)
    return record


def sanitize(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key not in {"hostname", "gpu_before_train", "gpu_before_eval"}
        }
    if isinstance(value, str):
        if value.startswith(str(REPOSITORY_ROOT)):
            return relative_path(Path(value))
        return value
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def attempt_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def collect_attempts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: dict[tuple[Any, ...], tuple[int, Path, dict[str, Any]]] = {}
    failed: list[dict[str, Any]] = []
    for status_path in sorted(root.rglob("status.json")):
        status = read_json(status_path)
        if status.get("state") not in {"complete", "failed"}:
            continue
        key = (
            status.get("candidate_type"),
            status.get("model_type"),
            status.get("ablation", "full"),
            status.get("seed"),
        )
        attempt_dir = status_path.parent
        entry = {
            "experiment": {
                "candidate_type": key[0],
                "model_type": key[1],
                "ablation": key[2],
                "seed": key[3],
            },
            "attempt": attempt_dir.name,
            "status": sanitize(status),
            "artifacts": [
                artifact(attempt_dir / filename)
                for filename in (
                    "best_reranker.pth",
                    "last_reranker.pth",
                    "train_rerank.log",
                    "eval_rerank.log",
                    "status.json",
                )
                if (attempt_dir / filename).exists()
            ],
        }
        if status.get("state") == "failed":
            failed.append(entry)
            continue
        candidate = (attempt_number(attempt_dir), attempt_dir, status)
        previous = complete.get(key)
        if previous is None or candidate[0] > previous[0]:
            complete[key] = candidate

    entries = []
    for key, (_, attempt_dir, status) in sorted(complete.items()):
        entries.append(
            {
                "experiment": {
                    "candidate_type": key[0],
                    "model_type": key[1],
                    "ablation": key[2],
                    "seed": key[3],
                },
                "attempt": attempt_dir.name,
                "source_commit": status.get("git_commit"),
                "params_sha256": status.get("params_sha256"),
                "train_command": sanitize(status.get("train_command", [])),
                "eval_command": sanitize(status.get("eval_command", [])),
                "cache_files": sanitize(status.get("cache_files", {})),
                "metadata": {
                    "max_train_queries": status.get("max_train_queries"),
                    "batch_size": status.get("batch_size"),
                    "exclude_val_query_indices": status.get("exclude_val_query_indices", []),
                    "ablation_overrides": status.get("ablation_overrides", {}),
                },
                "artifacts": [
                    artifact(attempt_dir / filename)
                    for filename in (
                        "best_reranker.pth",
                        "last_reranker.pth",
                        "train_rerank.log",
                        "eval_rerank.log",
                        "status.json",
                    )
                    if (attempt_dir / filename).exists()
                ],
            }
        )
    return entries, failed


def tracked_analysis_files() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    command = [
        "git",
        "ls-files",
        "analysis/mentor2026_experiments",
        "analysis/mentor2026_baseline",
    ]
    for line in subprocess.check_output(command, cwd=REPOSITORY_ROOT, text=True).splitlines():
        path = REPOSITORY_ROOT / line
        if path.is_file():
            output.append(artifact(path))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output

    main_runs: list[dict[str, Any]] = []
    main_failures: list[dict[str, Any]] = []
    for root in MAIN_ROOTS:
        complete, failed = collect_attempts(root)
        main_runs.extend(complete)
        main_failures.extend(failed)
    ablation_runs, ablation_failures = collect_attempts(ABLATION_ROOT)

    params = REPOSITORY_ROOT / "params.yaml"
    payload = {
        "schema_version": 1,
        "scope": "mentor review 2026-08-23",
        "repository_head_at_index": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "params": artifact(params),
        "experiment_groups": {
            "main_alignment_pointwise_relative": {
                "roots": [relative_path(root) for root in MAIN_ROOTS],
                "complete_runs": main_runs,
                "failed_attempts": main_failures,
                "expected_complete_runs": 36,
            },
            "alignment42_relative_ablations": {
                "root": relative_path(ABLATION_ROOT),
                "complete_runs": ablation_runs,
                "failed_attempts": ablation_failures,
                "expected_complete_runs": 30,
            },
        },
        "analysis_outputs": tracked_analysis_files(),
        "environment_files": [
            artifact(REPOSITORY_ROOT / "environment.yml"),
            artifact(REPOSITORY_ROOT / "reproducibility" / "conda-linux-64.lock"),
            artifact(REPOSITORY_ROOT / "reproducibility" / "pip-linux-64.lock"),
            artifact(REPOSITORY_ROOT / "reproducibility" / "runtime-snapshot.yaml"),
        ],
        "notes": [
            "Checkpoint, cache, training-log, and evaluation-log paths are repository-relative and refer to local ignored artifacts; they must be copied into a controlled supplement or release archive before submission.",
            "The complete per-run ranking metrics are in eval_rerank.log. The tracked hard-query JSON files contain selected-query ranking/prediction outputs for alignment-42 seed-42 pointwise and relative runs; the tracked mentor2026_baseline JSON files contain the local JESTR-style cosine controls.",
            "Cache metadata records source paths, sizes, and mtimes in each status file; raw MassSpecGym data and caches are not redistributed by this index.",
            "The failed-attempt list is retained for provenance; only the latest complete attempt for each experiment key is used in the complete_runs lists.",
            "All seed and alignment aggregates are descriptive; this index does not imply significance testing or bitwise retraining equivalence.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": relative_path(output),
        "main_complete": len(main_runs),
        "ablation_complete": len(ablation_runs),
        "failed_attempts": len(main_failures) + len(ablation_failures),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
