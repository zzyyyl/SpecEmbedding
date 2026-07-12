import csv
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from run_rerank_overlap_sensitivity import (
    SensitivityExperiment,
    execute_experiment,
    experiment_fingerprint,
    experiment_root,
    find_complete_attempt,
    load_source_attempts,
    resolve_runtime_paths,
    run_eval_subprocess,
    select_attempt,
    sensitivity_evaluation_complete,
    validate_path_isolation,
    write_summaries,
)
from SpecEmbedding.utils.gpu import parse_cuda_device
from SpecEmbedding.utils.rerank import parse_rerank_eval_metrics


def evaluation_log(
    total: int,
    *,
    excluded: int | None = None,
    rerank_top1: float = 70.0,
    upper_bound: float = 82.1512,
) -> str:
    lines = []
    if excluded is not None:
        lines.append(f"Excluded {excluded} cache queries for sensitivity evaluation: [5908, 5909, 5910]")
    lines.extend(
        [
            f"Total queries: {total}",
            f"Pre-retrieval upper bound: {upper_bound:.4f}%",
            "BASE RESULTS",
            "Top-1  Accuracy : 43.7817%",
            "Top-5  Accuracy : 61.0893%",
            "Top-10 Accuracy : 68.4270%",
            "Top-20 Accuracy : 75.5426%",
            "MRR              : 0.5196",
            "RERANK RESULTS",
            f"Top-1  Accuracy : {rerank_top1:.4f}%",
            "Top-5  Accuracy : 75.9000%",
            "Top-10 Accuracy : 78.4000%",
            "Top-20 Accuracy : 80.6000%",
            "MRR              : 0.7150",
            "MCES calculation skipped.",
        ]
    )
    return "\n".join(lines) + "\n"


class RerankSensitivityTest(unittest.TestCase):
    def _args(self, root: Path):
        return SimpleNamespace(
            output_root=root / "output",
            source_root=root / "source",
            cache_root=root / "cache",
            candidate_types=["mass"],
            model_types=["pointwise"],
            seeds=[42],
            git_commit="commit",
            params_sha256="params",
            dataset_type="massspecgym",
            run_prefix="run",
            topk=40,
            exclude_query_indices=[5908, 5909, 5910],
            rerun_completed=False,
            dry_run=False,
            device="cuda:1",
            min_free_mib=16_000,
            max_utilization=20,
            git_worktree_changes=[],
        )

    def _source(self, root: Path, seed: int = 42):
        source_attempt = root / f"legacy_source_seed{seed}"
        source_attempt.mkdir(parents=True, exist_ok=True)
        source_log = source_attempt / "eval_rerank.log"
        source_log.write_text(evaluation_log(17_556), encoding="utf-8")
        return {
            "attempt": str(source_attempt),
            "checkpoint": {
                "path": str(source_attempt / "best_reranker.pth"),
                "size_bytes": 10,
                "mtime_ns": 20,
                "sha256": "checkpoint",
            },
            "eval_log": {
                "path": str(source_log),
                "size_bytes": source_log.stat().st_size,
                "mtime_ns": source_log.stat().st_mtime_ns,
                "sha256": "source_log",
            },
            "metrics": parse_rerank_eval_metrics(source_log),
        }

    def _cache_metadata(self, root: Path):
        return {
            "path": str(root / "test.pt"),
            "size_bytes": 30,
            "mtime_ns": 40,
            "excluded_queries": [
                {"query_index": index, "spec_index": index, "true_smiles": "smiles"}
                for index in (5908, 5909, 5910)
            ],
            "original_total": 17_556,
            "original_labeled_queries": 14_423,
            "expected_filtered_total": 17_553,
            "filtered_labeled_queries": 14_420,
            "expected_filtered_upper_bound_pct": 82.15119922520367,
        }

    def _complete_attempt(self, args, experiment, source, cache_metadata, rerank_top1=69.99):
        attempt = experiment_root(args, experiment) / "attempt_001"
        attempt.mkdir(parents=True, exist_ok=True)
        (attempt / "eval_rerank.log").write_text(
            evaluation_log(17_553, excluded=3, rerank_top1=rerank_top1),
            encoding="utf-8",
        )
        status = {
            "state": "complete",
            "fingerprint": experiment_fingerprint(args, experiment, source, cache_metadata),
        }
        (attempt / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return attempt

    def test_cuda_device_must_be_explicit(self):
        self.assertEqual(parse_cuda_device("cuda:1"), 1)
        with self.assertRaises(ValueError):
            parse_cuda_device("cuda")

    def test_eval_log_parser_reads_sensitivity_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval.log"
            path.write_text(evaluation_log(17_553, excluded=3), encoding="utf-8")
            metrics = parse_rerank_eval_metrics(path)
        self.assertEqual(metrics["total_queries"], 17_553)
        self.assertEqual(metrics["excluded_queries"], 3)
        self.assertEqual(metrics["rerank_top1_pct"], 70.0)

    def test_source_summary_supports_legacy_full_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.source_root.mkdir(parents=True)
            attempt = root / "legacy" / "attempt_001"
            with (args.source_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["candidate_type", "model_type", "seed", "attempt_dir"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_type": "mass",
                        "model_type": "pointwise",
                        "seed": 42,
                        "attempt_dir": str(attempt),
                    }
                )
            attempts = load_source_attempts(args, root)
        self.assertEqual(attempts[("mass", "pointwise", 42)], attempt.resolve())

    def test_source_summary_returns_other_full_rows_and_ignores_ablations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.source_root.mkdir(parents=True)
            with (args.source_root / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "candidate_type",
                        "model_type",
                        "ablation",
                        "seed",
                        "attempt_dir",
                    ],
                )
                writer.writeheader()
                for seed, ablation, suffix in (
                    (42, "full", "full42"),
                    (43, "", "full43"),
                    (42, "no_residual", "ablation42"),
                ):
                    writer.writerow(
                        {
                            "candidate_type": "mass",
                            "model_type": "pointwise",
                            "ablation": ablation,
                            "seed": seed,
                            "attempt_dir": suffix,
                        }
                    )

            attempts = load_source_attempts(args, root)

        self.assertEqual(
            set(attempts),
            {("mass", "pointwise", 42), ("mass", "pointwise", 43)},
        )

    def test_runtime_paths_are_repo_relative_and_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary).resolve()
            args = SimpleNamespace(
                source_root=Path("source"),
                cache_root=Path("cache"),
                output_root=Path("output"),
            )
            resolve_runtime_paths(args, repo_root)
            self.assertEqual(args.source_root, repo_root / "source")
            self.assertEqual(args.cache_root, repo_root / "cache")
            self.assertEqual(args.output_root, repo_root / "output")
            validate_path_isolation(args)

            for source, output in (
                (repo_root / "source", repo_root / "source"),
                (repo_root / "source", repo_root / "source" / "child"),
                (repo_root / "output" / "child", repo_root / "output"),
            ):
                with self.subTest(source=source, output=output):
                    args.source_root = source
                    args.output_root = output
                    with self.assertRaises(RuntimeError):
                        validate_path_isolation(args)

            args.source_root = repo_root / "source"
            args.output_root = repo_root / "source_backup"
            validate_path_isolation(args)

    def test_completed_attempt_is_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            experiment = SensitivityExperiment("mass", "pointwise", 42)
            source = self._source(root)
            cache_metadata = self._cache_metadata(root)
            attempt = self._complete_attempt(args, experiment, source, cache_metadata)
            fingerprint = experiment_fingerprint(args, experiment, source, cache_metadata)

            action, selected = select_attempt(args, fingerprint, experiment, cache_metadata)
            self.assertEqual((action, selected), ("skip", attempt))
            self.assertTrue(sensitivity_evaluation_complete(attempt, cache_metadata))

            args.rerun_completed = True
            action, selected = select_attempt(args, fingerprint, experiment, cache_metadata)
            self.assertEqual(action, "eval")
            self.assertEqual(selected.name, "attempt_002")

    def test_wrong_upper_bound_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            experiment = SensitivityExperiment("mass", "pointwise", 42)
            source = self._source(root)
            cache_metadata = self._cache_metadata(root)
            attempt = self._complete_attempt(args, experiment, source, cache_metadata)
            (attempt / "eval_rerank.log").write_text(
                evaluation_log(17_553, excluded=3, upper_bound=83.1512),
                encoding="utf-8",
            )
            fingerprint = experiment_fingerprint(args, experiment, source, cache_metadata)

            self.assertFalse(sensitivity_evaluation_complete(attempt, cache_metadata))
            self.assertIsNone(find_complete_attempt(args, experiment, source, cache_metadata))
            action, selected = select_attempt(
                args,
                fingerprint,
                experiment,
                cache_metadata,
            )
            self.assertEqual(action, "eval")
            self.assertEqual(selected.name, "attempt_002")

    def test_summary_contains_filtered_delta_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            experiment = SensitivityExperiment("mass", "pointwise", 42)
            source = self._source(root)
            cache_metadata = self._cache_metadata(root)
            self._complete_attempt(args, experiment, source, cache_metadata, rerank_top1=69.99)

            write_summaries(
                args,
                {("mass", "pointwise", 42): source},
                {"mass": cache_metadata},
            )
            with (args.output_root / "summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (args.output_root / "summary_aggregate.csv").open(encoding="utf-8") as handle:
                aggregate = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["delta_rerank_top1_pct"]), -0.01)
        self.assertEqual(aggregate[0]["num_seeds"], "1")
        self.assertEqual(aggregate[0]["delta_rerank_top1_pct_num_seeds"], "1")

    def test_summary_retains_completed_attempts_from_other_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            cache_metadata = self._cache_metadata(root)
            first = SensitivityExperiment("mass", "pointwise", 42)
            second = SensitivityExperiment("mass", "pointwise", 43)
            first_source = self._source(root, seed=42)
            second_source = self._source(root, seed=43)
            sources = {
                experiment_key: source
                for experiment_key, source in (
                    (("mass", "pointwise", 42), first_source),
                    (("mass", "pointwise", 43), second_source),
                )
            }

            self._complete_attempt(args, first, first_source, cache_metadata)
            write_summaries(args, sources, {"mass": cache_metadata})
            (args.output_root / "summary.csv").unlink()
            (args.output_root / "summary_aggregate.csv").unlink()

            self._complete_attempt(args, second, second_source, cache_metadata)
            write_summaries(args, sources, {"mass": cache_metadata})
            with (args.output_root / "summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (args.output_root / "summary_aggregate.csv").open(encoding="utf-8") as handle:
                aggregate = list(csv.DictReader(handle))

        self.assertEqual([int(row["seed"]) for row in rows], [42, 43])
        self.assertEqual(aggregate[0]["num_seeds"], "2")

    def test_keyboard_interrupt_reaps_process_group(self):
        process = Mock(pid=123)
        process.wait.side_effect = [
            KeyboardInterrupt(),
            subprocess.TimeoutExpired(cmd="eval", timeout=10),
            0,
        ]
        with (
            patch("run_rerank_overlap_sensitivity.subprocess.Popen", return_value=process),
            patch("run_rerank_overlap_sensitivity.os.killpg") as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            run_eval_subprocess(["python", "eval.py"], Path("/repo"))

        self.assertEqual(
            killpg.call_args_list,
            [call(123, signal.SIGTERM), call(123, signal.SIGKILL)],
        )

    def test_keyboard_interrupt_marks_attempt_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            experiment = SensitivityExperiment("mass", "pointwise", 42)
            source = self._source(root)
            cache_metadata = self._cache_metadata(root)
            with (
                patch(
                    "run_rerank_overlap_sensitivity.require_available_gpu",
                    return_value={},
                ),
                patch(
                    "run_rerank_overlap_sensitivity.run_eval_subprocess",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                execute_experiment(args, root, experiment, source, cache_metadata)

            status_path = experiment_root(args, experiment) / "attempt_001" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["state"], "interrupted")
        self.assertEqual(status["error"], "Interrupted by user")
        self.assertIn("interrupted_at", status)
        self.assertNotIn("failed_at", status)


if __name__ == "__main__":
    unittest.main()
