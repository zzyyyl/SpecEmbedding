import csv
import hashlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path

import analyze_alignment_multiseed as analysis


class AlignmentAnalysisFixture:
    def __init__(self, root: Path):
        self.root = root
        self.cache_hashes = {}
        self._write_canonical_manifest()
        self.specs = []
        self.rerank_roots = {}
        for alignment_seed in analysis.EXPECTED_ALIGNMENT_SEEDS:
            self._write_run(alignment_seed)

    def _cache_bytes(self, alignment_seed: int, candidate: str, split: str) -> bytes:
        return f"cache-{alignment_seed}-{candidate}-{split}".encode()

    def _write_canonical_manifest(self) -> None:
        artifacts = []
        run_prefix = "alignseed42"
        for candidate in analysis.CANDIDATE_TYPES:
            for split in analysis.SPLITS:
                relative = (
                    Path("rerank_cache")
                    / f"{run_prefix}_{candidate}_topk{analysis.TOPK}"
                    / f"massspecgym_{candidate}_{split}.pt"
                )
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                data = self._cache_bytes(42, candidate, split)
                path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                self.cache_hashes[(42, candidate, split)] = digest
                artifacts.append(
                    {
                        "id": f"cache.{candidate}.{split}",
                        "location": f"repo:{relative.as_posix()}",
                        "sha256": digest,
                        "size_bytes": len(data),
                    }
                )
        manifest = self.root / "paper" / "adma2026_artifact_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")

    def delta(self, alignment_seed: int, candidate: str) -> float:
        return {
            (42, "mass"): 1.0,
            (42, "formula"): 0.5,
            (43, "mass"): 0.2,
            (43, "formula"): -0.3,
            (44, "mass"): -0.4,
            (44, "formula"): 0.1,
        }[(alignment_seed, candidate)]

    def _write_run(self, alignment_seed: int) -> None:
        align_root = self.root / "checkpoints_align" / f"alignseed{alignment_seed}"
        align_root.mkdir(parents=True)
        checkpoint = align_root / "best_model_stage2.pth"
        checkpoint.write_bytes(f"alignment-{alignment_seed}".encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selection = {
            "dataset_type": "massspecgym",
            "seed": alignment_seed,
            "exclude_val_query_indices": list(analysis.VALIDATION_EXCLUSIONS),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
        }
        selection_path = align_root / "alignment_selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")

        cache_records = {}
        run_prefix = f"alignseed{alignment_seed}"
        for candidate in analysis.CANDIDATE_TYPES:
            cache_records[candidate] = {}
            for split in analysis.SPLITS:
                cache_path = analysis.canonical_cache_path(
                    self.root, run_prefix, "massspecgym", candidate, split
                )
                if not cache_path.exists():
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(
                        self._cache_bytes(alignment_seed, candidate, split)
                    )
                stat = cache_path.stat()
                cache_records[candidate][split] = {
                    "path": str(cache_path.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }

        rerank_root = self.root / "checkpoints_rerank" / f"alignseed{alignment_seed}"
        rerank_root.mkdir(parents=True)
        summary_rows = []
        batch_results = []
        source_commit = f"commit-{alignment_seed}"
        for candidate_index, candidate in enumerate(analysis.CANDIDATE_TYPES):
            base_top1 = 20.0 + alignment_seed + candidate_index
            base_mrr = 0.2 + alignment_seed / 1000 + candidate_index / 100
            for model in analysis.MODEL_TYPES:
                for reranker_seed in analysis.RERANKER_SEEDS:
                    attempt = (
                        rerank_root
                        / candidate
                        / model
                        / f"seed{reranker_seed}"
                        / "attempt_001"
                    )
                    attempt.mkdir(parents=True)
                    status = {
                        "state": "complete",
                        "candidate_type": candidate,
                        "model_type": model,
                        "ablation": "full",
                        "seed": reranker_seed,
                        "git_commit": source_commit,
                        "params_sha256": "params-sha",
                        "exclude_val_query_indices": list(
                            analysis.VALIDATION_EXCLUSIONS
                        ),
                        "git_worktree_changes": [],
                        "attempt_dir": str(attempt.resolve()),
                        "ablation_overrides": analysis.FULL_RERANKER_OVERRIDES,
                    }
                    pointwise_top1 = base_top1 + 10 + reranker_seed / 100
                    pointwise_mrr = base_mrr + 0.3 + reranker_seed / 10000
                    model_delta = self.delta(alignment_seed, candidate)
                    if model == "transformer":
                        top1 = pointwise_top1 + model_delta
                        mrr = pointwise_mrr + model_delta / 100
                    else:
                        top1 = pointwise_top1
                        mrr = pointwise_mrr
                    status["training_selection"] = {
                        "metric_for_best": "mrr",
                        "best_epoch": 3,
                        "best_val_metric": mrr,
                        "stop_epoch": 8,
                        "early_stopped": True,
                    }
                    status["cache_files"] = cache_records[candidate]
                    status["fingerprint"] = {
                        "git_commit": source_commit,
                        "params_sha256": "params-sha",
                        "dataset_type": "massspecgym",
                        "run_prefix": run_prefix,
                        "topk": analysis.TOPK,
                        "candidate_type": candidate,
                        "model_type": model,
                        "ablation": "full",
                        "ablation_overrides": analysis.FULL_RERANKER_OVERRIDES,
                        "seed": reranker_seed,
                        "exclude_val_query_indices": list(
                            analysis.VALIDATION_EXCLUSIONS
                        ),
                        "cache_files": cache_records[candidate],
                    }
                    (attempt / "status.json").write_text(
                        json.dumps(status), encoding="utf-8"
                    )
                    (attempt / "best_reranker.pth").write_bytes(
                        f"best-{alignment_seed}-{candidate}-{model}-{reranker_seed}".encode()
                    )
                    (attempt / "last_reranker.pth").write_bytes(
                        f"last-{alignment_seed}-{candidate}-{model}-{reranker_seed}".encode()
                    )
                    (attempt / "train_rerank.log").write_text(
                        "\n".join(
                            [
                                *[
                                    f"Epoch {epoch}: train_loss=1.0"
                                    for epoch in range(1, 9)
                                ],
                                "Training finished.",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    (attempt / "eval_rerank.log").write_text(
                        self.eval_log_text(base_top1, base_mrr, top1, mrr),
                        encoding="utf-8",
                    )
                    summary_rows.append(
                        {
                            "candidate_type": candidate,
                            "model_type": model,
                            "ablation": "full",
                            "seed": reranker_seed,
                            "metric_for_best": "mrr",
                            "best_epoch": 3,
                            "best_val_metric": mrr,
                            "stop_epoch": 8,
                            "early_stopped": True,
                            "upper_bound_pct": base_top1 + 50,
                            "base_top1_pct": base_top1,
                            "base_top5_pct": base_top1 + 5,
                            "base_top10_pct": base_top1 + 10,
                            "base_top20_pct": base_top1 + 15,
                            "base_mrr_raw": base_mrr,
                            "rerank_top1_pct": top1,
                            "rerank_top5_pct": top1 + 5,
                            "rerank_top10_pct": top1 + 10,
                            "rerank_top20_pct": top1 + 15,
                            "rerank_mrr_raw": mrr,
                            "attempt_dir": str(attempt.resolve()),
                        }
                    )
                    batch_results.append(
                        {
                            "candidate_type": candidate,
                            "model_type": model,
                            "ablation": "full",
                            "seed": reranker_seed,
                            "state": "complete",
                            "attempt_dir": str(attempt.resolve()),
                        }
                    )
        self.write_summary(rerank_root, summary_rows)
        self.write_aggregate(rerank_root, summary_rows)
        batch = {
            "state": "complete",
            "git_commit": source_commit,
            "git_worktree_changes": [],
            "params_sha256": "params-sha",
            "exclude_val_query_indices": list(analysis.VALIDATION_EXCLUSIONS),
            "results": batch_results,
            "errors": [],
            "mces": False,
            "experiments": [
                {
                    "candidate_type": result["candidate_type"],
                    "model_type": result["model_type"],
                    "ablation": result["ablation"],
                    "seed": result["seed"],
                }
                for result in batch_results
            ],
        }
        (rerank_root / "batch_status.json").write_text(
            json.dumps(batch), encoding="utf-8"
        )
        self.specs.append(analysis.RunSpec(alignment_seed, selection_path, rerank_root))
        self.rerank_roots[alignment_seed] = rerank_root

    def eval_log_text(
        self, base_top1: float, base_mrr: float, rerank_top1: float, rerank_mrr: float
    ) -> str:
        return f"""Total queries: 17556
Pre-retrieval upper bound: {base_top1 + 50:.10f}%
BASE RESULTS
Top-1 Accuracy : {base_top1:.10f}%
Top-5 Accuracy : {base_top1 + 5:.10f}%
Top-10 Accuracy : {base_top1 + 10:.10f}%
Top-20 Accuracy : {base_top1 + 15:.10f}%
MRR : {base_mrr:.10f}
RERANK RESULTS
Top-1 Accuracy : {rerank_top1:.10f}%
Top-5 Accuracy : {rerank_top1 + 5:.10f}%
Top-10 Accuracy : {rerank_top1 + 10:.10f}%
Top-20 Accuracy : {rerank_top1 + 15:.10f}%
MRR : {rerank_mrr:.10f}
MCES calculation skipped.
==================================================
"""

    def write_summary(self, rerank_root: Path, rows: list[dict]) -> None:
        with (rerank_root / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(analysis.SUMMARY_REQUIRED_COLUMNS),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def read_summary(self, alignment_seed: int) -> list[dict[str, str]]:
        with (self.rerank_roots[alignment_seed] / "summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def write_aggregate(self, rerank_root: Path, rows: list[dict]) -> None:
        aggregate_rows = []
        for candidate in analysis.CANDIDATE_TYPES:
            for model in analysis.MODEL_TYPES:
                group = [
                    row
                    for row in rows
                    if row["candidate_type"] == candidate
                    and row["model_type"] == model
                ]
                aggregate = {
                    "candidate_type": candidate,
                    "model_type": model,
                    "ablation": "full",
                    "num_seeds": len(group),
                }
                for metric in analysis.AGGREGATE_METRICS:
                    values = [
                        (1.0 if row[metric] in (True, "True") else 0.0)
                        if metric == "early_stopped"
                        else float(row[metric])
                        for row in group
                    ]
                    aggregate[f"{metric}_num_seeds"] = len(values)
                    aggregate[f"{metric}_mean"] = statistics.mean(values)
                    aggregate[f"{metric}_std"] = statistics.stdev(values)
                aggregate_rows.append(aggregate)
        fields = []
        for row in aggregate_rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with (rerank_root / "summary_aggregate.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(aggregate_rows)


class AlignmentMultiseedAnalysisTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> AlignmentAnalysisFixture:
        return AlignmentAnalysisFixture(root)

    def test_success_writes_complete_deterministic_analysis_and_fails_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            output = root / "analysis"

            manifest = analysis.run_analysis(fixture.specs, output, root)
            first_bytes = {
                path.name: path.read_bytes() for path in sorted(output.iterdir())
            }
            second_manifest = analysis.run_analysis(fixture.specs, output, root)
            second_bytes = {
                path.name: path.read_bytes() for path in sorted(output.iterdir())
            }

            self.assertEqual(manifest, second_manifest)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(manifest["counts"]["summary_rows"], 36)
            self.assertEqual(manifest["counts"]["paired_rows"], 18)
            self.assertEqual(manifest["counts"]["alignment_candidate_rows"], 6)
            self.assertEqual(manifest["counts"]["cross_alignment_rows"], 2)
            self.assertEqual(manifest["counts"]["cache_artifacts"], 18)
            self.assertEqual(manifest["counts"]["attempts"], 36)
            self.assertEqual(manifest["design"]["n_alignment_seeds"], 3)
            self.assertEqual(
                manifest["evaluation_identity_protocol"],
                analysis.EVALUATION_IDENTITY_PROTOCOL,
            )
            self.assertFalse(
                manifest["evaluation_identity_protocol"][
                    "official_evaluator_equivalent"
                ]
            )
            self.assertEqual(manifest["claim_gate"]["status"], "fail")
            self.assertEqual(
                manifest["residual_vs_base_gate"]["status"], "pass_descriptive"
            )
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("produced by 3 source commit(s)", report)
            self.assertIn("exact-target-SMILES single-positive rule", report)
            self.assertIn("not official-evaluator-equivalent results", report)
            with (output / "paired_deltas.csv").open(encoding="utf-8") as handle:
                paired = list(csv.DictReader(handle))
            row = next(
                item
                for item in paired
                if item["alignment_seed"] == "42"
                and item["candidate_type"] == "mass"
                and item["reranker_seed"] == "42"
            )
            self.assertAlmostEqual(float(row["delta_top1_pct"]), 1.0)
            self.assertAlmostEqual(float(row["delta_mrr_raw"]), 0.01)

    def test_missing_summary_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture.read_summary(43)
            fixture.write_summary(fixture.rerank_roots[43], rows[:-1])

            with self.assertRaisesRegex(analysis.AnalysisError, "summary experiment matrix"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_duplicate_summary_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture.read_summary(43)
            fixture.write_summary(fixture.rerank_roots[43], [*rows, rows[0]])

            with self.assertRaisesRegex(analysis.AnalysisError, "Duplicate summary row"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_base_metric_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture.read_summary(43)
            rows[1]["base_top1_pct"] = str(float(rows[1]["base_top1_pct"]) + 1)
            fixture.write_summary(fixture.rerank_roots[43], rows)

            with self.assertRaisesRegex(analysis.AnalysisError, "invariant base_top1_pct"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_alignment_selection_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            selection_path = fixture.specs[1].selection_path
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["seed"] = 99
            selection_path.write_text(json.dumps(selection), encoding="utf-8")

            with self.assertRaisesRegex(analysis.AnalysisError, "selection seed"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_failed_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            batch_path = fixture.rerank_roots[44] / "batch_status.json"
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            batch["state"] = "failed"
            batch_path.write_text(json.dumps(batch), encoding="utf-8")

            with self.assertRaisesRegex(analysis.AnalysisError, "batch state"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_summary_and_aggregate_tampering_is_rejected_by_eval_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture.read_summary(43)
            rows[0]["rerank_top1_pct"] = str(
                float(rows[0]["rerank_top1_pct"]) + 1
            )
            fixture.write_summary(fixture.rerank_roots[43], rows)
            fixture.write_aggregate(fixture.rerank_roots[43], rows)

            with self.assertRaisesRegex(analysis.AnalysisError, "evaluation"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_training_selection_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            status_path = (
                fixture.rerank_roots[43]
                / "mass"
                / "pointwise"
                / "seed42"
                / "attempt_001"
                / "status.json"
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["training_selection"]["best_epoch"] = 99
            status_path.write_text(json.dumps(status), encoding="utf-8")

            with self.assertRaisesRegex(analysis.AnalysisError, "training selection"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_cache_size_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            status_path = (
                fixture.rerank_roots[44]
                / "formula"
                / "transformer"
                / "seed44"
                / "attempt_001"
                / "status.json"
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["cache_files"]["test"]["size_bytes"] += 1
            status["fingerprint"]["cache_files"]["test"]["size_bytes"] += 1
            status_path.write_text(json.dumps(status), encoding="utf-8")

            with self.assertRaisesRegex(analysis.AnalysisError, "cache size"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_nonfinite_summary_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture.read_summary(43)
            rows[0]["rerank_mrr_raw"] = "nan"
            fixture.write_summary(fixture.rerank_roots[43], rows)

            with self.assertRaisesRegex(analysis.AnalysisError, "finite"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            checkpoint = (
                fixture.rerank_roots[43]
                / "mass"
                / "pointwise"
                / "seed42"
                / "attempt_001"
                / "best_reranker.pth"
            )
            checkpoint.unlink()

            with self.assertRaisesRegex(analysis.AnalysisError, "checkpoint"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_canonical_cache_content_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            cache = analysis.canonical_cache_path(
                root, "alignseed42", "massspecgym", "mass", "test"
            )
            original = cache.read_bytes()
            cache.write_bytes(b"x" * len(original))
            stat = cache.stat()
            for status_path in fixture.rerank_roots[42].glob(
                "mass/*/seed*/attempt_001/status.json"
            ):
                status = json.loads(status_path.read_text(encoding="utf-8"))
                for container in (
                    status["cache_files"],
                    status["fingerprint"]["cache_files"],
                ):
                    container["test"]["mtime_ns"] = stat.st_mtime_ns
                status_path.write_text(json.dumps(status), encoding="utf-8")

            with self.assertRaisesRegex(analysis.AnalysisError, "canonical cache hash"):
                analysis.run_analysis(fixture.specs, root / "analysis", root)

    def test_uniform_positive_mrr_directions_require_manual_magnitude_review(self):
        rows = []
        for alignment_seed in analysis.EXPECTED_ALIGNMENT_SEEDS:
            for candidate in analysis.CANDIDATE_TYPES:
                rows.append(
                    {
                        "alignment_seed": alignment_seed,
                        "candidate_type": candidate,
                        "delta_mrr_raw_mean": 0.001,
                    }
                )

        gate = analysis.claim_gate(rows)

        self.assertEqual(gate["status"], "manual_review_magnitude")


if __name__ == "__main__":
    unittest.main()
