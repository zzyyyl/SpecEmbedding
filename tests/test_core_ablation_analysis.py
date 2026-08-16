import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import analyze_alignment_multiseed as common
import analyze_core_ablations as analysis


class CoreAblationAnalysisTest(unittest.TestCase):
    def make_runs(self):
        full_rows = {}
        ablation_rows = {}
        delta_patterns = {
            ("mass", "no_residual"): [1.0, 2.0, 3.0],
            ("formula", "no_residual"): [-1.0, -2.0, -3.0],
            ("mass", "no_base_score"): [1.0, -0.5, 1.0],
            ("formula", "no_base_score"): [1.0, -0.2, 0.5],
            ("mass", "no_rank_embedding"): [-1.0, -2.0, -1.5],
            ("formula", "no_rank_embedding"): [-2.0, -1.0, -2.5],
            ("mass", "no_interaction_features"): [1.0, 0.5, 0.2],
            ("formula", "no_interaction_features"): [-1.0, -0.5, -0.2],
            ("mass", "listwise_only"): [1.0, 0.5, 0.2],
            ("formula", "listwise_only"): [0.4, 0.3, 0.2],
        }
        for candidate_index, candidate in enumerate(analysis.CANDIDATE_TYPES):
            for seed_index, seed in enumerate(analysis.RERANKER_SEEDS):
                full = {
                    "candidate_type": candidate,
                    "model_type": "transformer",
                    "ablation": "full",
                    "seed": seed,
                }
                for metric in common.RERANK_METRICS:
                    if metric == "rerank_mrr_raw":
                        full[metric] = 0.7 + candidate_index / 10 + seed_index / 100
                    else:
                        full[metric] = 70.0 + candidate_index + seed_index
                full_rows[(candidate, "transformer", seed)] = full
                for ablation in analysis.ABLATIONS:
                    delta = delta_patterns[(candidate, ablation)][seed_index]
                    row = {
                        "candidate_type": candidate,
                        "model_type": "transformer",
                        "ablation": ablation,
                        "seed": seed,
                    }
                    for metric in common.RERANK_METRICS:
                        scaled = delta / 100 if metric == "rerank_mrr_raw" else delta
                        row[metric] = full[metric] - scaled
                    ablation_rows[(candidate, ablation, seed)] = row
        full = SimpleNamespace(rows=full_rows)
        ablations = SimpleNamespace(rows=ablation_rows)
        return full, ablations

    def test_expected_matrix_and_frozen_override_semantics(self):
        self.assertEqual(len(analysis.expected_ablation_keys()), 30)
        self.assertFalse(
            analysis.EXPECTED_OVERRIDES["no_residual"]["use_residual_score"]
        )
        self.assertTrue(
            analysis.EXPECTED_OVERRIDES["no_residual"]["use_base_score_feature"]
        )
        self.assertFalse(
            analysis.EXPECTED_OVERRIDES["no_base_score"]["use_residual_score"]
        )
        self.assertFalse(
            analysis.EXPECTED_OVERRIDES["no_base_score"]["use_base_score_feature"]
        )
        self.assertFalse(
            analysis.EXPECTED_OVERRIDES["no_interaction_features"][
                "use_product_feature"
            ]
        )
        self.assertFalse(
            analysis.EXPECTED_OVERRIDES["no_interaction_features"][
                "use_abs_diff_feature"
            ]
        )
        self.assertEqual(
            analysis.EXPECTED_OVERRIDES["listwise_only"]["lambda_pair"], 0.0
        )

    def test_paired_delta_summary_and_sample_standard_deviation(self):
        full, ablations = self.make_runs()
        paired = analysis.build_paired_rows(full, ablations)
        summary = analysis.build_summary_rows(paired)

        self.assertEqual(len(paired), 30)
        self.assertEqual(len(summary), 10)
        row = next(
            item
            for item in summary
            if item["candidate_type"] == "mass"
            and item["ablation"] == "no_residual"
        )
        self.assertAlmostEqual(row["delta_top1_pct_mean"], 2.0)
        self.assertAlmostEqual(row["delta_top1_pct_std"], 1.0)
        self.assertAlmostEqual(row["delta_mrr_raw_mean"], 0.02)
        self.assertAlmostEqual(row["delta_mrr_raw_std"], 0.01)
        self.assertEqual(row["delta_mrr_raw_positive"], 3)
        self.assertEqual(row["delta_mrr_raw_zero"], 0)
        self.assertEqual(row["delta_mrr_raw_negative"], 0)

    def test_component_gate_distinguishes_supported_mixed_and_harmful(self):
        full, ablations = self.make_runs()
        summary = analysis.build_summary_rows(
            analysis.build_paired_rows(full, ablations)
        )
        gate = analysis.build_component_gate(summary)
        decisions = {item["ablation"]: item for item in gate["decisions"]}

        self.assertEqual(gate["passed_components"], 1)
        self.assertEqual(
            decisions["listwise_only"]["status"], "pass_descriptive"
        )
        self.assertEqual(
            decisions["no_base_score"]["status"],
            "mean_favors_full_seed_mixed",
        )
        self.assertEqual(
            decisions["no_residual"]["status"], "candidate_dependent"
        )
        self.assertEqual(
            decisions["no_rank_embedding"]["status"],
            "removal_consistently_better",
        )

    def test_report_preserves_component_interpretation_boundaries(self):
        full, ablations = self.make_runs()
        full.batch = {"git_commit": "full-commit", "params_sha256": "params"}
        ablations.batch = {
            "git_commit": "ablation-commit",
            "params_sha256": "params",
            "results": [
                *[{"state": "skipped"} for _ in range(29)],
                {"state": "complete"},
            ],
        }
        summary = analysis.build_summary_rows(
            analysis.build_paired_rows(full, ablations)
        )
        gate = analysis.build_component_gate(summary)

        report = analysis.build_report(full, ablations, summary, gate, [])

        self.assertIn("Delta is defined as `full - ablation`", report)
        self.assertIn("candidate self-attention remains enabled", report)
        self.assertIn("not a single-factor feature ablation", report)
        self.assertIn("not justify selecting a new post-hoc main model", report)
        self.assertIn("exact-target-SMILES single-positive rule", report)
        self.assertIn("not official-evaluator-equivalent results", report)

    def test_retry_status_accepts_historical_failures_but_requires_success(self):
        history = [
            *[{"state": "complete"} for _ in range(30)],
            *[{"state": "failed"} for _ in range(8)],
        ]
        batch = {"git_commit": "commit", "params_sha256": "params"}
        retry = {
            "state": "complete",
            "child_pid": None,
            "progress": {
                "total_experiments": 30,
                "experiments_seen": 30,
                "not_started": 0,
                "latest_experiment_states": {"complete": 30},
                "all_attempt_states": {"complete": 30, "failed": 8},
                "attempt_status_files": 38,
            },
            "last_invocation": {"return_code": 0},
            "provenance": {"git_commit": "commit", "params_sha256": "params"},
        }

        analysis.validate_retry_status(retry, batch, history)
        retry["last_invocation"]["return_code"] = 1
        with self.assertRaisesRegex(common.AnalysisError, "return code"):
            analysis.validate_retry_status(retry, batch, history)

    def test_attempt_resolution_rejects_unexpected_or_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            root = repo_root / "ablations"
            attempt = (
                root
                / "mass"
                / "transformer"
                / "no_residual"
                / "seed42"
                / "attempt_001"
            )
            attempt.mkdir(parents=True)
            key = ("mass", "no_residual", 42)

            resolved = analysis.resolve_ablation_attempt(
                attempt.relative_to(repo_root).as_posix(), root, key, repo_root
            )
            self.assertEqual(resolved, attempt.resolve())
            with self.assertRaisesRegex(common.AnalysisError, "Unexpected attempt"):
                analysis.resolve_ablation_attempt(
                    "ablations/mass/transformer/no_residual/seed42/latest",
                    root,
                    key,
                    repo_root,
                )
            with self.assertRaisesRegex(common.AnalysisError, "mismatch"):
                analysis.resolve_ablation_attempt(
                    "../ablations/mass/transformer/no_residual/seed42/attempt_001",
                    root,
                    key,
                    repo_root,
                )

    @mock.patch("analyze_core_ablations.subprocess.run")
    def test_runtime_diff_is_parsed_and_git_failures_are_rejected(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="train_rerank.py\nparams.yaml\n", stderr=""
        )
        changed = analysis.runtime_changed_paths(Path("."), "full", "ablation")
        self.assertEqual(changed, ["train_rerank.py", "params.yaml"])
        self.assertIn("--name-only", run.call_args.args[0])

        run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="bad revision"
        )
        with self.assertRaisesRegex(common.AnalysisError, "bad revision"):
            analysis.runtime_changed_paths(Path("."), "full", "ablation")

    def test_output_root_must_be_an_isolated_analysis_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            full_root = repo_root / "checkpoints_rerank" / "full"
            ablation_root = repo_root / "checkpoints_rerank" / "ablations"
            accepted = repo_root / "analysis" / "core_ablations"

            self.assertEqual(
                analysis.validate_output_root(
                    accepted, repo_root, (full_root, ablation_root)
                ),
                accepted.resolve(),
            )
            with self.assertRaisesRegex(common.AnalysisError, "inside"):
                analysis.validate_output_root(
                    repo_root / "elsewhere",
                    repo_root,
                    (full_root, ablation_root),
                )
            with self.assertRaisesRegex(common.AnalysisError, "subdirectory"):
                analysis.validate_output_root(
                    repo_root / "analysis",
                    repo_root,
                    (full_root, ablation_root),
                )

    def test_nonfinite_deltas_do_not_silently_pass(self):
        full, ablations = self.make_runs()
        ablations.rows[("mass", "no_residual", 42)]["rerank_mrr_raw"] = math.nan
        with self.assertRaisesRegex(common.AnalysisError, "finite numeric"):
            analysis.build_paired_rows(full, ablations)


if __name__ == "__main__":
    unittest.main()
