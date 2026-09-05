import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from analysis.analyze_nplib1_multiseed import AnalysisError, analyze


class NPLIB1ResultAnalysisTest(unittest.TestCase):
    def materialize(self, root: Path, *, omit=None):
        for alignment_seed in (42, 43):
            prefix = f"nplib_alignseed{alignment_seed}"
            rerank_root = (
                root / "checkpoints_rerank" / f"{prefix}_topk256_multiseed"
            )
            rows = []
            experiments = []
            for model_index, model in enumerate(("pointwise", "relative")):
                for reranker_seed in (42, 43):
                    if omit == (alignment_seed, model, reranker_seed):
                        continue
                    attempt = (
                        rerank_root
                        / "formula"
                        / model
                        / f"seed{reranker_seed}"
                        / "attempt_001"
                    )
                    attempt.mkdir(parents=True)
                    status = {
                        "state": "complete",
                        "candidate_type": "formula",
                        "model_type": model,
                        "ablation": "full",
                        "seed": reranker_seed,
                    }
                    (attempt / "status.json").write_text(json.dumps(status))
                    (attempt / "eval_rerank.log").write_text(
                        "Total queries: 10\nMCES calculation skipped.\n"
                    )
                    gain = alignment_seed / 1000 + model_index + reranker_seed / 100
                    rows.append(
                        {
                            "candidate_type": "formula",
                            "model_type": model,
                            "ablation": "full",
                            "seed": reranker_seed,
                            "best_epoch": 3,
                            "best_val_metric": 0.5,
                            "stop_epoch": 8,
                            "early_stopped": "True",
                            "upper_bound_pct": 80.0,
                            "base_top1_pct": 10.0,
                            "base_top5_pct": 20.0,
                            "base_top10_pct": 30.0,
                            "base_top20_pct": 40.0,
                            "base_mrr_raw": 0.2,
                            "rerank_top1_pct": 10.0 + gain,
                            "rerank_top5_pct": 20.0 + gain,
                            "rerank_top10_pct": 30.0 + gain,
                            "rerank_top20_pct": 40.0 + gain,
                            "rerank_mrr_raw": 0.2 + gain / 100,
                            "attempt_dir": str(attempt.relative_to(root)),
                        }
                    )
                    experiments.append(
                        {
                            "candidate_type": "formula",
                            "model_type": model,
                            "ablation": "full",
                            "seed": reranker_seed,
                        }
                    )
            rerank_root.mkdir(parents=True, exist_ok=True)
            with (rerank_root / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (rerank_root / "batch_status.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "errors": [],
                        "git_commit": "commit",
                        "git_worktree_changes": [],
                        "params_sha256": "params",
                        "experiments": experiments,
                    }
                )
            )

    def args(self, root: Path):
        return SimpleNamespace(
            repo_root=root,
            run_prefix_base="nplib",
            dataset_type="nplib1",
            candidate_type="formula",
            topk=256,
            alignment_seeds=(42, 43),
            reranker_seeds=(42, 43),
            model_types=("pointwise", "relative"),
            expected_git_commit="commit",
        )

    def test_analyze_stratifies_alignment_and_pairs_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.materialize(root)
            report = analyze(self.args(root))

        overall = report["across_alignment"]
        self.assertEqual(
            overall["models"]["pointwise"]["rerank_alignment_means"][
                "rerank_top1_pct"
            ]["n"],
            2,
        )
        relative_delta = overall["relative_minus_pointwise_alignment_means"][
            "top1_pct"
        ]
        self.assertAlmostEqual(relative_delta["mean"], 1.0)
        self.assertEqual(len(report["raw_runs"]), 8)

    def test_analyze_rejects_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.materialize(root, omit=(43, "relative", 43))
            with self.assertRaisesRegex(AnalysisError, "Batch matrix mismatch"):
                analyze(self.args(root))


if __name__ == "__main__":
    unittest.main()
