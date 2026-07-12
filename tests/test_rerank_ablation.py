import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from eval_rerank import exclude_query_indices
from run_rerank_multiseed import (
    ABLATION_OVERRIDES,
    Experiment,
    build_train_command,
    cache_paths,
    experiment_dir,
    experiment_fingerprint,
    select_attempt,
    write_summaries,
)
from SpecEmbedding.models_rerank import CandidateReranker
from SpecEmbedding.utils.rerank import load_reranker


class RerankerAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model_kwargs = {
            "embedding_dim": 4,
            "hidden_dim": 8,
            "rank_emb_dim": 2,
            "max_rank": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "alpha_init": 1.0,
        }
        self.inputs = {
            "spec_emb": torch.randn(2, 4),
            "candidate_embs": torch.randn(2, 3, 4),
            "base_scores": torch.randn(2, 3),
            "base_ranks": torch.tensor([[1, 2, 3], [2, 1, 3]]),
            "candidate_mask": torch.ones(2, 3, dtype=torch.bool),
        }

    def _runner_args(self, root: Path):
        return SimpleNamespace(
            cache_root=root / "cache",
            output_root=root / "output",
            run_prefix="run",
            dataset_type="massspecgym",
            topk=40,
            git_commit="commit",
            params_sha256="params",
            mces=False,
            rerun_completed=False,
        )

    def _materialize_attempt(self, args, experiment, *, state="complete", with_eval=True):
        caches = cache_paths(args, experiment)
        for cache in caches.values():
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(b"cache")

        attempt = experiment_dir(args, experiment) / "attempt_001"
        attempt.mkdir(parents=True, exist_ok=True)
        overrides = ABLATION_OVERRIDES[experiment.ablation]
        model_config = {
            "model_type": experiment.model_type,
            **{key: value for key, value in overrides.items() if key.startswith("use_")},
        }
        training_config = {
            key: value for key, value in overrides.items() if not key.startswith("use_")
        }
        checkpoint = {
            "seed": experiment.seed,
            "model_config": model_config,
            "training_config": training_config,
        }
        torch.save(checkpoint, attempt / "best_reranker.pth")
        torch.save(checkpoint, attempt / "last_reranker.pth")
        (attempt / "train_rerank.log").write_text("Training finished.\n", encoding="utf-8")
        if with_eval:
            (attempt / "eval_rerank.log").write_text(
                "\n".join(
                    [
                        "Total queries: 10",
                        "Pre-retrieval upper bound: 80.0000%",
                        "BASE RESULTS",
                        "Top-1 Accuracy : 50.0000%",
                        "MRR : 0.6000",
                        "RERANK RESULTS",
                        "Top-1 Accuracy : 70.0000%",
                        "MRR : 0.7500",
                        "MCES calculation skipped.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        status = {
            "state": state,
            "candidate_type": experiment.candidate_type,
            "model_type": experiment.model_type,
            "ablation": experiment.ablation,
            "seed": experiment.seed,
            "fingerprint": experiment_fingerprint(args, experiment, caches),
        }
        (attempt / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return attempt

    def test_feature_ablations_keep_parameter_count(self):
        full_count = sum(parameter.numel() for parameter in CandidateReranker(**self.model_kwargs).parameters())
        variants = [
            {"use_base_score_feature": False},
            {"use_residual_score": False},
            {"use_rank_embedding": False},
            {"use_product_feature": False},
            {"use_abs_diff_feature": False},
            {"use_product_feature": False, "use_abs_diff_feature": False},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                model = CandidateReranker(**self.model_kwargs, **variant)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), full_count)

    def test_no_base_score_ignores_base_score_values(self):
        model = CandidateReranker(
            **self.model_kwargs,
            use_base_score_feature=False,
            use_residual_score=False,
        ).eval()
        first = model(**self.inputs)
        changed = dict(self.inputs)
        changed["base_scores"] = self.inputs["base_scores"] + 1000.0
        second = model(**changed)
        torch.testing.assert_close(first, second)

    def test_candidate_permutation_is_equivariant(self):
        permutation = torch.tensor([2, 0, 1])
        for n_layers in (0, 1):
            with self.subTest(n_layers=n_layers):
                model = CandidateReranker(**{**self.model_kwargs, "n_layers": n_layers}).eval()
                original = model(**self.inputs)
                permuted_inputs = dict(self.inputs)
                for key in ("candidate_embs", "base_scores", "base_ranks", "candidate_mask"):
                    permuted_inputs[key] = self.inputs[key][:, permutation]
                permuted = model(**permuted_inputs)
                torch.testing.assert_close(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)

    def test_old_checkpoint_config_defaults_to_full_features(self):
        old_config = {
            "model_type": "transformer",
            "embedding_dim": 4,
            "hidden_dim": 8,
            "rank_emb_dim": 2,
            "max_rank": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "alpha_init": 1.0,
        }
        original = CandidateReranker(**self.model_kwargs).eval()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "old_checkpoint.pth"
            torch.save({"state_dict": original.state_dict(), "model_config": old_config}, checkpoint)
            loaded = load_reranker(checkpoint, torch.device("cpu"))
        self.assertTrue(loaded.use_base_score_feature)
        self.assertTrue(loaded.use_residual_score)
        self.assertTrue(loaded.use_rank_embedding)
        self.assertTrue(loaded.use_product_feature)
        self.assertTrue(loaded.use_abs_diff_feature)
        torch.testing.assert_close(original(**self.inputs), loaded(**self.inputs))

    def test_ablation_commands_are_explicit(self):
        caches = {"train": Path("train.pt"), "val": Path("val.pt")}
        experiment = Experiment("mass", "pointwise", "no_base_score", 42)
        command = build_train_command(
            Path("/repo"),
            caches,
            Path("attempt"),
            experiment,
            "cuda:1",
            40,
        )
        self.assertIn("--no-base-score-feature", command)
        self.assertIn("--no-residual-score", command)
        self.assertEqual(command[command.index("--train-k") + 1], "40")

        ce_command = build_train_command(
            Path("/repo"),
            caches,
            Path("attempt"),
            Experiment("formula", "pointwise", "listwise_only", 44),
            "cuda:1",
            40,
        )
        self.assertEqual(ce_command[ce_command.index("--lambda-pair") + 1], "0.0")

    def test_failed_evaluation_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._runner_args(Path(temporary))
            experiment = Experiment("mass", "pointwise", "no_residual", 42)
            attempt = self._materialize_attempt(
                args,
                experiment,
                state="failed",
                with_eval=False,
            )
            action, selected = select_attempt(args, experiment)
        self.assertEqual(action, "eval")
        self.assertEqual(selected, attempt)

    def test_summary_discovers_completed_subset_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._runner_args(Path(temporary))
            first = Experiment("mass", "pointwise", "no_residual", 42)
            second = Experiment("mass", "pointwise", "listwise_only", 42)
            self._materialize_attempt(args, first)
            self._materialize_attempt(args, second)
            args.mces = True
            write_summaries(args, [second])

            with (args.output_root / "summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (args.output_root / "summary_aggregate.csv").open(encoding="utf-8") as handle:
                aggregate_rows = list(csv.DictReader(handle))
        self.assertEqual({row["ablation"] for row in rows}, {"no_residual", "listwise_only"})
        self.assertEqual(
            {row["ablation"] for row in aggregate_rows},
            {"no_residual", "listwise_only"},
        )

    def test_query_exclusion_uses_cache_indices(self):
        dataset = SimpleNamespace(queries=[{} for _ in range(5)], indices=list(range(5)))
        removed = exclude_query_indices(dataset, [1, 3, 3])
        self.assertEqual(removed, 2)
        self.assertEqual(dataset.indices, [0, 2, 4])
        with self.assertRaises(ValueError):
            exclude_query_indices(dataset, [5])


if __name__ == "__main__":
    unittest.main()
