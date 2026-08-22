import tempfile
import unittest
from pathlib import Path

import torch

from SpecEmbedding.data.datasets_rerank import (
    RerankCacheDataset,
    exclude_query_indices,
    rerank_collate_fn,
)
from SpecEmbedding.models_rerank import CandidateReranker, RelativeCandidateReranker
from SpecEmbedding.utils.rerank import load_reranker, spectrum_dependency_loss


class RerankerCoreTest(unittest.TestCase):
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

    def test_feature_ablations_keep_parameter_count(self):
        full_count = sum(
            parameter.numel()
            for parameter in CandidateReranker(**self.model_kwargs).parameters()
        )
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
                count = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(count, full_count)

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
                model = CandidateReranker(
                    **{**self.model_kwargs, "n_layers": n_layers}
                ).eval()
                original = model(**self.inputs)
                permuted_inputs = dict(self.inputs)
                for key in (
                    "candidate_embs",
                    "base_scores",
                    "base_ranks",
                    "candidate_mask",
                ):
                    permuted_inputs[key] = self.inputs[key][:, permutation]
                permuted = model(**permuted_inputs)
                torch.testing.assert_close(
                    permuted,
                    original[:, permutation],
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_old_checkpoint_defaults_to_full_features(self):
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
            torch.save(
                {"state_dict": original.state_dict(), "model_config": old_config},
                checkpoint,
            )
            loaded = load_reranker(checkpoint, torch.device("cpu"))

        self.assertTrue(loaded.use_base_score_feature)
        self.assertTrue(loaded.use_residual_score)
        self.assertTrue(loaded.use_rank_embedding)
        self.assertTrue(loaded.use_product_feature)
        self.assertTrue(loaded.use_abs_diff_feature)
        torch.testing.assert_close(original(**self.inputs), loaded(**self.inputs))

    def test_cache_dataset_collation_and_query_exclusion(self):
        payload = {
            "spec_embs": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            "mol_embs": torch.eye(4),
            "mol_smiles": ["CC", "CCC", "CO", "CN"],
            "queries": [
                {
                    "spec_index": 0,
                    "true_smiles": "CC",
                    "candidate_indices": torch.tensor([0, 1]),
                    "base_scores": torch.tensor([0.9, 0.1]),
                    "base_ranks": torch.tensor([1, 2]),
                    "label": 0,
                    "positive_in_base_topk": True,
                },
                {
                    "spec_index": 1,
                    "true_smiles": "CN",
                    "candidate_indices": torch.tensor([1, 2, 3]),
                    "base_scores": torch.tensor([0.8, 0.5, 0.2]),
                    "base_ranks": torch.tensor([1, 2, 3]),
                    "label": None,
                    "positive_in_base_topk": False,
                },
            ],
            "meta": {"split": "synthetic"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache.pt"
            torch.save(payload, cache)
            dataset = RerankCacheDataset(cache, require_label=False)
            labeled = RerankCacheDataset(cache, require_label=True)
            batch = rerank_collate_fn([dataset[0], dataset[1]])

            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(labeled), 1)
            self.assertEqual(tuple(batch["candidate_embs"].shape), (2, 3, 4))
            self.assertEqual(
                batch["candidate_mask"].tolist(),
                [[True, True, False], [True, True, True]],
            )
            self.assertEqual(batch["labels"].tolist(), [0, -1])
            self.assertEqual(exclude_query_indices(dataset, [1]), 1)
            self.assertEqual(len(dataset), 1)

    def test_dataset_truncation_does_not_force_positive_by_default(self):
        payload = {
            "spec_embs": torch.eye(2),
            "mol_embs": torch.eye(3),
            "mol_smiles": ["CC", "CCC", "CO"],
            "queries": [
                {
                    "spec_index": 0,
                    "true_smiles": "CO",
                    "candidate_indices": torch.tensor([0, 1, 2]),
                    "base_scores": torch.tensor([0.9, 0.8, 0.1]),
                    "base_ranks": torch.tensor([1, 2, 3]),
                    "label": 2,
                    "positive_in_base_topk": True,
                }
            ],
            "meta": {"split": "train"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache.pt"
            torch.save(payload, cache)
            dataset = RerankCacheDataset(cache, max_candidates=2, require_label=False)
            sample = dataset[0]
            self.assertEqual(sample["label"], -1)
            self.assertEqual(sample["candidate_indices"].tolist(), [0, 1])

    def test_relative_reranker_is_permutation_equivariant_and_antisymmetric(self):
        model = RelativeCandidateReranker(
            embedding_dim=4,
            hidden_dim=8,
            relation_dim=4,
            pair_chunk_size=2,
            dropout=0.0,
        ).eval()
        original = model(**self.inputs)
        permutation = torch.tensor([2, 0, 1])
        permuted_inputs = dict(self.inputs)
        for key in ("candidate_embs", "base_scores", "base_ranks", "candidate_mask"):
            permuted_inputs[key] = self.inputs[key][:, permutation]
        permuted = model(**permuted_inputs)
        torch.testing.assert_close(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)

        spec_rel = model.spec_relation(self.inputs["spec_emb"][:1])
        mol_rel = model.mol_relation(self.inputs["candidate_embs"][:1])
        mol_i = mol_rel[:, :1].unsqueeze(2)
        mol_j = mol_rel[:, 1:2].unsqueeze(1)
        base_i = self.inputs["base_scores"][:1, :1].unsqueeze(2)
        base_j = self.inputs["base_scores"][:1, 1:2].unsqueeze(1)
        spec_pair = spec_rel[:, None, None, :]
        forward = model.preference_mlp(
            model._relation_features(spec_pair, mol_i, mol_j, base_i, base_j)
        )
        reverse = model.preference_mlp(
            model._relation_features(spec_pair, mol_j, mol_i, base_j, base_i)
        )
        torch.testing.assert_close(forward - reverse, -(reverse - forward))

    def test_relative_model_uses_spectrum_dependency_loss(self):
        model = RelativeCandidateReranker(
            embedding_dim=4,
            hidden_dim=8,
            relation_dim=4,
            pair_chunk_size=2,
            dropout=0.0,
        )
        labels = torch.tensor([0, 1])
        loss = spectrum_dependency_loss(
            model,
            self.inputs["spec_emb"],
            self.inputs["candidate_embs"],
            self.inputs["base_scores"],
            self.inputs["base_ranks"],
            self.inputs["candidate_mask"],
            labels,
            margin=0.1,
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
