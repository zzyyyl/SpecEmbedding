import tempfile
import unittest
from pathlib import Path

import torch

from analysis.benchmark_reranker_efficiency import benchmark_checkpoint, parse_checkpoint_spec
from analysis.benchmark_reranker_efficiency import parse_args as parse_benchmark_args
from analysis.export_rerank_cases import export_cases, ranked_candidates
from SpecEmbedding.models_rerank import RelativeCandidateReranker


class RerankToolsTest(unittest.TestCase):
    def test_checkpoint_spec_requires_name_and_path(self):
        self.assertEqual(
            parse_checkpoint_spec("relative=/tmp/model.pth"),
            ("relative", Path("/tmp/model.pth")),
        )
        with self.assertRaisesRegex(ValueError, "NAME=PATH"):
            parse_checkpoint_spec("/tmp/model.pth")

    def test_efficiency_parser_preserves_measurement_protocol(self):
        args = parse_benchmark_args(
            [
                "--cache",
                "/tmp/cache.pt",
                "--checkpoint",
                "relative=/tmp/model.pth",
                "--candidate-counts",
                "40",
                "80",
                "--warmup-batches",
                "2",
                "--measure-batches",
                "4",
                "--output",
                "/tmp/efficiency.json",
            ]
        )
        self.assertEqual(args.candidate_counts, [40, 80])
        self.assertEqual(args.warmup_batches, 2)
        self.assertEqual(args.measure_batches, 4)

    def test_ranked_candidates_exports_identity_and_scores(self):
        result = ranked_candidates(
            torch.tensor([1, 0]),
            torch.tensor([0.1, 0.8]),
            torch.tensor([0.2, 0.7]),
            torch.tensor([2, 1]),
            torch.tensor([11, 12]),
            ["CCC", "CCO"],
            2,
        )
        self.assertEqual([item["candidate_index"] for item in result], [12, 11])
        self.assertEqual(result[0]["smiles"], "CCO")
        self.assertEqual(result[0]["base_rank"], 1)
        self.assertAlmostEqual(result[0]["score"], 0.8)

    def test_tools_run_on_synthetic_cache_and_checkpoint(self):
        torch.manual_seed(3)
        model = RelativeCandidateReranker(
            embedding_dim=4,
            hidden_dim=8,
            relation_dim=4,
            pair_chunk_size=2,
            relation_top_k=2,
            dropout=0.0,
        ).eval()
        model_config = {
            "model_type": "relative",
            "embedding_dim": 4,
            "hidden_dim": 8,
            "rank_emb_dim": 2,
            "max_rank": 8,
            "n_layers": 0,
            "n_heads": 1,
            "dropout": 0.0,
            "alpha_init": 1.0,
            "relation_dim": 4,
            "pair_chunk_size": 2,
            "relation_top_k": 2,
            "pair_mode": "antisymmetric",
        }
        payload = {
            "spec_embs": torch.randn(1, 4),
            "mol_embs": torch.randn(3, 4),
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
            "meta": {
                "split": "test",
                "candidate_type": "mass",
                "pre_top_k": 3,
                "force_include_positive": False,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache.pt"
            checkpoint = Path(temporary) / "model.pth"
            torch.save(payload, cache)
            torch.save(
                {"state_dict": model.state_dict(), "model_config": model_config},
                checkpoint,
            )

            benchmark = benchmark_checkpoint(
                cache,
                checkpoint,
                [2, 3],
                torch.device("cpu"),
                batch_size=1,
                warmup_batches=0,
                measure_batches=1,
            )
            cases = export_cases(
                cache,
                checkpoint,
                [0],
                torch.device("cpu"),
                top_k=2,
            )

        self.assertEqual([item["candidate_count"] for item in benchmark], [2, 3])
        self.assertEqual(cases["cases"][0]["query_index"], 0)
        self.assertEqual(len(cases["cases"][0]["rerank_topk"]), 2)


if __name__ == "__main__":
    unittest.main()
