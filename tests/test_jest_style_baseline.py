import tempfile
import unittest
from pathlib import Path

import torch

from analysis.run_jest_style_baseline import evaluate_cache, parse_args


class JestStyleBaselineTest(unittest.TestCase):
    def test_cosine_control_uses_saved_embedding_identity(self):
        payload = {
            "spec_embs": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "mol_embs": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            "mol_smiles": ["CC", "CCC", "CO"],
            "queries": [
                {
                    "spec_index": 0,
                    "true_smiles": "CC",
                    "candidate_indices": torch.tensor([0, 1, 2]),
                    "base_scores": torch.tensor([0.1, 0.2, 0.3]),
                    "base_ranks": torch.tensor([3, 2, 1]),
                    "label": 0,
                    "positive_in_base_topk": True,
                },
                {
                    "spec_index": 1,
                    "true_smiles": "CCC",
                    "candidate_indices": torch.tensor([0, 1]),
                    "base_scores": torch.tensor([0.1, 0.2]),
                    "base_ranks": torch.tensor([2, 1]),
                    "label": 1,
                    "positive_in_base_topk": True,
                },
            ],
            "meta": {"split": "test", "candidate_type": "mass"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache.pt"
            torch.save(payload, cache)
            result = evaluate_cache(cache, torch.device("cpu"), [1, 5], 2)

        self.assertEqual(result["total_queries"], 2)
        self.assertEqual(result["labeled_queries"], 2)
        self.assertEqual(result["top1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)

    def test_parser_rejects_invalid_candidate_cutoff(self):
        with self.assertRaises(SystemExit):
            parse_args(
                ["--cache", "/tmp/cache.pt", "--output", "/tmp/out.json", "--max-candidates", "0"]
            )


if __name__ == "__main__":
    unittest.main()
