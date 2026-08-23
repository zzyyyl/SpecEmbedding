import csv
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_mentor_experiments import aggregate, read_rows


class MentorExperimentSummaryTest(unittest.TestCase):
    def test_aggregate_preserves_alignment_and_pairs_relative(self):
        rows = [
            {
                "alignment": "align42",
                "candidate_type": "mass",
                "model_type": "pointwise",
                "ablation": "full",
                "seed": 42,
                "rerank_top1_pct": 10.0,
                "rerank_mrr_raw": 0.2,
            },
            {
                "alignment": "align42",
                "candidate_type": "mass",
                "model_type": "relative",
                "ablation": "full",
                "seed": 42,
                "rerank_top1_pct": 11.0,
                "rerank_mrr_raw": 0.21,
            },
        ]
        summary, paired = aggregate(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(paired[0]["delta_relative_minus_pointwise_top1_pct"], 1.0)
        self.assertAlmostEqual(paired[0]["delta_relative_minus_pointwise_mrr_raw"], 0.01)

    def test_read_rows_parses_csv_numbers_and_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["candidate_type", "model_type", "ablation", "seed", "rerank_mrr_raw"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_type": "formula",
                        "model_type": "relative",
                        "ablation": "full",
                        "seed": "43",
                        "rerank_mrr_raw": "0.3",
                    }
                )
            rows = read_rows([f"align43={path}"])

        self.assertEqual(rows[0]["alignment"], "align43")
        self.assertEqual(rows[0]["seed"], 43)
        self.assertAlmostEqual(rows[0]["rerank_mrr_raw"], 0.3)


if __name__ == "__main__":
    unittest.main()
