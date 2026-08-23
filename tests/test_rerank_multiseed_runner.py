import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_rerank_multiseed import (
    ABLATION_OVERRIDES,
    Experiment,
    build_train_command,
    experiment_fingerprint,
)


class RerankMultiseedRunnerTest(unittest.TestCase):
    def test_required_relative_ablation_variants_are_explicit(self):
        self.assertEqual(ABLATION_OVERRIDES["directed_pair"]["pair_mode"], "directed")
        self.assertEqual(
            ABLATION_OVERRIDES["antisymmetric_pair"]["pair_mode"],
            "antisymmetric",
        )
        self.assertFalse(ABLATION_OVERRIDES["no_spectrum_conditioning"]["use_spectrum_conditioning"])
        self.assertEqual(ABLATION_OVERRIDES["no_lambda_pair"]["lambda_pair"], 0.0)
        self.assertEqual(ABLATION_OVERRIDES["no_lambda_spec"]["lambda_spec"], 0.0)

    def test_train_command_records_scope_and_relative_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = {
                split: root / f"{split}.pt" for split in ("train", "val", "test")
            }
            args = SimpleNamespace(
                max_train_queries=20_000,
                exclude_val_query_indices=[1, 2],
                batch_size=32,
            )
            experiment = Experiment("mass", "relative", "directed_pair", 43)
            command = build_train_command(
                root,
                cache,
                root / "attempt",
                experiment,
                "cuda:0",
                40,
                args.max_train_queries,
                args.exclude_val_query_indices,
                args.batch_size,
            )
            command_text = " ".join(command)

        self.assertIn("--max-train-queries 20000", command_text)
        self.assertIn("--batch-size 32", command_text)
        self.assertIn("--pair-mode directed", command_text)
        self.assertIn("--lambda-spec 0.1", command_text)
        self.assertIn("--no-rank-embedding", command_text)
        self.assertIn("--exclude-val-query-indices 1 2", command_text)

    def test_fingerprint_separates_training_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = {}
            for split in ("train", "val", "test"):
                path = root / f"{split}.pt"
                path.write_bytes(b"cache")
                cache[split] = path
            experiment = Experiment("formula", "relative", "full", 42)
            args = SimpleNamespace(
                git_commit="commit",
                params_sha256="params",
                dataset_type="massspecgym",
                run_prefix="run",
                topk=40,
                max_train_queries=20_000,
                batch_size=32,
                exclude_val_query_indices=[],
                cache_root=root,
            )
            first = experiment_fingerprint(args, experiment, cache)
            args.max_train_queries = 50_000
            second = experiment_fingerprint(args, experiment, cache)

        self.assertNotEqual(first["max_train_queries"], second["max_train_queries"])


if __name__ == "__main__":
    unittest.main()
