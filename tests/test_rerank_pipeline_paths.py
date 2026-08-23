import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_rerank_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


class RerankPipelinePathTest(unittest.TestCase):
    @staticmethod
    def _prepare_args(*, force_include_positive=False):
        return SimpleNamespace(
            dataset_type="massspecgym",
            candidate_type="mass",
            data_path=None,
            device=None,
            candidate_path=None,
            pre_top_k=256,
            limit=None,
            mol_norm_type=None,
            mol_norm_eps=None,
            force_include_positive=force_include_positive,
        )

    def test_prepare_defaults_to_no_forcing_and_keeps_legacy_opt_in(self):
        default_args = self._prepare_args()
        train_command = run_rerank_pipeline.build_prepare_command(
            default_args,
            "/tmp/aligned.pth",
            "train",
            "/tmp/train.pt",
        )
        val_command = run_rerank_pipeline.build_prepare_command(
            default_args,
            "/tmp/aligned.pth",
            "val",
            "/tmp/val.pt",
        )
        self.assertIn("--no-force_include_positive", train_command)
        self.assertIn("--no-force_include_positive", val_command)
        self.assertNotIn("--force_include_positive", train_command)

        legacy_args = self._prepare_args(force_include_positive=True)
        legacy_train_command = run_rerank_pipeline.build_prepare_command(
            legacy_args,
            "/tmp/aligned.pth",
            "train",
            "/tmp/train.pt",
        )
        legacy_val_command = run_rerank_pipeline.build_prepare_command(
            legacy_args,
            "/tmp/aligned.pth",
            "val",
            "/tmp/val.pt",
        )
        self.assertIn("--force_include_positive", legacy_train_command)
        self.assertIn("--no-force_include_positive", legacy_val_command)

    def test_pipeline_help_exposes_relative_model(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run_rerank_pipeline.py"), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("relative", result.stdout)

    def test_git_and_child_processes_use_repository_root(self):
        with mock.patch.object(
            run_rerank_pipeline.subprocess,
            "check_output",
            return_value=b"abc123\n",
        ) as check_output:
            self.assertEqual(run_rerank_pipeline.get_commit_hash(), "abc123")

        check_output.assert_called_once_with(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=run_rerank_pipeline.REPOSITORY_ROOT,
        )

        completed = subprocess.CompletedProcess(["python"], returncode=0)
        with mock.patch.object(
            run_rerank_pipeline.subprocess,
            "run",
            return_value=completed,
        ) as run:
            run_rerank_pipeline.run_command(["python", "example.py"])

        run.assert_called_once_with(
            ["python", "example.py"],
            cwd=run_rerank_pipeline.REPOSITORY_ROOT,
        )

    def test_dry_run_is_independent_of_calling_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.pop("SPECEMBEDDING_CONFIG", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_rerank_pipeline.py"),
                    "--dry-run",
                    "--mode",
                    "all",
                    "--data_path",
                    "data/processed",
                    "--candidate_path",
                    "data/processed/MassSpecGym/candidates_mass.pkl",
                ],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn(str(REPO_ROOT / "prepare_rerank_cache.py"), result.stdout)
        self.assertIn(str(REPO_ROOT / "train_rerank.py"), result.stdout)
        self.assertIn(str(REPO_ROOT / "eval_rerank.py"), result.stdout)
        self.assertIn(str(REPO_ROOT / "data/processed"), result.stdout)


if __name__ == "__main__":
    unittest.main()
