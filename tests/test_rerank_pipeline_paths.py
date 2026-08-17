import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_rerank_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


class RerankPipelinePathTest(unittest.TestCase):
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
