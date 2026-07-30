import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_alignment_multiseed


class AlignmentMultiseedRunnerTest(unittest.TestCase):
    def base_cli(self, root: Path) -> list[str]:
        data_path = root / "processed"
        data_path.mkdir()
        tokenset_cache = root / "tokenset.pkl"
        tokenset_cache.write_bytes(b"tokens")
        pipeline_script = root / "pipeline.sh"
        pipeline_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s|%s|%s\\n" "$ALIGNMENT_SEED" "$RUN_PREFIX" "${1:-}" '
            f'>> "{root / "calls.txt"}"\n',
            encoding="utf-8",
        )
        pipeline_script.chmod(0o755)
        return [
            "--run-prefix-base",
            "transfer2026_massspecgym",
            "--alignment-seeds",
            "43",
            "44",
            "--alignment-device",
            "cuda:1",
            "--rerank-devices",
            "cuda:0",
            "cuda:1",
            "--data-path",
            str(data_path),
            "--tokenset-cache",
            str(tokenset_cache),
            "--pipeline-script",
            str(pipeline_script),
            "--status-root",
            str(root / "status"),
        ]

    def test_parse_args_rejects_duplicate_alignment_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            seed_start = cli.index("--alignment-seeds") + 1
            cli[seed_start : seed_start + 2] = ["43", "43"]
            with self.assertRaises(SystemExit):
                run_alignment_multiseed.parse_args(cli)

    def test_parse_args_requires_explicit_cuda_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            cli[cli.index("cuda:1")] = "cuda"
            with self.assertRaises(SystemExit):
                run_alignment_multiseed.parse_args(cli)

    def test_build_seed_run_isolates_alignment_outputs(self):
        repo_root = Path("/repo")
        args = SimpleNamespace(
            run_prefix_base="transfer2026",
            pipeline_script=repo_root / "pipeline.sh",
            dry_run=True,
            conda_env="specembedding",
            data_path=Path("/data/processed"),
            tokenset_cache=Path("/data/tokenset.pkl"),
            alignment_device="cuda:1",
            rerank_devices=["cuda:0", "cuda:1"],
            min_free_mib=16_000,
            max_utilization=20,
        )
        run = run_alignment_multiseed.build_seed_run(args, repo_root, 43)

        self.assertEqual(run["alignment_seed"], 43)
        self.assertEqual(run["run_prefix"], "transfer2026_alignseed43")
        self.assertEqual(run["environment"]["ALIGNMENT_SEED"], "43")
        self.assertEqual(run["environment"]["RERANK_DEVICE_LIST"], "cuda:0 cuda:1")
        self.assertEqual(
            run["alignment_dir"],
            "/repo/checkpoints_align/transfer2026_alignseed43",
        )
        self.assertTrue(run["rerank_output"].endswith("_topk40_multiseed"))
        self.assertEqual(run["command"][-1], "--dry-run")

    def test_dry_run_executes_each_inner_pipeline_without_outer_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = [*self.base_cli(root), "--dry-run"]

            returncode = run_alignment_multiseed.main(cli)

            self.assertEqual(returncode, 0)
            self.assertFalse((root / "status").exists())
            calls = (root / "calls.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                calls,
                [
                    "43|transfer2026_massspecgym_alignseed43|--dry-run",
                    "44|transfer2026_massspecgym_alignseed44|--dry-run",
                ],
            )

    def test_successful_real_run_writes_complete_outer_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            with (
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_commit",
                    return_value="deadbeef",
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_worktree_changes",
                    return_value=[],
                ),
            ):
                returncode = run_alignment_multiseed.main(cli)

            self.assertEqual(returncode, 0)
            status = json.loads(
                (root / "status" / "batch_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["git_commit"], "deadbeef")
            self.assertEqual(
                status["pipeline_script_sha256"],
                hashlib.sha256(
                    (root / "pipeline.sh").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(status["alignment_seeds"], [43, 44])
            self.assertEqual([run["state"] for run in status["runs"]], ["complete", "complete"])
            self.assertEqual([run["alignment_seed"] for run in status["runs"]], [43, 44])
            self.assertEqual(status["errors"], [])

    def test_keyboard_interrupt_is_persisted_and_remaining_run_is_not_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            with (
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_commit",
                    return_value="deadbeef",
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_worktree_changes",
                    return_value=[],
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "execute_seed_run",
                    side_effect=KeyboardInterrupt,
                ),
            ):
                returncode = run_alignment_multiseed.main(cli)

            self.assertEqual(returncode, 130)
            status = json.loads(
                (root / "status" / "batch_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(
                [run["state"] for run in status["runs"]],
                ["interrupted", "not_started"],
            )
            self.assertEqual(status["runs"][0]["returncode"], 130)
            self.assertEqual(status["errors"][0]["kind"], "keyboard_interrupt")

    def test_runner_exception_is_persisted_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            with (
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_commit",
                    return_value="deadbeef",
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_worktree_changes",
                    return_value=[],
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "execute_seed_run",
                    side_effect=OSError("cannot launch child"),
                ),
            ):
                returncode = run_alignment_multiseed.main(cli)

            self.assertEqual(returncode, 1)
            status = json.loads(
                (root / "status" / "batch_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "failed")
            self.assertEqual(
                [run["state"] for run in status["runs"]],
                ["failed", "not_started"],
            )
            self.assertEqual(status["errors"][0]["kind"], "runner_exception")
            self.assertEqual(status["errors"][0]["exception_type"], "OSError")
            self.assertEqual(status["errors"][0]["message"], "cannot launch child")

    def test_nonzero_child_exit_is_persisted_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            with (
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_commit",
                    return_value="deadbeef",
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_worktree_changes",
                    return_value=[],
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "execute_seed_run",
                    return_value=7,
                ),
            ):
                returncode = run_alignment_multiseed.main(cli)

            self.assertEqual(returncode, 7)
            status = json.loads(
                (root / "status" / "batch_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "failed")
            self.assertEqual(
                [run["state"] for run in status["runs"]],
                ["failed", "not_started"],
            )
            self.assertEqual(status["runs"][0]["returncode"], 7)
            self.assertEqual(status["errors"][0]["returncode"], 7)

    def test_dirty_worktree_is_rejected_before_status_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = self.base_cli(root)
            with (
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_commit",
                    return_value="deadbeef",
                ),
                mock.patch.object(
                    run_alignment_multiseed,
                    "git_worktree_changes",
                    return_value=[" M train_align.py"],
                ),
                self.assertRaisesRegex(RuntimeError, "dirty worktree"),
            ):
                run_alignment_multiseed.main(cli)

            self.assertFalse((root / "status").exists())


if __name__ == "__main__":
    unittest.main()
