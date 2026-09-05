import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_pipeline
import train_align
from SpecEmbedding.config import config


class _FakeAlignModel(torch.nn.Module):
    def __init__(self, spec_encoder, mol_encoder):
        super().__init__()
        self.spec_encoder = spec_encoder
        self.mol_encoder = mol_encoder
        self.spec_proj = torch.nn.Linear(1, 1)
        self.mol_proj = torch.nn.Linear(1, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(()))


class _FakeTrainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.stage_summaries = {}
        self.save_dir = Path(save_dir)

    def fit(self, *, epochs, optimizer, stage_name, patience, scheduler=None):
        self.stage_summaries[stage_name] = {"configured_epochs": epochs}
        if stage_name == "stage2":
            torch.save({}, self.save_dir / "best_model_stage2.pth")


class AlignmentSeedFunctionTest(unittest.TestCase):
    def test_train_align_seed_defaults_to_config(self):
        parameter = inspect.signature(train_align.train_align).parameters["seed"]
        self.assertEqual(parameter.default, config.general.seed)

    def test_train_align_rejects_negative_explicit_seed(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            train_align.train_align(
                train_data={},
                train_keys=[],
                val_data={},
                val_keys=[],
                spec_encoder=None,
                seed=-1,
            )

    def test_train_align_seeds_both_dataloaders_from_explicit_seed(self):
        loader_seeds = []

        def fake_loader(*args, **kwargs):
            loader_seeds.append(kwargs["generator"].initial_seed())
            return object()

        def fake_align_model(*, spec_encoder, mol_encoder, **kwargs):
            return _FakeAlignModel(spec_encoder, mol_encoder)

        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(train_align, "AlignGraphDataset", return_value=object()),
                mock.patch.object(train_align, "DataLoader", side_effect=fake_loader),
                mock.patch.object(train_align, "SiameseModel", return_value=torch.nn.Linear(1, 1)),
                mock.patch.object(train_align, "GINEEncoder", return_value=torch.nn.Linear(1, 1)),
                mock.patch.object(train_align, "SpecMolAlignModel", side_effect=fake_align_model),
                mock.patch.object(train_align, "TrainerAlign", _FakeTrainer),
            ):
                train_align.train_align(
                    train_data={},
                    train_keys=[],
                    val_data={},
                    val_keys=[],
                    spec_encoder=None,
                    save_dir=temporary,
                    device="cpu",
                    seed=17,
                )

        self.assertEqual(loader_seeds, [17, 17])


class TrainAlignCliSeedTest(unittest.TestCase):
    def test_cli_seed_reaches_global_rng_training_and_selection_metadata(self):
        classified_data = {
            "train_data": {},
            "train_keys": [],
            "val_data": {},
            "val_keys": [],
        }
        final_model = mock.Mock()
        final_model.state_dict.return_value = {}

        with tempfile.TemporaryDirectory() as temporary:
            argv = [
                "train_align.py",
                "--save_dir",
                temporary,
                "--cache_path",
                str(Path(temporary) / "cache"),
                "--tokenset_cache",
                str(Path(temporary) / "exact.pkl"),
                "--seed",
                "17",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(train_align, "setup_logging"),
                mock.patch.object(train_align, "startup_logging"),
                mock.patch.object(train_align, "set_seed") as set_seed,
                mock.patch.object(train_align, "resolve_device", return_value=torch.device("cpu")),
                mock.patch.object(
                    train_align,
                    "get_classified_data",
                    return_value=classified_data,
                ) as get_classified,
                mock.patch.object(train_align, "train_align", return_value=final_model) as train,
                mock.patch.object(train_align.torch, "save"),
            ):
                train_align.main()

        set_seed.assert_called_once_with(17)
        kwargs = train.call_args.kwargs
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(kwargs["selection_metadata"]["seed"], 17)
        classified_call = get_classified.call_args.kwargs
        self.assertEqual(classified_call["cache_path"], str(Path(temporary) / "cache"))
        self.assertEqual(classified_call["cache_file"], str(Path(temporary) / "exact.pkl"))

    def test_cli_rejects_negative_seed(self):
        with (
            mock.patch.object(sys, "argv", ["train_align.py", "--seed", "-1"]),
            self.assertRaises(SystemExit) as raised,
        ):
            train_align.main()

        self.assertEqual(raised.exception.code, 2)


class RunPipelineSeedTest(unittest.TestCase):
    def run_pipeline_commands(self, *arguments):
        commands = []
        argv = ["run_pipeline.py", *arguments]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(run_pipeline, "get_commit_hash", return_value="abc123"),
            mock.patch.object(run_pipeline, "run_command", side_effect=commands.append),
        ):
            run_pipeline.main()
        return commands

    def test_default_seed_keeps_legacy_output_path_and_is_forwarded(self):
        commands = self.run_pipeline_commands("--mode", "train", "--no-pretrain")

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--seed") + 1], str(config.general.seed))
        self.assertEqual(
            command[command.index("--save_dir") + 1],
            "checkpoints_align/abc123_massspecgym_nopretrain",
        )
        self.assertEqual(
            Path(command[1]),
            run_pipeline.REPOSITORY_ROOT / "train_align.py",
        )

    def test_nondefault_seed_gets_collision_safe_default_path(self):
        commands = self.run_pipeline_commands(
            "--mode",
            "train",
            "--no-pretrain",
            "--seed",
            "43",
        )

        command = commands[0]
        self.assertEqual(command[command.index("--seed") + 1], "43")
        self.assertEqual(
            command[command.index("--save_dir") + 1],
            "checkpoints_align/abc123_massspecgym_nopretrain_seed43",
        )

    def test_pipeline_rejects_negative_seed(self):
        with (
            mock.patch.object(sys, "argv", ["run_pipeline.py", "--seed", "-1"]),
            self.assertRaises(SystemExit) as raised,
        ):
            run_pipeline.main()

        self.assertEqual(raised.exception.code, 2)

    def test_pipeline_forwards_portable_train_paths(self):
        commands = self.run_pipeline_commands(
            "--mode",
            "train",
            "--data_path",
            "processed",
            "--cache_path",
            "cache",
            "--tokenset_cache",
            "cache/exact.pkl",
            "--pretrained_spec",
            "models/spec.ckpt",
        )

        command = commands[0]
        for flag, expected in (
            ("--data_path", "processed"),
            ("--cache_path", "cache"),
            ("--tokenset_cache", "cache/exact.pkl"),
            ("--pretrained_spec", "models/spec.ckpt"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_pipeline_forwards_portable_eval_paths(self):
        commands = self.run_pipeline_commands(
            "--mode",
            "eval",
            "--data_path",
            "processed",
            "--candidate_type",
            "formula",
            "--candidate_path",
            "candidates/formula.pkl",
        )

        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertEqual(command[command.index("--data_path") + 1], "processed")
            self.assertEqual(command[command.index("--candidate_type") + 1], "formula")
            self.assertEqual(
                command[command.index("--candidate_path") + 1],
                "candidates/formula.pkl",
            )

    def test_pipeline_accepts_nplib1_supplied_candidates(self):
        commands = self.run_pipeline_commands(
            "nplib1",
            "--mode",
            "eval",
            "--candidate_type",
            "supplied",
        )

        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertEqual(command[command.index("--dataset_type") + 1], "nplib1")
            self.assertEqual(
                command[command.index("--candidate_type") + 1],
                "supplied",
            )

    def test_subprocesses_run_from_repository_root(self):
        completed = subprocess.CompletedProcess(["python"], returncode=0)
        with mock.patch.object(
            run_pipeline.subprocess,
            "run",
            return_value=completed,
        ) as run:
            run_pipeline.run_command(["python", "example.py"])

        run.assert_called_once_with(
            ["python", "example.py"],
            cwd=run_pipeline.REPOSITORY_ROOT,
        )


if __name__ == "__main__":
    unittest.main()
