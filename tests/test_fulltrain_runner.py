import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import run_fulltrain_rerank as runner
from SpecEmbedding.utils import fulltrain


def cache(split="train"):
    return {"spec_embs": [0, 0], "queries": [
        {"spec_index": index, "candidate_indices": [0, 1], "label": 0 if index == 0 else None,
         "positive_forced_into_candidate_pool": False, "positive_forced_into_topk": False}
        for index in range(2)
    ], "meta": {"dataset_type": "massspecgym", "candidate_type": "mass", "split": split,
                "pre_top_k": 256, "limit": 0, "force_include_positive": False,
                "num_input_spectra": 2, "num_candidate_mapped_spectra": 2, "num_selected_spectra": 2,
                "num_queries": 2, "num_skipped_queries": 0, "checkpoint_sha256": "align", "candidate_sha256": "candidates"}}


def validate(value):
    return fulltrain.validate_cache(value, split="train", candidate_type="mass", expected_count=2)


def test_complete_cache_counts_only_eligible_labels():
    assert validate(cache()) == {"queries": 2, "eligible_queries": 1, "unlabeled_queries": 1}


@pytest.mark.parametrize("changes", [{"limit": 20000}, {"pre_top_k": 40}, {"force_include_positive": True},
                                     {"num_selected_spectra": 1}, {"num_skipped_queries": 1}, {"split": "test"}])
def test_reject_limited_legacy_or_incomplete_cache(changes):
    value = cache()
    value["meta"].update(changes)
    with pytest.raises(ValueError):
        validate(value)


def test_reject_unrecorded_limit_forcing_or_reordered_queries():
    for change in ("missing_limit", "forcing", "order", "bad_label"):
        value = cache()
        if change == "missing_limit":
            del value["meta"]["limit"]
        elif change == "forcing":
            value["queries"][0]["positive_forced_into_topk"] = True
        elif change == "order":
            value["queries"].reverse()
        else:
            value["queries"][0]["label"] = 5
        with pytest.raises(ValueError):
            validate(value)


def test_reject_different_input_fingerprints():
    with pytest.raises(ValueError, match="fingerprint"):
        fulltrain.validate_cache(cache(), split="train", candidate_type="mass", expected_count=2,
                                 fingerprints={"checkpoint_sha256": "wrong"})


def test_training_selection_rejects_a_cap_or_lost_eligible_query():
    class Dataset:
        def __init__(self, split):
            self.cache = cache(split)
            self.meta = self.cache["meta"]
            self.queries = self.cache["queries"]
            self.indices = [0]
            self.cache_path = Path("unused_mock_cache")

        def __len__(self):
            return len(self.indices)

    train, val = Dataset("train"), Dataset("val")
    args = SimpleNamespace(max_train_queries=None, train_k=256)
    with patch.object(fulltrain, "sha256_file", return_value="hash"):
        assert fulltrain.validate_training_data(args, train, val, {"train": 2, "val": 2})["actual_train_queries"] == 1
        for cap, indices in ((20000, [0]), (None, [])):
            args.max_train_queries, train.indices = cap, indices
            with pytest.raises(ValueError):
                fulltrain.validate_training_data(args, train, val, {"train": 2, "val": 2})


def args_for(root):
    return SimpleNamespace(device="cuda:1", gpu=1, data_path=root / "data", checkpoint=root / "align.pth", output_root=root / "output")


def test_matrix_has_six_caches_then_twelve_full_train_eval_pairs():
    stages = runner.stages(args_for(Path("/tmp/synthetic-fulltrain")))
    assert len(stages) == 30
    assert [stage["kind"] for stage in stages[:6]] == ["prepare"] * 6
    assert len([stage for stage in stages if stage["kind"] == "train"]) == 12
    for stage in stages:
        command = stage["command"]
        assert command[command.index("--device") + 1] == "cuda:1"
        assert "--max-train-queries" not in command
        if stage["kind"] == "prepare":
            assert command[command.index("--limit") + 1] == "0"
            assert "--no-force_include_positive" in command
        if stage["kind"] == "train":
            assert "--formal-fulltrain" in command
            assert command[command.index("--train-k") + 1] == "256"
        if stage["kind"] == "eval":
            assert "--no-mces" in command


def test_config_rejects_formal_limit():
    runner.validate_config()
    with patch.object(runner.config.rerank.prepare, "limit", 20000), pytest.raises(ValueError):
        runner.validate_config()


def test_stage_gpu_gate_resets_on_busy_and_failed_queries():
    state = {"uuid": "GPU-aaaa", "utilization_gpu_pct": 0, "memory_free_mib": 24000}
    samples = [state, {**state, "utilization_gpu_pct": 90}, state, ValueError("query error"), state, state, state]
    elapsed = [0]

    def query(*args, **kwargs):
        value = samples.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, ""

    def sleep(seconds):
        elapsed[0] += seconds

    settings = SimpleNamespace(max_utilization=10, min_free_mib=20000, hold_seconds=20, poll_seconds=10)
    with patch.object(fulltrain, "gpu_snapshot", side_effect=query):
        fulltrain.wait_for_gpu(1, settings, clock=lambda: elapsed[0], sleep=sleep)
    assert elapsed[0] == 60


def test_cuda_mapping_refuses_cpu_and_mismatches():
    args = args_for(Path("/tmp/synthetic-fulltrain"))
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb", "SPECEMBEDDING_REQUIRE_CUDA": "1",
                   "SPECEMBEDDING_EXPECTED_CUDA_DEVICE": "cuda:1"}
    with patch.dict("os.environ", environment), patch.object(runner, "gpu_inventory", return_value={1: "GPU-bbbb"}), patch.object(runner, "resolve_device") as resolve:
        runner.verify_cuda_mapping(args)
        resolve.assert_called_once_with("cuda:1")
        args.device = "cuda:0"
        with pytest.raises(ValueError):
            runner.verify_cuda_mapping(args)
        args.device = "cpu"
        with pytest.raises(ValueError):
            runner.verify_cuda_mapping(args)


def test_runner_records_wait_start_completion_and_stops_after_failure():
    with tempfile.TemporaryDirectory() as directory:
        args = args_for(Path(directory))
        args.output_root.mkdir()
        manifest = {"stages": [{"kind": "prepare", "command": ["MOCK_ONLY"]}]}
        with patch.object(runner.subprocess, "check_output", return_value=""), patch.object(runner, "verify_cuda_mapping"), patch.object(runner, "wait_for_gpu") as wait, patch.object(runner.subprocess, "run") as run, patch.object(runner, "audit_stage", return_value={"ok": True}):
            runner.run(args, manifest)
            wait.assert_called_once()
            run.assert_called_once()
            assert json.loads((args.output_root / "status.json").read_text())["state"] == "complete"
            with pytest.raises(FileExistsError):
                runner.run(args, manifest)
        failed_args = copy.copy(args)
        failed_args.output_root = Path(directory) / "failed"
        failed_args.output_root.mkdir()
        with patch.object(runner.subprocess, "check_output", return_value=""), patch.object(runner, "verify_cuda_mapping"), patch.object(runner, "wait_for_gpu"), patch.object(runner.subprocess, "run", side_effect=RuntimeError("MOCK failure")) as run:
            with pytest.raises(RuntimeError):
                runner.run(failed_args, {"stages": manifest["stages"] * 2})
            run.assert_called_once()
            assert json.loads((failed_args.output_root / "status.json").read_text())["state"] == "failed_or_interrupted"


def test_dry_run_never_writes_or_starts_processes():
    with patch.object(runner, "input_manifest", return_value={"stages": []}), patch.object(runner, "run") as run, patch.object(Path, "mkdir", new=Mock()) as mkdir:
        runner.main(["--gpu", "1", "--device", "cuda:1", "--data-path", "/tmp/data", "--checkpoint", "/tmp/align", "--output-root", "/tmp/dry-fulltrain", "--dry-run"])
        mkdir.assert_not_called()
        run.assert_not_called()
