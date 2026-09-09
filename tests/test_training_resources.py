import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import SpecEmbedding.utils.training_resources as resources
from SpecEmbedding.trainer.trainer_align import TrainerAlign


def test_cpu_phase_measurements_count_actual_queries_without_cuda_calls():
    model = torch.nn.Linear(2, 1)
    with patch.object(resources.time, "monotonic", side_effect=[0., 1., 3., 6., 10.]), \
         patch.object(resources.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=100)), \
         patch.object(resources.sys, "platform", "linux"), patch.object(torch.cuda, "synchronize") as sync:
        profile = resources.EpochResources(model, "cpu")
        profile.phase("train")
        profile.phase("contrastive_validation")
        profile.phase("retrieval_validation")
        result = profile.finish({"train": 17, "val": 5, "epoch": 1})
    sync.assert_not_called()
    assert result["parameter_count"] == result["trainable_parameter_count"] == 3
    assert result["phase_seconds"] == {"train": 1., "contrastive_validation": 2., "retrieval_validation": 3., "checkpoint_and_selection": 4.}
    assert result["epoch_seconds"] == 10 and result["train_queries_per_second"] == 17
    assert result["observed_queries"] == {"train": 17, "val": 5}
    assert result["cpu_main_process_lifetime_peak_rss_bytes"] == 102400
    assert result["cuda_peak_allocated_bytes"] is None


def test_cuda_measurements_bind_every_call_to_explicit_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic")
    device = torch.device("cuda:7")
    with patch.object(torch.cuda, "synchronize") as sync, patch.object(torch.cuda, "reset_peak_memory_stats") as reset, \
         patch.object(torch.cuda, "max_memory_allocated", return_value=1000) as allocated, \
         patch.object(torch.cuda, "max_memory_reserved", return_value=2048) as reserved:
        profile = resources.EpochResources(torch.nn.Linear(1, 1), device)
        for phase in ("train", "contrastive_validation", "retrieval_validation"):
            profile.phase(phase)
        result = profile.finish()
    reset.assert_called_once_with(device)
    assert sync.call_count == 5 and all(call.args == (device,) for call in sync.call_args_list)
    allocated.assert_called_once_with(device)
    reserved.assert_called_once_with(device)
    assert result["cuda_peak_allocated_bytes"] == 1000 and result["cuda_peak_reserved_bytes"] == 2048
    assert result["cuda_visible_devices"] == "GPU-synthetic"
    assert result["observed_queries"] is None and result["train_queries_per_second"] is None
    with pytest.raises(ValueError, match="explicit CUDA"):
        resources.EpochResources(torch.nn.Linear(1, 1), "cuda")


def test_profile_rejects_missing_and_duplicate_phases():
    profile = resources.EpochResources(torch.nn.Linear(1, 1), "cpu")
    profile.phase("train")
    with pytest.raises(ValueError, match="already recorded"):
        profile.phase("train")
    with pytest.raises(ValueError, match="Incomplete"):
        profile.finish()


def run_synthetic_fit(path, record):
    torch.manual_seed(42)
    model = torch.nn.Linear(1, 1)
    trainer = TrainerAlign(model, [], [], torch.device("cpu"), str(path), record_resources=record)
    trainer.expected_epoch_counts = {"train": 7, "val": 3}
    def train(optimizer, epoch, stage):
        with torch.no_grad():
            model.weight.fill_(torch.rand(()).item())
        trainer.epoch_counts.append({"stage": stage, "epoch": epoch, "train": 7})
        return 1.
    def validate(epoch, stage):
        trainer.epoch_counts[-1]["val"] = 3
        return [2., 1., 2., 3.][epoch - 1]
    trainer.train_epoch, trainer.validate = train, validate
    trainer.fit(10, torch.optim.SGD(model.parameters(), lr=.1), stage_name="stage2", patience=2)
    return trainer, torch.get_rng_state().clone(), copy.deepcopy(model.state_dict())


def test_recording_preserves_training_rng_selection_and_early_stop_epoch(tmp_path):
    normal, rng_a, weights_a = run_synthetic_fit(tmp_path / "normal", False)
    measured, rng_b, weights_b = run_synthetic_fit(tmp_path / "measured", True)
    assert torch.equal(rng_a, rng_b)
    assert all(torch.equal(weights_a[key], weights_b[key]) for key in weights_a)
    summary = measured.stage_summaries["stage2"]
    assert {key: value for key, value in summary.items() if key != "resource_profiles"} == normal.stage_summaries["stage2"]
    assert summary["best_epoch"] == 2 and summary["stop_epoch"] == 4
    assert len(summary["resource_profiles"]) == 4
    for epoch, profile in enumerate(summary["resource_profiles"], 1):
        path = tmp_path / "measured" / "resources" / f"stage2_epoch{epoch:03d}.json"
        assert json.loads(path.read_text()) == profile
        assert profile["observed_queries"] == {"train": 7, "val": 3}
        assert profile["train_queries_per_second"] > 0
    report, hashes = resources.audit_resource_profiles(tmp_path / "measured", summary, {"train": 7, "val": 3}, "cpu")
    assert report["state"] == "verified" and report["epochs"] == 4 and len(hashes) == 4


@pytest.mark.parametrize("damage", ["count", "timing", "device", "memory", "file"])
def test_resource_audit_rejects_inconsistent_measurements(tmp_path, damage):
    trainer, _, _ = run_synthetic_fit(tmp_path, True)
    stage = trainer.stage_summaries["stage2"]
    row = stage["resource_profiles"][0]
    if damage == "count":
        row["observed_queries"]["train"] = 6
    elif damage == "timing":
        row["phase_seconds"]["train"] = -1.
    elif damage == "device":
        row["device"] = "cuda:0"
    elif damage == "memory":
        row["cuda_peak_allocated_bytes"] = 1
    else:
        (tmp_path / "resources/stage2_epoch001.json").write_text("{}")
    with pytest.raises(ValueError, match="(Resource|resource)"):
        resources.audit_resource_profiles(tmp_path, stage, {"train": 7, "val": 3}, "cpu")


def test_legacy_runs_have_no_invented_resource_measurements(tmp_path):
    report, hashes = resources.audit_resource_profiles(tmp_path, {"stop_epoch": 19}, {"train": 7, "val": 3}, "cuda:0")
    assert report["state"] == "not_recorded" and not hashes
