import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_fulltrain_rerank as rerank
import run_massspecgym_v15 as v15
from SpecEmbedding.config import config
from SpecEmbedding.utils.gpu_pool import pin_pool, validate_gpu_arguments, wait_for_any_gpu


def state(gpu, *, busy=False, uuid=None):
    return {"index": gpu, "uuid": uuid or f"GPU-{gpu}", "utilization_gpu_pct": 95 if busy else 0,
            "memory_free_mib": 4000 if busy else 24000}


SETTINGS = SimpleNamespace(max_utilization=10, min_free_mib=20000, hold_seconds=20, poll_seconds=10)
PINNED = {"0": "GPU-0", "1": "GPU-1"}


def test_busy_or_failed_peer_does_not_reset_free_card_timer():
    elapsed = [0]
    calls = []

    def query(device, **kwargs):
        calls.append((device, elapsed[0]))
        if device == "cuda:1":
            raise subprocess.TimeoutExpired("nvidia-smi", 10)
        return state(0), ""

    def sleep(seconds):
        elapsed[0] += seconds

    selected = wait_for_any_gpu([0, 1], SETTINGS, PINNED, snapshot=query, clock=lambda: elapsed[0], sleep=sleep)
    assert selected["index"] == 0
    assert elapsed[0] == 20
    assert sum(d == "cuda:1" for d, _ in calls) == 3


def test_alternating_free_cards_cannot_share_a_timer_and_second_gpu_can_win():
    elapsed = [0]

    def query(device, **kwargs):
        gpu = int(device[-1])
        busy = (gpu == 0 and elapsed[0] >= 10) or (gpu == 1 and elapsed[0] == 0)
        return state(gpu, busy=busy), ""

    def sleep(seconds):
        elapsed[0] += seconds

    result = wait_for_any_gpu([0, 1], SETTINGS, PINNED, snapshot=query, clock=lambda: elapsed[0], sleep=sleep)
    assert result["index"] == 1
    assert elapsed[0] == 30


def test_tie_uses_priority_and_final_check_can_reject_a_card():
    seen = {0: 0, 1: 0}
    callback = []

    def query(device, **kwargs):
        gpu = int(device[-1])
        seen[gpu] += 1
        return state(gpu, busy=(gpu == 1 and seen[gpu] == 2)), ""

    settings = SimpleNamespace(**{**vars(SETTINGS), "hold_seconds": 0})
    selected = wait_for_any_gpu([1, 0], settings, PINNED, snapshot=query, before_select=lambda: callback.append(True))
    assert selected["index"] == 0
    assert len(callback) == 2


def test_uuid_change_and_preflight_failure_never_select():
    with pytest.raises(RuntimeError, match="identity changed"):
        wait_for_any_gpu([0, 1], SETTINGS, PINNED, snapshot=lambda device, **kw: (state(int(device[-1]), uuid="GPU-new"), ""))
    settings = SimpleNamespace(**{**vars(SETTINGS), "hold_seconds": 0})
    with pytest.raises(ValueError, match="changed input"):
        wait_for_any_gpu([0], settings, {"0": "GPU-0"}, snapshot=lambda *a, **k: (state(0), ""),
                         before_select=lambda: (_ for _ in ()).throw(ValueError("changed input")))


@pytest.mark.parametrize("gpus,device", [([], "cuda:0"), ([0, 0], "cuda:0"), ([-1, 0], "cuda:0"), ([0, 1], "cuda:1"), ([0], "cpu")])
def test_invalid_gpu_pool_arguments(gpus, device):
    with pytest.raises(ValueError):
        validate_gpu_arguments(SimpleNamespace(gpu=None, gpus=gpus, device=device))


def test_nonexistent_gpu_is_rejected_before_wait():
    with patch("SpecEmbedding.utils.gpu_pool.gpu_inventory", return_value={0: "GPU-0"}), pytest.raises(ValueError, match="nonexistent"):
        pin_pool([0, 1])


def test_rerank_pool_reselects_each_stage_and_binds_real_child_environment(tmp_path):
    args = SimpleNamespace(gpu=None, gpus=[0, 1], device="cuda:0", output_root=tmp_path)
    steps = [{"kind": "prepare", "command": [sys.executable, "-c",
              "import os,json; print(json.dumps([os.environ['CUDA_VISIBLE_DEVICES'],os.environ['SPECEMBEDDING_EXPECTED_CUDA_DEVICE']]))"]}] * 2
    outputs = []
    original = subprocess.run

    def child(command, **kwargs):
        result = original(command, capture_output=True, text=True, **kwargs)
        outputs.append(json.loads(result.stdout))
        return result

    with patch.object(rerank.subprocess, "check_output", return_value=""), \
         patch.object(rerank, "pin_pool", return_value=PINNED), \
         patch.object(rerank, "wait_for_any_gpu", side_effect=[state(1), state(0)]) as wait, \
         patch.object(rerank, "verify_cuda_mapping") as fixed, \
         patch.object(rerank, "audit_stage", return_value={}), \
         patch.object(rerank.subprocess, "run", side_effect=child):
        rerank.run(args, {"gpu_pool": PINNED, "stages": steps})
    assert outputs == [["GPU-1", "cuda:0"], ["GPU-0", "cuda:0"]]
    assert wait.call_count == 2
    fixed.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert [s["gpu_selection"]["index"] for s in status["stages"]] == [1, 0]


def test_v15_pool_binds_alignment_and_forwards_pool_to_baseline(tmp_path):
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    args = SimpleNamespace(gpu=None, gpus=[0, 1], device="cuda:0", output_root=tmp_path,
                           source_dir=tmp_path / "raw", legacy_tsv=tmp_path / "old", prepared_data=None)
    steps = v15.commands(args)
    assert steps[-1]["command"][steps[-1]["command"].index("--gpus") + 1:][:2] == ["0", "1"]
    manifest = {"runtime_config": config.to_dict(), "stages": steps, "gpu_pool": PINNED}
    environments = []

    def child(command, **kwargs):
        environments.append(kwargs["env"])
        if "run_fulltrain_rerank.py" in command[1]:
            (tmp_path / "rerank_topk256").mkdir()
            completed = [{"kind": "prepare", "split": split, "candidate": "mass", "state": "complete"}
                         for split in ("train", "val", "test")]
            completed += [{"kind": kind, "candidate": "mass", "model": "relative", "seed": 42, "state": "complete"}
                          for kind in ("train", "eval")]
            (tmp_path / "rerank_topk256/status.json").write_text(json.dumps({"state": "complete", "stages": completed}))

    with patch.object(v15, "gpu_inventory", return_value={0: "GPU-0", 1: "GPU-1"}), \
         patch.object(v15, "preflight", return_value=manifest), \
         patch.object(v15, "wait_for_any_gpu", return_value=state(1)), \
         patch.object(v15, "verify_dataset", return_value={"target_audit": {}}), \
         patch.object(v15, "audit_alignment", return_value={}), \
         patch.object(v15.subprocess, "run", side_effect=child):
        v15.execute(args, manifest)
    assert [e["CUDA_VISIBLE_DEVICES"] for e in environments] == ["", "GPU-1", ""]
    assert environments[1]["SPECEMBEDDING_EXPECTED_CUDA_DEVICE"] == "cuda:0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_visible


def test_prepared_import_failure_cannot_dispatch_gpu_or_children(tmp_path):
    args = SimpleNamespace(gpu=None, gpus=[0, 1], device="cuda:0", output_root=tmp_path,
                           source_dir=tmp_path / "raw", legacy_tsv=tmp_path / "old", prepared_data=tmp_path / "prepared")
    manifest = {"runtime_config": config.to_dict(), "stages": v15.commands(args), "gpu_pool": PINNED,
                "prepared_input": {"directory": str(args.prepared_data), "manifest_sha256": "fake"}}
    with patch.object(v15, "gpu_inventory", return_value={0: "GPU-0", 1: "GPU-1"}), \
         patch.object(v15, "preflight", return_value=manifest), \
         patch.object(v15, "import_prepared_dataset", side_effect=ValueError("changed prepared data")), \
         patch.object(v15, "wait_for_any_gpu") as wait, patch.object(v15.subprocess, "run") as child:
        with pytest.raises(ValueError, match="changed prepared data"):
            v15.execute(args, manifest)
    wait.assert_not_called()
    child.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed_or_interrupted"
    assert status["stages"][0]["name"] == "import_v15"
