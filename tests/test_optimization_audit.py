import copy
import json
from importlib.metadata import version

import pytest
import torch
import yaml

import SpecEmbedding.utils.optimization_audit as audit
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.retrieval_validation import PROTOCOL, retrieval_metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_snapshot(path, index, ranks):
    scores = -torch.arange(25, dtype=torch.float32).repeat(5, 1)
    for row, rank in enumerate(ranks):
        scores[row, 0] = -(rank - .5)
    metrics, actual = retrieval_metrics(scores, index["positive_mask"], index["candidate_indices"] >= 0)
    assert actual.tolist() == ranks
    metrics["seconds"] = 1.
    snapshot = {"scores": scores, "ranks": actual, "metrics": metrics, "protocol": PROTOCOL,
                "raw_query_indices": index["raw_query_indices"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    return metrics


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    directory = run / "alignment42_topk256"
    data = run / "data" / "MassSpecGym"
    write_json(data / "dataset_manifest.json", {"synthetic": True})
    outputs = {"synthetic": "not a formal dataset"}
    ids = torch.arange(25).repeat(5, 1)
    ids[-1] = -1
    labels = torch.zeros_like(ids, dtype=torch.bool)
    labels[:4, 0] = True
    index = {"protocol": PROTOCOL, "raw_query_indices": [0, 1, 3, 4, 6], "candidate_indices": ids,
             "positive_mask": labels, "dataset_manifest_sha256": sha256_file(data / "dataset_manifest.json"),
             "dataset_outputs": outputs}
    index_path = run / "validation" / "mass_val_topk256.pt"
    index_path.parent.mkdir(parents=True)
    torch.save(index, index_path)
    monkeypatch.setattr(audit, "load_validation_index", lambda *args: index)
    history = []
    for epoch, ranks in enumerate(([2, 2, 2, 2, 0], [1, 1, 20, 22, 0]), 1):
        metrics = make_snapshot(directory / "validation_retrieval" / f"stage2_epoch{epoch:03d}.pt", index, ranks)
        name = f"candidate_stage2_epoch{epoch:03d}.pth"
        torch.save({"weight": torch.tensor([float(epoch)])}, directory / name)
        history.append({"epoch": epoch, "train_loss": .1, "val_loss": epoch / 10,
                        "candidate_checkpoint": name, **metrics})
    torch.save({"weight": torch.tensor([2.])}, directory / "best_model_stage2.pth")
    baseline_dir = run / "baseline_validation"
    baseline = make_snapshot(baseline_dir / "baseline_epoch000.pt", index, [2, 2, 2, 2, 0])
    baseline_ckpt = tmp_path / "baseline.pth"
    torch.save({"weight": torch.tensor([1.])}, baseline_ckpt)
    write_json(baseline_dir / "metrics.json", {"protocol": PROTOCOL, "index_sha256": sha256_file(index_path),
                                               "checkpoint_sha256": sha256_file(baseline_ckpt), "metrics": baseline})
    runtime = {"train": {"align": {"batching": "random", "batch_size": 128}}, "model": {"synthetic": True},
               "augmentation": {"prob": .5, "node_drop_rate": .1, "edge_mask_rate": .1},
               "fulltrain": {"v15": {"alignment_seed": 42}, "expected_counts": {"train": 7, "val": 7, "test": 2},
                             "exclude_val_query_indices": [2, 5]}, "data": {"tokenizer": {}}}
    runtime_path = run / "runtime_params.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime))
    source = tmp_path / "source_params.yaml"
    source.write_text(runtime_path.read_text())
    write_json(run / "inputs_and_commands.json", {"runtime_config": runtime, "source_config": str(source),
        "source_config_sha256": sha256_file(source), "git_commit": "synthetic-version", "device": "cuda:0",
        "runtime_versions": {"torch": version("torch")},
        "inputs": {"baseline_checkpoint": {"path": str(baseline_ckpt), "sha256": sha256_file(baseline_ckpt)}}})
    write_json(run / "status.json", {"state": "complete", "runtime_config_sha256": sha256_file(runtime_path),
        "stages": [{"name": name, "state": "complete", "audit": {"sha256": sha256_file(index_path)}}
                   for name in ("import_v15", "prepare_validation", "baseline_validation", "alignment42")]})
    stage = {"retrieval_history": history, "stop_epoch": 2, "best_epoch": 2, "best_val_loss": .2,
             "best_retrieval": history[1], "metric_for_best": "validation_top1_then_mrr", "patience": 5,
             "early_stopped": False, "configured_epochs": 2,
             "pareto_frontier": [{"epoch": row["epoch"], "checkpoint": row["candidate_checkpoint"],
                                  "metrics": {key: row[key] for key in audit.METRICS}} for row in history]}
    selection = {"training_config": runtime["train"]["align"], "model_config": runtime["model"], "seed": 42,
                 "config_snapshot": copy.deepcopy(runtime),
                 "device": "cuda:0", "exclude_val_query_indices": [2, 5],
                 "checkpoint_sha256": sha256_file(directory / "best_model_stage2.pth"),
                 "validation_index": {"sha256": sha256_file(index_path)}, "stages": {"stage2": stage},
                 "fulltrain_audit": {"formal_fulltrain": True, "dataset_version": "1.5", "training_batching": "random",
                     "expected_epoch_counts": {"train": 7, "val": 5}, "input_outputs": outputs,
                     "dataset_manifest_sha256": index["dataset_manifest_sha256"],
                     "epochs": [{"stage": "stage2", "epoch": n, "train": 7, "val": 5} for n in (1, 2)]}}
    write_json(directory / "alignment_selection.json", selection)
    return run, index, selection


def test_complete_audit_preserves_full_denominator_and_reports_topk_tradeoff(completed_run):
    run, _, _ = completed_run
    report = audit.audit_optimization_run(run)
    assert report["full_epoch_counts"] == {"train": 7, "val": 5}
    assert report["selected_epoch"] == 2 and report["minimum_loss_epoch_in_observed_trajectory"] == 1
    assert report["selected_metrics"]["top1"] == .4  # Fifth, empty-positive query stays in the denominator.
    assert report["selected_vs_baseline"]["outcome"] == "tradeoff"
    assert "top5" in report["selected_vs_baseline"]["material_regressions"]
    assert len(report["pareto_candidates"]) == 2 and not report["test_evaluated_by_this_audit"]


def test_completion_rejects_changed_effective_training_augmentation(completed_run):
    run, _, selection = completed_run
    selection["config_snapshot"]["augmentation"]["node_drop_rate"] = 0.0
    write_json(run / "alignment42_topk256" / "alignment_selection.json", selection)
    with pytest.raises(ValueError, match="augmentation configuration"):
        audit.audit_optimization_run(run)


@pytest.mark.parametrize("damage", ["rank", "score", "query", "metric"])
def test_snapshot_corruption_is_detected(completed_run, damage):
    run, index, _ = completed_run
    path = run / "baseline_validation" / "baseline_epoch000.pt"
    snapshot = torch.load(path, weights_only=False)
    if damage == "rank":
        snapshot["ranks"][0] = 1
    elif damage == "score":
        snapshot["scores"][0, 0] = float("nan")
    elif damage == "query":
        snapshot["raw_query_indices"].reverse()
    else:
        snapshot["metrics"]["mrr"] += .1
    torch.save(snapshot, path)
    with pytest.raises(ValueError):
        audit.audit_snapshot(path, index)


@pytest.mark.parametrize("damage", ["running", "coverage", "selection", "frontier", "checkpoint", "fingerprint"])
def test_completion_audit_rejects_incomplete_or_inconsistent_artifacts(completed_run, damage):
    run, _, selection = completed_run
    directory = run / "alignment42_topk256"
    if damage == "running":
        status = json.loads((run / "status.json").read_text())
        status["state"] = "running"
        write_json(run / "status.json", status)
    elif damage == "coverage":
        selection["fulltrain_audit"]["epochs"][1]["train"] -= 1
    elif damage == "selection":
        selection["stages"]["stage2"]["best_epoch"] = 1
    elif damage == "frontier":
        selection["stages"]["stage2"]["pareto_frontier"] = []
    elif damage == "checkpoint":
        torch.save({"weight": torch.tensor([99.])}, directory / "candidate_stage2_epoch002.pth")
    else:
        (run / "runtime_params.yaml").write_text("changed: true\n")
    write_json(directory / "alignment_selection.json", selection)
    with pytest.raises(ValueError):
        audit.audit_optimization_run(run)


def test_tolerance_is_raw_units_and_mrr_cannot_replace_topk_improvement():
    baseline = {key: .5 for key in audit.METRICS}
    candidate = copy.deepcopy(baseline)
    candidate["top1"] += .002
    candidate["mrr"] += .1
    assert audit.compare_metrics(candidate, baseline)["outcome"] == "no_material_improvement"
    candidate["top1"] += .001
    assert audit.compare_metrics(candidate, baseline)["outcome"] == "eligible_for_incumbent_review"
    candidate["top20"] -= .003
    assert audit.compare_metrics(candidate, baseline)["outcome"] == "tradeoff"


@pytest.mark.parametrize("mode", ["legacy_a01", "mass_blocks"])
def test_pinned_a01_compatibility_and_a02_mass_batch_receipts(completed_run, mode):
    run, _, selection = completed_run
    manifest = json.loads((run / "inputs_and_commands.json").read_text())
    settings = manifest["runtime_config"]["train"]["align"]
    counts = selection["fulltrain_audit"]
    if mode == "legacy_a01":
        manifest["git_commit"] = audit.LEGACY_A01_COMMIT
        settings.pop("batching")
        counts.pop("training_batching")
    else:
        settings.update(batching="mass_blocks", mass_block_size=32)
        counts["training_batching"] = "mass_blocks"
        for row in counts["epochs"]:
            row["batching"] = {"scheme": "mass_blocks", "epoch": row["epoch"], "seed": 42,
                               "queries": 7, "unique_queries": 7, "batch_size": 128, "block_size": 32,
                               "mass_sha256": "a" * 64, "order_sha256": "b" * 64}
    selection["training_config"] = settings
    runtime_path = run / "runtime_params.yaml"
    runtime_path.write_text(yaml.safe_dump(manifest["runtime_config"]))
    write_json(run / "inputs_and_commands.json", manifest)
    status = json.loads((run / "status.json").read_text())
    status["runtime_config_sha256"] = sha256_file(runtime_path)
    write_json(run / "status.json", status)
    selection_path = run / "alignment42_topk256" / "alignment_selection.json"
    write_json(selection_path, selection)
    report = audit.audit_optimization_run(run)
    assert report["training_batching"] == ("random" if mode == "legacy_a01" else "mass_blocks")
    if mode == "legacy_a01":
        manifest["git_commit"] = "unrecognized-version-with-missing-config"
        write_json(run / "inputs_and_commands.json", manifest)
        with pytest.raises(KeyError):
            audit.audit_optimization_run(run)
    else:
        counts["epochs"][0]["batching"]["unique_queries"] -= 1
        write_json(selection_path, selection)
        with pytest.raises(ValueError, match="batching coverage"):
            audit.audit_optimization_run(run)
