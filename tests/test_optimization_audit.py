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


@pytest.mark.parametrize('damage', [None, 'baseline', 'selection', 'snapshot', 'cache_bytes', 'unpinned'])
def test_completion_checks_full_graph_cache_and_every_usage_receipt(completed_run, damage):
    from SpecEmbedding.utils.molecule_graph_cache import audit_graph_cache, build_graph_cache, graph_cache_provenance
    from SpecEmbedding.utils.retrieval_validation import load_validation_graph_cache

    run, index, selection = completed_run
    index['mol_smiles'] = ['C' * n for n in range(1, 26)]
    index_path = run / 'validation/mass_val_topk256.pt'
    torch.save(index, index_path)
    index_sha = sha256_file(index_path)
    root = run.parent / 'graphs'
    provenance = graph_cache_provenance(index['mol_smiles'], index_sha256=index_sha,
                                        dataset_manifest_sha256=index['dataset_manifest_sha256'])
    build_graph_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=8)
    audit_graph_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=8)
    _, receipt = load_validation_graph_cache(index_path, index, root)
    fingerprint = {key: receipt[key] for key in ('directory', 'manifest_sha256', 'audit_sha256')}
    manifest_path = run / 'inputs_and_commands.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['validation_graph_cache'] = receipt
    files = {entry['filename']: entry['sha256'] for entry in receipt['files'].values()}
    files.update({'manifest.json': receipt['manifest_sha256'], 'audit.json': receipt['audit_sha256']})
    for name, digest in files.items():
        manifest['inputs'][f'validation_graph_{name}'] = {'path': str(root / name), 'sha256': digest}
    if damage == 'unpinned':
        del manifest['inputs']['validation_graph_x.bin']
    write_json(manifest_path, manifest)
    status = json.loads((run / 'status.json').read_text())
    status['stages'][1]['audit']['sha256'] = index_sha
    write_json(run / 'status.json', status)
    selection['validation_index']['sha256'] = index_sha
    selection['validation_graph_cache'] = None if damage == 'selection' else receipt
    write_json(run / 'alignment42_topk256/alignment_selection.json', selection)
    baseline_path = run / 'baseline_validation/metrics.json'
    baseline = json.loads(baseline_path.read_text())
    baseline.update(index_sha256=index_sha, validation_graph_cache=None if damage == 'baseline' else receipt)
    write_json(baseline_path, baseline)
    for path in [run / 'baseline_validation/baseline_epoch000.pt',
                 *sorted((run / 'alignment42_topk256/validation_retrieval').glob('*.pt'))]:
        snapshot = torch.load(path, weights_only=False)
        snapshot['validation_graph_cache'] = fingerprint
        if damage == 'snapshot' and path.name == 'stage2_epoch002.pt':
            snapshot['validation_graph_cache'] = None
        torch.save(snapshot, path)
    if damage == 'cache_bytes':
        (root / 'x.bin').write_bytes((root / 'x.bin').read_bytes() + b'changed')
    if damage is None:
        result = audit.audit_optimization_run(run)
        assert result['validation_graph_cache'] == receipt
        assert all(str(root / name) in result['artifact_sha256'] for name in files)
    else:
        with pytest.raises(ValueError, match='[Gg]raph cache|fingerprint'):
            audit.audit_optimization_run(run)


def test_completion_binds_imported_validation_to_original_source_and_receipt(completed_run):
    run, _, _ = completed_run
    source = run.parent / 'prepared.pt'
    index = run / 'validation/mass_val_topk256.pt'
    source.write_bytes(index.read_bytes())
    source.with_suffix('.json').write_text(json.dumps({'sha256': sha256_file(source)}))
    index.with_suffix('.json').write_bytes(source.with_suffix('.json').read_bytes())
    prepared = {'path': str(source), 'sha256': sha256_file(source),
                'receipt_sha256': sha256_file(source.with_suffix('.json'))}
    manifest_path, status_path = run / 'inputs_and_commands.json', run / 'status.json'
    manifest, status = json.loads(manifest_path.read_text()), json.loads(status_path.read_text())
    manifest['prepared_validation'] = prepared
    manifest['inputs'].update(prepared_validation_index={'path': str(source), 'sha256': prepared['sha256']},
                              prepared_validation_receipt={'path': str(source.with_suffix('.json')),
                                                           'sha256': prepared['receipt_sha256']})
    status['stages'][1].update(name='import_validation', imported_validation={**prepared, 'path': str(index)})
    write_json(manifest_path, manifest)
    write_json(status_path, status)
    assert audit.audit_optimization_run(run)['state'] == 'complete_validation_audit'
    status['stages'][1]['imported_validation']['sha256'] = '0'*64
    write_json(status_path, status)
    with pytest.raises(ValueError, match='Imported validation'):
        audit.audit_optimization_run(run)
    status['stages'][1]['imported_validation']['sha256'] = prepared['sha256']
    write_json(status_path, status)
    index.with_suffix('.json').write_text('{}')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        audit.audit_optimization_run(run)


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
