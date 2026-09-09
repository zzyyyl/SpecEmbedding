"""CPU audit of completed alignment optimization runs; never starts inference or training."""

import json
import logging
import math
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.retrieval_validation import PROTOCOL, load_validation_graph_cache, load_validation_index
from SpecEmbedding.utils.training_resources import audit_resource_profiles

METRICS = ("top1", "top5", "top10", "top20", "mrr")
TOLERANCE = {key: .002 for key in METRICS}  # Top-k: 0.2 percentage points; MRR: 0.002 raw.
LEGACY_A01_COMMIT = "cba38a459fe438d29bc21bd19c54995f8d7e298e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_metrics(candidate, baseline):
    """Apply the predeclared validation tolerance; this is not a significance test."""
    delta = {key: candidate[key] - baseline[key] for key in METRICS}
    gains = [key for key in METRICS[:-1] if delta[key] > TOLERANCE[key] + 1e-12]
    losses = [key for key in METRICS if delta[key] < -TOLERANCE[key] - 1e-12]
    outcome = "eligible_for_incumbent_review" if gains and not losses else "tradeoff" if gains else "no_material_improvement"
    return {"delta_raw": delta, "topk_gains": gains, "material_regressions": losses, "outcome": outcome}


def audit_snapshot(path, index, *, expected_spectrum_control=None, expected_graph_cache=None):
    """Independently reconstruct all ranks from saved float32 scores, including missing positives."""
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    require(snapshot.get("spectrum_control") == expected_spectrum_control, "Unexpected spectrum control in validation snapshot")
    require(snapshot.get("validation_graph_cache") == expected_graph_cache, "Unexpected graph cache in validation snapshot")
    require(snapshot["protocol"] == index["protocol"] == PROTOCOL, "Snapshot protocol mismatch")
    require(snapshot["raw_query_indices"] == index["raw_query_indices"], "Snapshot query order/coverage mismatch")
    scores, labels = snapshot["scores"], index["positive_mask"]
    valid = index["candidate_indices"] >= 0
    require(scores.dtype == torch.float32 and scores.shape == labels.shape == valid.shape, "Snapshot score shape/dtype mismatch")
    require(bool(torch.isfinite(scores[valid]).all()) and not bool((labels & ~valid).any()), "Invalid scores or labels")
    ranks = torch.zeros(len(scores), dtype=torch.int64)
    stable_ranks = torch.zeros_like(ranks)
    for row in range(len(scores)):
        values, positive = scores[row, valid[row]], labels[row, valid[row]]
        if bool(positive.any()):
            for stable, output in ((False, ranks), (True, stable_ranks)):
                positions = torch.nonzero(positive[torch.argsort(values, descending=True, stable=stable)]).flatten()
                output[row] = positions[0] + 1
    require(snapshot["ranks"].dtype == torch.int64 and torch.equal(ranks, snapshot["ranks"]), "Saved ranks disagree with scores")
    n = len(ranks)
    require(n > 0, "Empty validation snapshot")
    covered = ranks > 0
    metrics = {"queries": n, "positive_queries": int(covered.sum()), "coverage": float(covered.float().mean())}
    for prefix, values in (("", ranks), ("stable_", stable_ranks)):
        for k in (1, 5, 10, 20):
            metrics[f"{prefix}top{k}"] = float(((values > 0) & (values <= k)).double().mean())
        metrics[f"{prefix}mrr"] = sum(1 / rank if rank else 0 for rank in values.tolist()) / n
    for key, value in metrics.items():
        require(math.isclose(value, snapshot["metrics"][key], rel_tol=0, abs_tol=1e-12), f"Saved metric mismatch: {key}")
    logging.info("Audited %s: all %s validation queries", Path(path).name, n)
    return metrics


def audit_trajectory(directory, index, stage, baseline, *, expected_graph_cache=None):
    history = stage["retrieval_history"]
    require([row["epoch"] for row in history] == list(range(1, stage["stop_epoch"] + 1)), "Incomplete epoch trajectory")
    require(bool(history) and stage["metric_for_best"] == "validation_top1_then_mrr", "Wrong selection protocol")
    frontier = []
    hashes = {}
    best = None
    stale = 0
    for record in history:
        epoch = record["epoch"]
        path = directory / "validation_retrieval" / f"stage2_epoch{epoch:03d}.pt"
        metrics = audit_snapshot(path, index, expected_graph_cache=expected_graph_cache)
        hashes[str(path)] = sha256_file(path)
        for key, value in metrics.items():
            require(math.isclose(value, record[key], rel_tol=0, abs_tol=1e-12), f"Trajectory metric mismatch: epoch {epoch} {key}")
        require(all(math.isfinite(record[key]) for key in ("train_loss", "val_loss", "seconds")), "Non-finite trajectory loss/time")
        if best is None or (record["top1"], record["mrr"]) > (best["top1"], best["mrr"]):
            best, stale = record, 0
        else:
            stale += 1
        require(stale < stage["patience"] or epoch == stage["stop_epoch"], "Training continued after prescribed early stop")
        if not any(all(other[key] >= record[key] for key in METRICS) for other in frontier):
            frontier = [other for other in frontier if not all(record[key] >= other[key] for key in METRICS)]
            frontier.append(record)
    require(stage["best_epoch"] == best["epoch"] and stage["best_val_loss"] == best["val_loss"], "Wrong selected epoch/loss")
    require(stage["early_stopped"] == (stale >= stage["patience"]), "Early-stop flag mismatch")
    require(stage["early_stopped"] or stage["stop_epoch"] == stage["configured_epochs"], "Training stopped without completing its rule")
    for key in METRICS:
        require(stage["best_retrieval"][key] == best[key], "Selected metrics disagree with trajectory")
    require([row["epoch"] for row in stage["pareto_frontier"]] == [row["epoch"] for row in frontier], "Pareto frontier mismatch")
    for saved, record in zip(stage["pareto_frontier"], frontier, strict=True):
        filename = f"candidate_stage2_epoch{record['epoch']:03d}.pth"
        require(saved["checkpoint"] == record["candidate_checkpoint"] == filename, "Candidate checkpoint association mismatch")
        require(saved["metrics"] == {key: record[key] for key in METRICS}, "Candidate checkpoint metrics mismatch")
        path = directory / filename
        hashes[str(path)] = sha256_file(path)
    selected = torch.load(directory / "best_model_stage2.pth", map_location="cpu", weights_only=True)
    selected_candidate = directory / f"candidate_stage2_epoch{best['epoch']:03d}.pth"
    hashes[str(selected_candidate)] = sha256_file(selected_candidate)
    candidate = torch.load(selected_candidate, map_location="cpu", weights_only=True)
    require(selected.keys() == candidate.keys() and all(torch.equal(selected[key], candidate[key]) for key in selected),
            "Selected checkpoint tensors differ from the selected epoch candidate")
    loss_best = min(history, key=lambda row: row["val_loss"])
    return {"selected_epoch": best["epoch"], "selected_metrics": {key: best[key] for key in METRICS},
            "selected_vs_baseline": compare_metrics(best, baseline),
            "minimum_loss_epoch_in_observed_trajectory": loss_best["epoch"],
            "minimum_loss_metrics": {key: loss_best[key] for key in METRICS},
            "selected_vs_minimum_loss": compare_metrics(best, loss_best),
            "pareto_candidates": [{"epoch": row["epoch"], "metrics": {key: row[key] for key in METRICS},
                                   "vs_baseline": compare_metrics(row, baseline)} for row in frontier],
            "epochs": len(history), "total_retrieval_validation_seconds": sum(row["seconds"] for row in history),
            "artifact_sha256": hashes}


def audit_optimization_run(run):
    run = Path(run).resolve()
    hashes = {}

    def fingerprint(path, expected=None):
        path = Path(path)
        digest = sha256_file(path)
        require(expected is None or digest == expected, f"Input fingerprint mismatch: {path}")
        hashes[str(path)] = digest
        return digest

    def read_json(path):
        fingerprint(path)
        return json.loads(Path(path).read_text())

    status = read_json(run / "status.json")
    require(status["state"] == "complete" and all(row["state"] == "complete" for row in status["stages"]),
            "Optimization run is not complete; refusing a completion audit")
    require([row["name"] for row in status["stages"]] in (
        ["import_v15", "prepare_validation", "baseline_validation", "alignment42"],
        ["import_v15", "import_validation", "baseline_validation", "alignment42"],
        ["prepare_v15", "prepare_validation", "baseline_validation", "alignment42"]), "Unexpected optimization stages")
    manifest = read_json(run / "inputs_and_commands.json")
    audit_versions = {name: version(name) for name in ("torch", "torchmetrics")}
    require(audit_versions["torch"] == manifest["runtime_versions"]["torch"] and audit_versions["torchmetrics"] == "1.8.2",
            "Audit sorting dependencies differ from the pinned evaluation protocol")
    runtime_path = run / "runtime_params.yaml"
    fingerprint(runtime_path, status["runtime_config_sha256"])
    runtime = yaml.safe_load(runtime_path.read_text())
    require(runtime == manifest["runtime_config"], "Runtime configuration differs from preflight")
    fingerprint(manifest["source_config"], manifest["source_config_sha256"])
    for item in manifest["inputs"].values():
        fingerprint(item["path"], item["sha256"])
    settings, fulltrain = runtime["train"]["align"], runtime["fulltrain"]
    counts, exclusions = fulltrain["expected_counts"], fulltrain["exclude_val_query_indices"]
    expected = {"train": counts["train"], "val": counts["val"] - len(exclusions)}
    index_path = run / "validation" / "mass_val_topk256.pt"
    index = load_validation_index(index_path, run / "data" / "MassSpecGym", counts, exclusions, runtime["data"]["tokenizer"])
    index_sha = fingerprint(index_path, status["stages"][1]["audit"]["sha256"])
    if status["stages"][1]["name"] == "import_validation":
        prepared = manifest.get("prepared_validation")
        require(isinstance(prepared, dict), "Missing prepared validation input provenance")
        imported = status["stages"][1].get("imported_validation")
        require(imported == {**prepared, "path": str(index_path)} and prepared["sha256"] == index_sha,
                "Imported validation index differs from the pinned source")
        require(manifest["inputs"].get("prepared_validation_index") == {
                    "path": prepared["path"], "sha256": index_sha}
                and manifest["inputs"].get("prepared_validation_receipt") == {
                    "path": str(Path(prepared["path"]).with_suffix('.json')), "sha256": prepared["receipt_sha256"]},
                "Prepared validation input files differ from preflight")
        fingerprint(index_path.with_suffix('.json'), prepared["receipt_sha256"])
    else:
        require("prepared_validation" not in manifest, "Prepared validation was not imported")
    directory = run / "alignment42_topk256"
    selection = read_json(directory / "alignment_selection.json")
    fingerprint(directory / "best_model_stage2.pth", selection["checkpoint_sha256"])
    fingerprint(run / "data" / "MassSpecGym" / "dataset_manifest.json", index["dataset_manifest_sha256"])
    audit, stage = selection["fulltrain_audit"], selection["stages"]["stage2"]
    require(selection["training_config"] == settings and selection["model_config"] == runtime["model"], "Selected model/config mismatch")
    require(selection.get("config_snapshot", {}).get("augmentation") == runtime["augmentation"],
            "Training augmentation configuration differs from preflight")
    require(selection["seed"] == fulltrain["v15"]["alignment_seed"] == 42 and selection["device"] == manifest["device"]
            and selection["device"].startswith("cuda:") and selection["exclude_val_query_indices"] == exclusions,
            "Selected seed/device/exclusions mismatch")
    require(selection["validation_index"]["sha256"] == index_sha, "Training used another validation index")
    require(audit["formal_fulltrain"] and audit["dataset_version"] == "1.5" and audit["expected_epoch_counts"] == expected
            and audit["dataset_manifest_sha256"] == index["dataset_manifest_sha256"]
            and audit["input_outputs"] == index["dataset_outputs"], "Training input/coverage mismatch")
    require(len(audit["epochs"]) == stage["stop_epoch"], "Training epoch count mismatch")
    if manifest["git_commit"] == LEGACY_A01_COMMIT:
        # This immutable A01 version predates both fields and only implements random batching.
        require("batching" not in settings and "training_batching" not in audit, "Unexpected sampler fields in A01")
        batching = "random"
    else:
        batching = settings["batching"]
        require(audit["training_batching"] == batching, "Training batching mismatch")
    for number, record in enumerate(audit["epochs"], 1):
        require({key: record[key] for key in ("stage", "epoch", "train", "val")}
                == {"stage": "stage2", "epoch": number, **expected}, "Incomplete full-spectrum epoch")
        if batching == "mass_blocks":
            b = record["batching"]
            require(b["scheme"] == batching and b["epoch"] == number and b["seed"] == selection["seed"]
                    and b["queries"] == b["unique_queries"] == expected["train"] and b["batch_size"] == settings["batch_size"]
                    and b["block_size"] == settings["mass_block_size"]
                    and all(len(b[key]) == 64 and all(c in "0123456789abcdef" for c in b[key]) for key in ("mass_sha256", "order_sha256")),
                    "Invalid mass batching coverage/configuration receipt")
        else:
            require(batching == "random" and "batching" not in record, "Unexpected training batch audit")
    from SpecEmbedding.utils.candidate_training import audit_candidate_training, candidate_source_inputs
    candidate_path = run / "candidate_training_input.json"
    candidate_input = candidate_path if candidate_path.exists() else None
    candidate_fingerprint = ({"path": str(candidate_path), "sha256": fingerprint(candidate_path)}
                             if candidate_input is not None else None)
    require(selection.get("candidate_training_input") == candidate_fingerprint,
            "Candidate training receipt differs from the actual training input")
    if candidate_input is not None:
        receipt = json.loads(candidate_path.read_text())
        require(receipt == manifest.get("candidate_training_input")
                and status.get("candidate_training_input_sha256") == candidate_fingerprint["sha256"],
                "Candidate training input differs from preflight/status")
        require(all(manifest["inputs"].get(name) == item for name, item in candidate_source_inputs(receipt).items()),
                "Candidate sources differ from the pinned manifest")
    else:
        require("candidate_training_input" not in manifest and "candidate_training_input_sha256" not in status,
                "Missing candidate input from a pinned run")
    candidate_report, candidate_hashes = audit_candidate_training(
        directory, stage, candidate_input, settings.get("candidate_supervision"), selection["seed"],
        settings["batch_size"], run / "data" / "MassSpecGym", counts, exclusions,
    )
    hashes.update(candidate_hashes)
    baseline_receipt = read_json(run / "baseline_validation" / "metrics.json")
    require(baseline_receipt["index_sha256"] == index_sha and baseline_receipt["protocol"] == PROTOCOL
            and baseline_receipt["checkpoint_sha256"] == manifest["inputs"]["baseline_checkpoint"]["sha256"], "Baseline provenance mismatch")
    baseline_path = run / "baseline_validation" / "baseline_epoch000.pt"
    graph_receipt = manifest.get('validation_graph_cache')
    require(selection.get('validation_graph_cache') == baseline_receipt.get('validation_graph_cache') == graph_receipt,
            'Validation graph cache differs between preflight, training and baseline')
    graph_fingerprint = None
    if graph_receipt is not None:
        _, verified_graph = load_validation_graph_cache(index_path, index, graph_receipt['directory'])
        require(verified_graph == graph_receipt, 'Validation graph cache changed since preflight')
        graph_fingerprint = {key: graph_receipt[key] for key in ('directory', 'manifest_sha256', 'audit_sha256')}
        graph_files = {entry['filename']: entry['sha256'] for entry in graph_receipt['files'].values()}
        graph_files.update({'manifest.json': graph_receipt['manifest_sha256'], 'audit.json': graph_receipt['audit_sha256']})
        for name, digest in graph_files.items():
            path = Path(graph_receipt['directory']) / name
            require(manifest['inputs'].get(f'validation_graph_{name}') == {'path': str(path), 'sha256': digest},
                    'Graph cache file was not pinned in preflight')
            fingerprint(path, digest)
    baseline = audit_snapshot(baseline_path, index, expected_graph_cache=graph_fingerprint)
    fingerprint(baseline_path)
    require(all(math.isclose(baseline[key], baseline_receipt["metrics"][key], rel_tol=0, abs_tol=1e-12) for key in baseline),
            "Baseline receipt metrics mismatch")
    report = audit_trajectory(directory, index, stage, baseline, expected_graph_cache=graph_fingerprint)
    report['validation_graph_cache'] = graph_receipt
    resource_report, resource_hashes = audit_resource_profiles(directory, stage, expected, selection["device"])
    report["resource_measurements"] = resource_report
    report["candidate_training"] = candidate_report
    hashes.update(resource_hashes)
    report["artifact_sha256"].update(hashes)
    report.update(run=str(run), source_commit=manifest["git_commit"], protocol=PROTOCOL,
                  audit_versions=audit_versions,
                  full_epoch_counts=expected, training_batching=batching, baseline_metrics={key: baseline[key] for key in METRICS},
                  tolerance_raw=TOLERANCE, state="complete_validation_audit", test_evaluated_by_this_audit=False,
                  limits=["Saved-score CPU verification; not new model inference or an official-loader test result.",
                          "Comparison tolerances are engineering decisions, not statistical significance.",
                          "Loss-based selection is compared within the observed trajectory, not a counterfactual rerun.",
                          "Incumbent eligibility is advisory; no checkpoint, queue or paper is changed."])
    return report
