"""One isolated v1.5 baseline: CPU audit, GPU alignment-42, Mass/relative/seed42."""

import argparse
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from SpecEmbedding.config import DEFAULT_CONFIG_PATH, config
from SpecEmbedding.utils.candidate_training import (
    audit_candidate_training,
    build_candidate_training_input,
    candidate_source_inputs,
    validate_candidate_settings,
)
from SpecEmbedding.utils.fingerprint_alignment_inputs import fingerprint_input_files, load_alignment_fingerprints
from SpecEmbedding.utils.formal_alignment import formal_model_type, read_formal_alignment_checkpoint
from SpecEmbedding.utils.fulltrain import (
    sha256_file,
    validate_baseline_completion,
    validate_baseline_scope,
    wait_for_gpu,
)
from SpecEmbedding.utils.gpu import gpu_inventory, parse_cuda_device
from SpecEmbedding.utils.gpu_pool import (
    add_gpu_arguments,
    pin_pool,
    pool_environment,
    validate_gpu_arguments,
    wait_for_any_gpu,
)
from SpecEmbedding.utils.massspecgym_v15 import import_prepared_dataset, prepared_source, verify_dataset, write_json
from SpecEmbedding.utils.retrieval_validation import (
    import_validation_index,
    load_validation_graph_cache,
    load_validation_index,
    prepared_validation_input,
)
from SpecEmbedding.utils.storage import storage_receipt

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def commands(args):
    data = args.output_root / "data" / "MassSpecGym"
    alignment = args.output_root / "alignment42_topk256"
    pool = getattr(args, "gpus", None)
    gpu_args = ["--gpus", *map(str, pool)] if pool is not None else ["--gpu", str(args.gpu)]
    result = [
        {"name": "prepare_v15", "gpu": False, "command": [sys.executable, str(ROOT / "prepare_massspecgym_v15.py"),
         "--source-dir", str(args.source_dir), "--legacy-tsv", str(args.legacy_tsv), "--output-dir", str(data)]},
        {"name": "alignment42", "gpu": True, "command": [sys.executable, str(ROOT / "train_align.py"),
         "--dataset_type", "massspecgym", "--data_path", str(data), "--save_dir", str(alignment),
         "--device", args.device, "--seed", str(config.fulltrain.v15.alignment_seed), "--formal-fulltrain",
         "--exclude-val-query-indices", *map(str, config.fulltrain.exclude_val_query_indices)]},
        {"name": "rerank_baseline", "gpu": False, "command": [sys.executable, str(ROOT / "run_fulltrain_rerank.py"),
         *gpu_args, "--device", args.device, "--data-path", str(data),
         "--checkpoint", str(alignment / "best_model_stage2.pth"), "--output-root", str(args.output_root / "rerank_topk256")]},
    ]
    if getattr(args, "prepared_data", None) is not None:
        result[0] = {"name": "import_v15", "gpu": False, "source": str(args.prepared_data), "output": str(data)}
    if getattr(args, "optimize_alignment", False):
        fingerprint_model = getattr(args, 'molecule_input', 'gine') == 'fingerprint'
        index = args.output_root / "validation" / "mass_val_topk256.pt"
        result[1]["command"] += ["--validation-index", str(index)]
        if getattr(args, "alignment_training_candidates", None) is not None:
            result[1]["command"] += ["--candidate-training-input", str(args.output_root / "candidate_training_input.json")]
        result = [result[0],
                  {"name": "prepare_validation", "gpu": False, "command": [sys.executable, str(ROOT / "alignment_validation.py"),
                   "--data-path", str(data), "--index", str(index), "--prepare-only"]},
                  {"name": "baseline_validation", "gpu": True, "command": [sys.executable, str(ROOT / "alignment_validation.py"),
                   "--data-path", str(data), "--index", str(index), "--checkpoint", str(args.baseline_checkpoint),
                   "--output", str(args.output_root / "baseline_validation"), "--device", args.device]},
                  result[1]]
        if getattr(args, "prepared_validation_index", None) is not None:
            result[1] = {"name": "import_validation", "gpu": False,
                         "source": str(args.prepared_validation_index), "output": str(index)}
        if getattr(args, "validation_graph_cache", None) is not None:
            result[2]["command"] += ["--graph-cache", str(args.validation_graph_cache)]
            if not fingerprint_model:
                result[3]["command"] += ["--validation-graph-cache", str(args.validation_graph_cache)]
        if getattr(args, "checkpoint_model_config", False) or fingerprint_model:
            result[2]["command"] += ["--checkpoint-model-config"]
        if getattr(args, 'baseline_fingerprint_cache', None) is not None:
            result[2]['command'] += ['--fingerprint-cache', str(args.baseline_fingerprint_cache)]
        if fingerprint_model:
            for flag in ('fingerprint_training_index', 'training_fingerprint_cache', 'validation_fingerprint_cache'):
                result[3]['command'] += ['--' + flag.replace('_', '-'), str(getattr(args, flag))]
    return result


def preflight(args):
    storage = storage_receipt(args.output_root)
    fingerprint_model = getattr(args, 'molecule_input', 'gine') == 'fingerprint'
    independent_baseline = getattr(args, 'checkpoint_model_config', False) or fingerprint_model
    if hasattr(config.model.spec_encoder, 'precursor_delta') and not independent_baseline:
        raise ValueError('Precursor delta optimization requires independent baseline checkpoint construction')
    if independent_baseline and (not getattr(args, 'optimize_alignment', False)
                                  or getattr(args, 'prepared_validation_index', None) is None):
        raise ValueError('Independent baseline construction requires optimization and a prepared full validation index')
    if config.fulltrain.v15.graph_policy != "rdkit_sanitized" or config.fulltrain.v15.alignment_seed != 42:
        raise ValueError("Approved v1.5 protocol requires sanitized graphs and alignment seed 42")
    if config.fulltrain.v15.audit_workers < 1:
        raise ValueError("Candidate audit requires at least one CPU worker")
    validate_baseline_scope(config.fulltrain)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("v1.5 queue requires a clean, fixed source worktree")
    inputs = {}
    for name, digest in config.fulltrain.v15.sources.to_dict().items():
        path = args.source_dir / name
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"Pinned v1.5 input changed: {name}")
        inputs[name] = {"path": str(path), "sha256": actual}
    inputs["legacy_tsv"] = {"path": str(args.legacy_tsv), "sha256": sha256_file(args.legacy_tsv)}
    runtime_config = config.to_dict()
    runtime_config["model"]["mol_encoder"]["graph_policy"] = config.fulltrain.v15.graph_policy
    runtime_config["general"]["device"] = args.device
    if fingerprint_model:
        runtime_config['model']['type'] = 'fingerprint'
        runtime_config['model']['mol_encoder'] = {**config.fingerprint_encoder.to_dict(), 'graph_policy': 'rdkit_sanitized'}
    formal_model_type(runtime_config['model'])
    mol_augmentation = getattr(args, "alignment_mol_augmentation", None)
    if mol_augmentation is not None:
        if not getattr(args, "optimize_alignment", False) or not isinstance(mol_augmentation, bool):
            raise ValueError("Molecular augmentation override requires the optimization branch and a boolean")
        if not mol_augmentation:
            # Keep spectrum augmentation and its probability unchanged; disable only graph perturbations.
            runtime_config["augmentation"]["node_drop_rate"] = 0.0
            runtime_config["augmentation"]["edge_mask_rate"] = 0.0
    if fingerprint_model and (runtime_config['augmentation']['node_drop_rate'] != 0
                              or runtime_config['augmentation']['edge_mask_rate'] != 0
                              or runtime_config['model']['mol_encoder']['input_bits'] != config.molecule_fingerprints.bits):
        raise ValueError('Fingerprint training requires explicitly disabled graph augmentation and matching input width')
    batching = getattr(args, "alignment_batching", None)
    if batching is not None:
        if not getattr(args, "optimize_alignment", False) or batching not in {"random", "mass_blocks"}:
            raise ValueError("Alignment batching override requires the optimization branch")
        runtime_config["train"]["align"]["batching"] = batching
    align_settings = runtime_config["train"]["align"]
    if align_settings["batching"] not in {"random", "mass_blocks"}:
        raise ValueError("Unknown alignment batching in configuration")
    if align_settings["batching"] == "mass_blocks" and (
        align_settings["mass_block_size"] < 1 or align_settings["batch_size"] % align_settings["mass_block_size"]
    ):
        raise ValueError("Mass block size must divide the alignment batch size")
    if getattr(args, "optimize_alignment", False):
        if args.baseline_checkpoint is None:
            raise ValueError("Alignment optimization requires the frozen baseline checkpoint")
        runtime_config["train"]["align"]["metric_for_best"] = "validation_top1_then_mrr"
        for name, path in (("baseline_checkpoint", args.baseline_checkpoint),
                           ("baseline_selection", args.baseline_checkpoint.parent / "alignment_selection.json")):
            inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
        selection = json.loads((args.baseline_checkpoint.parent / "alignment_selection.json").read_text())
        if (selection["checkpoint_sha256"] != inputs["baseline_checkpoint"]["sha256"]
                or selection["seed"] != 42
                or (not independent_baseline and selection["model_config"] != runtime_config["model"])
                or selection["graph_policy"] != "rdkit_sanitized"
                or not selection["fulltrain_audit"]["formal_fulltrain"]
                or selection["fulltrain_audit"]["dataset_version"] != "1.5"):
            raise ValueError("Optimization baseline checkpoint/configuration provenance mismatch")
    source_config = Path(os.environ.get("SPECEMBEDDING_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    extra = {"storage": storage} if storage is not None else {}
    validation_path = getattr(args, "prepared_validation_index", None)
    if validation_path is not None:
        if not getattr(args, "optimize_alignment", False) or getattr(args, "prepared_data", None) is None:
            raise ValueError("Prepared validation index requires optimization and a fully prepared dataset")
        prepared = prepared_validation_input(validation_path, args.prepared_data,
                                             config.fulltrain.expected_counts.to_dict(),
                                             config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
        extra["prepared_validation"] = prepared
        inputs["prepared_validation_index"] = {"path": str(validation_path), "sha256": prepared["sha256"]}
        inputs["prepared_validation_receipt"] = {"path": str(validation_path.with_suffix('.json')),
                                                 "sha256": prepared["receipt_sha256"]}
    graph_path = getattr(args, "validation_graph_cache", None)
    if independent_baseline:
        baseline_index = load_validation_index(validation_path, args.prepared_data,
                                               config.fulltrain.expected_counts.to_dict(),
                                               config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
        parent_selection, model_receipt = read_formal_alignment_checkpoint(
            args.baseline_checkpoint, dataset_outputs=baseline_index['dataset_outputs'],
            dataset_manifest_sha256=baseline_index['dataset_manifest_sha256'],
            tokenizer_config=config.data.tokenizer.to_dict(),
            expected_counts={'train': config.fulltrain.expected_counts.train,
                             'val': config.fulltrain.expected_counts.val - len(config.fulltrain.exclude_val_query_indices)},
            exclusions=config.fulltrain.exclude_val_query_indices,
        )
        baseline_fingerprints = getattr(args, 'baseline_fingerprint_cache', None)
        if (model_receipt['model_type'] == 'fingerprint') != bool(baseline_fingerprints):
            raise ValueError('Baseline model type requires matching explicit molecular inputs')
        if baseline_fingerprints:
            expected_cache = parent_selection['validation_fingerprint_cache']
            _, baseline_input = load_alignment_fingerprints(
                args.prepared_data, validation_path, 'validation', baseline_fingerprints,
                counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
                tokenizer_config=config.data.tokenizer.to_dict(), settings=expected_cache['provenance']['options'],
                pool_cache_size=config.train.align.candidate_supervision.pool_cache_size)
            if baseline_input['cache'] != expected_cache:
                raise ValueError('Baseline fingerprints differ from its own selected model inputs')
            extra['baseline_fingerprint_input'] = baseline_input
            inputs.update(fingerprint_input_files(baseline_input, 'baseline_fingerprint'))
        extra['checkpoint_model'] = model_receipt
    if graph_path is not None:
        if validation_path is None:
            raise ValueError("Validation graph cache requires a verified prepared validation index")
        index = load_validation_index(validation_path, args.prepared_data, config.fulltrain.expected_counts.to_dict(),
                                      config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
        _, graph_receipt = load_validation_graph_cache(validation_path, index, graph_path)
        extra["validation_graph_cache"] = graph_receipt
        graph_files = {entry['filename']: entry['sha256'] for entry in graph_receipt['files'].values()}
        graph_files.update({'manifest.json': graph_receipt['manifest_sha256'], 'audit.json': graph_receipt['audit_sha256']})
        inputs.update({f"validation_graph_{name}": {"path": str(graph_path / name), "sha256": digest}
                       for name, digest in graph_files.items()})
    candidate_settings = align_settings["candidate_supervision"]
    validate_candidate_settings(candidate_settings)
    candidate_path = getattr(args, "alignment_training_candidates", None)
    if candidate_path is None and candidate_settings["enabled"]:
        raise ValueError("Enabled candidate supervision requires --alignment-training-candidates")
    if candidate_path is not None:
        if not getattr(args, "optimize_alignment", False) or getattr(args, "prepared_data", None) is None:
            raise ValueError("Candidate supervision requires optimization and a fully prepared dataset")
        candidate_settings["enabled"] = True
        _, receipt = build_candidate_training_input(candidate_path, args.prepared_data, candidate_settings,
                                                   config.fulltrain.expected_counts.to_dict(),
                                                   config.fulltrain.exclude_val_query_indices)
        extra["candidate_training_input"] = receipt
        inputs.update(candidate_source_inputs(receipt))
    if fingerprint_model:
        extra['fingerprint_inputs'] = {}
        for split, index_path, cache_path in (('train', args.fingerprint_training_index, args.training_fingerprint_cache),
                                              ('validation', validation_path, args.validation_fingerprint_cache)):
            _, receipt = load_alignment_fingerprints(
                args.prepared_data, index_path, split, cache_path, counts=config.fulltrain.expected_counts.to_dict(),
                exclusions=config.fulltrain.exclude_val_query_indices, tokenizer_config=config.data.tokenizer.to_dict(),
                settings=config.molecule_fingerprints.to_dict(), pool_cache_size=candidate_settings['pool_cache_size'])
            extra['fingerprint_inputs'][split] = receipt
            inputs.update(fingerprint_input_files(receipt, f'fingerprint_{split}'))
        if candidate_path is not None and (extra['fingerprint_inputs']['train']['source']['sha256']
                                           != extra['candidate_training_input']['provenance']['sha256']):
            raise ValueError('Fingerprint and negative-sampling training inventories differ')
    if getattr(args, "gpus", None) is not None:
        extra["gpu_pool"] = pin_pool(args.gpus)
    if getattr(args, "prepared_data", None) is not None:
        extra["prepared_input"] = prepared_source(args.prepared_data, args.source_dir, args.legacy_tsv,
                                                 config.fulltrain.v15, config.fulltrain.expected_counts.to_dict(),
                                                 config.fulltrain.exclude_val_query_indices)
    return {**extra, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runtime_versions": {"python": sys.version, **{name: version(name) for name in ("torch", "torch-geometric", "rdkit", "matchms")}},
            "source_config": str(source_config), "source_config_sha256": sha256_file(source_config),
            "runtime_config": runtime_config, "inputs": inputs, "device": args.device, "gpu": args.gpu,
            "stages": commands(args), "protocol": "v1.5 source order, sanitized graphs; cache-local exact-target-SMILES sensitivity; independent 2D result audit pending"}


def audit_alignment(args, alignment_settings, augmentation_settings):
    data = args.output_root / "data" / "MassSpecGym"
    verify_dataset(data, config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices)
    directory = args.output_root / "alignment42_topk256"
    selection = json.loads((directory / "alignment_selection.json").read_text())
    expected = {"train": config.fulltrain.expected_counts.train,
                "val": config.fulltrain.expected_counts.val - len(config.fulltrain.exclude_val_query_indices)}
    audit = selection["fulltrain_audit"]
    stage = selection["stages"]["stage2"]
    if (selection["graph_policy"] != "rdkit_sanitized" or selection["seed"] != 42
            or selection["device"] != args.device or not audit["formal_fulltrain"]
            or audit["dataset_version"] != "1.5" or audit["expected_epoch_counts"] != expected
            or audit["dataset_manifest_sha256"] != sha256_file(data / "dataset_manifest.json")
            or selection["checkpoint_sha256"] != sha256_file(directory / "best_model_stage2.pth")
            or selection["exclude_val_query_indices"] != config.fulltrain.exclude_val_query_indices
            or selection["training_config"] != alignment_settings
            or selection.get("config_snapshot", {}).get("augmentation") != augmentation_settings
            or audit["training_batching"] != alignment_settings["batching"]
            or len(audit["epochs"]) != stage["stop_epoch"] or stage["best_epoch"] is None
            or stage["stop_epoch"] <= 0):
        raise ValueError("Formal v1.5 alignment completion audit failed")
    for index, epoch in enumerate(audit["epochs"], 1):
        counts = {key: value for key, value in epoch.items() if key != "batching"}
        if counts != {"stage": "stage2", "epoch": index, **expected}:
            raise ValueError("Alignment did not visit all train/validation spectra every epoch")
        batch_audit = epoch.get("batching")
        if alignment_settings["batching"] == "mass_blocks":
            if (not isinstance(batch_audit, dict) or batch_audit.get("scheme") != "mass_blocks"
                    or batch_audit.get("epoch") != index or batch_audit.get("seed") != selection["seed"]
                    or batch_audit.get("queries") != expected["train"] or batch_audit.get("unique_queries") != expected["train"]
                    or batch_audit.get("batch_size") != alignment_settings["batch_size"]
                    or batch_audit.get("block_size") != alignment_settings["mass_block_size"]
                    or any(len(batch_audit.get(key, "")) != 64 for key in ("mass_sha256", "order_sha256"))):
                raise ValueError("Mass batching coverage/configuration audit failed")
        elif batch_audit is not None:
            raise ValueError("Unexpected mass batching in a random-batching trial")
    candidate_path = args.output_root / "candidate_training_input.json"
    candidate_input = candidate_path if candidate_path.exists() else None
    expected_input = ({"path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path)}
                      if candidate_input is not None else None)
    if selection.get("candidate_training_input") != expected_input:
        raise ValueError("Alignment used a different candidate training input")
    candidate_report, candidate_hashes = audit_candidate_training(
        directory, stage, candidate_input, alignment_settings.get("candidate_supervision"), selection["seed"],
        alignment_settings["batch_size"], data, config.fulltrain.expected_counts.to_dict(),
        config.fulltrain.exclude_val_query_indices,
        fingerprint_cache=selection.get('training_fingerprint_cache'),
    )
    return {"checkpoint_sha256": selection["checkpoint_sha256"], "epochs": len(audit["epochs"]),
            "expected_epoch_counts": expected, "graph_policy": selection["graph_policy"],
            "candidate_training": candidate_report, "candidate_artifact_sha256": candidate_hashes}


def execute(args, manifest):
    for name in ("status.json", "data", "alignment42_topk256", "rerank_topk256", "logs", "validation", "baseline_validation",
                 "candidate_training_input.json"):
        if (args.output_root / name).exists():
            raise FileExistsError(f"Refusing existing v1.5 run artifact: {name}")
    runtime_path = args.output_root / "runtime_params.yaml"
    runtime_path.write_text(yaml.safe_dump(manifest["runtime_config"], sort_keys=False))
    environment = os.environ.copy()
    environment["SPECEMBEDDING_CONFIG"] = str(runtime_path)
    inventory = gpu_inventory()
    pool = getattr(args, "gpus", None)
    if pool is None and (args.gpu not in inventory or parse_cuda_device(args.device) != args.gpu):
        raise ValueError("v1.5 all-GPU UUID mapping requires cuda:N to match physical GPU N")
    ordered = [inventory[index] for index in sorted(inventory)]
    if sorted(inventory) != list(range(len(inventory))):
        raise ValueError("Unexpected noncontiguous physical GPU inventory")
    environment.update(CUDA_VISIBLE_DEVICES="" if pool is not None else ",".join(ordered), SPECEMBEDDING_REQUIRE_CUDA="1",
                       SPECEMBEDDING_EXPECTED_CUDA_DEVICE=args.device,
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.setdefault('NUMBA_CACHE_DIR', '/tmp/specembedding-v15-numba')
    environment.setdefault('MPLCONFIGDIR', '/tmp/specembedding-v15-mpl')
    logs = args.output_root / "logs"
    logs.mkdir()
    status = {"state": "running", "started_at": now(), "stages": [], "gpu_uuid_order": ordered,
              "runtime_config_sha256": sha256_file(runtime_path)}
    if "candidate_training_input" in manifest:
        candidate_path = args.output_root / "candidate_training_input.json"
        write_json(candidate_path, manifest["candidate_training_input"])
        status["candidate_training_input_sha256"] = sha256_file(candidate_path)
    if pool is not None:
        status["gpu_pool"] = manifest["gpu_pool"]

    def check_preflight():
        if sha256_file(runtime_path) != status["runtime_config_sha256"] or preflight(args) != manifest:
            raise ValueError("Inputs/source/configuration changed during queue wait")
        if "candidate_training_input_sha256" in status:
            if sha256_file(args.output_root / "candidate_training_input.json") != status["candidate_training_input_sha256"]:
                raise ValueError("Pinned candidate training input changed during queue wait")
        for item in status["stages"]:
            if item["name"] in ("prepare_validation", "import_validation") and item["state"] == "complete":
                if sha256_file(args.output_root / "validation" / "mass_val_topk256.pt") != item["audit"]["sha256"]:
                    raise ValueError("Prepared validation index changed during queue wait")

    try:
        for stage in manifest["stages"]:
            progress = {**stage, "state": "waiting_gpu" if stage["gpu"] else "running", "queued_at": now()}
            status["stages"].append(progress)
            write_json(args.output_root / "status.json", status)
            child_environment = environment.copy()
            if stage["gpu"]:
                if pool is None:
                    wait_for_gpu(args.gpu, config.fulltrain)
                else:
                    selected = wait_for_any_gpu(pool, config.fulltrain, manifest["gpu_pool"], before_select=check_preflight)
                    child_environment = pool_environment(selected, environment)
                    progress["gpu_selection"] = {**selected, "device": args.device}
                if pool is None and gpu_inventory() != inventory:
                    raise ValueError("GPU UUID mapping changed")
            if pool is None or not stage["gpu"]:
                check_preflight()
            progress.update(state="running", started_at=now())
            write_json(args.output_root / "status.json", status)
            logging.info("START %s: %s", stage["name"], shlex.join(stage.get("command", [])))
            if stage["name"] == "import_v15":
                import_prepared_dataset(args.prepared_data, args.output_root / "data" / "MassSpecGym",
                                        manifest["prepared_input"], config.fulltrain.expected_counts.to_dict(),
                                        config.fulltrain.exclude_val_query_indices)
                progress["prepared_input"] = manifest["prepared_input"]
            elif stage["name"] == "import_validation":
                progress["imported_validation"] = import_validation_index(
                    args.prepared_validation_index, args.output_root / "validation" / "mass_val_topk256.pt",
                    manifest["prepared_validation"], args.output_root / "data" / "MassSpecGym",
                    config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices,
                    config.data.tokenizer.to_dict(),
                )
            else:
                with (logs / f"{stage['name']}.log").open("x") as handle:
                    subprocess.run(stage["command"], cwd=ROOT, env=child_environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
            if stage["name"] in ("prepare_v15", "import_v15"):
                data_report = verify_dataset(args.output_root / "data" / "MassSpecGym", config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices)
                progress["audit"] = data_report["target_audit"]
            elif stage["name"] == "alignment42":
                progress["audit"] = audit_alignment(args, manifest["runtime_config"]["train"]["align"],
                                                    manifest["runtime_config"]["augmentation"])
                if getattr(args, "optimize_alignment", False):
                    selection = json.loads((args.output_root / "alignment42_topk256" / "alignment_selection.json").read_text())
                    from SpecEmbedding.utils.fingerprint_alignment_inputs import audit_model_fingerprint_inputs
                    if selection['model_config'] != manifest['runtime_config']['model']:
                        raise ValueError('Trained model construction differs from preflight')
                    fp_report, fp_hashes = audit_model_fingerprint_inputs(
                        manifest, selection, args.output_root / 'data/MassSpecGym',
                        args.output_root / 'validation/mass_val_topk256.pt')
                    if fp_report is not None:
                        progress['audit']['fingerprint_inputs'] = fp_report
                        progress['audit']['fingerprint_artifact_sha256'] = fp_hashes
                    summary = selection["stages"]["stage2"]
                    if (summary["metric_for_best"] != "validation_top1_then_mrr"
                            or summary["best_retrieval"]["queries"] != config.fulltrain.expected_counts.val - len(config.fulltrain.exclude_val_query_indices)):
                        raise ValueError("Alignment retrieval checkpoint-selection audit failed")
                    progress["retrieval_selection"] = summary
                    if selection.get("validation_graph_cache") != manifest.get("validation_graph_cache"):
                        raise ValueError("Training used a different validation graph cache")
            elif stage["name"] in ("prepare_validation", "import_validation"):
                receipt = json.loads((args.output_root / "validation" / "mass_val_topk256.json").read_text())
                if (receipt["sha256"] != sha256_file(args.output_root / "validation" / "mass_val_topk256.pt")
                        or receipt["queries"] != config.fulltrain.expected_counts.val - len(config.fulltrain.exclude_val_query_indices)):
                    raise ValueError("Validation index preparation audit failed")
                progress["audit"] = receipt
            elif stage["name"] == "baseline_validation":
                progress["audit"] = json.loads((args.output_root / "baseline_validation" / "metrics.json").read_text())
                if progress["audit"]["checkpoint_sha256"] != manifest["inputs"]["baseline_checkpoint"]["sha256"]:
                    raise ValueError("Baseline checkpoint changed")
                if progress["audit"].get("validation_graph_cache") != manifest.get("validation_graph_cache"):
                    raise ValueError("Baseline used a different validation graph cache")
                if progress['audit'].get('checkpoint_model') != manifest.get('checkpoint_model'):
                    raise ValueError('Baseline model construction changed after preflight')
                baseline_fingerprint = manifest.get('baseline_fingerprint_input')
                if progress['audit'].get('validation_fingerprint_cache') != (
                    baseline_fingerprint['cache'] if baseline_fingerprint else None
                ):
                    raise ValueError('Baseline fingerprint inputs changed after preflight')
            else:
                rerank = json.loads((args.output_root / "rerank_topk256" / "status.json").read_text())
                validate_baseline_completion(rerank)
            progress.update(state="complete", completed_at=now())
            write_json(args.output_root / "status.json", status)
        status.update(state="complete", completed_at=now(), result_identity_audit="pending", paper_update="pending",
                      scope="alignment_validation_only" if getattr(args, "optimize_alignment", False) else "rerank_baseline",
                      sota_goal="not_achieved")
    except BaseException as error:
        status.update(state="failed_or_interrupted", error=repr(error), stopped_at=now())
        if status["stages"]:
            status["stages"][-1]["state"] = "failed_or_interrupted"
        raise
    finally:
        write_json(args.output_root / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "legacy-tsv", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    add_gpu_arguments(parser)
    parser.add_argument("--device", required=True)
    parser.add_argument("--prepared-data", type=Path, help="Import a complete, fingerprint-verified CPU dataset into a new run")
    parser.add_argument("--prepared-validation-index", type=Path,
                        help="Reuse verified CPU validation index bytes; all model embeddings are still freshly encoded")
    parser.add_argument("--validation-graph-cache", type=Path, help="Reuse fully audited fixed validation molecule graphs")
    parser.add_argument("--optimize-alignment", action="store_true", help="Full validation baseline then one alignment seed42 selected by retrieval; no test/rerank matrix")
    parser.add_argument("--baseline-checkpoint", type=Path, help="Frozen v1.5 seed42 baseline for the alignment optimization branch")
    parser.add_argument("--checkpoint-model-config", action="store_true",
                        help="Build the baseline from its own verified model configuration, sharing the full data protocol")
    parser.add_argument('--molecule-input', choices=('gine', 'fingerprint'), default='gine')
    parser.add_argument('--fingerprint-training-index', type=Path)
    parser.add_argument('--training-fingerprint-cache', type=Path)
    parser.add_argument('--validation-fingerprint-cache', type=Path)
    parser.add_argument('--baseline-fingerprint-cache', type=Path)
    parser.add_argument("--alignment-batching", choices=["random", "mass_blocks"],
                        help="Optimization trial: override only the training batch assembly; block size comes from params.yaml")
    parser.add_argument("--alignment-mol-augmentation", action=argparse.BooleanOptionalAction, default=None,
                        help="Optimization trial: --no-alignment-mol-augmentation disables graph perturbations only")
    parser.add_argument("--alignment-training-candidates", type=Path,
                        help="Opt in to natural candidate supervision using fully audited training metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-preflight", action="store_true")
    args = parser.parse_args(argv)
    fp_paths = (args.fingerprint_training_index, args.training_fingerprint_cache, args.validation_fingerprint_cache)
    if ((args.molecule_input == 'fingerprint') != any(fp_paths)
            or (args.molecule_input == 'fingerprint' and not all(fp_paths))):
        parser.error('Fingerprint model requires all three explicit fixed-input paths')
    if args.molecule_input == 'fingerprint' and (not args.optimize_alignment or args.prepared_data is None
                                                or args.prepared_validation_index is None):
        parser.error('Fingerprint optimization requires complete prepared data and validation index')
    if args.baseline_fingerprint_cache and (args.validation_graph_cache or not (
        args.checkpoint_model_config or args.molecule_input == 'fingerprint'
    )):
        parser.error('Baseline fingerprints require independent model loading without baseline graph cache')
    for key in ('fingerprint_training_index', 'training_fingerprint_cache', 'validation_fingerprint_cache',
                'baseline_fingerprint_cache'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    try:
        validate_gpu_arguments(args, matching_single=True)
    except ValueError as error:
        parser.error(str(error))
    for name in ("source_dir", "legacy_tsv", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.prepared_data is not None:
        args.prepared_data = args.prepared_data.expanduser().resolve()
    if args.prepared_validation_index is not None:
        args.prepared_validation_index = args.prepared_validation_index.expanduser().resolve()
        if not args.optimize_alignment or args.prepared_data is None:
            parser.error("--prepared-validation-index requires --optimize-alignment and --prepared-data")
    if args.baseline_checkpoint is not None:
        args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    if args.validation_graph_cache is not None:
        args.validation_graph_cache = args.validation_graph_cache.expanduser().resolve()
        if args.prepared_validation_index is None:
            parser.error("--validation-graph-cache requires --prepared-validation-index")
    if args.alignment_training_candidates is not None:
        args.alignment_training_candidates = args.alignment_training_candidates.expanduser().resolve()
        if not args.optimize_alignment or args.prepared_data is None:
            parser.error("--alignment-training-candidates requires --optimize-alignment and --prepared-data")
    if args.optimize_alignment != (args.baseline_checkpoint is not None):
        parser.error("--optimize-alignment and --baseline-checkpoint must be provided together")
    if args.alignment_batching is not None and not args.optimize_alignment:
        parser.error("--alignment-batching requires --optimize-alignment")
    if args.alignment_mol_augmentation is not None and not args.optimize_alignment:
        parser.error("--alignment-mol-augmentation requires --optimize-alignment")
    manifest = preflight(args)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output_root / "inputs_and_commands.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("v1.5 preflight fingerprint changed")
        if not path.exists():
            write_json(path, manifest)
        if args.write_preflight:
            print(f"Saved v1.5 preflight: {path}; no CPU preparation or GPU work started")
            return
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.FileHandler(args.output_root / "runner.log"), logging.StreamHandler()])
        execute(args, manifest)


if __name__ == "__main__":
    main()
