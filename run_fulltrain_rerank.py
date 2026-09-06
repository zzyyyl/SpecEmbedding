"""Approved MassSpecGym alignment-42 full-training matrix, sequential and GPU-only."""

import argparse
import fcntl
import json
import logging
import os
import pickle
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from SpecEmbedding.config import DEFAULT_CONFIG_PATH, config
from SpecEmbedding.data.datasets_rerank import load_rerank_cache
from SpecEmbedding.utils.fulltrain import sha256_file, validate_cache, wait_for_gpu
from SpecEmbedding.utils.gpu import gpu_inventory, parse_cuda_device
from SpecEmbedding.utils.rerank import parse_rerank_eval_metrics
from SpecEmbedding.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi index, independent of CUDA mapping")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory containing train/val/test and candidate pickles")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen alignment-42 stage2 checkpoint")
    parser.add_argument("--output-root", type=Path, required=True, help="New output root; no reuse/overwrite of partial runs")
    parser.add_argument("--dry-run", action="store_true", help="Read-only inputs/config audit and command listing, no GPU work")
    parser.add_argument("--write-preflight", action="store_true", help="Save input fingerprints and commands only; no GPU work")
    args = parser.parse_args(argv)
    try:
        parse_cuda_device(args.device)
        if args.gpu < 0:
            raise ValueError("Physical GPU index must be nonnegative")
    except ValueError as error:
        parser.error(str(error))
    for name in ("data_path", "checkpoint", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def validate_config():
    if config.rerank.prepare.limit != 0 or config.rerank.prepare.force_include_positive:
        raise ValueError("Formal preparation forbids limiting queries or forcing positives")
    if config.rerank.prepare.pre_top_k != 256 or config.rerank.train.train_k != 256:
        raise ValueError("Approved full-training protocol requires topk=256")
    if config.rerank.train.use_rank_embedding or config.rerank.train.relation_top_k != 40:
        raise ValueError("Approved model requires no rank embedding and coarse top-40")
    if config.rerank.train.metric_for_best != "mrr" or config.rerank.train.epochs != 30 or config.rerank.train.patience != 5:
        raise ValueError("Approved model selection requires validation MRR, 30 epochs, patience=5")
    if config.fulltrain.candidate_types != ["mass", "formula"] or config.fulltrain.model_types != ["relative", "pointwise"] or config.fulltrain.seeds != [42, 43, 44]:
        raise ValueError("Approved first batch is 2 candidate protocols x 2 models x 3 seeds")


def stages(args):
    result = []
    for candidate in config.fulltrain.candidate_types:
        caches = {split: args.output_root / "cache" / f"{candidate}_topk256" / f"massspecgym_{candidate}_{split}.pt"
                  for split in ("train", "val", "test")}
        for split, path in caches.items():
            result.append({"kind": "prepare", "candidate": candidate, "split": split, "output": str(path), "command": [
                sys.executable, str(ROOT / "prepare_rerank_cache.py"), "--checkpoint", str(args.checkpoint),
                "--dataset_type", "massspecgym", "--data_path", str(args.data_path), "--split", split,
                "--candidate_type", candidate, "--save_path", str(path), "--device", args.device,
                "--pre_top_k", "256", "--limit", "0", "--no-force_include_positive",
            ]})
        for model in config.fulltrain.model_types:
            for seed in config.fulltrain.seeds:
                output = args.output_root / "checkpoints" / candidate / model / f"seed{seed}_topk256"
                common = {"candidate": candidate, "model": model, "seed": seed, "output": str(output)}
                result.append({**common, "kind": "train", "command": [
                    sys.executable, str(ROOT / "train_rerank.py"), "--train_cache", str(caches["train"]),
                    "--val_cache", str(caches["val"]), "--save_dir", str(output), "--model_type", model,
                    "--seed", str(seed), "--device", args.device, "--train-k", "256", "--formal-fulltrain",
                    "--exclude-val-query-indices", *map(str, config.fulltrain.exclude_val_query_indices),
                ]})
                result.append({**common, "kind": "eval", "command": [
                    sys.executable, str(ROOT / "eval_rerank.py"), "--cache", str(caches["test"]),
                    "--checkpoint", str(output / "best_reranker.pth"), "--save_dir", str(output),
                    "--device", args.device, "--no-mces",
                ]})
    # Audit all six caches before any training; no inherited legacy caches.
    return [stage for stage in result if stage["kind"] == "prepare"] + [stage for stage in result if stage["kind"] != "prepare"]


def input_manifest(args):
    validate_config()
    inputs = {}
    for name in ("train", "val", "test", "candidates_mass", "candidates_formula"):
        path = args.data_path / f"{name}.pkl"
        inputs[name] = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if name in ("train", "val", "test"):
            with path.open("rb") as handle:
                count = len(pickle.load(handle))  # Trusted, existing project artifacts only.
            if count != config.fulltrain.expected_counts[name]:
                raise ValueError(f"Input split {name} has {count} records, expected {config.fulltrain.expected_counts[name]}")
            inputs[name]["records"] = count
    inputs["alignment"] = {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)}
    selection_path = args.checkpoint.parent / "alignment_selection.json"
    selection = json.loads(selection_path.read_text())
    if (selection["seed"] != 42 or selection["checkpoint_sha256"] != inputs["alignment"]["sha256"]
            or selection["exclude_val_query_indices"] != config.fulltrain.exclude_val_query_indices):
        raise ValueError("Alignment-42 selection/checkpoint/exclusion provenance mismatch")
    inputs["alignment_selection"] = {"path": str(selection_path), "sha256": sha256_file(selection_path)}
    config_path = Path(os.environ.get("SPECEMBEDDING_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "config": config.to_dict(), "inputs": inputs, "device": args.device, "physical_gpu": args.gpu,
            "stages": stages(args), "evaluation_protocol": "local exact-target-SMILES; new identity audit pending"}


def verify_cuda_mapping(args):
    logical = parse_cuda_device(args.device)
    expected_uuid = gpu_inventory()[args.gpu]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if logical >= len(visible) or visible[logical] != expected_uuid:
        raise ValueError("Full-training runner requires explicit UUID CUDA mapping; launch via watch_gpu_tmux.py")
    if os.environ.get("SPECEMBEDDING_REQUIRE_CUDA") != "1" or os.environ.get("SPECEMBEDDING_EXPECTED_CUDA_DEVICE") != args.device:
        raise ValueError("Strict CUDA launch environment is missing or mismatched")
    resolve_device(args.device)  # Availability only; no tensors/model are allocated by the runner.


def audit_stage(stage, manifest, cache_audits):
    output = Path(stage["output"])
    if stage["kind"] == "prepare":
        cache = load_rerank_cache(output)
        summary = validate_cache(cache, split=stage["split"], candidate_type=stage["candidate"],
                                 expected_count=config.fulltrain.expected_counts[stage["split"]], fingerprints={
                                     "checkpoint_sha256": manifest["inputs"]["alignment"]["sha256"],
                                     "candidate_sha256": manifest["inputs"][f"candidates_{stage['candidate']}"]["sha256"],
                                 })
        summary["sha256"] = sha256_file(output)
        cache_audits[f"{stage['candidate']}/{stage['split']}"] = summary
        return summary
    if stage["kind"] == "train":
        expected = cache_audits[f"{stage['candidate']}/train"]["eligible_queries"]
        for name in ("best_reranker.pth", "last_reranker.pth"):
            checkpoint = load_rerank_cache(output / name)
            training = checkpoint["training_config"]
            summary = checkpoint["data_summary"]
            if (checkpoint["seed"] != stage["seed"] or checkpoint["model_config"]["model_type"] != stage["model"]
                    or training["max_train_queries"] is not None or training["device"] != manifest["device"]
                    or not training["formal_fulltrain"] or training["train_k"] != 256
                    or training["exclude_val_query_indices"] != config.fulltrain.exclude_val_query_indices
                    or summary["labeled_train_queries"] != expected or summary["last_epoch_train_queries"] != expected
                    or summary["fulltrain_audit"]["actual_train_queries"] != expected):
                raise ValueError(f"Full-training checkpoint audit failed: {output / name}")
            for split in ("train", "val"):
                if summary["fulltrain_audit"][f"{split}_cache_sha256"] != cache_audits[f"{stage['candidate']}/{split}"]["sha256"]:
                    raise ValueError("Training used a changed cache")
        if "Training finished." not in (output / "train_rerank.log").read_text():
            raise ValueError("Training completion marker missing")
        return {"actual_train_queries": expected}
    metrics = parse_rerank_eval_metrics(output / "eval_rerank.log")
    required = [f"{model}_top{k}_pct" for model in ("base", "rerank") for k in (1, 5, 10, 20)] + ["base_mrr_raw", "rerank_mrr_raw", "upper_bound_pct"]
    if metrics.get("total_queries") != config.fulltrain.expected_counts.test or any(key not in metrics for key in required):
        raise ValueError("Incomplete full-test evaluation")
    return metrics


def run(args, manifest):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Formal runner requires a clean, fixed source worktree")
    for name in ("cache", "checkpoints", "status.json"):
        if (args.output_root / name).exists():
            raise FileExistsError(f"Refusing to reuse/overwrite existing run artifacts: {args.output_root / name}")
    verify_cuda_mapping(args)
    status = {"started_at": now(), "state": "running", "stages": [], "cache_audits": {}}
    try:
        for stage in manifest["stages"]:
            progress = {**stage, "state": "waiting_gpu", "queued_at": now()}
            status["stages"].append(progress)
            write_json(args.output_root / "status.json", status)
            wait_for_gpu(args.gpu, config.fulltrain)
            verify_cuda_mapping(args)
            # Each child is foreground/sequential; no peer tasks are signalled on error.
            progress.update(state="running", started_at=now())
            write_json(args.output_root / "status.json", status)
            logging.info("START %s: %s", stage["kind"], shlex.join(stage["command"]))
            subprocess.run(stage["command"], cwd=ROOT, check=True)
            progress["audit"] = audit_stage(stage, manifest, status["cache_audits"])
            progress.update(state="complete", completed_at=now())
            write_json(args.output_root / "status.json", status)
        status.update(state="complete", completed_at=now())
    except BaseException as error:
        status.update(state="failed_or_interrupted", error=repr(error), stopped_at=now())
        if status["stages"]:
            status["stages"][-1]["state"] = "failed_or_interrupted"
        raise
    finally:
        write_json(args.output_root / "status.json", status)


def main(argv=None):
    args = parse_args(argv)
    manifest = input_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output_root / "inputs_and_commands.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Inputs, source, configuration or commands changed since preflight")
        else:
            write_json(path, manifest)
        if args.write_preflight:
            print(f"Preflight saved: {path}; no GPU work started")
            return
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.FileHandler(args.output_root / "runner.log"), logging.StreamHandler()])
        run(args, manifest)


if __name__ == "__main__":
    main()
