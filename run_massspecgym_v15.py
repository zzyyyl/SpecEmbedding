"""One isolated v1.5 queue: full CPU audit, GPU alignment-42, six caches and 12 rerank runs."""

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
from SpecEmbedding.utils.fulltrain import sha256_file, wait_for_gpu
from SpecEmbedding.utils.gpu import gpu_inventory, parse_cuda_device
from SpecEmbedding.utils.massspecgym_v15 import verify_dataset, write_json

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def commands(args):
    data = args.output_root / "data" / "MassSpecGym"
    alignment = args.output_root / "alignment42_topk256"
    return [
        {"name": "prepare_v15", "gpu": False, "command": [sys.executable, str(ROOT / "prepare_massspecgym_v15.py"),
         "--source-dir", str(args.source_dir), "--legacy-tsv", str(args.legacy_tsv), "--output-dir", str(data)]},
        {"name": "alignment42", "gpu": True, "command": [sys.executable, str(ROOT / "train_align.py"),
         "--dataset_type", "massspecgym", "--data_path", str(data), "--save_dir", str(alignment),
         "--device", args.device, "--seed", str(config.fulltrain.v15.alignment_seed), "--formal-fulltrain",
         "--exclude-val-query-indices", *map(str, config.fulltrain.exclude_val_query_indices)]},
        {"name": "rerank_matrix", "gpu": False, "command": [sys.executable, str(ROOT / "run_fulltrain_rerank.py"),
         "--gpu", str(args.gpu), "--device", args.device, "--data-path", str(data),
         "--checkpoint", str(alignment / "best_model_stage2.pth"), "--output-root", str(args.output_root / "rerank_topk256")]},
    ]


def preflight(args):
    if config.fulltrain.v15.graph_policy != "rdkit_sanitized" or config.fulltrain.v15.alignment_seed != 42:
        raise ValueError("Approved v1.5 protocol requires sanitized graphs and alignment seed 42")
    if config.fulltrain.v15.audit_workers < 1:
        raise ValueError("Candidate audit requires at least one CPU worker")
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
    source_config = Path(os.environ.get("SPECEMBEDDING_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runtime_versions": {"python": sys.version, **{name: version(name) for name in ("torch", "torch-geometric", "rdkit", "matchms")}},
            "source_config": str(source_config), "source_config_sha256": sha256_file(source_config),
            "runtime_config": runtime_config, "inputs": inputs, "device": args.device, "gpu": args.gpu,
            "stages": commands(args), "protocol": "v1.5 source order, sanitized graphs; cache-local exact-target-SMILES sensitivity; independent 2D result audit pending"}


def audit_alignment(args):
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
            or len(audit["epochs"]) != stage["stop_epoch"] or stage["best_epoch"] is None
            or stage["stop_epoch"] <= 0):
        raise ValueError("Formal v1.5 alignment completion audit failed")
    for index, epoch in enumerate(audit["epochs"], 1):
        if epoch != {"stage": "stage2", "epoch": index, **expected}:
            raise ValueError("Alignment did not visit all train/validation spectra every epoch")
    return {"checkpoint_sha256": selection["checkpoint_sha256"], "epochs": len(audit["epochs"]),
            "expected_epoch_counts": expected, "graph_policy": selection["graph_policy"]}


def execute(args, manifest):
    for name in ("status.json", "data", "alignment42_topk256", "rerank_topk256", "logs"):
        if (args.output_root / name).exists():
            raise FileExistsError(f"Refusing existing v1.5 run artifact: {name}")
    runtime_path = args.output_root / "runtime_params.yaml"
    runtime_path.write_text(yaml.safe_dump(manifest["runtime_config"], sort_keys=False))
    environment = os.environ.copy()
    environment["SPECEMBEDDING_CONFIG"] = str(runtime_path)
    inventory = gpu_inventory()
    if args.gpu not in inventory or parse_cuda_device(args.device) != args.gpu:
        raise ValueError("v1.5 all-GPU UUID mapping requires cuda:N to match physical GPU N")
    ordered = [inventory[index] for index in sorted(inventory)]
    if sorted(inventory) != list(range(len(inventory))):
        raise ValueError("Unexpected noncontiguous physical GPU inventory")
    environment.update(CUDA_VISIBLE_DEVICES=",".join(ordered), SPECEMBEDDING_REQUIRE_CUDA="1",
                       SPECEMBEDDING_EXPECTED_CUDA_DEVICE=args.device,
                       NUMBA_CACHE_DIR="/tmp/specembedding-v15-numba", MPLCONFIGDIR="/tmp/specembedding-v15-mpl",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    logs = args.output_root / "logs"
    logs.mkdir()
    status = {"state": "running", "started_at": now(), "stages": [], "gpu_uuid_order": ordered,
              "runtime_config_sha256": sha256_file(runtime_path)}
    try:
        for stage in manifest["stages"]:
            progress = {**stage, "state": "waiting_gpu" if stage["gpu"] else "running", "queued_at": now()}
            status["stages"].append(progress)
            write_json(args.output_root / "status.json", status)
            if stage["gpu"]:
                wait_for_gpu(args.gpu, config.fulltrain)
                if gpu_inventory() != inventory:
                    raise ValueError("GPU UUID mapping changed")
            if sha256_file(runtime_path) != status["runtime_config_sha256"] or preflight(args) != manifest:
                raise ValueError("Inputs/source/configuration changed during queue wait")
            progress.update(state="running", started_at=now())
            write_json(args.output_root / "status.json", status)
            logging.info("START %s: %s", stage["name"], shlex.join(stage["command"]))
            with (logs / f"{stage['name']}.log").open("x") as handle:
                subprocess.run(stage["command"], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
            if stage["name"] == "prepare_v15":
                data_report = verify_dataset(args.output_root / "data" / "MassSpecGym", config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices)
                progress["audit"] = data_report["target_audit"]
            elif stage["name"] == "alignment42":
                progress["audit"] = audit_alignment(args)
            else:
                rerank = json.loads((args.output_root / "rerank_topk256" / "status.json").read_text())
                if rerank["state"] != "complete" or len(rerank["stages"]) != 30:
                    raise ValueError("Incomplete rerank matrix")
            progress.update(state="complete", completed_at=now())
            write_json(args.output_root / "status.json", status)
        status.update(state="complete", completed_at=now(), result_identity_audit="pending", paper_update="pending")
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
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0 or parse_cuda_device(args.device) != args.gpu:
        parser.error("Requires matching physical GPU N and explicit cuda:N")
    for name in ("source_dir", "legacy_tsv", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
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
