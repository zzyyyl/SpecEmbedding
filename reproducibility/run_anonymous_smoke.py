#!/usr/bin/env python3
"""Run a self-contained synthetic prepare -> train -> eval CPU smoke."""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMILES = ("CC", "CCC", "CO", "CN", "CCO", "CCN")
SPLIT_SMILES = {
    "train": SMILES,
    "val": SMILES[:4],
    "test": SMILES[2:],
}
GENERATED_SUFFIXES = {".ckpt", ".log", ".pkl", ".pt", ".pth"}


def _package_files() -> set[Path]:
    return {
        path.relative_to(REPOSITORY_ROOT)
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
    }


def build_environment(work_dir: Path, config_path: Path) -> dict[str, str]:
    cache_root = work_dir / "runtime-cache"
    cache_paths = {
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath_parts = [str(REPOSITORY_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    environment.update(
        {
            "SPECEMBEDDING_CONFIG": str(config_path),
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            **{key: str(value) for key, value in cache_paths.items()},
        }
    )
    return environment


def write_smoke_config(path: Path, data_root: Path) -> None:
    base = yaml.safe_load((REPOSITORY_ROOT / "params.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    config["general"].update({"seed": 7, "device": "cpu"})
    config["data"]["data_path"] = str(data_root)
    config["data"]["tokenizer"].update(
        {"max_len": 8, "show_progress_bar": False}
    )
    config["model"]["spec_encoder"].update(
        {
            "embedding_dim": 8,
            "n_head": 2,
            "n_layer": 1,
            "dim_feedward": 8,
            "dim_target": 8,
            "feedward_activation": "relu",
        }
    )
    config["model"]["mol_encoder"].update(
        {
            "emb_dim": 8,
            "n_layers": 1,
            "dropout_rate": 0.0,
            "size_feature_dim": 4,
            "norm_type": "layernorm",
            "norm_eps": 1e-5,
        }
    )
    config["model"]["align"].update(
        {"final_dim": 8, "dropout_rate": 0.0, "tau": 0.07}
    )
    config["rerank"]["prepare"].update(
        {
            "spec_batch_size": 4,
            "mol_batch_size": 4,
            "candidate_chunk_size": 8,
            "mol_embedding_storage": "cpu",
            "mol_embedding_dtype": "float32",
            "num_workers": 0,
            "pre_top_k": 3,
            "limit": 0,
        }
    )
    config["rerank"]["train"].update(
        {
            "model_type": "relative",
            "train_k": 3,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "hidden_dim": 8,
            "rank_emb_dim": 2,
            "max_rank": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "relation_dim": 4,
            "pair_chunk_size": 2,
            "relation_top_k": 2,
            "use_rank_embedding": False,
            "shuffle_candidates": False,
            "num_workers": 0,
            "top_k": [1, 3],
            "metric_for_best": "mrr",
        }
    )
    config["rerank"]["eval"].update(
        {
            "batch_size": 2,
            "num_workers": 0,
            "top_k": [1, 3],
            "compute_mces": False,
        }
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def configure_current_process(environment: dict[str, str]) -> None:
    for key in (
        "SPECEMBEDDING_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
    ):
        os.environ[key] = environment[key]
    sys.dont_write_bytecode = True
    repository_text = str(REPOSITORY_ROOT)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)


def write_synthetic_inputs(data_root: Path, candidate_path: Path) -> None:
    import numpy as np
    from matchms import Spectrum

    dataset_dir = data_root / "MassSpecGym"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, smiles_values in SPLIT_SMILES.items():
        spectra = []
        for index, smiles in enumerate(smiles_values):
            offset = float(index) + {"train": 0.0, "val": 0.2, "test": 0.4}[split]
            spectra.append(
                Spectrum(
                    mz=np.array([40.0 + offset, 65.0 + offset, 90.0 + offset]),
                    intensities=np.array([0.4, 1.0, 0.7]),
                    metadata={
                        "smiles": smiles,
                        "precursor_mz": 120.0 + offset,
                    },
                )
            )
        with (dataset_dir / f"{split}.pkl").open("wb") as handle:
            pickle.dump(spectra, handle)

    candidates = {
        smiles: [SMILES[(index + 1) % len(SMILES)], SMILES[(index + 2) % len(SMILES)]]
        for index, smiles in enumerate(SMILES)
    }
    with candidate_path.open("wb") as handle:
        pickle.dump(candidates, handle)


def write_alignment_checkpoint(path: Path) -> None:
    from SpecEmbedding.utils.align import create_align_model

    torch.manual_seed(7)
    model = create_align_model(mol_norm_type="layernorm", mol_norm_eps=1e-5)
    torch.save(model.state_dict(), path)


def run_checked(
    command: list[str],
    *,
    work_dir: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=work_dir,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Smoke command failed:\n"
            + " ".join(command)
            + f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def load_torch_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_cache(path: Path, *, split: str, expected_queries: int) -> None:
    payload = load_torch_payload(path)
    required = {"spec_embs", "mol_embs", "mol_smiles", "queries", "meta"}
    if set(payload) != required:
        raise RuntimeError(f"{split} cache keys are invalid: {sorted(payload)}")

    spec_embs = payload["spec_embs"]
    mol_embs = payload["mol_embs"]
    queries = payload["queries"]
    if tuple(spec_embs.shape) != (expected_queries, 8):
        raise RuntimeError(f"{split} spectrum embedding shape is invalid")
    if mol_embs.ndim != 2 or mol_embs.shape[1] != 8:
        raise RuntimeError(f"{split} molecule embedding shape is invalid")
    if not torch.isfinite(spec_embs).all() or not torch.isfinite(mol_embs).all():
        raise RuntimeError(f"{split} cache contains non-finite embeddings")
    if len(payload["mol_smiles"]) != mol_embs.shape[0] or len(queries) != expected_queries:
        raise RuntimeError(f"{split} cache lengths are inconsistent")

    for query_index, query in enumerate(queries):
        candidate_indices = query["candidate_indices"]
        scores = query["base_scores"]
        ranks = query["base_ranks"]
        label = query["label"]
        if candidate_indices.dtype != torch.long or candidate_indices.numel() != 3:
            raise RuntimeError(f"{split} query {query_index} has invalid candidates")
        if scores.shape != candidate_indices.shape or ranks.shape != candidate_indices.shape:
            raise RuntimeError(f"{split} query {query_index} candidate tensors disagree")
        if int(candidate_indices.min()) < 0 or int(candidate_indices.max()) >= mol_embs.shape[0]:
            raise RuntimeError(f"{split} query {query_index} has an invalid molecule index")
        if label is None or not 0 <= int(label) < candidate_indices.numel():
            raise RuntimeError(f"{split} query {query_index} has an invalid label")
        if not query["positive_in_base_topk"]:
            raise RuntimeError(f"{split} query {query_index} unexpectedly lost its positive")

    meta = payload["meta"]
    if meta["split"] != split or meta["num_queries"] != expected_queries:
        raise RuntimeError(f"{split} cache metadata is inconsistent")
    if bool(meta["force_include_positive"]):
        raise RuntimeError(f"{split} unexpectedly enabled positive forcing")


def validate_checkpoint(model_dir: Path) -> None:
    best_path = model_dir / "best_reranker.pth"
    last_path = model_dir / "last_reranker.pth"
    train_log = model_dir / "train_rerank.log"
    for path in (best_path, last_path, train_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Synthetic training did not create {path.name}")

    checkpoint = load_torch_payload(best_path)
    expected_config = {
        "model_type": "relative",
        "embedding_dim": 8,
        "hidden_dim": 8,
        "n_layers": 1,
        "n_heads": 2,
    }
    for key, expected in expected_config.items():
        if checkpoint["model_config"].get(key) != expected:
            raise RuntimeError(f"Unexpected checkpoint model_config.{key}")
    if checkpoint.get("best_epoch") != 1:
        raise RuntimeError("Synthetic training did not select epoch 1")
    best_metric = float(checkpoint.get("best_metric", float("nan")))
    if not math.isfinite(best_metric) or not 0.0 <= best_metric <= 1.0:
        raise RuntimeError("Synthetic training produced an invalid best metric")
    for name, value in checkpoint["state_dict"].items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"Checkpoint tensor is non-finite: {name}")
    if "Using device: cpu" not in train_log.read_text(encoding="utf-8"):
        raise RuntimeError("Training log does not confirm CPU execution")


def _extract_metric(log: str, pattern: str, label: str) -> float:
    match = re.search(pattern, log)
    if not match:
        raise RuntimeError(f"Synthetic evaluation log is missing {label}")
    return float(match.group(1))


def validate_evaluation(eval_dir: Path, expected_queries: int) -> None:
    log_path = eval_dir / "eval_rerank.log"
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise RuntimeError("Synthetic evaluation did not create eval_rerank.log")
    log = log_path.read_text(encoding="utf-8")
    for expected in (
        f"Total queries: {expected_queries}",
        "Pre-retrieval upper bound: 100.0000%",
        "BASE RESULTS",
        "RERANK RESULTS",
        "MCES calculation skipped.",
        "Using device: cpu",
    ):
        if expected not in log:
            raise RuntimeError(f"Synthetic evaluation log is missing: {expected}")

    base_section, rerank_section = log.split("RERANK RESULTS", maxsplit=1)
    base_top3 = _extract_metric(
        base_section,
        r"Top-3\s+Accuracy\s+:\s+([0-9.]+)%",
        "base Top-3",
    )
    if base_top3 != 100.0:
        raise RuntimeError("Synthetic base Top-3 should be exactly 100%")
    for section_name, section in (("base", base_section), ("rerank", rerank_section)):
        top1 = _extract_metric(
            section,
            r"Top-1\s+Accuracy\s+:\s+([0-9.]+)%",
            f"{section_name} Top-1",
        )
        mrr = _extract_metric(section, r"MRR\s+:\s+([0-9.]+)", f"{section_name} MRR")
        if not 0.0 <= top1 <= 100.0 or not 0.0 <= mrr <= 1.0:
            raise RuntimeError(f"Synthetic {section_name} metrics are outside valid ranges")


def execute_smoke(work_dir: Path) -> dict[str, object]:
    work_dir = work_dir.resolve()
    try:
        work_dir.relative_to(REPOSITORY_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("--work-dir must be outside the extracted source package")

    before_files = _package_files()
    work_dir.mkdir(parents=True, exist_ok=True)
    data_root = work_dir / "data"
    candidate_path = work_dir / "candidates_mass.pkl"
    config_path = work_dir / "smoke_params.yaml"
    align_checkpoint = work_dir / "synthetic_alignment.pth"
    cache_dir = work_dir / "rerank-cache"
    model_dir = work_dir / "rerank-model"
    eval_dir = work_dir / "rerank-eval"
    cache_dir.mkdir(parents=True, exist_ok=True)

    write_smoke_config(config_path, data_root)
    environment = build_environment(work_dir, config_path)
    configure_current_process(environment)
    write_synthetic_inputs(data_root, candidate_path)
    write_alignment_checkpoint(align_checkpoint)

    cache_paths: dict[str, Path] = {}
    for split, smiles_values in SPLIT_SMILES.items():
        cache_path = cache_dir / f"{split}.pt"
        cache_paths[split] = cache_path
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "prepare_rerank_cache.py"),
            "--checkpoint",
            str(align_checkpoint),
            "--dataset_type",
            "massspecgym",
            "--split",
            split,
            "--data_path",
            str(data_root),
            "--save_path",
            str(cache_path),
            "--device",
            "cpu",
            "--candidate_type",
            "mass",
            "--candidate_path",
            str(candidate_path),
            "--pre_top_k",
            "3",
            "--mol_norm_type",
            "layernorm",
            "--mol_norm_eps",
            "0.00001",
        ]
        command.append("--no-force_include_positive")
        run_checked(command, work_dir=work_dir, environment=environment)
        validate_cache(cache_path, split=split, expected_queries=len(smiles_values))
        prepare_log = cache_dir / f"prepare_rerank_cache_{split}.log"
        if "Using device: cpu" not in prepare_log.read_text(encoding="utf-8"):
            raise RuntimeError(f"{split} cache preparation did not confirm CPU execution")

    run_checked(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "train_rerank.py"),
            "--train_cache",
            str(cache_paths["train"]),
            "--val_cache",
            str(cache_paths["val"]),
            "--save_dir",
            str(model_dir),
            "--device",
            "cpu",
            "--model_type",
            "relative",
            "--seed",
            "7",
            "--train-k",
            "3",
            "--no-shuffle-candidates",
        ],
        work_dir=work_dir,
        environment=environment,
    )
    validate_checkpoint(model_dir)

    run_checked(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "eval_rerank.py"),
            "--cache",
            str(cache_paths["test"]),
            "--checkpoint",
            str(model_dir / "best_reranker.pth"),
            "--save_dir",
            str(eval_dir),
            "--device",
            "cpu",
            "--no-mces",
        ],
        work_dir=work_dir,
        environment=environment,
    )
    validate_evaluation(eval_dir, expected_queries=len(SPLIT_SMILES["test"]))

    new_package_files = _package_files() - before_files
    forbidden = sorted(
        path.as_posix()
        for path in new_package_files
        if path.suffix.lower() in GENERATED_SUFFIXES or "__pycache__" in path.parts
    )
    if forbidden:
        raise RuntimeError(f"Smoke wrote runtime artifacts into the source package: {forbidden}")

    return {
        "device": "cpu",
        "prepared_queries": {
            split: len(smiles_values) for split, smiles_values in SPLIT_SMILES.items()
        },
        "reranker": "tiny relative reranker, one epoch",
        "mces": "skipped",
        "reported_metrics_reproduced": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the anonymous supplement's synthetic CPU cold smoke."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional external directory in which to retain generated smoke artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.work_dir:
        result = execute_smoke(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="specembedding-smoke-") as temporary:
            result = execute_smoke(Path(temporary))
    print("Synthetic prepare -> train -> eval CPU smoke passed.")
    print(yaml.safe_dump(result, sort_keys=True).strip())


if __name__ == "__main__":
    main()
