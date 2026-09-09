"""Fail-closed protocol checks for full-training reranker experiments."""

import hashlib
import logging
import subprocess
import time
from pathlib import Path

from SpecEmbedding.utils.gpu import gpu_snapshot


def validate_baseline_scope(settings):
    if settings.candidate_types != ["mass"] or settings.model_types != ["relative"] or settings.seeds != [42]:
        raise ValueError("Current optimization requires one baseline: mass/relative/seed42; matrix expansion is deferred")


def validate_baseline_completion(status):
    expected = [("prepare", split) for split in ("train", "val", "test")] + [("train", None), ("eval", None)]
    stages = status["stages"]
    actual = [(item["kind"], item.get("split")) for item in stages]
    if (status["state"] != "complete" or actual != expected
            or any(item["state"] != "complete" or item["candidate"] != "mass" for item in stages)
            or any(item.get("model") != "relative" or item.get("seed") != 42 for item in stages[3:])):
        raise ValueError("Incomplete or unexpected rerank baseline")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(cache, *, split, candidate_type, expected_count, topk=256, fingerprints=None):
    meta = cache.get("meta", {})
    required = {
        "dataset_type": "massspecgym", "split": split, "candidate_type": candidate_type,
        "pre_top_k": topk, "limit": 0, "force_include_positive": False,
        "num_input_spectra": expected_count, "num_candidate_mapped_spectra": expected_count,
        "num_selected_spectra": expected_count, "num_queries": expected_count,
        "num_skipped_queries": 0,
    }
    for key, expected in required.items():
        if key not in meta or meta[key] != expected:
            raise ValueError(f"Full-training cache {split}: {key}={meta.get(key)!r}, expected {expected!r}")
    if fingerprints:
        for key, expected in fingerprints.items():
            if meta.get(key) != expected:
                raise ValueError(f"Full-training cache fingerprint mismatch: {key}")
    queries = cache["queries"]
    if len(queries) != expected_count or len(cache["spec_embs"]) != expected_count:
        raise ValueError("Full-training cache has missing/extra queries or spectrum embeddings")
    labeled = 0
    for index, query in enumerate(queries):
        if query["spec_index"] != index:
            raise ValueError("Cache query order changed; index-based validation exclusions are unsafe")
        if any(query.get(key) is not False for key in ("positive_forced_into_candidate_pool", "positive_forced_into_topk")):
            raise ValueError("Forced or unknown positive-inclusion semantics in cache")
        size = len(query["candidate_indices"])
        label = query.get("label")
        if not 0 < size <= topk or (label is not None and not 0 <= int(label) < size):
            raise ValueError("Invalid candidate count/label in full-training cache")
        labeled += label is not None
    return {"queries": len(queries), "eligible_queries": labeled, "unlabeled_queries": len(queries) - labeled}


def validate_training_data(args, train, val, expected_counts):
    if args.max_train_queries is not None or args.train_k != 256:
        raise ValueError("Formal full training forbids query caps and requires train-k=256")
    candidate_type = train.meta["candidate_type"]
    summary = validate_cache(train.cache, split="train", candidate_type=candidate_type,
                             expected_count=expected_counts["train"])
    validate_cache(val.cache, split="val", candidate_type=candidate_type, expected_count=expected_counts["val"],
                   fingerprints={key: train.meta[key] for key in ("checkpoint_sha256", "candidate_sha256")})
    expected_indices = [index for index, query in enumerate(train.queries) if query.get("label") is not None]
    if train.indices != expected_indices or len(train) != summary["eligible_queries"]:
        raise ValueError("Formal training does not include every eligible training query")
    return {**summary, "actual_train_queries": len(train), "train_cache_sha256": sha256_file(train.cache_path),
            "val_cache_sha256": sha256_file(val.cache_path)}


def wait_for_gpu(gpu, settings, *, clock=time.monotonic, sleep=time.sleep):
    """Stage gate; queries only, never reserves GPU memory or stops other processes."""
    since = None
    uuid = None
    while True:
        try:
            state, _ = gpu_snapshot(f"cuda:{gpu}", timeout=10, include_table=False)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            logging.warning("Stage GPU query failed, timer reset: %s", error)
            since = None
        else:
            if uuid is not None and uuid != state["uuid"]:
                raise RuntimeError("Physical GPU identity changed during stage wait")
            uuid = state["uuid"]
            logging.info("Stage GPU %s: util=%s%% free=%s MiB", gpu, state["utilization_gpu_pct"], state["memory_free_mib"])
            if state["utilization_gpu_pct"] <= settings.max_utilization and state["memory_free_mib"] >= settings.min_free_mib:
                since = clock() if since is None else since
                if clock() - since >= settings.hold_seconds:
                    return state
            else:
                since = None
                logging.info("Waiting for stage GPU headroom; continuous timer reset")
        sleep(settings.poll_seconds)
