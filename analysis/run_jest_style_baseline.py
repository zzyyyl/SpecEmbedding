"""Evaluate a JESTR-style saved-embedding cosine ranking control.

This is a local reimplementation control, not an invocation of the official
JESTR training code or released checkpoint.  It uses the same rerank cache,
candidate order, identity labels, and metric functions as the local reranker.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.utils.rerank import (
    init_ranking_metrics,
    rank_from_scores,
    summarize_ranking_metrics,
    update_ranking_metrics,
)
from SpecEmbedding.utils.runtime import configure_runtime_cache, resolve_device


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.inference_mode()
def evaluate_cache(
    cache: str | Path,
    device: torch.device,
    top_k: list[int],
    batch_size: int,
    max_candidates: int | None = None,
) -> dict:
    dataset = RerankCacheDataset(
        cache,
        max_candidates=max_candidates,
        return_smiles=False,
        require_label=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=rerank_collate_fn,
        num_workers=0,
    )
    metrics = init_ranking_metrics(top_k)
    labeled_queries = 0
    for batch in tqdm(loader, desc="JESTR-style cosine", ascii=True):
        spec_emb = F.normalize(batch["spec_emb"].to(device).float(), dim=-1)
        candidate_embs = F.normalize(batch["candidate_embs"].to(device).float(), dim=-1)
        candidate_scores = torch.einsum("bd,bkd->bk", spec_emb, candidate_embs)
        candidate_mask = batch["candidate_mask"].to(device)
        labels = batch["labels"].to(device)
        for row in range(labels.numel()):
            label = int(labels[row].item())
            if label < 0:
                update_ranking_metrics(metrics, None, top_k)
                continue
            labeled_queries += 1
            rank = rank_from_scores(candidate_scores[row], label, candidate_mask[row])
            update_ranking_metrics(metrics, rank, top_k)

    result = summarize_ranking_metrics(metrics, top_k)
    result["total_queries"] = metrics["total"]
    result["labeled_queries"] = labeled_queries
    result["coverage_upper_bound"] = labeled_queries / max(metrics["total"], 1)
    result["max_candidates"] = max_candidates
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a local JESTR-style cosine ranking control on a rerank cache."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10, 20])
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be greater than 0")
    if any(value <= 0 for value in args.top_k):
        parser.error("--top-k values must be greater than 0")
    return args


def main(argv=None):
    configure_runtime_cache()
    args = parse_args(argv)
    device = resolve_device(args.device)
    result = evaluate_cache(
        args.cache,
        device,
        sorted(set(args.top_k)),
        args.batch_size,
        args.max_candidates,
    )
    payload = {
        "baseline_status": "reimplemented_control",
        "baseline_name": "JESTR-style saved-embedding cosine",
        "source_commit": git_commit(),
        "cache": str(args.cache),
        "cache_meta": RerankCacheDataset(args.cache).meta,
        "evaluation": result,
        "metric_definition": "cosine(normalized saved spectrum embedding, normalized saved candidate embedding)",
        "official_jest_reproduction": False,
        "protocol_note": (
            "Uses the local cache order, local two-dimensional identity labels, and local metrics; "
            "does not claim official JESTR encoder training or checkpoint equivalence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
