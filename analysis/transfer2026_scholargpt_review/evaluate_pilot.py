"""Save query-level reranker predictions and paired pilot analyses.

This utility evaluates an existing full-pool cache without inserting positives.
K summaries are evaluation-only truncations of the same 256-candidate cache; they
must not be read as separately trained K-specific models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.utils.rerank import load_reranker, rank_from_scores


def summarize_ranks(ranks: list[int], *, total: int, ks: list[int]) -> dict:
    result = {
        "queries": total,
        "positive_queries": len(ranks),
        "positive_fraction": len(ranks) / max(total, 1),
        "mrr": sum(1.0 / rank for rank in ranks) / max(total, 1),
    }
    for k in ks:
        result[f"recall_at_{k}"] = sum(rank <= k for rank in ranks) / max(total, 1)
    return result


def bootstrap_ci(values: np.ndarray, *, seed: int = 20260823, draws: int = 2000) -> dict:
    if values.size == 0:
        return {"mean": None, "low": None, "high": None, "draws": 0}
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    chunk = 100
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means.append(values[indices].mean(axis=1))
    distribution = np.concatenate(means)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(distribution, 0.025)),
        "high": float(np.quantile(distribution, 0.975)),
        "draws": draws,
        "seed": seed,
    }


def difficulty_summary(rows: list[dict], ks: list[int]) -> dict:
    bins = {
        "base_rank_1": lambda rank: rank == 1,
        "base_rank_2_5": lambda rank: 2 <= rank <= 5,
        "base_rank_6_20": lambda rank: 6 <= rank <= 20,
        "base_rank_21_40": lambda rank: 21 <= rank <= 40,
        "base_rank_41_80": lambda rank: 41 <= rank <= 80,
        "base_rank_81_256": lambda rank: 81 <= rank <= 256,
    }
    output = {}
    for name, predicate in bins.items():
        selected = [row for row in rows if predicate(row["base_rank"])]
        if not selected:
            continue
        output[name] = {
            "queries": len(selected),
            "base": summarize_ranks(
                [row["base_rank"] for row in selected], total=len(selected), ks=ks
            ),
            "rerank": summarize_ranks(
                [row["rerank_rank"] for row in selected], total=len(selected), ks=ks
            ),
        }
    return output


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    dataset = RerankCacheDataset(
        args.cache,
        return_smiles=False,
        require_label=False,
    )
    if args.max_queries is not None:
        dataset.indices = dataset.indices[: args.max_queries]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rerank_collate_fn,
        num_workers=0,
    )
    device = torch.device(args.device)
    model = load_reranker(args.checkpoint, device)
    rows: list[dict] = []
    warmup_batches = 2
    benchmark_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        spec_emb = batch["spec_emb"].to(device)
        candidate_embs = batch["candidate_embs"].to(device)
        base_scores = batch["base_scores"].to(device)
        base_ranks = batch["base_ranks"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        labels = batch["labels"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        rerank_scores = model(spec_emb, candidate_embs, base_scores, base_ranks, candidate_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if batch_index >= warmup_batches:
            benchmark_times.append(elapsed / max(labels.numel(), 1))
        for row, label_tensor in enumerate(labels):
            label = int(label_tensor.item())
            if label < 0:
                continue
            mask = candidate_mask[row]
            base_rank = rank_from_scores(base_scores[row], label, mask)
            rerank_rank = rank_from_scores(rerank_scores[row], label, mask)
            rows.append(
                {
                    "query_index": len(rows),
                    "candidate_count": int(mask.sum().item()),
                    "base_rank": base_rank,
                    "rerank_rank": rerank_rank,
                }
            )
    ks = sorted(set(args.ks))
    base_ranks = [row["base_rank"] for row in rows]
    rerank_ranks = [row["rerank_rank"] for row in rows]
    base_mrr = np.asarray([1.0 / rank for rank in base_ranks], dtype=np.float64)
    rerank_mrr = np.asarray([1.0 / rank for rank in rerank_ranks], dtype=np.float64)
    paired = {
        "recall_at_1_difference": bootstrap_ci(
            np.asarray(
                [int(rerank <= 1) - int(base <= 1) for base, rerank in zip(base_ranks, rerank_ranks)],
                dtype=np.float64,
            )
        ),
        "mrr_difference": bootstrap_ci(rerank_mrr - base_mrr),
    }
    result = {
        "cache": str(Path(args.cache)),
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "candidate_protocol": "full cached candidate pool; no positive forcing; K summaries are evaluation-only",
        "total_queries": len(rows),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "base": summarize_ranks(base_ranks, total=len(rows), ks=ks),
        "rerank": summarize_ranks(rerank_ranks, total=len(rows), ks=ks),
        "paired_bootstrap": paired,
        "difficulty_by_base_rank": difficulty_summary(rows, ks),
        "benchmark": {
            "mean_forward_ms_per_query": (
                1000.0 * float(np.mean(benchmark_times)) if benchmark_times else None
            ),
            "benchmark_batches": len(benchmark_times),
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        },
        "predictions": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 40, 80, 100, 256])
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
