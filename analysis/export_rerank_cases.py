"""Export query-level reranker ranking changes for manual case analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.utils.rerank import load_reranker


def ranked_candidates(
    order: torch.Tensor,
    scores: torch.Tensor,
    base_scores: torch.Tensor,
    base_ranks: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_smiles: list[str],
    top_k: int,
) -> list[dict]:
    result = []
    for position in order[:top_k].tolist():
        result.append(
            {
                "candidate_index": int(candidate_indices[position].item()),
                "smiles": candidate_smiles[position],
                "score": float(scores[position].item()),
                "base_score": float(base_scores[position].item()),
                "base_rank": int(base_ranks[position].item()),
            }
        )
    return result


def export_cases(
    cache: Path,
    checkpoint: Path,
    query_indices: list[int],
    device: torch.device,
    top_k: int,
) -> dict:
    dataset = RerankCacheDataset(
        cache,
        return_smiles=True,
        require_label=False,
    )
    available = set(dataset.indices)
    missing = sorted(set(query_indices) - available)
    if missing:
        raise ValueError(f"Query indices are out of range or unavailable: {missing}")

    model = load_reranker(checkpoint, device)
    cases = []
    for query_index in query_indices:
        sample_position = dataset.indices.index(query_index)
        batch = rerank_collate_fn([dataset[sample_position]])
        tensors = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        with torch.inference_mode():
            rerank_scores = model(
                tensors["spec_emb"],
                tensors["candidate_embs"],
                tensors["base_scores"],
                tensors["base_ranks"],
                tensors["candidate_mask"],
            )[0].cpu()

        candidate_mask = batch["candidate_mask"][0]
        valid_positions = candidate_mask.nonzero(as_tuple=False).flatten()
        base_scores = batch["base_scores"][0]
        base_ranks = batch["base_ranks"][0]
        candidate_indices = batch["candidate_indices"][0]
        candidate_smiles = batch["candidate_smiles"][0]
        base_order = valid_positions[torch.argsort(base_scores[valid_positions], descending=True)]
        rerank_order = valid_positions[
            torch.argsort(rerank_scores[valid_positions], descending=True)
        ]
        label = int(batch["labels"][0].item())
        cases.append(
            {
                "query_index": query_index,
                "true_smiles": batch["true_smiles"][0],
                "label_position": None if label < 0 else label,
                "positive_in_base_topk": bool(batch["positive_in_base_topk"][0].item()),
                "candidate_count": int(valid_positions.numel()),
                "base_topk": ranked_candidates(
                    base_order,
                    base_scores,
                    base_scores,
                    base_ranks,
                    candidate_indices,
                    candidate_smiles,
                    top_k,
                ),
                "rerank_topk": ranked_candidates(
                    rerank_order,
                    rerank_scores,
                    base_scores,
                    base_ranks,
                    candidate_indices,
                    candidate_smiles,
                    top_k,
                ),
            }
        )
    return {
        "cache": str(cache),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "top_k": top_k,
        "query_indices": query_indices,
        "cases": cases,
        "interpretation": (
            "Ranking evidence only; no fragment attribution or chemical causal "
            "interpretation is inferred by this exporter."
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export base and reranked candidate lists for selected cache queries."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-indices", type=int, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(index < 0 for index in args.query_indices):
        parser.error("--query-indices must contain non-negative integers")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    result = export_cases(
        args.cache,
        args.checkpoint,
        args.query_indices,
        torch.device(args.device),
        args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
