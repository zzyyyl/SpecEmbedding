"""Evaluate existing top-40 caches under local and 2D-InChIKey identities.

This is an evaluation-only audit: it does not retrain models or claim official
MassSpecGym evaluator equivalence. It uses the saved canonical seed-42
reranker checkpoints and the tracked cache provenance.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import inchi
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.utils.rerank import load_reranker


@lru_cache(maxsize=200_000)
def inchikey_2d(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = inchi.MolToInchiKey(mol)
    return key.split("-", 1)[0] if key else None


def rank_for_keys(order: torch.Tensor, candidate_keys: list[str | None], target_key: str | None) -> int | None:
    for pos, idx in enumerate(order.tolist(), start=1):
        if candidate_keys[idx] is not None and candidate_keys[idx] == target_key:
            return pos
    return None


def summarize(ranks: list[int | None], total: int) -> dict[str, float | int]:
    valid = [rank for rank in ranks if rank is not None]
    return {
        "queries": total,
        "positive_queries": len(valid),
        "positive_fraction": len(valid) / max(total, 1),
        "recall_at_1": sum(rank <= 1 for rank in valid) / max(total, 1),
        "recall_at_5": sum(rank <= 5 for rank in valid) / max(total, 1),
        "recall_at_20": sum(rank <= 20 for rank in valid) / max(total, 1),
        "mrr": sum(1.0 / rank for rank in valid) / max(total, 1),
    }


@torch.no_grad()
def evaluate(cache_path: Path, checkpoint: Path, device: str, batch_size: int) -> dict:
    dataset = RerankCacheDataset(cache_path, return_smiles=True, require_label=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=rerank_collate_fn)
    model = load_reranker(checkpoint, device)
    local_base: list[int | None] = []
    local_rerank: list[int | None] = []
    ref_base: list[int | None] = []
    ref_rerank: list[int | None] = []
    multiple_positive_queries = 0
    reference_candidate_collisions = 0

    for batch in loader:
        spec_emb = batch["spec_emb"].to(device)
        candidate_embs = batch["candidate_embs"].to(device)
        base_scores = batch["base_scores"].to(device)
        base_ranks = batch["base_ranks"].to(device)
        mask = batch["candidate_mask"].to(device)
        rerank_scores = model(spec_emb, candidate_embs, base_scores, base_ranks, mask)
        for row, target_smiles in enumerate(batch["true_smiles"]):
            candidates = batch["candidate_smiles"][row]
            local_target = target_smiles
            local_keys = [smiles for smiles in candidates]
            target_key = inchikey_2d(target_smiles)
            candidate_keys = [inchikey_2d(smiles) for smiles in candidates]
            matching = sum(key is not None and key == target_key for key in candidate_keys)
            if matching > 1:
                multiple_positive_queries += 1
            reference_candidate_collisions += max(matching - 1, 0)
            valid_indices = mask[row].nonzero(as_tuple=False).flatten()
            base_order = valid_indices[torch.argsort(base_scores[row, valid_indices], descending=True, stable=True)]
            rerank_order = valid_indices[torch.argsort(rerank_scores[row, valid_indices], descending=True, stable=True)]
            local_base.append(rank_for_keys(base_order, local_keys, local_target))
            local_rerank.append(rank_for_keys(rerank_order, local_keys, local_target))
            ref_base.append(rank_for_keys(base_order, candidate_keys, target_key))
            ref_rerank.append(rank_for_keys(rerank_order, candidate_keys, target_key))

    total = len(local_base)
    return {
        "cache": str(cache_path),
        "checkpoint": str(checkpoint),
        "total_queries": total,
        "multiple_positive_queries": multiple_positive_queries,
        "candidate_identity_collisions": reference_candidate_collisions,
        "local_exact_smiles": {
            "base": summarize(local_base, total),
            "rerank": summarize(local_rerank, total),
        },
        "reference_2d_inchikey": {
            "base": summarize(ref_base, total),
            "rerank": summarize(ref_rerank, total),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    specs = {
        "mass": (
            root / "rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_mass_topk40/massspecgym_mass_test.pt",
            root / "checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/mass/transformer/seed42/attempt_001/best_reranker.pth",
        ),
        "formula": (
            root / "rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_formula_topk40/massspecgym_formula_test.pt",
            root / "checkpoints_rerank/a2280d2_massspecgym_nopretrain_valoverlapclean_topk40_multiseed/formula/transformer/seed42/attempt_001/best_reranker.pth",
        ),
    }
    results = {name: evaluate(cache, checkpoint, args.device, args.batch_size) for name, (cache, checkpoint) in specs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
