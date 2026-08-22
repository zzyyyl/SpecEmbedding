"""Evaluate saved rerank checkpoints with the official MassSpecGym candidate order.

The local caches contain the same candidate *sets* as the cached MassSpecGym
1.3.1 retrieval JSON but not always the same order.  This script reconstructs
the official order, recomputes cached-embedding cosine scores and base ranks,
and evaluates the reranker on those ordered lists.  A prior tracked 2D
InChIKey audit verifies that the local lists contain one official positive per
test query; this script also checks that each official list contains the exact
target once.  It does not retrain an alignment model or an external baseline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from SpecEmbedding.data.datasets_rerank import rerank_collate_fn
from SpecEmbedding.utils.rerank import load_reranker, rank_from_scores


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialCandidateDataset(Dataset):
    def __init__(self, cache_path: Path, candidate_json_path: Path):
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.spec_embs = self.cache["spec_embs"]
        self.mol_embs = self.cache["mol_embs"]
        self.mol_smiles = self.cache["mol_smiles"]
        self.lookup = {smiles: index for index, smiles in enumerate(self.mol_smiles)}
        candidate_json = json.loads(candidate_json_path.read_text(encoding="utf-8"))
        self.rows = []
        for query_index, query in enumerate(self.cache["queries"]):
            true_smiles = query["true_smiles"]
            official = candidate_json.get(true_smiles)
            if official is None:
                raise ValueError(f"Official candidate JSON misses query {query_index}")
            if any(smiles not in self.lookup for smiles in official):
                raise ValueError(f"Official candidate set has molecules missing from cache at query {query_index}")
            exact_positions = [position for position, smiles in enumerate(official) if smiles == true_smiles]
            if len(exact_positions) != 1:
                raise ValueError(f"Expected one exact target in official list at query {query_index}: {exact_positions}")
            candidate_indices = torch.tensor([self.lookup[smiles] for smiles in official], dtype=torch.long)
            spec_index = int(query["spec_index"])
            candidate_embs = self.mol_embs[candidate_indices].float()
            spec_emb = self.spec_embs[spec_index].float()
            base_scores = candidate_embs @ spec_emb
            order = torch.argsort(base_scores, descending=True, stable=True)
            base_ranks = torch.empty_like(order)
            base_ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.long)
            self.rows.append(
                {
                    "spec_emb": spec_emb,
                    "candidate_embs": candidate_embs,
                    "candidate_indices": candidate_indices,
                    "base_scores": base_scores,
                    "base_ranks": base_ranks,
                    "label": exact_positions[0],
                    "true_smiles": true_smiles,
                    "positive_in_base_topk": True,
                }
            )
        self.cache_path = cache_path
        self.candidate_json_path = candidate_json_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def summarize(ranks: list[int], total: int) -> dict:
    return {
        "queries": total,
        "positive_queries": len(ranks),
        "positive_fraction": len(ranks) / total,
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "recall_at_20": sum(rank <= 20 for rank in ranks) / total,
        "recall_at_40": sum(rank <= 40 for rank in ranks) / total,
        "recall_at_256": sum(rank <= 256 for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
    }


@torch.no_grad()
def evaluate(pool: str, cache_path: Path, checkpoint_path: Path, candidate_json_path: Path, device: str, batch_size: int) -> dict:
    dataset = OfficialCandidateDataset(cache_path, candidate_json_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=rerank_collate_fn)
    model = load_reranker(checkpoint_path, torch.device(device))
    base_ranks: list[int] = []
    rerank_ranks: list[int] = []
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        scores = model(
            tensors["spec_emb"],
            tensors["candidate_embs"],
            tensors["base_scores"],
            tensors["base_ranks"],
            tensors["candidate_mask"],
        )
        for row, label in enumerate(batch["labels"].tolist()):
            mask = batch["candidate_mask"][row]
            base_ranks.append(rank_from_scores(batch["base_scores"][row], label, mask))
            rerank_ranks.append(rank_from_scores(scores[row], label, mask.to(device)))
    total = len(dataset)
    result = {
        "pool": pool,
        "official_protocol": {
            "package": "massspecgym",
            "version": importlib.metadata.version("massspecgym"),
            "candidate_json": candidate_json_path.name,
            "candidate_json_sha256": sha256(candidate_json_path),
            "candidate_order": "official retrieval candidate JSON",
            "identity_rule": "exact target is unique in each official list; prior tracked 2D-InChIKey audit found no extra positive",
            "scope": "official candidate order with saved alignment embeddings; no retraining",
        },
        "cache": str(cache_path),
        "checkpoint": str(checkpoint_path),
        "base": summarize(base_ranks, total),
        "rerank": summarize(rerank_ranks, total),
    }
    del model, loader, dataset
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True, metavar="NAME:CACHE:CHECKPOINT:CANDIDATES")
    args = parser.parse_args()
    results = {}
    for spec in args.pool:
        name, cache, checkpoint, candidates = spec.split(":", 3)
        results[name] = evaluate(name, Path(cache), Path(checkpoint), Path(candidates), args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
