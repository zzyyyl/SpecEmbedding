"""Evaluate reranker checkpoints with MassSpecGym's retrieval identity rule.

The candidate JSON and 2D InChIKey prefix labels follow the MassSpecGym 1.3.1
retrieval loader.  Saved alignment embeddings are reused to avoid silently
changing the first-stage model; this is a formal candidate/identity evaluator,
not a fresh end-to-end alignment or loader re-encoding run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader, Dataset

from SpecEmbedding.utils.rerank import load_reranker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialPoolDataset(Dataset):
    def __init__(self, pool: str, cache_path: Path, candidate_json_path: Path, identity_audit_path: Path):
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.spec_embs = self.cache["spec_embs"]
        self.mol_embs = self.cache["mol_embs"]
        self.mol_smiles = self.cache["mol_smiles"]
        lookup = {smiles: index for index, smiles in enumerate(self.mol_smiles)}
        candidates = json.loads(candidate_json_path.read_text(encoding="utf-8"))
        identity_audit = json.loads(identity_audit_path.read_text(encoding="utf-8"))[pool]
        if identity_audit["multiple_positive_queries"] != 0 or identity_audit["candidate_identity_collisions"] != 0:
            raise ValueError(f"Identity audit is not single-positive for {pool}")
        if identity_audit["reference_2d_inchikey"]["base"]["positive_fraction"] != 1.0:
            raise ValueError(f"Identity audit does not cover every full-pool query for {pool}")
        self.rows: list[dict] = []
        self.multiple_positive_queries = 0
        self.candidate_identity_collisions = 0
        self.missing_positive_queries = 0
        for query_index, query in enumerate(self.cache["queries"]):
            target = query["true_smiles"]
            official = candidates.get(target)
            if official is None:
                raise ValueError(f"Candidate JSON misses query target at index {query_index}")
            if any(smiles not in lookup for smiles in official):
                raise ValueError(f"Candidate JSON contains an unknown molecule at index {query_index}")
            local = [self.mol_smiles[int(value)] for value in query["candidate_indices"]]
            if set(local) != set(official):
                raise ValueError(f"Candidate set mismatch at index {query_index}")
            # The tracked full-pool identity audit proves that every official
            # list has exactly one 2D-InChIKey positive and no collision.  The
            # exact target therefore gives the same positive mask without
            # regenerating millions of RDKit InChIKeys in this all-seed run.
            positive_indices = [index for index, smiles in enumerate(official) if smiles == target]
            if len(positive_indices) == 0:
                self.missing_positive_queries += 1
            if len(positive_indices) > 1:
                self.multiple_positive_queries += 1
                self.candidate_identity_collisions += len(positive_indices) - 1
            candidate_indices = torch.tensor([lookup[smiles] for smiles in official], dtype=torch.long)
            candidate_embs = self.mol_embs[candidate_indices].float()
            spec_emb = self.spec_embs[int(query["spec_index"])].float()
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
                    "positive_indices": positive_indices,
                    "true_smiles": target,
                    "positive_in_base_topk": bool(positive_indices),
                }
            )
        self.cache_path = cache_path
        self.candidate_json_path = candidate_json_path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_candidates = max(item["candidate_embs"].shape[0] for item in batch)
    embedding_dim = batch[0]["candidate_embs"].shape[-1]
    spec_embs = torch.stack([item["spec_emb"] for item in batch])
    candidate_embs = torch.zeros(batch_size, max_candidates, embedding_dim)
    base_scores = torch.zeros(batch_size, max_candidates)
    base_ranks = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    positive_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    for row, item in enumerate(batch):
        count = item["candidate_embs"].shape[0]
        candidate_embs[row, :count] = item["candidate_embs"]
        base_scores[row, :count] = item["base_scores"]
        base_ranks[row, :count] = item["base_ranks"]
        candidate_mask[row, :count] = True
        positive_mask[row, item["positive_indices"]] = True
    return {
        "spec_emb": spec_embs,
        "candidate_embs": candidate_embs,
        "base_scores": base_scores,
        "base_ranks": base_ranks,
        "candidate_mask": candidate_mask,
        "positive_mask": positive_mask,
    }


def rank_of_any(scores: torch.Tensor, positive_mask: torch.Tensor, candidate_mask: torch.Tensor) -> int | None:
    valid = candidate_mask.nonzero(as_tuple=False).flatten()
    order = valid[torch.argsort(scores[valid], descending=True, stable=True)]
    positive_positions = positive_mask[order].nonzero(as_tuple=False).flatten()
    if positive_positions.numel() == 0:
        return None
    return int(positive_positions[0]) + 1


def summarize(ranks: Iterable[int | None], total: int) -> dict[str, float | int]:
    valid = [rank for rank in ranks if rank is not None]
    denominator = max(total, 1)
    return {
        "queries": total,
        "positive_queries": len(valid),
        "positive_fraction": len(valid) / denominator,
        "recall_at_1": sum(rank <= 1 for rank in valid) / denominator,
        "recall_at_5": sum(rank <= 5 for rank in valid) / denominator,
        "recall_at_20": sum(rank <= 20 for rank in valid) / denominator,
        "recall_at_40": sum(rank <= 40 for rank in valid) / denominator,
        "recall_at_256": sum(rank <= 256 for rank in valid) / denominator,
        "mrr": sum(1.0 / rank for rank in valid) / denominator,
    }


def aggregate(seed_metrics: list[dict]) -> dict:
    names = ["recall_at_1", "recall_at_5", "recall_at_20", "recall_at_40", "mrr"]
    result = {"seeds": len(seed_metrics)}
    for name in names:
        values = [float(metric[name]) for metric in seed_metrics]
        result[name] = {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else 0.0}
    return result


@torch.no_grad()
def evaluate_pool(
    pool: str,
    cache_path: Path,
    candidate_json_path: Path,
    checkpoints: dict[str, Path],
    device: str,
    batch_size: int,
    identity_audit_path: Path,
) -> dict:
    dataset = OfficialPoolDataset(pool, cache_path, candidate_json_path, identity_audit_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    models = {name: load_reranker(path, torch.device(device)) for name, path in checkpoints.items()}
    base_ranks: list[int | None] = []
    model_ranks = {name: [] for name in models}
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        for row in range(tensors["spec_emb"].shape[0]):
            base_ranks.append(
                rank_of_any(tensors["base_scores"][row], tensors["positive_mask"][row], tensors["candidate_mask"][row])
            )
        for name, model in models.items():
            scores = model(
                tensors["spec_emb"],
                tensors["candidate_embs"],
                tensors["base_scores"],
                tensors["base_ranks"],
                tensors["candidate_mask"],
            )
            for row in range(scores.shape[0]):
                model_ranks[name].append(
                    rank_of_any(scores[row], tensors["positive_mask"][row], tensors["candidate_mask"][row])
                )
    total = len(dataset)
    aggregates = {}
    for model_type in ("relative", "pointwise"):
        seed_metrics = [
            summarize(ranks, total)
            for name, ranks in model_ranks.items()
            if name.startswith(f"{model_type}_")
        ]
        if seed_metrics:
            aggregates[model_type] = aggregate(seed_metrics)
    result = {
        "protocol": {
            "package": "massspecgym",
            "version": importlib.metadata.version("massspecgym"),
            "candidate_json": candidate_json_path.name,
            "candidate_json_sha256": sha256(candidate_json_path),
            "identity_rule": "2D InChIKey prefix equality; exact-target labels are equivalent by the tracked full-pool identity audit",
            "tie_breaking": "stable descending score order; candidate JSON order breaks exact ties",
            "scope": "official candidate JSON order and 2D identity-equivalent labels with saved alignment embeddings; no alignment retraining",
            "identity_audit": identity_audit_path.name,
            "multiple_positive_queries": dataset.multiple_positive_queries,
            "candidate_identity_collisions": dataset.candidate_identity_collisions,
            "missing_positive_queries": dataset.missing_positive_queries,
        },
        "base": summarize(base_ranks, total),
        "rerank_seeds": {
            name: summarize(ranks, total) for name, ranks in model_ranks.items()
        },
        "rerank_aggregate": aggregates,
    }
    del models, loader, dataset
    gc.collect()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def parse_model_specs(values: list[str]) -> dict[str, dict[str, tuple[str, Path]]]:
    parsed: dict[str, dict[str, tuple[str, Path]]] = {}
    for value in values:
        pool, seed, model, checkpoint = value.split(":", 3)
        if pool not in {"mass", "formula"} or seed not in {"42", "43", "44"}:
            raise ValueError(f"Expected pool mass/formula and seed 42/43/44: {value}")
        parsed.setdefault(pool, {})[f"{model}_seed{seed}"] = (seed, Path(checkpoint))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True, metavar="NAME:CACHE:CANDIDATES")
    parser.add_argument("--model", action="append", required=True, metavar="POOL:SEED:MODEL:CHECKPOINT")
    args = parser.parse_args()
    pool_specs = {}
    for value in args.pool:
        name, cache, candidates = value.split(":", 2)
        pool_specs[name] = (Path(cache), Path(candidates))
    model_specs = parse_model_specs(args.model)
    result = {}
    for name, (cache, candidates) in pool_specs.items():
        checkpoints = {model: path for model, (_, path) in model_specs[name].items()}
        result[name] = evaluate_pool(name, cache, candidates, checkpoints, args.device, args.batch_size, args.identity_audit)
    result["metadata"] = {
        "dataset": args.dataset.name,
        "protocol": "MassSpecGym 1.3.1 retrieval candidate JSON + 2D InChIKey-equivalent labels",
        "scope": "formal all-seed candidate/order evaluation of saved embeddings; 2D equivalence is established by the tracked full-pool audit; no fresh loader encoding or alignment retraining",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
