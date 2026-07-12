import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_rerank_cache(cache_path: str | Path):
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def exclude_query_indices(
    dataset,
    query_indices: list[int],
    *,
    split_name: str = "evaluation",
) -> int:
    excluded = set(query_indices)
    invalid = sorted(index for index in excluded if index < 0 or index >= len(dataset.queries))
    if invalid:
        raise ValueError(f"Excluded {split_name} query indices are out of range: {invalid}")
    original_size = len(dataset.indices)
    dataset.indices = [index for index in dataset.indices if index not in excluded]
    if not dataset.indices:
        raise ValueError(f"Query exclusion removed every {split_name} sample")
    return original_size - len(dataset.indices)


class RerankCacheDataset(Dataset):
    """Dataset backed by a rerank cache generated from a frozen SpecMolAlignModel."""

    def __init__(
        self,
        cache_path: str | Path,
        max_candidates: int | None = None,
        shuffle_candidates: bool = False,
        require_label: bool = False,
        return_smiles: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.cache = load_rerank_cache(self.cache_path)
        self.spec_embs = self.cache["spec_embs"]
        self.mol_embs = self.cache["mol_embs"]
        self.mol_smiles = self.cache["mol_smiles"]
        self.queries = self.cache["queries"]
        self.meta = self.cache.get("meta", {})
        self.max_candidates = max_candidates
        self.shuffle_candidates = shuffle_candidates
        self.return_smiles = return_smiles

        self.indices = list(range(len(self.queries)))
        if require_label:
            self.indices = [idx for idx in self.indices if self.queries[idx].get("label") is not None]

        if not self.indices:
            raise ValueError(f"No valid rerank samples found in {self.cache_path}")

    @property
    def embedding_dim(self) -> int:
        return int(self.spec_embs.shape[-1])

    def __len__(self):
        return len(self.indices)

    def _select_candidates(self, query):
        candidate_indices = query["candidate_indices"].clone().long()
        base_scores = query["base_scores"].clone().float()
        base_ranks = query["base_ranks"].clone().long()
        label = query.get("label")
        label = None if label is None else int(label)

        if self.max_candidates is not None and candidate_indices.numel() > self.max_candidates:
            if label is None:
                selected = torch.arange(self.max_candidates)
                new_label = None
            else:
                if label < self.max_candidates:
                    selected = torch.arange(self.max_candidates)
                    new_label = label
                else:
                    keep = list(range(self.max_candidates - 1)) + [label]
                    selected = torch.tensor(keep, dtype=torch.long)
                    new_label = self.max_candidates - 1

            candidate_indices = candidate_indices[selected]
            base_scores = base_scores[selected]
            base_ranks = base_ranks[selected]
            label = new_label

        if self.shuffle_candidates:
            order = list(range(candidate_indices.numel()))
            random.shuffle(order)
            order_tensor = torch.tensor(order, dtype=torch.long)
            candidate_indices = candidate_indices[order_tensor]
            base_scores = base_scores[order_tensor]
            base_ranks = base_ranks[order_tensor]
            if label is not None:
                label = order.index(label)

        return candidate_indices, base_scores, base_ranks, label

    def __getitem__(self, item):
        query_index = self.indices[item]
        query = self.queries[query_index]
        spec_index = int(query["spec_index"])
        candidate_indices, base_scores, base_ranks, label = self._select_candidates(query)
        candidate_embs = self.mol_embs[candidate_indices].float()

        sample = {
            "spec_emb": self.spec_embs[spec_index].float(),
            "candidate_embs": candidate_embs,
            "candidate_indices": candidate_indices,
            "base_scores": base_scores,
            "base_ranks": base_ranks,
            "label": -1 if label is None else label,
            "true_smiles": query["true_smiles"],
            "positive_in_base_topk": bool(query.get("positive_in_base_topk", False)),
        }

        if self.return_smiles:
            sample["candidate_smiles"] = [self.mol_smiles[int(idx)] for idx in candidate_indices.tolist()]

        return sample


def rerank_collate_fn(batch):
    batch_size = len(batch)
    max_candidates = max(item["candidate_embs"].shape[0] for item in batch)
    embedding_dim = batch[0]["candidate_embs"].shape[-1]

    spec_embs = torch.stack([item["spec_emb"] for item in batch], dim=0)
    candidate_embs = torch.zeros(batch_size, max_candidates, embedding_dim, dtype=torch.float32)
    base_scores = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    base_ranks = torch.zeros(batch_size, max_candidates, dtype=torch.long)
    candidate_indices = torch.full((batch_size, max_candidates), -1, dtype=torch.long)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    for row, item in enumerate(batch):
        num_candidates = item["candidate_embs"].shape[0]
        candidate_embs[row, :num_candidates] = item["candidate_embs"]
        base_scores[row, :num_candidates] = item["base_scores"]
        base_ranks[row, :num_candidates] = item["base_ranks"]
        candidate_indices[row, :num_candidates] = item["candidate_indices"]
        candidate_mask[row, :num_candidates] = True

    result = {
        "spec_emb": spec_embs,
        "candidate_embs": candidate_embs,
        "candidate_indices": candidate_indices,
        "base_scores": base_scores,
        "base_ranks": base_ranks,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "true_smiles": [item["true_smiles"] for item in batch],
        "positive_in_base_topk": torch.tensor(
            [item["positive_in_base_topk"] for item in batch],
            dtype=torch.bool,
        ),
    }

    if "candidate_smiles" in batch[0]:
        result["candidate_smiles"] = [item["candidate_smiles"] for item in batch]

    return result
