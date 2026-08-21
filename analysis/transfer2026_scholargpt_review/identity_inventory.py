"""Inventory 2D-InChIKey multiplicity in existing top-40 test caches."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import inchi


@lru_cache(maxsize=200_000)
def key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    value = inchi.MolToInchiKey(mol)
    return value.split("-", 1)[0] if value else None


def summarize(path: Path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    smiles = cache["mol_smiles"]
    queries = cache["queries"]
    keys = [key(value) for value in smiles]
    total = len(queries)
    local_base_hits = 0
    reference_base_hits = 0
    reference_positive_queries = 0
    multiple_positive_queries = 0
    extra_positive_candidates = 0
    for query in queries:
        target_smiles = query["true_smiles"]
        target_key = key(target_smiles)
        indices = query["candidate_indices"].tolist()
        base_ranks = query["base_ranks"].tolist()
        local_index = None
        reference_indices = []
        for pos, index in enumerate(indices):
            if smiles[index] == target_smiles and local_index is None:
                local_index = pos
            if target_key is not None and keys[index] == target_key:
                reference_indices.append(pos)
        if local_index is not None and local_index < 1:
            local_base_hits += 1
        if reference_indices:
            reference_positive_queries += 1
            if min(reference_indices) < 1:
                reference_base_hits += 1
            if len(reference_indices) > 1:
                multiple_positive_queries += 1
                extra_positive_candidates += len(reference_indices) - 1
    return {
        "cache": str(path),
        "queries": total,
        "local_exact_smiles_base_r1": local_base_hits / max(total, 1),
        "reference_2d_inchikey_base_upper_bound": reference_positive_queries / max(total, 1),
        "reference_2d_inchikey_base_r1": reference_base_hits / max(total, 1),
        "reference_positive_queries": reference_positive_queries,
        "multiple_positive_queries": multiple_positive_queries,
        "extra_positive_candidates": extra_positive_candidates,
        "note": "Base-only identity inventory from saved top-40 cache; not a full official evaluator rerun.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = {
        "mass": root / "rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_mass_topk40/massspecgym_mass_test.pt",
        "formula": root / "rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_formula_topk40/massspecgym_formula_test.pt",
    }
    result = {name: summarize(path) for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
