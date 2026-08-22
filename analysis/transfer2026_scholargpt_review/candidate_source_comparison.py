"""Compare local rerank-cache candidate lists with official MassSpecGym JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def compare(cache_path: Path, candidate_json_path: Path, test_smiles: list[str]) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    candidate_json = json.loads(candidate_json_path.read_text(encoding="utf-8"))
    mol_smiles = cache["mol_smiles"]
    exact = 0
    jaccards = []
    overlaps = []
    local_target = 0
    official_target = 0
    local_counts = []
    official_counts = []
    for index, (query, true_smiles) in enumerate(zip(cache["queries"], test_smiles)):
        if query["true_smiles"] != true_smiles:
            raise ValueError(f"Query order mismatch at index {index}")
        local = [mol_smiles[int(value)] for value in query["candidate_indices"]]
        official = candidate_json[true_smiles]
        local_set, official_set = set(local), set(official)
        overlap = len(local_set & official_set)
        overlaps.append(overlap)
        jaccards.append(overlap / len(local_set | official_set))
        exact += local == official
        local_target += true_smiles in local_set
        official_target += true_smiles in official_set
        local_counts.append(len(local))
        official_counts.append(len(official))
    return {
        "cache": str(cache_path),
        "official_candidate_json": candidate_json_path.name,
        "queries": len(test_smiles),
        "exact_ordered_list_matches": exact,
        "exact_ordered_list_match_fraction": exact / len(test_smiles),
        "mean_set_overlap": sum(overlaps) / len(overlaps),
        "mean_set_jaccard": sum(jaccards) / len(jaccards),
        "local_candidate_count_min_mean_max": [min(local_counts), sum(local_counts) / len(local_counts), max(local_counts)],
        "official_candidate_count_min_mean_max": [min(official_counts), sum(official_counts) / len(official_counts), max(official_counts)],
        "local_target_present_queries": local_target,
        "official_target_present_queries": official_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True, metavar="NAME:CACHE:CANDIDATES")
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset_tsv, sep="\t")
    test_smiles = dataset.loc[dataset["fold"] == "test", "smiles"].tolist()
    result = {}
    for spec in args.pool:
        name, cache, candidates = spec.split(":", 2)
        result[name] = compare(Path(cache), Path(candidates), test_smiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
