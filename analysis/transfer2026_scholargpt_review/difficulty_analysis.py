"""Stratify saved seed-42 reranking predictions by query difficulty.

This is a query-level descriptive analysis on the local exact-target-SMILES
cache.  It does not change training or claim official-evaluator equivalence.
The strata cover candidate-pool size, base rank, base-score margin, top-
candidate Morgan similarity, and spectrum peak count.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def morgan(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def metric(rows: list[dict]) -> dict:
    if not rows:
        return {"queries": 0, "base_r1": None, "rerank_r1": None, "delta_r1": None, "base_mrr": None, "rerank_mrr": None, "delta_mrr": None}
    base_r1 = sum(row["base_rank"] <= 1 for row in rows) / len(rows)
    rerank_r1 = sum(row["rerank_rank"] <= 1 for row in rows) / len(rows)
    base_mrr = sum(1.0 / row["base_rank"] for row in rows) / len(rows)
    rerank_mrr = sum(1.0 / row["rerank_rank"] for row in rows) / len(rows)
    return {
        "queries": len(rows),
        "base_r1": base_r1,
        "rerank_r1": rerank_r1,
        "delta_r1": rerank_r1 - base_r1,
        "base_mrr": base_mrr,
        "rerank_mrr": rerank_mrr,
        "delta_mrr": rerank_mrr - base_mrr,
    }


def bins() -> dict[str, list[tuple[str, Callable[[float], bool]]]]:
    return {
        "candidate_count": [
            ("1_63", lambda value: value < 64),
            ("64_127", lambda value: 64 <= value < 128),
            ("128_255", lambda value: 128 <= value < 256),
            ("256", lambda value: value >= 256),
        ],
        "base_rank": [
            ("1", lambda value: value == 1),
            ("2_5", lambda value: 2 <= value <= 5),
            ("6_20", lambda value: 6 <= value <= 20),
            ("21_40", lambda value: 21 <= value <= 40),
            ("41_80", lambda value: 41 <= value <= 80),
            ("81_256", lambda value: 81 <= value <= 256),
        ],
        "top1_morgan": [
            ("lt_0.25", lambda value: value < 0.25),
            ("0.25_0.5", lambda value: 0.25 <= value < 0.5),
            ("ge_0.5", lambda value: value >= 0.5),
        ],
        "base_score_margin": [
            ("lt_0.02", lambda value: value < 0.02),
            ("0.02_0.05", lambda value: 0.02 <= value < 0.05),
            ("ge_0.05", lambda value: value >= 0.05),
        ],
        "peak_count": [
            ("lt_20", lambda value: value < 20),
            ("20_39", lambda value: 20 <= value < 40),
            ("ge_40", lambda value: value >= 40),
        ],
    }


def analyze_pool(cache_path: Path, prediction_path: Path, dataset_tsv: Path) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))["predictions"]
    test = pd.read_csv(dataset_tsv, sep="\t")
    test = test.loc[test["fold"] == "test"].reset_index(drop=True)
    if len(predictions) != len(cache["queries"]) or len(predictions) != len(test):
        raise ValueError("Prediction, cache, and test-fold lengths differ")
    fingerprint_cache = {}
    rows: list[dict] = []
    for index, (query, prediction, record) in enumerate(zip(cache["queries"], predictions, test.to_dict("records"))):
        if query["true_smiles"] != record["smiles"]:
            raise ValueError(f"Query order mismatch at index {index}")
        candidate_indices = query["candidate_indices"].tolist()
        base_scores = query["base_scores"].float().tolist()
        base_top_order = sorted(range(len(base_scores)), key=lambda position: (-base_scores[position], position))
        target = query["true_smiles"]
        target_fp = fingerprint_cache.setdefault(target, morgan(target))
        top1 = cache["mol_smiles"][candidate_indices[base_top_order[0]]]
        top5 = [cache["mol_smiles"][candidate_indices[position]] for position in base_top_order[:5]]
        top1_fp = fingerprint_cache.setdefault(top1, morgan(top1))
        top1_similarity = DataStructs.TanimotoSimilarity(target_fp, top1_fp)
        top5_similarity = sum(
            DataStructs.TanimotoSimilarity(target_fp, fingerprint_cache.setdefault(smiles, morgan(smiles)))
            for smiles in top5
        ) / len(top5)
        label = int(query["label"])
        margin = base_scores[base_top_order[0]] - base_scores[label]
        rows.append(
            {
                "candidate_count": len(candidate_indices),
                "base_rank": int(prediction["base_rank"]),
                "rerank_rank": int(prediction["rerank_rank"]),
                "top1_morgan": top1_similarity,
                "top5_morgan_mean": top5_similarity,
                "base_score_margin": margin,
                "peak_count": len([value for value in str(record["mzs"]).split(",") if value]),
            }
        )
    output = {"queries": len(rows), "overall": metric(rows), "strata": {}}
    for field, definitions in bins().items():
        output["strata"][field] = {
            name: metric([row for row in rows if predicate(row[field])])
            for name, predicate in definitions
        }
    output["notes"] = {
        "identity_protocol": "local exact-target-SMILES; official 2D identity is audited separately",
        "morgan": "RDKit Morgan radius 2, 2048 bits",
        "interpretation": "descriptive seed-42 strata; no causal or significance claim",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", action="append", required=True, metavar="NAME:CACHE:PREDICTIONS")
    args = parser.parse_args()
    result = {}
    for spec in args.pool:
        name, cache, predictions = spec.split(":", 2)
        result[name] = analyze_pool(Path(cache), Path(predictions), args.dataset_tsv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
