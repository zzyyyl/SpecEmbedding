"""Aggregate query-level pilot evaluation files without flattening seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_input(value: str) -> tuple[str, str, int, Path]:
    try:
        pool, model, seed, path = value.split("=", 3)
        return pool, model, int(seed), Path(path)
    except ValueError as exc:
        raise SystemExit("--input must use POOL=MODEL=SEED=PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files = []
    for raw in args.input:
        pool, model, seed, path = parse_input(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[(pool, model)].append({"seed": seed, "metrics": payload})
        files.append({"pool": pool, "model": model, "seed": seed, "path": str(path)})

    summary = {"files": files, "groups": {}}
    for (pool, model), entries in sorted(grouped.items()):
        metrics = {}
        for metric in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_20",
            "recall_at_40",
            "recall_at_80",
            "recall_at_100",
            "recall_at_256",
            "mrr",
        ):
            values = np.asarray(
                [entry["metrics"]["rerank"][metric] for entry in entries], dtype=float
            )
            base_values = np.asarray(
                [entry["metrics"]["base"][metric] for entry in entries], dtype=float
            )
            metrics[metric] = {
                "rerank_mean": float(values.mean()),
                "rerank_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "base_mean": float(base_values.mean()),
                "base_sample_sd": float(base_values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
            }
        summary["groups"][f"{pool}/{model}"] = {
            "seeds": [entry["seed"] for entry in entries],
            "metrics": metrics,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
