"""Summarize evaluation-only candidate-cutoff reranking logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpecEmbedding.utils.rerank import parse_rerank_eval_metrics  # noqa: E402


def parse_cutoff(log_path: Path) -> int:
    pattern = re.compile(r"Evaluation candidate truncation:\s+K=(\d+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"Candidate cutoff not found in {log_path}")


def summarize(root: Path, *, candidate_type: str, model_type: str) -> list[dict]:
    rows = []
    prefix = f"{candidate_type}_{model_type}_k"
    for log_path in sorted(root.glob(f"{prefix}*/eval_rerank.log")):
        metrics = parse_rerank_eval_metrics(log_path)
        cutoff = parse_cutoff(log_path)
        required = (
            "total_queries",
            "upper_bound_pct",
            "base_top1_pct",
            "base_top5_pct",
            "base_top10_pct",
            "base_top20_pct",
            "base_mrr_raw",
            "rerank_top1_pct",
            "rerank_top5_pct",
            "rerank_top10_pct",
            "rerank_top20_pct",
            "rerank_mrr_raw",
        )
        missing = [field for field in required if field not in metrics]
        if missing:
            raise ValueError(f"Missing {missing} in {log_path}")
        rows.append(
            {
                "candidate_type": candidate_type,
                "model_type": model_type,
                "candidate_cutoff": cutoff,
                "total_queries": metrics["total_queries"],
                "upper_bound_pct": metrics["upper_bound_pct"],
                "base_top1_pct": metrics["base_top1_pct"],
                "base_top5_pct": metrics["base_top5_pct"],
                "base_top10_pct": metrics["base_top10_pct"],
                "base_top20_pct": metrics["base_top20_pct"],
                "base_mrr_raw": metrics["base_mrr_raw"],
                "rerank_top1_pct": metrics["rerank_top1_pct"],
                "rerank_top5_pct": metrics["rerank_top5_pct"],
                "rerank_top10_pct": metrics["rerank_top10_pct"],
                "rerank_top20_pct": metrics["rerank_top20_pct"],
                "rerank_mrr_raw": metrics["rerank_mrr_raw"],
                "log_path": str(log_path),
            }
        )
    if not rows:
        raise ValueError(f"No evaluation logs found under {root}")
    return sorted(rows, key=lambda row: int(row["candidate_cutoff"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-type", choices=("mass", "formula"), required=True)
    parser.add_argument("--model-type", choices=("pointwise", "relative"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize(args.root, candidate_type=args.candidate_type, model_type=args.model_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "candidate_protocol": "alignment-42 saved-embedding top-40 cache; evaluation-only truncation; no positive forcing",
        "candidate_type": args.candidate_type,
        "model_type": args.model_type,
        "rows": len(rows),
        "cutoffs": [row["candidate_cutoff"] for row in rows],
        "source_root": str(args.root),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
