"""Summarize alignment-by-reranker experiments without hiding seed structure."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
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


def parse_input(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise ValueError("--input must use ALIGNMENT=SUMMARY_CSV") from exc
    return label, Path(path)


def read_rows(inputs: list[str]) -> list[dict]:
    rows = []
    for raw in inputs:
        alignment, path = parse_input(raw)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["alignment"] = alignment
                for field in ("seed", *METRICS):
                    if field in row and row[field] not in {"", None}:
                        row[field] = int(row[field]) if field == "seed" else float(row[field])
                rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["alignment"], row["candidate_type"], row["model_type"], row["ablation"])
        ].append(row)

    summary = []
    for key, group in sorted(grouped.items()):
        alignment, candidate_type, model_type, ablation = key
        output = {
            "alignment": alignment,
            "candidate_type": candidate_type,
            "model_type": model_type,
            "ablation": ablation,
            "num_seeds": len(group),
        }
        for metric in METRICS:
            values = [row[metric] for row in group if metric in row]
            if not values:
                continue
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(output)

    paired = []
    full = [row for row in rows if row.get("ablation") in {"", "full"}]
    by_key = {
        (row["alignment"], row["candidate_type"], row["seed"], row["model_type"]): row
        for row in full
    }
    for alignment, candidate_type, seed in sorted(
        {(row["alignment"], row["candidate_type"], row["seed"]) for row in full}
    ):
        pointwise = by_key.get((alignment, candidate_type, seed, "pointwise"))
        relative = by_key.get((alignment, candidate_type, seed, "relative"))
        if pointwise is None or relative is None:
            continue
        paired.append(
            {
                "alignment": alignment,
                "candidate_type": candidate_type,
                "seed": seed,
                "pointwise_rerank_top1_pct": pointwise["rerank_top1_pct"],
                "relative_rerank_top1_pct": relative["rerank_top1_pct"],
                "delta_relative_minus_pointwise_top1_pct": relative["rerank_top1_pct"]
                - pointwise["rerank_top1_pct"],
                "pointwise_rerank_mrr_raw": pointwise["rerank_mrr_raw"],
                "relative_rerank_mrr_raw": relative["rerank_mrr_raw"],
                "delta_relative_minus_pointwise_mrr_raw": relative["rerank_mrr_raw"]
                - pointwise["rerank_mrr_raw"],
            }
        )
    return summary, paired


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="ALIGNMENT=SUMMARY_CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    rows = read_rows(args.input)
    summary, paired = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "alignment_summary.csv", summary)
    write_csv(args.output_dir / "paired_relative_minus_pointwise.csv", paired)
    manifest = {
        "source_inputs": args.input,
        "row_count": len(rows),
        "alignment_summary_rows": len(summary),
        "paired_rows": len(paired),
        "interpretation": "descriptive seed/alignment summaries; no significance claim",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
