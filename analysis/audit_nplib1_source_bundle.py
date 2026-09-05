#!/usr/bin/env python
"""Audit the pinned NPLIB1 split, spectra, and supplied candidate sources."""

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.nplib1_candidates import unpack_candidate_inchikeys

FOLD_MAP = {"train": "train", "valid": "val", "test": "test"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"NPLIB1 source file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def two_dimensional_inchikey(value: str) -> str:
    return str(value).split("-", maxsplit=1)[0]


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def split_key_coverage(candidate_keys: set[str], split: dict) -> dict:
    return {
        output_fold: {
            "covered_inchikeys": len(candidate_keys.intersection(split[source_fold])),
            "total_inchikeys": len(split[source_fold]),
            "coverage": len(candidate_keys.intersection(split[source_fold]))
            / max(len(split[source_fold]), 1),
        }
        for source_fold, output_fold in FOLD_MAP.items()
    }


def summarize_split(split: dict) -> dict:
    full_keys = {name: set(split[name]) for name in FOLD_MAP}
    two_dimensional_keys = {
        name: {two_dimensional_inchikey(value) for value in split[name]}
        for name in FOLD_MAP
    }
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    return {
        "folds": {
            FOLD_MAP[name]: {
                "entries": len(split[name]),
                "unique_full_inchikeys": len(full_keys[name]),
                "unique_2d_inchikeys": len(two_dimensional_keys[name]),
            }
            for name in FOLD_MAP
        },
        "pairwise_overlap": {
            f"{FOLD_MAP[left]}_vs_{FOLD_MAP[right]}": {
                "shared_full_inchikeys": len(full_keys[left] & full_keys[right]),
                "shared_2d_inchikeys": len(
                    two_dimensional_keys[left] & two_dimensional_keys[right]
                ),
            }
            for left, right in pairs
        },
    }


def summarize_candidate_source(candidate_dict: dict, split: dict) -> dict:
    candidate_sizes = []
    source_formats = Counter()
    exact_positive_sets = 0
    two_dimensional_positive_sets = 0
    candidate_entry_count = 0

    for query_inchikey, candidate_value in candidate_dict.items():
        candidate_inchikeys, source_format = unpack_candidate_inchikeys(
            candidate_value
        )
        source_formats[source_format] += 1
        candidate_sizes.append(len(candidate_inchikeys))
        candidate_entry_count += len(candidate_inchikeys)
        exact_positive_sets += query_inchikey in candidate_inchikeys
        query_2d = two_dimensional_inchikey(query_inchikey)
        two_dimensional_positive_sets += any(
            two_dimensional_inchikey(candidate) == query_2d
            for candidate in candidate_inchikeys
        )

    candidate_keys = set(candidate_dict)
    official_split_keys = set().union(*(set(split[name]) for name in FOLD_MAP))
    candidate_set_count = len(candidate_dict)
    return {
        "candidate_sets": candidate_set_count,
        "candidate_entries": candidate_entry_count,
        "source_formats": dict(sorted(source_formats.items())),
        "candidate_size_quantiles": {
            str(fraction): round(quantile(candidate_sizes, fraction), 6)
            for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
        "exact_inchikey_positive_sets": exact_positive_sets,
        "exact_inchikey_positive_coverage": exact_positive_sets
        / max(candidate_set_count, 1),
        "two_dimensional_inchikey_positive_sets": two_dimensional_positive_sets,
        "two_dimensional_inchikey_positive_coverage": (
            two_dimensional_positive_sets / max(candidate_set_count, 1)
        ),
        "query_key_coverage_by_split": split_key_coverage(candidate_keys, split),
        "query_keys_outside_official_split": len(candidate_keys - official_split_keys),
        "candidate_policy": "audit_supplied_membership_without_positive_insertion",
    }


def summarize_spectra_source(data_dict: dict, split: dict) -> dict:
    spectra_by_identity = Counter(
        info.get("inchikey")
        for info in data_dict.values()
        if isinstance(info, dict) and info.get("inchikey")
    )
    official_split_keys = set().union(*(set(split[name]) for name in FOLD_MAP))
    return {
        "source_entries": len(data_dict),
        "entries_with_inchikey": sum(spectra_by_identity.values()),
        "unique_inchikeys": len(spectra_by_identity),
        "spectra_by_split": {
            output_fold: sum(spectra_by_identity[key] for key in split[source_fold])
            for source_fold, output_fold in FOLD_MAP.items()
        },
        "split_inchikeys_without_spectra": {
            output_fold: sum(key not in spectra_by_identity for key in split[source_fold])
            for source_fold, output_fold in FOLD_MAP.items()
        },
        "spectra_outside_official_split": sum(
            count
            for key, count in spectra_by_identity.items()
            if key not in official_split_keys
        ),
    }


def audit_source_bundle(raw_dir: Path) -> dict:
    split = load_pickle(raw_dir / "split.pkl")
    data_dict = load_pickle(raw_dir / "data_dict.pkl")
    train_candidates = load_pickle(raw_dir / "cand_dict_train_updated.pkl")
    test_candidates = load_pickle(raw_dir / "cand_dict_large.pkl")
    acquisition_manifest = raw_dir / "nplib1_acquisition_manifest.json"

    if not isinstance(split, dict):
        raise TypeError("Expected split.pkl to contain a dictionary")
    for source_fold in FOLD_MAP:
        if source_fold not in split:
            raise KeyError(f"Missing NPLIB1 split key: {source_fold}")
    if not isinstance(data_dict, dict):
        raise TypeError("Expected data_dict.pkl to contain a dictionary")
    if not isinstance(train_candidates, dict) or not isinstance(test_candidates, dict):
        raise TypeError("Expected NPLIB1 candidate sources to contain dictionaries")
    if not acquisition_manifest.is_file():
        raise FileNotFoundError(
            f"Verified acquisition manifest not found: {acquisition_manifest}"
        )

    train_summary = summarize_candidate_source(train_candidates, split)
    test_summary = summarize_candidate_source(test_candidates, split)
    return {
        "schema_version": 1,
        "dataset": "NPLIB1",
        "identity_policy": "full and two-dimensional InChIKey audits",
        "split_summary": summarize_split(split),
        "split_inchikeys": {
            output_fold: len(split[source_fold])
            for source_fold, output_fold in FOLD_MAP.items()
        },
        "spectra_source": summarize_spectra_source(data_dict, split),
        "candidate_sources": {
            "cand_dict_train_updated.pkl": train_summary,
            "cand_dict_large.pkl": test_summary,
            "query_key_overlap_between_files": len(
                set(train_candidates).intersection(test_candidates)
            ),
        },
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NPLIB1 source split and supplied candidate coverage."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_source_bundle(args.raw_dir.expanduser().resolve())
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split_inchikeys": report["split_inchikeys"],
                "split_summary": report["split_summary"],
                "candidate_sources": report["candidate_sources"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
