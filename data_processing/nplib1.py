import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matchms import Spectrum
from matchms.filtering import default_filters
from rdkit import Chem
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.utils.clean import (
    clean_metadata,
    clean_metadata2,
    count_annotations,
    filter_by_precursor_mz,
    is_annotated,
    minimal_processing,
)
from SpecEmbedding.utils.runtime import configure_runtime_cache, setup_logging

SOURCE_FILENAMES = (
    "split.pkl",
    "data_dict.pkl",
    "inchikey_to_smiles.pkl",
    "cand_dict_large.pkl",
)
FOLD_MAP = {"train": "train", "valid": "val", "test": "test"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def two_dimensional_inchikey(value: str) -> str:
    return str(value).split("-", maxsplit=1)[0]


def canonicalize_2d_smiles(value: str) -> str | None:
    """Return a canonical non-isomeric SMILES for the benchmark identity rule."""

    molecule = Chem.MolFromSmiles(str(value)) if value else None
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def load_nplib1_inputs(raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    missing = [name for name in SOURCE_FILENAMES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"NPLIB1 source files missing from {raw_dir}: {', '.join(missing)}"
        )

    inputs = {}
    for filename in SOURCE_FILENAMES:
        path = raw_dir / filename
        with path.open("rb") as handle:
            inputs[filename] = pickle.load(handle)
    return inputs


def build_candidate_mapping(cand_dict_large: dict, ik_to_smiles: dict) -> tuple[dict, dict]:
    """Map supplied InChIKey candidates to SMILES without adding the target.

    Candidate order and duplicates are intentionally preserved here. Downstream
    cache construction may remove duplicate molecular embeddings, but this
    conversion never repairs candidate coverage by inserting the ground truth.
    """

    candidates_smiles = {}
    missing_query_smiles = 0
    missing_candidate_smiles = 0
    invalid_query_smiles = 0
    invalid_candidate_smiles = 0
    exact_positive_sets = 0
    two_d_positive_sets = 0
    mapped_positive_sets = 0
    empty_mapped_sets = 0
    mapped_candidate_count = 0
    source_candidate_count = 0
    mapped_candidate_sizes = []

    for query_inchikey, candidate_inchikeys in cand_dict_large.items():
        source_query_smiles = ik_to_smiles.get(query_inchikey)
        if not source_query_smiles:
            missing_query_smiles += 1
            continue
        query_smiles = canonicalize_2d_smiles(source_query_smiles)
        if query_smiles is None:
            invalid_query_smiles += 1
            continue

        candidate_inchikeys = list(candidate_inchikeys)
        source_candidate_count += len(candidate_inchikeys)
        query_2d = two_dimensional_inchikey(query_inchikey)
        if query_inchikey in candidate_inchikeys:
            exact_positive_sets += 1
        if any(two_dimensional_inchikey(item) == query_2d for item in candidate_inchikeys):
            two_d_positive_sets += 1

        mapped_candidates = []
        for candidate_inchikey in candidate_inchikeys:
            source_candidate_smiles = ik_to_smiles.get(candidate_inchikey)
            if not source_candidate_smiles:
                missing_candidate_smiles += 1
                continue
            candidate_smiles = canonicalize_2d_smiles(source_candidate_smiles)
            if candidate_smiles is None:
                invalid_candidate_smiles += 1
                continue
            mapped_candidates.append(candidate_smiles)

        if not mapped_candidates:
            empty_mapped_sets += 1
        if query_smiles in mapped_candidates:
            mapped_positive_sets += 1
        mapped_candidate_count += len(mapped_candidates)
        mapped_candidate_sizes.append(len(mapped_candidates))

        previous = candidates_smiles.get(query_smiles)
        if previous is not None and previous != mapped_candidates:
            raise ValueError(
                "Multiple NPLIB1 query identities map to the same SMILES with "
                f"different candidate lists: {query_smiles}"
            )
        candidates_smiles[query_smiles] = mapped_candidates

    mapped_sets = len(candidates_smiles)
    summary = {
        "source_candidate_sets": len(cand_dict_large),
        "mapped_candidate_sets": mapped_sets,
        "missing_query_smiles": missing_query_smiles,
        "invalid_query_smiles": invalid_query_smiles,
        "empty_mapped_candidate_sets": empty_mapped_sets,
        "source_candidate_count": source_candidate_count,
        "mapped_candidate_count": mapped_candidate_count,
        "missing_candidate_smiles": missing_candidate_smiles,
        "invalid_candidate_smiles": invalid_candidate_smiles,
        "exact_inchikey_positive_sets": exact_positive_sets,
        "two_dimensional_inchikey_positive_sets": two_d_positive_sets,
        "mapped_2d_smiles_positive_sets": mapped_positive_sets,
        "exact_inchikey_source_coverage": (
            exact_positive_sets / len(cand_dict_large) if cand_dict_large else 0.0
        ),
        "two_dimensional_inchikey_source_coverage": (
            two_d_positive_sets / len(cand_dict_large) if cand_dict_large else 0.0
        ),
        "mapped_2d_smiles_positive_coverage": (
            mapped_positive_sets / mapped_sets if mapped_sets else 0.0
        ),
        "mapped_candidate_size": {
            "min": min(mapped_candidate_sizes, default=0),
            "median": (
                float(np.median(mapped_candidate_sizes)) if mapped_candidate_sizes else 0.0
            ),
            "mean": (
                float(np.mean(mapped_candidate_sizes)) if mapped_candidate_sizes else 0.0
            ),
            "max": max(mapped_candidate_sizes, default=0),
        },
        "candidate_policy": "preserve_supplied_candidates_without_positive_insertion",
        "smiles_identity_policy": "rdkit_canonical_non_isomeric_smiles",
    }
    return candidates_smiles, summary


def spectrum_from_entry(
    info: dict,
    inchikey: str,
    smiles: str,
    *,
    source_smiles: str | None = None,
) -> Spectrum | None:
    """Convert one source entry without mutating the loaded source dictionary."""

    if "ms" not in info:
        return None
    raw_ms = np.asarray(info["ms"])
    if raw_ms.size == 0 or raw_ms.ndim != 2 or raw_ms.shape[1] < 2:
        return None

    intensities = raw_ms[:, 0].astype(float, copy=True)
    mzs = raw_ms[:, 1].astype(float, copy=True)
    order = np.argsort(mzs, kind="stable")
    mzs = mzs[order]
    intensities = intensities[order]

    metadata = {key: value for key, value in info.items() if key != "ms"}
    precursor_type = metadata.pop("Precursor", metadata.get("precursor_type"))
    precursor_mz = metadata.pop("PrecursorMZ", metadata.get("precursor_mz", 0.0))
    metadata.update(
        {
            "precursor_type": precursor_type,
            "precursor_mz": float(precursor_mz),
            "smiles": smiles,
            "source_smiles": source_smiles or smiles,
            "inchikey": inchikey,
            "inchikey_2d": two_dimensional_inchikey(inchikey),
        }
    )
    return Spectrum(mz=mzs, intensities=intensities, metadata=metadata)


def build_fold_spectra(
    split_inchikeys: list,
    data_dict: dict,
    ik_to_smiles: dict,
) -> tuple[list[Spectrum], dict]:
    entries_by_inchikey = {}
    for info in data_dict.values():
        inchikey = info.get("inchikey")
        if inchikey:
            entries_by_inchikey.setdefault(inchikey, []).append(info)

    spectra = []
    missing_data = 0
    missing_smiles = 0
    invalid_spectra = 0
    for inchikey in split_inchikeys:
        entries = entries_by_inchikey.get(inchikey)
        if not entries:
            missing_data += 1
            continue
        source_smiles = ik_to_smiles.get(inchikey)
        if not source_smiles:
            missing_smiles += 1
            continue
        smiles = canonicalize_2d_smiles(source_smiles)
        if smiles is None:
            invalid_spectra += len(entries)
            continue
        for info in entries:
            spectrum = spectrum_from_entry(
                info,
                inchikey,
                smiles,
                source_smiles=source_smiles,
            )
            if spectrum is None:
                invalid_spectra += 1
                continue
            spectra.append(spectrum)

    return spectra, {
        "requested_inchikeys": len(split_inchikeys),
        "raw_spectra": len(spectra),
        "missing_data_inchikeys": missing_data,
        "missing_smiles_inchikeys": missing_smiles,
        "invalid_spectra": invalid_spectra,
    }


def filters_nplib1(spectra):
    spectra = [default_filters(s) for s in tqdm(spectra, desc="Apply filters")]
    spectra = [s for s in spectra if s is not None]
    spectra = filter_by_precursor_mz(spectra)
    count_annotations(spectra, "10 < precursor_mz < 1000")
    spectra = [clean_metadata(s) for s in tqdm(spectra, desc="Clean metadata")]
    spectra = [clean_metadata2(s) for s in tqdm(spectra, desc="Clean metadata 2")]
    spectra = [minimal_processing(s) for s in tqdm(spectra, desc="Minimal processing")]
    spectra = [s for s in spectra if s is not None]
    count_annotations(spectra, "peak num >= 5")
    spectra = is_annotated(spectra)
    count_annotations(spectra, "annotated")
    return spectra


def process_nplib1(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    raw_dir = Path(raw_dir) if raw_dir is not None else Path(config.data.raw_path) / "NPLIB1"
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(config.data.data_path) / "NPLIB1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading NPLIB1 source files from %s", raw_dir)
    inputs = load_nplib1_inputs(raw_dir)
    split = inputs["split.pkl"]
    data_dict = inputs["data_dict.pkl"]
    ik_to_smiles = inputs["inchikey_to_smiles.pkl"]
    cand_dict_large = inputs["cand_dict_large.pkl"]

    candidates_smiles, candidate_summary = build_candidate_mapping(
        cand_dict_large,
        ik_to_smiles,
    )
    candidate_path = output_dir / "candidates_supplied.pkl"
    with candidate_path.open("wb") as handle:
        pickle.dump(candidates_smiles, handle)
    logging.info(
        "Saved %s supplied candidate sets without positive insertion to %s",
        len(candidates_smiles),
        candidate_path,
    )

    fold_summaries = {}
    for source_fold, output_fold in FOLD_MAP.items():
        logging.info("Processing NPLIB1 fold %s -> %s", source_fold, output_fold)
        raw_spectra, fold_summary = build_fold_spectra(
            list(split.get(source_fold, [])),
            data_dict,
            ik_to_smiles,
        )
        filtered_spectra = filters_nplib1(raw_spectra)
        output_path = output_dir / f"{output_fold}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(filtered_spectra, handle)
        fold_summary["filtered_spectra"] = len(filtered_spectra)
        fold_summary["filter_retained_fraction"] = len(filtered_spectra) / max(
            len(raw_spectra), 1
        )
        fold_summary["output_sha256"] = sha256_file(output_path)
        fold_summaries[output_fold] = fold_summary
        logging.info("Saved %s filtered spectra to %s", len(filtered_spectra), output_path)

    source_files = {
        filename: {
            "bytes": (raw_dir / filename).stat().st_size,
            "sha256": sha256_file(raw_dir / filename),
        }
        for filename in SOURCE_FILENAMES
    }
    manifest = {
        "schema_version": 1,
        "dataset": "NPLIB1",
        "source_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_files": source_files,
        "candidate_file": {
            "path": str(candidate_path.resolve()),
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
        },
        "candidate_summary": candidate_summary,
        "folds": fold_summaries,
    }
    manifest_path = output_dir / "nplib1_processing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Saved NPLIB1 processing manifest to %s", manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NPLIB1 splits and supplied candidates without positive insertion."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(Path(config.data.raw_path) / "NPLIB1"),
        help="Directory containing the four official NPLIB1 pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(config.data.data_path) / "NPLIB1"),
        help="Output directory for train/val/test and candidates_supplied.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    configure_runtime_cache()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "nplib1_processing.log")
    process_nplib1(args.raw_dir, output_dir)


if __name__ == "__main__":
    main()
