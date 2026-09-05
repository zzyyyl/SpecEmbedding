"""Audit NPLIB1 split integrity and overlap with MassSpecGym."""

import argparse
import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_value(record, key: str):
    if isinstance(record, dict):
        metadata = record.get("metadata", record)
        return metadata.get(key)
    getter = getattr(record, "get", None)
    return getter(key) if getter is not None else None


def canonical_2d_smiles(record) -> str | None:
    smiles = metadata_value(record, "smiles")
    molecule = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def identity_2d(record) -> str | None:
    inchikey_2d = metadata_value(record, "inchikey_2d")
    if inchikey_2d:
        return str(inchikey_2d).split("-", maxsplit=1)[0]
    inchikey = metadata_value(record, "inchikey")
    if inchikey:
        return str(inchikey).split("-", maxsplit=1)[0]
    smiles = canonical_2d_smiles(record)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return None
    return Chem.MolToInchiKey(molecule).split("-", maxsplit=1)[0]


def peak_arrays(record) -> tuple[np.ndarray, np.ndarray] | None:
    peaks = getattr(record, "peaks", None)
    if peaks is not None and hasattr(peaks, "mz") and hasattr(peaks, "intensities"):
        return np.asarray(peaks.mz, dtype=np.float64), np.asarray(
            peaks.intensities,
            dtype=np.float64,
        )
    if not isinstance(record, dict):
        return None
    if "peaks" in record:
        pairs = np.asarray(record["peaks"], dtype=np.float64)
        if pairs.ndim == 2 and pairs.shape[1] >= 2:
            return pairs[:, 0], pairs[:, 1]
    if "mz" in record and "intensities" in record:
        return np.asarray(record["mz"], dtype=np.float64), np.asarray(
            record["intensities"],
            dtype=np.float64,
        )
    return None


def spectrum_signature(record) -> str | None:
    arrays = peak_arrays(record)
    if arrays is None:
        return None
    mzs, intensities = arrays
    if mzs.size == 0 or mzs.shape != intensities.shape:
        return None
    finite = np.isfinite(mzs) & np.isfinite(intensities)
    mzs = mzs[finite]
    intensities = intensities[finite]
    if mzs.size == 0:
        return None
    order = np.argsort(mzs, kind="stable")
    mzs = np.round(mzs[order], decimals=6).astype("<f8", copy=False)
    intensities = intensities[order]
    max_intensity = float(np.max(np.abs(intensities)))
    if max_intensity > 0:
        intensities = intensities / max_intensity
    intensities = np.round(intensities, decimals=6).astype("<f8", copy=False)
    precursor = metadata_value(record, "precursor_mz")
    try:
        precursor_value = round(float(precursor), 6)
    except (TypeError, ValueError):
        precursor_value = float("nan")

    digest = hashlib.sha256()
    digest.update(struct.pack("<d", precursor_value))
    digest.update(struct.pack("<Q", mzs.size))
    digest.update(mzs.tobytes())
    digest.update(intensities.tobytes())
    return digest.hexdigest()


def load_split(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(f"Processed split not found: {path}")
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"Expected a sequence in {path}, got {type(records).__name__}")
    return list(records)


def split_inventory(records: list) -> dict:
    identities = [identity_2d(record) for record in records]
    smiles = [canonical_2d_smiles(record) for record in records]
    signatures = [spectrum_signature(record) for record in records]
    return {
        "records": records,
        "identities": identities,
        "smiles": smiles,
        "signatures": signatures,
        "summary": {
            "spectra": len(records),
            "unique_2d_inchikeys": len({value for value in identities if value}),
            "unique_canonical_2d_smiles": len({value for value in smiles if value}),
            "unique_spectrum_signatures": len({value for value in signatures if value}),
            "missing_2d_identity": sum(value is None for value in identities),
            "missing_canonical_smiles": sum(value is None for value in smiles),
            "missing_spectrum_signature": sum(value is None for value in signatures),
        },
    }


def overlap_summary(left: dict, right: dict) -> dict:
    left_identities = {value for value in left["identities"] if value}
    right_identities = {value for value in right["identities"] if value}
    left_smiles = {value for value in left["smiles"] if value}
    right_smiles = {value for value in right["smiles"] if value}
    left_signatures = {value for value in left["signatures"] if value}
    right_signatures = {value for value in right["signatures"] if value}
    right_identities_by_signature = {}
    for identity, signature in zip(right["identities"], right["signatures"]):
        if signature:
            right_identities_by_signature.setdefault(signature, set()).add(identity)
    different_identity_signature_examples = []
    for left_index, (identity, signature) in enumerate(
        zip(left["identities"], left["signatures"])
    ):
        if (
            signature
            and signature in right_identities_by_signature
            and identity not in right_identities_by_signature[signature]
        ):
            different_identity_signature_examples.append(
                {
                    "left_record_index": left_index,
                    "left_2d_inchikey": identity,
                    "right_2d_inchikeys": sorted(
                        right_identities_by_signature[signature],
                        key=lambda value: "" if value is None else value,
                    ),
                    "spectrum_signature": signature,
                }
            )
    return {
        "shared_2d_inchikeys": len(left_identities & right_identities),
        "left_spectra_with_shared_2d_identity": sum(
            value in right_identities for value in left["identities"] if value
        ),
        "shared_canonical_2d_smiles": len(left_smiles & right_smiles),
        "left_spectra_with_shared_canonical_smiles": sum(
            value in right_smiles for value in left["smiles"] if value
        ),
        "shared_spectrum_signatures": len(left_signatures & right_signatures),
        "left_spectra_with_shared_signature": sum(
            value in right_signatures for value in left["signatures"] if value
        ),
        "left_spectra_with_shared_signature_but_different_2d_identity": (
            len(different_identity_signature_examples)
        ),
        "different_2d_identity_signature_examples": (
            different_identity_signature_examples
        ),
    }


def near_neighbor_summary(query_smiles: list[str], reference_smiles: list[str]) -> dict:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reference_fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in sorted(set(reference_smiles))
    ]
    maxima = []
    for smiles in sorted(set(query_smiles)):
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints)
        maxima.append(max(similarities, default=0.0))
    values = np.asarray(maxima, dtype=float)
    return {
        "fingerprint": "Morgan radius=2, 2048 bits",
        "query_structures": int(values.size),
        "reference_structures": len(reference_fingerprints),
        "max_tanimoto_quantiles": {
            str(quantile): float(np.quantile(values, quantile)) if values.size else None
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
    }


def run_audit(
    nplib1_dir: Path,
    massspecgym_dir: Path,
    *,
    compute_near_neighbors: bool = False,
) -> tuple[dict, list]:
    inventories = {"nplib1": {}, "massspecgym": {}}
    input_files = {"nplib1": {}, "massspecgym": {}}
    for dataset, root in (("nplib1", nplib1_dir), ("massspecgym", massspecgym_dir)):
        for split in SPLITS:
            path = root / f"{split}.pkl"
            inventories[dataset][split] = split_inventory(load_split(path))
            input_files[dataset][split] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    within_nplib1 = {}
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            within_nplib1[f"{left_split}_vs_{right_split}"] = overlap_summary(
                inventories["nplib1"][left_split],
                inventories["nplib1"][right_split],
            )

    cross_dataset = {
        f"nplib1_{nplib_split}_vs_massspecgym_{massspec_split}": overlap_summary(
            inventories["nplib1"][nplib_split],
            inventories["massspecgym"][massspec_split],
        )
        for nplib_split in SPLITS
        for massspec_split in SPLITS
    }

    nplib_test = inventories["nplib1"]["test"]
    massspec_train = inventories["massspecgym"]["train"]
    massspec_train_ids = {value for value in massspec_train["identities"] if value}
    massspec_train_signatures = {value for value in massspec_train["signatures"] if value}
    identity_filtered_test = [
        record
        for record, identity in zip(nplib_test["records"], nplib_test["identities"])
        if identity and identity not in massspec_train_ids
    ]
    strict_filtered_test = [
        record
        for record, identity, signature in zip(
            nplib_test["records"],
            nplib_test["identities"],
            nplib_test["signatures"],
        )
        if identity
        and identity not in massspec_train_ids
        and (signature is None or signature not in massspec_train_signatures)
    ]

    report = {
        "schema_version": 1,
        "identity_policy": "2D InChIKey with canonical non-isomeric SMILES cross-check",
        "spectrum_signature_policy": (
            "SHA-256 of precursor m/z and sorted peaks rounded to 1e-6 after max-intensity normalization"
        ),
        "input_files": input_files,
        "splits": {
            dataset: {
                split: inventories[dataset][split]["summary"] for split in SPLITS
            }
            for dataset in inventories
        },
        "within_nplib1": within_nplib1,
        "cross_dataset": cross_dataset,
        "zero_shot_nplib1_test": {
            "original_spectra": len(nplib_test["records"]),
            "after_massspecgym_train_2d_identity_exclusion": len(identity_filtered_test),
            "after_identity_and_exact_spectrum_exclusion": len(strict_filtered_test),
        },
    }
    if compute_near_neighbors:
        reference_smiles = [value for value in massspec_train["smiles"] if value]
        identity_filtered_inventory = split_inventory(identity_filtered_test)
        report["nplib1_test_vs_massspecgym_train_near_neighbors"] = {
            "all_nplib1_test": near_neighbor_summary(
                [value for value in nplib_test["smiles"] if value],
                reference_smiles,
            ),
            "after_2d_identity_exclusion": near_neighbor_summary(
                [value for value in identity_filtered_inventory["smiles"] if value],
                reference_smiles,
            ),
        }
    return report, strict_filtered_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nplib1-dir", type=Path, required=True)
    parser.add_argument("--massspecgym-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filtered-test-output", type=Path)
    parser.add_argument("--near-neighbors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, filtered_test = run_audit(
        args.nplib1_dir.expanduser().resolve(),
        args.massspecgym_dir.expanduser().resolve(),
        compute_near_neighbors=args.near_neighbors,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.filtered_test_output is not None:
        filtered_output = args.filtered_test_output.expanduser().resolve()
        filtered_output.parent.mkdir(parents=True, exist_ok=True)
        with filtered_output.open("wb") as handle:
            pickle.dump(filtered_test, handle)
    print(json.dumps(report["zero_shot_nplib1_test"], indent=2))


if __name__ == "__main__":
    main()
