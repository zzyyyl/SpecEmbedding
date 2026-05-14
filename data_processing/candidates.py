import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm


rdBase.DisableLog("rdApp.*")


DATASET_NAMES = {
    "gnps": "GNPS",
    "massbank": "MassBank",
    "massspecgym": "MassSpecGym",
    "mona": "MoNA",
    "nplib1": "NPLIB1",
}


def canonicalize_smiles(smiles: str) -> str | None:
    if not smiles or smiles.upper() in {"N/A", "NA"}:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_mol(smiles: str):
    if not smiles or smiles.upper() in {"N/A", "NA"}:
        return None
    return Chem.MolFromSmiles(smiles)


def calc_formula(smiles: str) -> str | None:
    mol = get_mol(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def calc_exact_mass(smiles: str) -> float | None:
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol)


def extract_smiles(record) -> str | None:
    if hasattr(record, "get"):
        return record.get("smiles")
    if isinstance(record, dict):
        return record.get("smiles")
    return None


def load_test_smiles(dataset_dir: Path) -> list[str]:
    test_path = dataset_dir / "test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    smiles = []
    seen = set()
    for record in test_data:
        value = extract_smiles(record)
        if value and value not in seen:
            seen.add(value)
            smiles.append(value)

    if not smiles:
        raise ValueError(f"No SMILES found in {test_path}")

    logging.info("Loaded %d unique test SMILES from %s", len(smiles), test_path)
    return smiles


def iter_cid_smiles(cid_smiles_path: Path) -> Iterable[str]:
    with open(cid_smiles_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            yield parts[1]


def append_unique(values: list[str], seen: set[str], value: str):
    if value not in seen:
        seen.add(value)
        values.append(value)


def build_formula_candidates(
    test_smiles: list[str],
    cid_smiles_path: Path,
) -> dict[str, list[str]]:
    formula_to_targets: dict[str, list[str]] = defaultdict(list)
    candidates = {smiles: [] for smiles in test_smiles}
    seen = {smiles: set() for smiles in test_smiles}

    for smiles in test_smiles:
        formula = calc_formula(smiles)
        if formula is None:
            logging.warning("Invalid test SMILES skipped: %s", smiles)
            continue
        formula_to_targets[formula].append(smiles)

    if not formula_to_targets:
        logging.warning("No valid test SMILES found for formula candidate generation.")
        return candidates

    for raw_smiles in tqdm(iter_cid_smiles(cid_smiles_path), desc="Scan CID-SMILES", unit="mol"):
        formula = calc_formula(raw_smiles)
        if formula not in formula_to_targets:
            continue

        canonical = canonicalize_smiles(raw_smiles)
        if canonical is None:
            continue

        for target in formula_to_targets[formula]:
            append_unique(candidates[target], seen[target], canonical)

    return candidates


def build_mass_candidates(
    test_smiles: list[str],
    cid_smiles_path: Path,
    mass_tolerance: float,
) -> dict[str, list[str]]:
    if mass_tolerance <= 0:
        raise ValueError("--mass-tolerance must be positive")

    bin_to_targets: dict[int, list[tuple[str, float]]] = defaultdict(list)
    candidates = {smiles: [] for smiles in test_smiles}
    seen = {smiles: set() for smiles in test_smiles}

    for smiles in test_smiles:
        mass = calc_exact_mass(smiles)
        if mass is None:
            logging.warning("Invalid test SMILES skipped: %s", smiles)
            continue
        bin_to_targets[int(mass / mass_tolerance)].append((smiles, mass))

    if not bin_to_targets:
        logging.warning("No valid test SMILES found for mass candidate generation.")
        return candidates

    for raw_smiles in tqdm(iter_cid_smiles(cid_smiles_path), desc="Scan CID-SMILES", unit="mol"):
        mass = calc_exact_mass(raw_smiles)
        if mass is None:
            continue

        mass_bin = int(mass / mass_tolerance)
        possible_targets = (
            bin_to_targets.get(mass_bin - 1, [])
            + bin_to_targets.get(mass_bin, [])
            + bin_to_targets.get(mass_bin + 1, [])
        )
        if not possible_targets:
            continue

        canonical = canonicalize_smiles(raw_smiles)
        if canonical is None:
            continue

        for target, target_mass in possible_targets:
            if abs(mass - target_mass) <= mass_tolerance:
                append_unique(candidates[target], seen[target], canonical)

    return candidates


def ensure_true_smiles_in_candidates(candidates: dict[str, list[str]]):
    for smiles, values in candidates.items():
        if smiles not in values:
            values.insert(0, smiles)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SMILES candidate dictionaries for test molecules."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["mass", "formula"],
        help="Candidate type: approximate exact mass or identical molecular formula.",
    )
    parser.add_argument(
        "--cid-smiles",
        default="data/candidates/CID-SMILES",
        help="Path to the PubChem CID-SMILES file.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Base directory containing processed datasets.",
    )
    parser.add_argument(
        "--mass-tolerance",
        type=float,
        default=0.01,
        help="Exact-mass tolerance in Dalton for --type mass.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from train import setup_logging

    setup_logging("generate_candidates.log")

    dataset_key = args.dataset.lower()
    if dataset_key not in DATASET_NAMES:
        raise ValueError(f"--dataset must be one of: {', '.join(sorted(DATASET_NAMES))}")

    dataset = DATASET_NAMES[dataset_key]
    dataset_dir = Path(args.processed_dir) / dataset
    cid_smiles_path = Path(args.cid_smiles)

    if not cid_smiles_path.exists():
        raise FileNotFoundError(f"CID-SMILES file not found: {cid_smiles_path}")

    test_smiles = load_test_smiles(dataset_dir)
    logging.info(
        "Generating %s candidates for %s using %s",
        args.type,
        dataset,
        cid_smiles_path,
    )

    if args.type == "formula":
        candidates = build_formula_candidates(test_smiles, cid_smiles_path)
    else:
        candidates = build_mass_candidates(test_smiles, cid_smiles_path, args.mass_tolerance)

    ensure_true_smiles_in_candidates(candidates)

    output_path = dataset_dir / f"candidates_{args.type}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(candidates, f)

    sizes = [len(values) for values in candidates.values()]
    logging.info(
        "Saved %s with %d keys. Candidate size min=%d mean=%.2f max=%d",
        output_path,
        len(candidates),
        min(sizes),
        sum(sizes) / len(sizes),
        max(sizes),
    )


if __name__ == "__main__":
    main()
