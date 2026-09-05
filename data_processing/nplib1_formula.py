"""Formula-conditioned candidate construction from the frozen NPLIB1 molecule library."""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def as_rdkit_molecule(value):
    if isinstance(value, str):
        return Chem.MolFromSmiles(value)
    if value is None or not hasattr(value, "GetAtoms"):
        return None
    return value


def molecule_formula(molecule) -> str | None:
    try:
        return rdMolDescriptors.CalcMolFormula(molecule)
    except (RuntimeError, TypeError, ValueError):
        return None


def canonical_smiles(molecule, *, isomeric: bool) -> str | None:
    try:
        value = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=isomeric,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    return value or None


def build_formula_library(
    mol_dict: dict,
    query_inchikeys: list[str],
    required_inchikeys: set[str],
    *,
    show_progress: bool = False,
) -> tuple[dict, dict]:
    """Index 2D structures by formula without inserting any query into a bucket."""

    query_records = {}
    missing_query_molecules = 0
    invalid_query_molecules = 0
    for inchikey in query_inchikeys:
        molecule = as_rdkit_molecule(mol_dict.get(inchikey))
        if molecule is None:
            missing_query_molecules += 1
            continue
        formula = molecule_formula(molecule)
        smiles_2d = canonical_smiles(molecule, isomeric=False)
        source_smiles = canonical_smiles(molecule, isomeric=True)
        if formula is None or smiles_2d is None or source_smiles is None:
            invalid_query_molecules += 1
            continue
        query_records[inchikey] = {
            "formula": formula,
            "smiles_2d": smiles_2d,
            "source_smiles": source_smiles,
        }

    query_formulas = {record["formula"] for record in query_records.values()}
    formula_to_smiles = defaultdict(list)
    seen_smiles_by_formula = defaultdict(set)
    canonical_by_required_inchikey = {}
    invalid_library_molecules = 0
    library_molecules_in_query_formulas = 0

    library_items = mol_dict.items()
    if show_progress:
        from tqdm import tqdm

        library_items = tqdm(
            library_items,
            total=len(mol_dict),
            desc="Build NPLIB1 formula library",
            unit="mol",
        )

    for inchikey, value in library_items:
        molecule = as_rdkit_molecule(value)
        if molecule is None:
            invalid_library_molecules += 1
            continue
        formula = molecule_formula(molecule)
        if formula is None:
            invalid_library_molecules += 1
            continue
        if formula not in query_formulas:
            continue
        smiles_2d = canonical_smiles(molecule, isomeric=False)
        if smiles_2d is None:
            invalid_library_molecules += 1
            continue
        library_molecules_in_query_formulas += 1
        if inchikey in required_inchikeys:
            canonical_by_required_inchikey[inchikey] = smiles_2d
        if smiles_2d in seen_smiles_by_formula[formula]:
            continue
        seen_smiles_by_formula[formula].add(smiles_2d)
        formula_to_smiles[formula].append(smiles_2d)

    candidate_mapping = {}
    positive_query_inchikeys = 0
    conflicting_query_smiles = 0
    candidate_sizes = []
    for inchikey in query_inchikeys:
        query = query_records.get(inchikey)
        if query is None:
            continue
        candidates = formula_to_smiles.get(query["formula"], [])
        if query["smiles_2d"] in seen_smiles_by_formula[query["formula"]]:
            positive_query_inchikeys += 1
        previous = candidate_mapping.get(query["smiles_2d"])
        if previous is not None and previous is not candidates:
            conflicting_query_smiles += 1
            continue
        candidate_mapping[query["smiles_2d"]] = candidates
        candidate_sizes.append(len(candidates))

    summary = {
        "source_library_molecules": len(mol_dict),
        "query_inchikeys": len(query_inchikeys),
        "mapped_query_inchikeys": len(query_records),
        "mapped_query_smiles": len(candidate_mapping),
        "missing_query_molecules": missing_query_molecules,
        "invalid_query_molecules": invalid_query_molecules,
        "query_formulas": len(query_formulas),
        "library_molecules_in_query_formulas": library_molecules_in_query_formulas,
        "unique_2d_library_smiles_in_query_formulas": sum(
            len(values) for values in seen_smiles_by_formula.values()
        ),
        "invalid_library_molecules": invalid_library_molecules,
        "required_inchikeys": len(required_inchikeys),
        "mapped_required_inchikeys": len(canonical_by_required_inchikey),
        "missing_required_inchikeys": len(
            required_inchikeys - set(canonical_by_required_inchikey)
        ),
        "positive_query_inchikeys": positive_query_inchikeys,
        "natural_positive_coverage": positive_query_inchikeys
        / max(len(query_inchikeys), 1),
        "conflicting_query_smiles": conflicting_query_smiles,
        "candidate_size": {
            "min": min(candidate_sizes, default=0),
            "median": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
            "mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
            "max": max(candidate_sizes, default=0),
        },
        "formula_algorithm": "RDKit CalcMolFormula on the pinned mol_dict molecule",
        "identity_policy": "RDKit canonical non-isomeric SMILES",
        "candidate_order": "first occurrence in pinned mol_dict insertion order",
        "candidate_policy": "fixed-library formula buckets without per-query insertion",
    }
    return {
        "query_records": query_records,
        "formula_to_smiles": dict(formula_to_smiles),
        "candidate_mapping": candidate_mapping,
        "canonical_by_required_inchikey": canonical_by_required_inchikey,
    }, summary
