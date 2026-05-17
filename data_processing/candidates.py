import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import bisect
import logging
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from SpecEmbedding.config import config

rdBase.DisableLog("rdApp.*")


DATASET_NAMES = {
    "gnps": "GNPS",
    "massbank": "MassBank",
    # "massspecgym": "MassSpecGym",
    "mona": "MoNA",
    "nplib1": "NPLIB1",
}


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


def calc_mol_properties(smiles: str) -> tuple[str, str, float] | None:
    mol = get_mol(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mass = Descriptors.ExactMolWt(mol)
    return canonical, formula, mass


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

    logging.info(f"Loaded {len(smiles)} unique test SMILES from {test_path}")
    return smiles


def iter_cid_smiles(cid_smiles_path: Path) -> Iterable[str]:
    with open(cid_smiles_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            yield parts[1]


def normalize_num_workers(num_workers: int) -> int:
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if num_workers == 0:
        return max(1, min(16, cpu_count() - 1))
    return num_workers


def iter_cid_mol_properties(
    cid_smiles_path: Path,
    num_workers: int,
    chunk_size: int,
    desc: str,
) -> Iterable[tuple[str, str, float] | None]:
    smiles_iter = iter_cid_smiles(cid_smiles_path)
    if num_workers == 1:
        for raw_smiles in tqdm(smiles_iter, desc=desc, unit="mol"):
            yield calc_mol_properties(raw_smiles)
        return

    with Pool(processes=num_workers) as pool:
        yield from tqdm(
            pool.imap(calc_mol_properties, smiles_iter, chunksize=chunk_size),
            desc=desc,
            unit="mol",
        )


def append_unique(values: list[str], seen: set[str], value: str):
    if value not in seen:
        seen.add(value)
        values.append(value)


def source_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_is_valid(cache: dict, cid_smiles_path: Path) -> bool:
    return cache.get("source") == source_metadata(cid_smiles_path)


def load_cache(cache_path: Path, cid_smiles_path: Path, rebuild_cache: bool) -> dict | None:
    if rebuild_cache or not cache_path.exists():
        return None

    logging.info(f"Loading candidate cache from {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict) or not cache_is_valid(cache, cid_smiles_path):
        logging.info(f"Cache {cache_path} is stale and will be rebuilt.")
        return None

    return cache


def save_cache(cache_path: Path, cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    logging.info(f"Saved candidate cache to {cache_path}")


def load_or_build_formula_cache(
    cid_smiles_path: Path,
    cache_dir: Path,
    rebuild_cache: bool,
    num_workers: int,
    chunk_size: int,
) -> dict[str, list[str]]:
    cache_path = cache_dir / "CID-SMILES_formula_cache.pkl"
    cache = load_cache(cache_path, cid_smiles_path, rebuild_cache)
    if cache is not None:
        return cache["formula_to_smiles"]

    logging.info(f"Building formula cache with {num_workers} worker(s).")
    formula_to_smiles: dict[str, list[str]] = defaultdict(list)
    seen_by_formula: dict[str, set[str]] = defaultdict(set)

    for props in iter_cid_mol_properties(
        cid_smiles_path,
        num_workers,
        chunk_size,
        "Build formula cache",
    ):
        if props is None:
            continue

        canonical, formula, _ = props
        append_unique(formula_to_smiles[formula], seen_by_formula[formula], canonical)

    cache = {
        "source": source_metadata(cid_smiles_path),
        "formula_to_smiles": dict(formula_to_smiles),
    }
    save_cache(cache_path, cache)
    return cache["formula_to_smiles"]


def load_or_build_mass_cache(
    cid_smiles_path: Path,
    cache_dir: Path,
    rebuild_cache: bool,
    num_workers: int,
    chunk_size: int,
) -> tuple[list[float], list[str]]:
    cache_path = cache_dir / "CID-SMILES_mass_cache.pkl"
    cache = load_cache(cache_path, cid_smiles_path, rebuild_cache)
    if cache is not None:
        return cache["masses"], cache["smiles"]

    logging.info(f"Building mass cache with {num_workers} worker(s).")
    mass_smiles: list[tuple[float, str]] = []
    seen_smiles = set()

    for props in iter_cid_mol_properties(
        cid_smiles_path,
        num_workers,
        chunk_size,
        "Build mass cache",
    ):
        if props is None:
            continue

        canonical, _, mass = props
        if canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)
        mass_smiles.append((mass, canonical))

    mass_smiles.sort(key=lambda item: item[0])
    masses = [mass for mass, _ in mass_smiles]
    smiles = [smiles for _, smiles in mass_smiles]

    cache = {
        "source": source_metadata(cid_smiles_path),
        "masses": masses,
        "smiles": smiles,
    }
    save_cache(cache_path, cache)
    return masses, smiles


def build_formula_candidates(
    test_smiles: list[str],
    formula_to_smiles: dict[str, list[str]],
) -> dict[str, list[str]]:
    candidates = {smiles: [] for smiles in test_smiles}

    for smiles in tqdm(test_smiles):
        formula = calc_formula(smiles)
        if formula is None:
            logging.warning(f"Invalid test SMILES skipped: {smiles}")
            continue
        candidates[smiles] = list(formula_to_smiles.get(formula, []))

    return candidates


def build_mass_candidates(
    test_smiles: list[str],
    cid_masses: list[float],
    cid_smiles: list[str],
    mass_tolerance: float,
) -> dict[str, list[str]]:
    if mass_tolerance <= 0:
        raise ValueError("--mass-tolerance must be positive")

    candidates = {smiles: [] for smiles in test_smiles}

    for smiles in tqdm(test_smiles):
        mass = calc_exact_mass(smiles)
        if mass is None:
            logging.warning(f"Invalid test SMILES skipped: {smiles}")
            continue
        left = bisect.bisect_left(cid_masses, mass - mass_tolerance)
        right = bisect.bisect_right(cid_masses, mass + mass_tolerance)
        candidates[smiles] = cid_smiles[left:right]

    return candidates


def ensure_true_smiles_in_candidates(candidates: dict[str, list[str]]):
    for smiles, values in candidates.items():
        if smiles not in values:
            values.insert(0, smiles)


def limit_candidate_sizes(
    candidates: dict[str, list[str]],
    max_candidates: int | None,
    seed: int,
):
    if max_candidates is None:
        return
    if max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")

    rng = random.Random(seed)
    for smiles, values in candidates.items():
        if len(values) <= max_candidates:
            continue

        pool = [value for value in values if value != smiles]
        keep_count = max_candidates - 1
        if keep_count <= 0:
            candidates[smiles] = [smiles]
            continue

        sampled = rng.sample(pool, min(keep_count, len(pool)))
        candidates[smiles] = [smiles] + sampled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SMILES candidate dictionaries for test molecules."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_NAMES.keys()),
        nargs="+",
        help="Dataset name(s). Supports '--dataset massbank gnps' or repeated '--dataset'. If omitted, all datasets are processed.",
    )
    parser.add_argument(
        "--type",
        action="append",
        choices=["mass", "formula"],
        nargs="+",
        help="Candidate type(s). Supports '--type mass formula' or repeated '--type'. If omitted, both are processed.",
    )
    parser.add_argument(
        "--cid-smiles",
        default=config.data.cid_smiles_path,
        help="Path to the PubChem CID-SMILES file.",
    )
    parser.add_argument(
        "--processed-dir",
        default=config.data.data_path,
        help="Base directory containing processed datasets.",
    )
    parser.add_argument(
        "--mass-tolerance",
        type=float,
        default=0.01,
        help="Exact-mass tolerance in Dalton for --type mass.",
    )
    parser.add_argument(
        "--cache-dir",
        default=config.data.candidates_path,
        help="Directory for derived CID-SMILES mass/formula caches.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuilding the selected CID-SMILES cache.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers for cache building. Use 0 for auto, 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of SMILES sent to each worker task while building caches.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum number of candidates kept for each test SMILES.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --max-candidates truncates candidate lists.",
    )
    return parser.parse_args()


def save_candidates(
    dataset: str,
    candidate_type: str,
    dataset_dir: Path,
    candidates: dict[str, list[str]],
):
    output_path = dataset_dir / f"candidates_{candidate_type}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(candidates, f)

    sizes = [len(values) for values in candidates.values()]
    logging.info(
        f"Saved {output_path} with {len(candidates)} keys. "
        f"Candidate size min={min(sizes)} "
        f"mean={sum(sizes) / len(sizes):.2f} "
        f"max={max(sizes)}"
    )


def build_and_save_candidates(
    dataset: str,
    candidate_type: str,
    args,
    cid_smiles_path: Path,
    cache_dir: Path,
    num_workers: int,
    cache_store: dict,
):
    dataset_dir = Path(args.processed_dir) / dataset
    test_smiles = load_test_smiles(dataset_dir)
    logging.info(f"Generating {candidate_type} candidates for {dataset} using {cid_smiles_path}")

    if candidate_type == "formula":
        if "formula" not in cache_store:
            cache_store["formula"] = load_or_build_formula_cache(
                cid_smiles_path,
                cache_dir,
                args.rebuild_cache,
                num_workers,
                args.chunk_size,
            )
        candidates = build_formula_candidates(test_smiles, cache_store["formula"])
    else:
        if "mass" not in cache_store:
            cache_store["mass"] = load_or_build_mass_cache(
                cid_smiles_path,
                cache_dir,
                args.rebuild_cache,
                num_workers,
                args.chunk_size,
            )
        cid_masses, cid_smiles = cache_store["mass"]
        candidates = build_mass_candidates(
            test_smiles,
            cid_masses,
            cid_smiles,
            args.mass_tolerance,
        )

    ensure_true_smiles_in_candidates(candidates)
    limit_candidate_sizes(candidates, args.max_candidates, args.seed)
    save_candidates(dataset, candidate_type, dataset_dir, candidates)


def main():
    args = parse_args()

    # Keep this import local so --help and multiprocessing workers do not load training dependencies.
    from train import setup_logging

    setup_logging("generate_candidates.log")

    cid_smiles_path = Path(args.cid_smiles)
    cache_dir = Path(args.cache_dir)
    num_workers = normalize_num_workers(args.num_workers)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    if not cid_smiles_path.exists():
        raise FileNotFoundError(f"CID-SMILES file not found: {cid_smiles_path}")

    dataset_keys = [item for group in args.dataset for item in group] if args.dataset else sorted(DATASET_NAMES.keys())
    type_keys = [item for group in args.type for item in group] if args.type else ["mass", "formula"]
    datasets = [DATASET_NAMES[dataset_key] for dataset_key in dict.fromkeys(dataset_keys)]
    candidate_types = list(dict.fromkeys(type_keys))
    cache_store = {}

    logging.info(
        f"Candidate generation targets: datasets={', '.join(datasets)}, "
        f"types={', '.join(candidate_types)}"
    )

    for candidate_type in candidate_types:
        for dataset in datasets:
            build_and_save_candidates(
                dataset,
                candidate_type,
                args,
                cid_smiles_path,
                cache_dir,
                num_workers,
                cache_store,
            )


if __name__ == "__main__":
    main()
