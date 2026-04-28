import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matchms import Spectrum
from matchms.filtering import (
    default_filters,
)

from train import setup_logging

from SpecEmbedding.utils.clean import (
    apply_filters,
    filter_by_precursor_mz,
    clean_metadata,
    clean_metadata2,
    minimal_processing,
    is_annotated,
    count_annotations,
)

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

def process_nplib1():
    # Define paths
    base_dir = Path("data/NPLIB1")
    output_dir = Path("data")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        logging.error(f"Directory {base_dir} not found.")
        return

    logging.info("Loading original pickle files...")
    # 1. Load data
    try:
        with open(base_dir / "split.pkl", "rb") as f:
            split = pickle.load(f)
        with open(base_dir / "data_dict.pkl", "rb") as f:
            data_dict = pickle.load(f)
        with open(base_dir / "inchikey_to_smiles.pkl", "rb") as f:
            ik_to_smiles = pickle.load(f)
        with open(base_dir / "cand_dict_large.pkl", "rb") as f:
            cand_dict_large = pickle.load(f)
    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}")
        return

    # 2. Build NPLIB1_candidates.pkl
    # Format: { "$smiles": ["$smiles", ...] }
    logging.info("Generating NPLIB1_candidates.pkl...")
    candidates_smiles = {}
    for q_ik, cand_iks in cand_dict_large.items():
        if q_ik not in ik_to_smiles:
            continue

        q_smiles = ik_to_smiles[q_ik]
        c_smiles_list = []
        for c_ik in cand_iks:
            if c_ik in ik_to_smiles:
                c_smiles_list.append(ik_to_smiles[c_ik])

        if c_smiles_list:
            # Ensure ground truth is present
            if q_smiles not in c_smiles_list:
                c_smiles_list.insert(0, q_smiles)
            candidates_smiles[q_smiles] = c_smiles_list

    with open(output_dir / "NPLIB1_candidates.pkl", "wb") as f:
        pickle.dump(candidates_smiles, f)
    logging.info(f"Saved NPLIB1_candidates.pkl with {len(candidates_smiles)} unique SMILES.")

    # 3. Build fold files
    ik_to_data_entries = {}
    for entry_id, info in data_dict.items():
        ik = info.get('inchikey')
        if not ik: continue
        if ik not in ik_to_data_entries:
            ik_to_data_entries[ik] = []
        ik_to_data_entries[ik].append(info)

    fold_map = {"train": "train", "valid": "val", "test": "test"}

    for in_fold, out_fold in fold_map.items():
        logging.info(f"Processing fold: {out_fold}...")
        fold_iks = split.get(in_fold, [])
        spectra = []

        for ik in fold_iks:
            if ik not in ik_to_data_entries:
                logging.warning(f"Unrecognized {ik}, skipped")
                continue

            for info in ik_to_data_entries[ik]:
                smiles = ik_to_smiles.get(ik)
                if not smiles:
                    logging.warning(f"Unrecognized smiles of {ik}, skipped")
                    continue

                # peaks: [int, mz] -> [mz, int]
                # Assuming info['ms'] is a numpy array or list of pairs
                if 'ms' not in info:
                    logging.warning(f"MZ peaks of {ik} not found, skipped")
                    continue

                raw_ms = info.pop('ms')
                if raw_ms.size == 0:
                    logging.warning(f"MZ peaks of {ik} is empty, skipped")
                    continue

                mzs = [float(p[1]) for p in raw_ms]
                ints = [float(p[0]) for p in raw_ms]

                if sorted(mzs) != mzs:
                    mzs.reverse()
                    ints.reverse()

                if sorted(mzs) != mzs:
                    logging.warning(f"MZ peaks of {ik} is not sorted, skipped")
                    continue

                metadata = info
                metadata['precursor_type'] = metadata.pop('Precursor')
                metadata['precursor_mz'] = metadata.pop('PrecursorMZ')
                metadata.update({
                    'smiles': smiles,
                    'inchikey': ik
                })
                spectrum = Spectrum(
                    mz=np.array(mzs).astype(float),
                    intensities=np.array(ints).astype(float),
                    metadata=metadata
                )
                spectra.append(spectrum)

        spectra = filters_nplib1(spectra)
        out_file = output_dir / f"NPLIB1_{out_fold}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(spectra, f)
        logging.info(f"Saved {out_file} with {len(spectra)} spectra.")

if __name__ == "__main__":
    setup_logging()
    process_nplib1()
