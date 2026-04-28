import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from pathlib import Path
import logging
from matchms import Spectrum

from train import setup_logging

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
    # Index data_dict by inchikey (handling multiple spectra per InChIKey)
    ik_to_data_entries = {}
    # Based on user description, data_dict is a dict: { "id": { ... } }
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
        processed_data = []
        
        for ik in fold_iks:
            if ik not in ik_to_data_entries:
                logging.warning(f"Unrecognized {ik}, skipped")
                continue
            
            for info in ik_to_data_entries[ik]:
                smiles = ik_to_smiles.get(ik)
                if not smiles:
                    logging.warning(f"Unrecognized smiles of {ik}, skipped")
                    continue
                
                try:
                    precursor_mz = float(info['PrecursorMZ'])
                except (ValueError, TypeError, KeyError):
                    precursor_mz = 0.0
                
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
                
                spectrum = Spectrum(
                    mz=np.array(mzs).astype(float),
                    intensities=np.array(ints).astype(float),
                    metadata={
                        'smiles': smiles,
                        'precursor_mz': precursor_mz,
                        'inchikey': ik
                    }
                )
                processed_data.append(spectrum)
        
        out_file = output_dir / f"NPLIB1_{out_fold}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(processed_data, f)
        logging.info(f"Saved {out_file} with {len(processed_data)} spectra.")

if __name__ == "__main__":
    setup_logging()
    process_nplib1()
