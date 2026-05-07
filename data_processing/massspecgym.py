import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from rdkit import Chem
from matchms import Spectrum

from train import setup_logging

def is_valid_smiles(smiles):
    """校验 SMILES 的合法性"""
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def process_massspecgym():
    output_dir = Path("data/processed/MassSpecGym")
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "roman-bushuiev/MassSpecGym"
    candidates_filenames = {
        "candidates_mass": "data/molecules/MassSpecGym_retrieval_candidates_mass.json",
        "candidates_formula": "data/molecules/MassSpecGym_retrieval_candidates_formula.json"
    }
    spectra_filename = "data/MassSpecGym.tsv"

    for save_filename, filename in candidates_filenames.items():
        logging.info(f"Downloading MassSpecGym candidates: {filename}...")
        cand_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        with open(cand_path, 'r') as f:
            candidates = json.load(f)

        out_file = output_dir / f"{save_filename}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(candidates, f)

    logging.info(f"Downloading MassSpecGym dataset...")
    data_path = hf_hub_download(repo_id=repo_id, filename=spectra_filename, repo_type="dataset")
    raw_data = pd.read_csv(data_path, sep="\t")

    parsed_results = {
        'train': [],
        'val': [],
        'test': [],
    }

    for _, row in tqdm(raw_data.iterrows(), desc="Processing spectra"):
        fold = row['fold']
        smiles = row['smiles']

        if not is_valid_smiles(smiles):
            logging.warning(f"invalid SMILES {smiles}, skip {row['identifier']}")
            continue

        # MassSpecGym 的峰数据是逗号分隔的字符串
        mzs = [float(x) for x in str(row['mzs']).split(',')]
        ints = [float(x) for x in str(row['intensities']).split(',')]

        metadata = row.to_dict()
        metadata.pop('mzs', None)
        metadata.pop('intensities', None)

        spectrum = Spectrum(
            mz=np.array(mzs).astype(float),
            intensities=np.array(ints).astype(float),
            metadata=metadata
        )

        parsed_results[fold].append(spectrum)

    for mode, spectra in parsed_results.items():
        out_file = output_dir / f"{mode}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(spectra, f)
        logging.info(f"Saved {out_file} with {len(spectra)} spectra.")

if __name__ == "__main__":
    setup_logging()
    process_massspecgym()
