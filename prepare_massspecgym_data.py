import os
import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from matchms import Spectrum

from SpecEmbedding.utils.clean import (
    get_ref_query
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_smiles
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

MASSSPECGYM_REPO_ID =  "roman-bushuiev/MassSpecGym"
MASSSPECGYM_FILENAME = "data/MassSpecGym.tsv"

def main():
    parser = argparse.ArgumentParser(description="Prepare MassSpecGym evaluation datasets (Spectrum objects in .npy).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated .npy files.")
    parser.add_argument("--replications", type=int, default=10, help="Number of replicated splits to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    setup_logging()

    np.random.seed(args.seed)
    path_dir = Path(args.save_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    replica_suffix = "-replication-{}.npy"

    # 2. Download and read MassSpecGym data
    logging.info(f"Downloading/Locating {MASSSPECGYM_FILENAME} from {MASSSPECGYM_REPO_ID}...")
    try:
        data_path = hf_hub_download(repo_id=MASSSPECGYM_REPO_ID, filename=MASSSPECGYM_FILENAME, repo_type="dataset")
        df = pd.read_csv(data_path, sep="\t")
        df = df[df['fold'] == 'test'] # use test fold only
    except Exception as e:
        logging.error(f"Failed to load MassSpecGym data: {e}")
        return

    # 3. Process into Spectrum objects
    logging.info("Converting TSV rows to Spectrum objects...")
    instrument2spectra = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ascii=True):
        smiles = row['smiles']
        if not is_valid_smiles(smiles):
            continue

        mzs = np.array([float(x) for x in str(row['mzs']).split(',')], dtype=np.float64)
        ints = np.array([float(x) for x in str(row['intensities']).split(',')], dtype=np.float64)
        pmz = float(row.get('precursor_mz', 0.0))
        instrument_type = str(row.get('instrument_type', 'unknown')).lower()

        s = Spectrum(mz=mzs, intensities=ints, metadata={'smiles': smiles, 'precursor_mz': pmz})

        instrument2spectra["all"].append(s)
        if 'orbitrap' in instrument_type:
             instrument2spectra["Orbitrap"].append(s)
        elif 'qtof' in instrument_type or 'qtof' in instrument_type:
             instrument2spectra["QTOF"].append(s)

    # 4. Generate splits
    for inst, spectra in instrument2spectra.items():
        if inst not in ["Orbitrap", "QTOF", "all"]: continue

        logging.info(f"Splitting for instrument: {inst}")
        smiles_seq = np.array(list(set(s.get('smiles') for s in spectra)))
        logging.info(f"  Total unique SMILES: {len(smiles_seq)}")

        for i in range(args.replications):
            query, reference = get_ref_query(spectra, smiles_seq)
            np.save(path_dir / f"{inst}-query{replica_suffix.format(i + 1)}", query)
            np.save(path_dir / f"{inst}-reference{replica_suffix.format(i + 1)}", reference)

    logging.info("Done!")

if __name__ == "__main__":
    main()
