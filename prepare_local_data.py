import os
import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
from matchms import Spectrum

from SpecEmbedding.utils.clean import (
    get_ref_query
)
from src.data import MSPProvider

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare Local MSP evaluation datasets (Spectrum objects in .npy).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the local .msp file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated .npy files.")
    parser.add_argument("--replications", type=int, default=10, help="Number of replicated splits to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    setup_logging()

    np.random.seed(args.seed)
    path_dir = Path(args.save_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    replica_suffix = "-replication-{}.npy"

    # 1. Load local test data using MSPProvider
    logging.info(f"Loading local test data from {args.data_path}...")
    provider = MSPProvider(args.data_path)
    test_raw = provider.load_data(mode='test')
    
    if not test_raw:
        logging.error("No test data loaded. Make sure the MSP file exists and was split correctly.")
        return

    # 2. Process into Spectrum objects
    logging.info("Converting records to Spectrum objects...")
    spectra = []
    for item in tqdm(test_raw, desc="Processing", ascii=True):
        mzs = np.array([float(p[0]) for p in item['peaks']], dtype=np.float64)
        ints = np.array([float(p[1]) for p in item['peaks']], dtype=np.float64)
        pmz = float(item.get('precursor_mz', 0.0))
        smiles = item['smiles']
        
        s = Spectrum(mz=mzs, intensities=ints, metadata={'smiles': smiles, 'precursor_mz': pmz})
        spectra.append(s)

    # 3. Generate splits (only for "all" category as local MSP may not have unified instrument tags)
    logging.info(f"Total spectra: {len(spectra)}")
    unique_smiles = np.unique([s.get('smiles') for s in spectra])
    logging.info(f"Total unique SMILES: {len(unique_smiles)}")

    for i in range(args.replications):
        logging.info(f"Generating replication {i+1}/{args.replications}...")
        query, reference = get_ref_query(spectra, unique_smiles)
        
        # Save in the format eval.py expects
        np.save(path_dir / f"all-query{replica_suffix.format(i + 1)}", query)
        np.save(path_dir / f"all-reference{replica_suffix.format(i + 1)}", reference)

    logging.info("Done! Local data prepared for eval.py.")

if __name__ == "__main__":
    main()
