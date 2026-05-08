import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from tqdm import tqdm
from matchms.importing import load_from_msp

from train import setup_logging
from massbank import split_and_save, filters_massbank
from src.utils.clean import canonicalize_smiles

def process_mona():
    # Define paths
    base_dir = Path("data/raw/MoNA")
    output_dir = Path("data/processed/MoNA")

    if not base_dir.exists():
        logging.error(f"Directory {base_dir} not found.")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading msp files...")

    spectra = list(load_from_msp(base_dir / "MoNA-export-LC-MS-MS_Positive_Mode.msp"))
    spectra = [canonicalize_smiles(spectrum=s) for s in tqdm(spectra, desc="Canonicalize SMILES")]
    spectra = filters_massbank(spectra)
    split_and_save(spectra, output_dir)

if __name__ == "__main__":
    setup_logging("data_processing.log")
    process_mona()
