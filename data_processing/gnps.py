import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matchms.importing import load_from_mgf

from SpecEmbedding.utils.clean import (
    count_annotations,
    is_annotated,
)
from train import setup_logging

def process_gnps():
    # Define paths
    base_dir = Path("data/GNPS")
    output_dir = Path("data")

    if not base_dir.exists():
        logging.error(f"Directory {base_dir} not found.")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading mgf files...")

    orbitrap_test1 = list(load_from_mgf(base_dir / "Orbitrap_test_subset1.mgf"))
    orbitrap_test2 = list(load_from_mgf(base_dir / "Orbitrap_test_subset2.mgf"))
    orbitrap_train = list(load_from_mgf(base_dir / "Orbitrap_train_subset.mgf"))

    orbitrap_test = orbitrap_test1 + orbitrap_test2
    del orbitrap_test1, orbitrap_test2
    orbitrap = orbitrap_test + orbitrap_train
    qtof = list(load_from_mgf(base_dir / "QTOF.mgf"))
    other = list(load_from_mgf(base_dir / "Other.mgf"))

    count_annotations(orbitrap, "gnps orbitrap")
    count_annotations(qtof, "gnps qtof")
    count_annotations(other, "gnps other")

    orbitrap_test = is_annotated(orbitrap_test)
    orbitrap_train = is_annotated(orbitrap_train)
    orbitrap = orbitrap_test + orbitrap_train
    qtof = is_annotated(qtof)
    other = is_annotated(other)

    count_annotations(orbitrap, "gnps orbitrap")
    count_annotations(qtof, "gnps qtof")
    count_annotations(other, "gnps other")

    gnps_all = orbitrap + qtof + other

    out_folds = {
        'train': orbitrap_train,
        'test': orbitrap_test,
        'all': gnps_all
    }

    for out_fold, spectra in out_folds.items():
        out_file = output_dir / f"GNPS_{out_fold}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(spectra, f)
        logging.info(f"Saved {out_file} with {len(spectra)} spectra.")

if __name__ == "__main__":
    setup_logging()
    process_gnps()
