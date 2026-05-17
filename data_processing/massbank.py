import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pickle
import random
from pathlib import Path

import numpy as np
from matchms import Spectrum
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.utils.clean import (
    apply_filters,
    clean_metadata,
    clean_metadata2,
    count_annotations,
    filter_by_precursor_mz,
    is_annotated,
    minimal_processing,
    seperate_spectra_by_ionmode,
)
from src.utils.clean import (
    canonicalize_smiles,
    is_valid_smiles,
)
from train import setup_logging


def filters_massbank(spectra):
    spectra = [apply_filters(s) for s in tqdm(spectra, desc="Apply filters")]
    spectra = [s for s in spectra if s is not None]
    positive, _ = seperate_spectra_by_ionmode(spectra)
    spectra = filter_by_precursor_mz(positive)
    count_annotations(spectra, "10 < precursor_mz < 1000")
    spectra = [clean_metadata(s) for s in tqdm(spectra, desc="Clean metadata")]
    spectra = [clean_metadata2(s) for s in tqdm(spectra, desc="Clean metadata 2")]
    spectra = [canonicalize_smiles(spectrum=s) for s in tqdm(spectra, desc="Canonicalize SMILES")]
    spectra = [minimal_processing(s) for s in tqdm(spectra, desc="Minimal processing")]
    spectra = [s for s in spectra if s is not None]
    count_annotations(spectra, "peak num >= 5")
    spectra = is_annotated(spectra)
    count_annotations(spectra, "annotated")
    return spectra

def create_spectrum(record):
    if not record or 'peaks' not in record or not record['peaks']:
        return None
    
    mz, intensities = zip(*record['peaks'])
    metadata = {k: v for k, v in record.items() if k != 'peaks'}
    if 'precursor_mz' not in metadata.keys():
        metadata['precursor_mz'] = 0.0
    return Spectrum(
        mz=np.array(mz).astype(float),
        intensities=np.array(intensities).astype(float),
        metadata=metadata
    )

def is_valid_record(record):
    return record and \
        'smiles' in record and \
        'peaks' in record and \
        'inchi' in record and \
        is_valid_smiles(record['smiles']) and \
        record['smiles'].upper() not in ['N/A', 'NA'] and \
        record['peaks'] and \
        record['inchi'].upper() not in ['N/A', 'NA']

def parse_msp(file_path):
    """解析原始 MSP 文件"""
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return []

    spectra = []
    logging.info(f"Starting to parse MSP file: {file_path} ...")
    
    # 预先设置 RDKit 静默模式
    try:
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.*')
    except ImportError:
        pass

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        record = {}
        in_peaks = False
        for line in f:
            line = line.strip()
            if not line:
                if is_valid_record(record):
                    spectrum = create_spectrum(record)
                    if spectrum:
                        spectra.append(spectrum)
                record = {}
                in_peaks = False
                continue

            if not in_peaks:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().lower()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    
                    if key == 'precursormz':
                        try:
                            record['precursor_mz'] = float(value)
                        except ValueError:
                            record['precursor_mz'] = 0.0
                    elif key == 'num peaks':
                        record['peaks'] = []
                        in_peaks = True
                    else:
                        record[key] = value
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        record['peaks'].append([mz, intensity])
                    except ValueError:
                        continue

        if is_valid_record(record):
            spectrum = create_spectrum(record)
            if spectrum:
                spectra.append(spectrum)

    logging.info(f"Parsing completed! Total valid data: {len(spectra)}")
    return spectra

def split_and_save(data, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    按 InchiKey 进行分组划分，防止数据泄露。
    """
    if not data:
        logging.error("No data to split.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Grouping records by InchiKey for strict splitting...")
    
    inchikey_groups = {}
    for spectrum in data:
        ik = spectrum.get("inchikey")[:14]
        if ik not in inchikey_groups:
            inchikey_groups[ik] = []
        inchikey_groups[ik].append(spectrum)
        
    full_inchikey_counts = len(set([s.get("inchikey") for s in data]))
    logging.info(f"Total Unique InchiKeys: {full_inchikey_counts}")
    unique_inchikeys = list(inchikey_groups.keys())
    logging.info(f"Total Unique InchiKeys[:14]: {len(unique_inchikeys)}")

    random.seed(seed)
    random.shuffle(unique_inchikeys)
    
    n = len(unique_inchikeys)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_inchikey_set = set(unique_inchikeys[:train_end])
    val_inchikey_set = set(unique_inchikeys[train_end:val_end])
    test_inchikey_set = set(unique_inchikeys[val_end:])

    train_data = []
    val_data = []
    test_data = []
    
    for ik, records in inchikey_groups.items():
        if ik in train_inchikey_set:
            train_data.extend(records)
        elif ik in val_inchikey_set:
            val_data.extend(records)
        else:
            test_data.extend(records)

    logging.info("Split results by Molecule:")
    logging.info(f"  Train: {len(train_data)} records ({len(train_inchikey_set)} unique InchiKeys)")
    logging.info(f"  Val:   {len(val_data)} records ({len(val_inchikey_set)} unique InchiKeys)")
    logging.info(f"  Test:  {len(test_data)} records ({len(test_inchikey_set)} unique InchiKeys)")

    out_folds = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'all': train_data + val_data + test_data
    }

    for out_fold, spectra in out_folds.items():
        out_file = output_dir / f"{out_fold}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(spectra, f)
        logging.info(f"Saved {out_file} with {len(spectra)} spectra.")


def process_massbank():
    base_dir = Path(config.data.raw_path) / "MassBank"
    output_dir = Path(config.data.data_path) / "MassBank"

    spectra = parse_msp(base_dir / "MassBank_NISTformat.msp")
    spectra = filters_massbank(spectra)
    split_and_save(spectra, output_dir)

if __name__ == "__main__":
    setup_logging("data_processing.log")
    process_massbank()
