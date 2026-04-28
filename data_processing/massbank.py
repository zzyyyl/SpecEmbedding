import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import logging
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

def parse_msp(file_path, limit=None):
    """解析原始 MSP 文件"""
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return []

    parsed_results = []
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
                        parsed_results.append(spectrum)
                        if limit and len(parsed_results) >= limit:
                            break
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
        
        if not limit or len(parsed_results) < limit:
            if is_valid_record(record):
                spectrum = create_spectrum(record)
                if spectrum:
                    parsed_results.append(spectrum)

    logging.info(f"Parsing completed! Total valid data: {len(parsed_results)}")
    return parsed_results

def split_and_save(data, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    按 SMILES 进行分组划分（Group Split），防止数据泄露。
    """
    if not data:
        logging.error("No data to split.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Grouping records by SMILES for strict splitting...")
    
    smiles_groups = {}
    for item in data:
        s = item.get("smiles")
        if s not in smiles_groups:
            smiles_groups[s] = []
        smiles_groups[s].append(item)
        
    unique_smiles = list(smiles_groups.keys())
    logging.info(f"Total Unique SMILES: {len(unique_smiles)}")

    random.seed(seed)
    random.shuffle(unique_smiles)
    
    n = len(unique_smiles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_smiles_set = set(unique_smiles[:train_end])
    val_smiles_set = set(unique_smiles[train_end:val_end])
    test_smiles_set = set(unique_smiles[val_end:])

    train_data = []
    val_data = []
    test_data = []
    
    for s, records in smiles_groups.items():
        if s in train_smiles_set:
            train_data.extend(records)
        elif s in val_smiles_set:
            val_data.extend(records)
        else:
            test_data.extend(records)

    logging.info(f"Split results by Molecule:")
    logging.info(f"  Train: {len(train_data)} records ({len(train_smiles_set)} unique SMILES)")
    logging.info(f"  Val:   {len(val_data)} records ({len(val_smiles_set)} unique SMILES)")
    logging.info(f"  Test:  {len(test_data)} records ({len(test_smiles_set)} unique SMILES)")

    paths = {
        'train': output_dir / "MassBank_train.pkl",
        'val': output_dir / "MassBank_val.pkl",
        'test': output_dir / "MassBank_test.pkl"
    }

    try:
        with open(paths['train'], 'wb') as f:
            pickle.dump(train_data, f)
        with open(paths['val'], 'wb') as f:
            pickle.dump(val_data, f)
        with open(paths['test'], 'wb') as f:
            pickle.dump(test_data, f)
        logging.info(f"Saved split data to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save split data: {e}")

def process_massbank():
    base_dir = Path("data/MassBank")
    output_dir = Path("data")

    data = parse_msp(base_dir / "MassBank_NISTformat.msp")
    split_and_save(data, output_dir)

if __name__ == "__main__":
    setup_logging()
    process_massbank()
