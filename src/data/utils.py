import random
import numpy as np
import pickle
import logging
from rdkit import Chem
from pathlib import Path

def is_valid_smiles(smiles):
    """校验 SMILES 的合法性"""
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

class DataProvider:
    def __init__(self, dataset_name, base_data_dir="data/processed"):
        self.base_data_dir = Path(base_data_dir)
        self.dataset_name = dataset_name
        self.data_dir = self.base_data_dir / dataset_name

    def load_data(self, mode):
        """
        加载数据
        mode: 'train', 'val', 'test'
        """
        file_path = self.data_dir / f"{mode}.pkl"

        if not file_path.exists():
            # WORKAROUND
            if mode == 'val' and self.dataset_name == "GNPS":
                file_path = self.data_dir / "test.pkl"
            
            if not file_path.exists():
                logging.error(f"{self.dataset_name} data not found at {file_path}")
                return []

        logging.info(f"Loading {self.dataset_name} {mode} data from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} records.")
        return data

    def load_candidates(self, type=None):
        """候选集加载"""
        if type:
            file_path = self.data_dir / f"candidates_{type}.pkl"
        else:
            file_path = self.data_dir / "candidates.pkl"

        if not file_path.exists():
            logging.warning(f"{self.dataset_name} candidates not found at {file_path}")
            return {}

        logging.info(f"Loading candidates from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} candidate sets.")
        return data

class MZBatchSampler:
    """按前体离子质量 (m/z) 进行分组取样的 BatchSampler，实现 Hard Negative Mining"""
    def __init__(self, data_list, batch_size, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 1. 提取所有记录的 m/z 并排序，记录原始索引
        mzs = []
        for i, item in enumerate(data_list):
            mzs.append((item.get('precursor_mz', 0.0), i))
        
        # 按 m/z 升序排列
        mzs.sort(key=lambda x: x[0])
        self.sorted_indices = [x[1] for x in mzs]
        
    def __iter__(self):
        # 2. 将排序后的索引分块
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        
        # 3. 如果需要打乱，打乱的是 Batch 的顺序，而不是 Batch 内部的顺序
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.sorted_indices) // self.batch_size
