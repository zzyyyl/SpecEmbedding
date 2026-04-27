import random
import numpy as np
from rdkit import Chem

def is_valid_smiles(smiles):
    """校验 SMILES 的合法性"""
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

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
