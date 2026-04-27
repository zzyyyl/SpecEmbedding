import pickle
import logging
from pathlib import Path

class NPLIB1Provider:
    """加载预处理后的 NPLIB1 数据集"""
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_data(self, mode='test'):
        """
        加载数据。mode: 'train', 'val', 'test'
        """
        file_path = self.data_dir / f"NPLIB1_{mode}.pkl"

        if not file_path.exists():
            logging.error(f"NPLIB1 data not found at {file_path}")
            return []

        logging.info(f"Loading NPLIB1 {mode} data from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} records.")
        return data

    def load_candidates(self):
        """
        加载 NPLIB1 的候选集。格式: {smiles: [cand_smiles, ...]}
        """
        file_path = self.data_dir / "NPLIB1_candidates.pkl"
        if not file_path.exists():
            logging.error(f"NPLIB1 candidates not found at {file_path}")
            return {}

        logging.info(f"Loading NPLIB1 candidates from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} candidate sets.")
        return data
