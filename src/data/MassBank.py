import pickle
import logging
from pathlib import Path

class MassBankProvider:
    """加载预处理好的 MassBank 数据集"""
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_data(self, mode='test'):
        """
        加载数据。mode: 'train', 'val', 'test'
        """
        file_path = self.data_dir / f"MassBank_{mode}.pkl"
        
        if not file_path.exists():
            logging.error(f"MassBank data not found at {file_path}")
            return []
            
        logging.info(f"Loading MassBank {mode} data from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} records.")
        return data

    def load_candidates(self):
        """
        加载预处理好的候选集。
        如果不使用预处理好的候选集，应当通过预处理脚本生成。
        假设候选集文件为 MassBank_candidates.pkl
        """
        file_path = self.data_dir / "MassBank_candidates.pkl"
        if not file_path.exists():
            logging.error(f"MassBank candidates not found at {file_path}. Please generate them using prepare_pubchem_candidates.py")
            return {}
            
        logging.info(f"Loading MassBank candidates from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} candidate sets.")
        return data
