import pickle
import logging
from pathlib import Path

class GNPSProvider:
    """GNPS 数据加载器"""
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_data(self, mode='train'):
        """
        加载数据。mode: 'train', 'val', 'test'
        """
        file_path = self.data_dir / f"GNPS_{mode}.pkl"

        if not file_path.exists():
            if mode == 'val':
                file_path = self.data_dir / "GNPS_test.pkl"
            
            if not file_path.exists():
                logging.error(f"GNPS data not found at {file_path}")
                return []

        logging.info(f"Loading GNPS {mode} data from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} records.")
        return data

    '''
    def load_candidates(self):
        """GNPS 候选集加载"""
        file_path = self.data_dir / "GNPS_candidates.pkl"
        if not file_path.exists():
            logging.warning(f"GNPS candidates not found at {file_path}")
            return {}
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    '''
