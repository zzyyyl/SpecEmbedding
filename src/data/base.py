import logging
import pickle
from pathlib import Path


class DataProvider:
    def __init__(self, dataset_name, base_data_dir):
        self.base_data_dir = Path(base_data_dir)
        self.dataset_name = dataset_name
        if self.base_data_dir.name == dataset_name:
            self.data_dir = self.base_data_dir
        else:
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

    def load_candidates(self, type):
        """候选集加载"""
        file_path = self.data_dir / f"candidates_{type}.pkl"

        if not file_path.exists():
            logging.warning(f"{self.dataset_name} candidates not found at {file_path}")
            return {}

        logging.info(f"Loading candidates from {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded {len(data)} candidate sets.")
        return data
