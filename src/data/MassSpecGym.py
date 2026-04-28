import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from matchms import Spectrum

from .utils import is_valid_smiles

class MassSpecGymProvider:
    """MassSpecGym (HuggingFace) 数据加载器，输出格式与 MSPProvider 一致"""
    def __init__(self, repo_id="roman-bushuiev/MassSpecGym", filename="data/MassSpecGym.tsv", use_cache=True, cache_dir="data"):
        self.repo_id = repo_id
        self.filename = filename
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def load_data(self, mode='train'):
        """
        mode: 'train', 'val', 'test'
        """
        if self.use_cache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, f"MassSpecGym_{mode}.pkl")
            if os.path.exists(cache_path):
                logging.info(f"Loading MassSpecGym {mode} data from cache: {cache_path} ...")
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load cache {cache_path}: {e}. Downloading from HF...")
        else:
            cache_path = None

        try:
            import pandas as pd
            from huggingface_hub import hf_hub_download
        except ImportError:
            logging.error("Please install pandas and huggingface_hub: pip install pandas huggingface_hub")
            return []

        logging.info(f"Downloading/Loading MassSpecGym dataset ({mode})...")
        try:
            data_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type="dataset")
            df = pd.read_csv(data_path, sep="\t")
            
            # 过滤对应 fold (MassSpecGym 使用 'train', 'val', 'test')
            df = df[df['fold'] == mode]
            logging.info(f"MassSpecGym {mode} size: {len(df)}")

            parsed_results = []
            
            for _, row in tqdm(df.iterrows(), desc="Processing spectra"):
                smiles = row['smiles']

                if not is_valid_smiles(smiles):
                    continue

                # MassSpecGym 的峰数据是逗号分隔的字符串
                mzs = [float(x) for x in str(row['mzs']).split(',')]
                ints = [float(x) for x in str(row['intensities']).split(',')]

                metadata = row.to_dict()
                metadata.pop('mzs', None)
                metadata.pop('intensities', None)

                spectrum = Spectrum(
                    mz=np.array(mzs).astype(float),
                    intensities=np.array(ints).astype(float),
                    metadata=metadata
                )

                parsed_results.append(spectrum)
            
            # 保存到缓存
            if self.use_cache and cache_path:
                logging.info(f"Saving MassSpecGym {mode} data to cache: {cache_path} ...")
                with open(cache_path, 'wb') as f:
                    pickle.dump(parsed_results, f)

            return parsed_results
        except Exception as e:
            logging.error(f"Failed to load MassSpecGym data: {e}")
            return []

    def load_candidates(self, type="mass"):
        """
        加载 MassSpecGym 的候选集 (默认为质量过滤后的候选集)
        type: "mass" (质量匹配), "retrieval" (检索任务)
        """
        if self.use_cache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, f"MassSpecGym_candidates_{type}.pkl")
            if os.path.exists(cache_path):
                logging.info(f"Loading MassSpecGym candidates ({type}) from cache...")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        else:
            cache_path = None

        try:
            import json
            from huggingface_hub import hf_hub_download
            
            filename = f"data/molecules/MassSpecGym_retrieval_candidates_{type}.json"
            logging.info(f"Downloading MassSpecGym candidates: {filename}...")
            cand_path = hf_hub_download(repo_id=self.repo_id, filename=filename, repo_type="dataset")
            with open(cand_path, 'r') as f:
                data = json.load(f)
            
            if self.use_cache and cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            return data
        except Exception as e:
            logging.error(f"Failed to load MassSpecGym candidates: {e}")
            return {}
