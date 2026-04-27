import os
import pickle
import logging
import random
import numpy as np
from tqdm import tqdm
from .utils import is_valid_smiles

class MSPProvider:
    """NIST format (.msp) 数据解析器，支持数据划分（Train/Val/Test）、Pickle 缓存与加速读取"""
    def __init__(self, file_path, limit=None, use_cache=True, seed=42):
        self.file_path = file_path
        self.limit = limit
        self.use_cache = use_cache
        self.seed = seed
        self.cache_path = self.file_path + ".pkl"
        self.train_cache_path = self.file_path + "_train.pkl"
        self.val_cache_path = self.file_path + "_val.pkl"
        self.test_cache_path = self.file_path + "_test.pkl"

    def load_data(self, mode='all'):
        """
        加载数据。
        mode: 'all' (全部), 'train' (训练), 'val' (验证), 'test' (测试)
        """
        modes = {
            'all': self.cache_path,
            'train': self.train_cache_path,
            'val': self.val_cache_path,
            'test': self.test_cache_path
        }
        target_cache = modes.get(mode, self.cache_path)

        # 1. 尝试从缓存加载
        if self.use_cache and not self.limit and os.path.exists(target_cache):
            logging.info(f"Loading {mode} data from cache: {target_cache} ...")
            try:
                with open(target_cache, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"Cache loaded! Total records: {len(data)}")
                return data
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}. Re-parsing/Re-splitting...")

        # 如果请求特定集合但缓存不存在，则先加载全部再重新执行划分
        if mode in ['train', 'val', 'test']:
            logging.info(f"{mode} cache not found, loading all data to split...")
            all_data = self.load_data(mode='all')
            if not all_data:
                return []
            self.split_and_save(all_data, seed=self.seed)
            return self.load_data(mode=mode)

        # 2. 解析原始 MSP 文件
        if not os.path.exists(self.file_path):
            logging.error(f"File {self.file_path} does not exist.")
            return []

        parsed_results = []
        logging.info(f"Starting to parse MSP file: {self.file_path} ...")
        
        # 预先设置 RDKit 静默模式
        try:
            from rdkit import Chem, rdBase
            rdBase.DisableLog('rdApp.*')
        except ImportError:
            Chem = None

        def is_valid_record(record):
            return record and \
                'smiles' in record and \
                'peaks' in record and \
                'inchi' in record and \
                is_valid_smiles(record['smiles']) and \
                record['smiles'].upper() not in ['N/A', 'NA'] and \
                record['peaks'] and \
                record['inchi'].upper() not in ['N/A', 'NA']

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            record = {}
            in_peaks = False
            for line in f:
                line = line.strip()
                if not line:
                    if is_valid_record(record):
                        parsed_results.append(record)
                        if self.limit and len(parsed_results) >= self.limit:
                            break
                    record = {}
                    in_peaks = False
                    continue

                if not in_peaks:
                    if ':' in line:
                        parts = line.split(':', 1)
                        key = parts[0].strip().lower()
                        value = parts[1].strip() if len(parts) > 1 else ""
                        
                        if key == 'smiles':
                            record['smiles'] = value
                        elif key == 'inchi':
                            record['inchi'] = value
                        elif key == 'precursormz':
                            try:
                                record['precursor_mz'] = float(value)
                            except ValueError:
                                record['precursor_mz'] = 0.0
                        elif key == 'num peaks':
                            record['peaks'] = []
                            in_peaks = True
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            record['peaks'].append([mz, intensity])
                        except ValueError:
                            continue
            
            if not self.limit or len(parsed_results) < self.limit:
                if is_valid_record(record):
                    parsed_results.append(record)

        logging.info(f"Parsing completed! Total valid training data: {len(parsed_results)}")

        # 3. 保存到全量缓存
        if self.use_cache and not self.limit and parsed_results:
            logging.info(f"Saving parsed data to cache: {self.cache_path} ...")
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(parsed_results, f)
                logging.info("Full data cache saved successfully!")
            except Exception as e:
                logging.error(f"Failed to save cache: {e}")

        return parsed_results

    def split_and_save(self, data, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        按 SMILES 进行分组划分（Group Split），确保同一个分子的所有记录
        只出现在一个集合中，防止数据泄露。
        """
        if not data:
            logging.error("No data to split.")
            return

        logging.info("Grouping records by SMILES for strict splitting...")
        
        # 1. 按 SMILES 分组
        smiles_groups = {}
        for item in data:
            s = item['smiles']
            if s not in smiles_groups:
                smiles_groups[s] = []
            smiles_groups[s].append(item)
            
        unique_smiles = list(smiles_groups.keys())
        logging.info(f"Total Unique SMILES: {len(unique_smiles)}")

        # 2. 对唯一 SMILES 列表进行随机打乱
        random.seed(seed)
        random.shuffle(unique_smiles)
        
        # 3. 计算划分索引
        n = len(unique_smiles)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_smiles_set = set(unique_smiles[:train_end])
        val_smiles_set = set(unique_smiles[train_end:val_end])
        test_smiles_set = set(unique_smiles[val_end:])

        # 4. 根据 SMILES 集合分配对应的记录
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

        try:
            with open(self.train_cache_path, 'wb') as f:
                pickle.dump(train_data, f)
            with open(self.val_cache_path, 'wb') as f:
                pickle.dump(val_data, f)
            with open(self.test_cache_path, 'wb') as f:
                pickle.dump(test_data, f)
            logging.info(f"Saved to {self.train_cache_path}, {self.val_cache_path} and {self.test_cache_path}")
        except Exception as e:
            logging.error(f"Failed to save split data: {e}")

    def load_candidates(self, mode='test', mz_tolerance=0.1):
        """
        为指定模式的数据生成候选集。
        """
        data = self.load_data(mode=mode)
        if not data:
            return {}

        logging.info(f"Generating candidates for {mode} data (mz_tolerance={mz_tolerance})...")
        
        # 获取所有唯一的 SMILES 及其对应的典型 precursor_mz (取均值)
        smiles_to_mz = {}
        for item in data:
            s = item['smiles']
            mz = item.get('precursor_mz', 0.0)
            if s not in smiles_to_mz:
                smiles_to_mz[s] = []
            smiles_to_mz[s].append(mz)
        
        unique_smiles_list = list(smiles_to_mz.keys())
        avg_mzs = np.array([np.mean(smiles_to_mz[s]) for s in unique_smiles_list])

        candidates_dict = {}
        for item in tqdm(data, desc="Building candidates", ascii=True):
            true_smiles = item['smiles']
            query_mz = item.get('precursor_mz', 0.0)
            
            if query_mz <= 0:
                candidates_dict[true_smiles] = unique_smiles_list
                continue

            diffs = np.abs(avg_mzs - query_mz)
            mask = diffs <= mz_tolerance
            
            indices = np.where(mask)[0]
            cands = [unique_smiles_list[i] for i in indices]
            
            if true_smiles not in cands:
                cands.append(true_smiles)
                
            candidates_dict[true_smiles] = cands

        logging.info(f"Candidates generated for {len(candidates_dict)} queries.")
        return candidates_dict
