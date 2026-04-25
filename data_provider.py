import os
import pickle
import logging
import random
import numpy as np

from tqdm import tqdm
from rdkit import Chem

def is_valid_smiles(smiles):
    """校验 SMILES 的合法性"""
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return True
    return False

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

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            record = {}
            in_peaks = False
            for line in f:
                line = line.strip()
                if not line:
                    if record and 'smiles' in record and record.get('peaks') and is_valid_smiles(record['smiles']):
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
                            if value and value.upper() not in ['N/A', 'NA']:
                                record['smiles'] = value
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
            
            if record and 'smiles' in record and record.get('peaks') and is_valid_smiles(record['smiles']) and (not self.limit or len(parsed_results) < self.limit):
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
        默认从该模式的所有唯一 SMILES 中，筛选出 precursor_mz 在容差范围内的 SMILES。
        返回格式: {true_smiles: [cand_smiles1, cand_smiles2, ...]}
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
                # 如果没有有效的 m/z，则将所有唯一 SMILES 作为候选集
                candidates_dict[true_smiles] = unique_smiles_list
                continue

            # 找到在容差范围内的所有 SMILES
            diffs = np.abs(avg_mzs - query_mz)
            mask = diffs <= mz_tolerance
            
            indices = np.where(mask)[0]
            cands = [unique_smiles_list[i] for i in indices]
            
            # 确保正确答案在里面
            if true_smiles not in cands:
                cands.append(true_smiles)
                
            candidates_dict[true_smiles] = cands

        logging.info(f"Candidates generated for {len(candidates_dict)} queries.")
        return candidates_dict

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

            # 转换为统一格式: [{'smiles': ..., 'peaks': [[mz, int], ...]}, ...]
            parsed_results = []
            
            for _, row in df.iterrows():
                smiles = row['smiles']

                if not is_valid_smiles(smiles):
                    continue

                # MassSpecGym 的峰数据是逗号分隔的字符串
                mzs = [float(x) for x in str(row['mzs']).split(',')]
                ints = [float(x) for x in str(row['intensities']).split(',')]
                peaks = list(zip(mzs, ints))
                
                # 尝试获取 precursor_mz，如果没有则设为 0
                pmz = row.get('precursor_mz', 0.0)
                
                parsed_results.append({
                    'smiles': smiles,
                    'peaks': peaks,
                    'precursor_mz': pmz
                })
            
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
