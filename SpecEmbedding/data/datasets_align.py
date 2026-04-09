import torch
from torch_geometric.data import Data, Batch
from SpecEmbedding.data.datasets import TrainDataset
from SpecEmbedding.data.graph_utils import smiles_to_graph

class AlignGraphDataset(TrainDataset):
    """
    质谱-分子图对齐数据集。
    将 SMILES 转换为 PyG 的 Data 对象，用于 GNN (如 GINE) 训练。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # 1. 获取质谱视图和 label (注意: 原版 Dataset 返回的是整数 label, 而非 smiles 字符串)
        views, label = super().__getitem__(index)
        
        # 2. 从原始数据中提取真正的 SMILES 字符串
        # self._data[label] 是一个列表，里面存放了所有拥有该 label (SMILES) 的质谱序列
        smiles_str = self._data[label][0]["smiles"]
        
        # 3. 将 SMILES 转换为分子图 (PyG Data 对象)
        mol_graph = smiles_to_graph(smiles_str)
        if mol_graph is None:
            # 如果解析失败，返回 None
            return None
            
        # 4. 提取质谱数据
        spec_view = views[0]
        spec_mz = torch.tensor(spec_view[0], dtype=torch.float32)
        spec_intensity = torch.tensor(spec_view[1], dtype=torch.float32)
        spec_mask = torch.tensor(spec_view[2], dtype=torch.bool)
        
        return {
            "spec_mz": spec_mz,
            "spec_intensity": spec_intensity,
            "spec_mask": spec_mask,
            "mol_graph": mol_graph,
            "smiles": smiles_str
        }

def align_collate_fn(batch):
    """
    自定义 Collate 函数，用于处理 PyG 图数据的 Batch 组装。
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    # 质谱数据组装
    spec_mzs = torch.stack([b["spec_mz"] for b in batch])
    spec_intensities = torch.stack([b["spec_intensity"] for b in batch])
    spec_masks = torch.stack([b["spec_mask"] for b in batch])
    
    # 分子图数据组装 (使用 PyG Batch.from_data_list)
    mol_graphs = Batch.from_data_list([b["mol_graph"] for b in batch])
    
    smiles = [b["smiles"] for b in batch]
    
    return {
        "spec_mz": spec_mzs,
        "spec_intensity": spec_intensities,
        "spec_mask": spec_masks,
        "mol_graph": mol_graphs,
        "smiles": smiles
    }
