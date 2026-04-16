import numpy as np
import torch
from rdkit import Chem
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
            return None

        # [新增] 图结构扰动 (Graph Structure Perturbation)
        if self.is_augment and np.random.random() < self.augment_config["prob"]:
            # A. 节点丢弃 (Node Dropping)
            if self.augment_config.get("node_drop_rate", 0) > 0:
                num_nodes = mol_graph.x.size(0)
                if num_nodes > 1: # 至少保留一个节点
                    # 生成保留节点的掩码
                    keep_prob = 1 - self.augment_config["node_drop_rate"]
                    node_mask = torch.rand(num_nodes) < keep_prob
                    
                    # 确保至少保留一个节点，防止空图
                    if not node_mask.any():
                        node_mask[np.random.randint(0, num_nodes)] = True
                    
                    # 更新节点特征
                    mol_graph.x = mol_graph.x[node_mask]
                    if hasattr(mol_graph, 'node_mass'):
                        mol_graph.node_mass = mol_graph.node_mass[node_mask]
                    
                    # 更新边索引 (过滤掉包含被删除节点的边)
                    # 1. 创建节点索引映射
                    assoc = torch.full((num_nodes,), -1, dtype=torch.long)
                    assoc[node_mask] = torch.arange(node_mask.sum())
                    
                    # 2. 过滤并重映射 edge_index
                    row, col = mol_graph.edge_index
                    mask = node_mask[row] & node_mask[col]
                    mol_graph.edge_index = assoc[mol_graph.edge_index[:, mask]]
                    if mol_graph.edge_attr is not None:
                        mol_graph.edge_attr = mol_graph.edge_attr[mask]

            # B. 边遮盖 (Edge Masking / Dropping)
            if self.augment_config.get("edge_mask_rate", 0) > 0 and mol_graph.edge_index.size(1) > 0:
                num_edges = mol_graph.edge_index.size(1)
                # 由于是无向图，edge_index 通常是成对出现的 [i,j] 和 [j,i]
                # 这里为了简单直接按比例随机丢弃半数以上的边（保持对称性略复杂，此处采取简单独立采样）
                keep_prob = 1 - self.augment_config["edge_mask_rate"]
                edge_mask = torch.rand(num_edges) < keep_prob
                
                mol_graph.edge_index = mol_graph.edge_index[:, edge_mask]
                if mol_graph.edge_attr is not None:
                    mol_graph.edge_attr = mol_graph.edge_attr[edge_mask]
            
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
