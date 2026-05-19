from collections import OrderedDict

import numpy as np
import torch
from torch_geometric.data import Batch

from SpecEmbedding.data.datasets import TrainDataset
from SpecEmbedding.data.graph_utils import smiles_to_graph


class AlignGraphDataset(TrainDataset):
    """
    质谱-分子图对齐数据集。
    将 SMILES 转换为 PyG 的 Data 对象，用于 GNN (如 GINE) 训练。
    """
    def __init__(self, *args, graph_cache_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if graph_cache_size < -1:
            raise ValueError("graph_cache_size must be -1, 0, or a positive integer")
        self.graph_cache_size = graph_cache_size
        self._mol_cache = OrderedDict()

    def get_mol_graph(self, label):
        if self.graph_cache_size != 0 and label in self._mol_cache:
            mol = self._mol_cache.pop(label)
            self._mol_cache[label] = mol
            return mol

        smiles = self._data[label][0]["smiles"]
        mol = smiles_to_graph(smiles)

        if self.graph_cache_size != 0:
            self._mol_cache[label] = mol
            if self.graph_cache_size > 0 and len(self._mol_cache) > self.graph_cache_size:
                self._mol_cache.popitem(last=False)

        return mol

    def aug_mol(self, item):
        """
        数据增强：图结构扰动
        """
        mol_graph = item.clone()
        # [新增] 图结构扰动 (Graph Structure Perturbation)
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
        return mol_graph

    def __getitem__(self, index):
        spec_views, label = super().__getitem__(index)
        mol = self.get_mol_graph(label)

        mzs, ints, masks, mols = [], [], [], []

        for spec_view in spec_views:
            mzs.append(torch.tensor(spec_view[0], dtype=torch.float32))
            ints.append(torch.tensor(spec_view[1], dtype=torch.float32))
            masks.append(torch.tensor(spec_view[2], dtype=torch.bool))

            if self.is_augment and np.random.random() < self.augment_config["prob"]:
                mols.append(self.aug_mol(mol))
            else:
                mols.append(mol.clone())

        mzs = torch.stack(mzs, dim=0)
        ints = torch.stack(ints, dim=0)
        masks = torch.stack(masks, dim=0)
        labels = [label] * len(mols)

        # Tuple[Tensor, Tensor, Tensor, List[Graph], List[str]]
        return mzs, ints, masks, mols, labels

def align_collate_fn(batch):
    """
    自定义 Collate 函数，支持多视图展开。
    """
    mzs, ints, masks, graphs, labels = zip(*batch)

    # 质谱数据组装
    mzs = torch.cat(mzs, dim=0)
    ints = torch.cat(ints, dim=0)
    masks = torch.cat(masks, dim=0)
    graphs = [g for views in graphs for g in views]
    labels = [label for views in labels for label in views]
    label_to_id = {label: idx for idx, label in enumerate(dict.fromkeys(labels))}
    labels = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)

    # 分子图数据组装 (使用 PyG Batch.from_data_list)
    mols = Batch.from_data_list(graphs)

    return mzs, ints, masks, mols, labels
