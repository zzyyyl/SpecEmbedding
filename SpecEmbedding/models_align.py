import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from typing import Tuple, Union, Iterable
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.data.graph_utils import ATOM_FEATURES, BOND_FEATURES

class GINEEncoder(nn.Module):
    """基于 GINE (Graph Isomorphism Network with Edge features) 的分子编码器"""
    def __init__(self, emb_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.emb_dim = emb_dim

        # 节点与边特征嵌入层 (根据 graph_utils.py 动态初始化)
        self.atom_embedding = nn.ModuleList([
            nn.Embedding(len(values), emb_dim) 
            for values in ATOM_FEATURES.values()
        ])

        self.bond_embedding = nn.ModuleList([
            nn.Embedding(len(values), emb_dim) 
            for values in BOND_FEATURES.values()
        ])

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            # GINEConv 需要一个 MLP 
            mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.ReLU(),
                nn.Linear(emb_dim * 2, emb_dim)
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(emb_dim))

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. 节点与边特征的 Embedding 汇总
        h_node = 0
        for i, embedding in enumerate(self.atom_embedding):
            h_node += embedding(x[:, i])

        h_edge = 0
        for i, embedding in enumerate(self.bond_embedding):
            h_edge += embedding(edge_attr[:, i])

        # 2. GINE 消息传递
        for conv, norm in zip(self.convs, self.norms):
            h_res = h_node # 保存残差
            h_node = conv(h_node, edge_index, edge_attr=h_edge)
            h_node = norm(h_node)
            h_node = F.relu(h_node)
            h_node = h_node + h_res # 残差相加
            h_node = F.dropout(h_node, p=0.1, training=self.training)

        # 3. 全局池化 (Graph-level representation)
        graph_repr = global_add_pool(h_node, batch)
        return F.relu(self.fc(graph_repr))

class SpecMolAlignModel(nn.Module):
    """质谱-分子对齐模型：双塔结构 (Spec Siamese + Mol GINE)"""
    def __init__(
        self,
        spec_encoder: SiameseModel,
        mol_encoder: GINEEncoder,
        spec_dim: int,
        hidden_dim: int = 512,
        final_dim: int = 512,
        dropout_rate = 0.2,
        tau = 0.07
    ):
        super().__init__()
        self.spec_encoder = spec_encoder
        self.mol_encoder = mol_encoder

        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, final_dim)
        )

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_encoder.emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, final_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / tau)))

    def forward(self, spec_mz, spec_intensity, spec_mask, mol_graph):
        # 质谱特征提取
        f_spec = self.spec_encoder(spec_mz, spec_intensity, spec_mask)

        # 分子图特征提取
        f_mol = self.mol_encoder(
            mol_graph.x, 
            mol_graph.edge_index, 
            mol_graph.edge_attr, 
            mol_graph.batch
        )

        # 投影层
        f_spec = self.spec_proj(f_spec)
        f_mol = self.mol_proj(f_mol)

        # 归一化
        f_spec = F.normalize(f_spec, dim=-1)
        f_mol = F.normalize(f_mol, dim=-1)

        # 限制温度系数上限，防止过于 confident 导致过拟合
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        return f_spec, f_mol, scale
