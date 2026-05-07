import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from typing import Tuple, Union, Iterable
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.data.graph_utils import ATOM_FEATURES, BOND_FEATURES

class GINEEncoder(nn.Module):
    """基于 GINE (Graph Isomorphism Network with Edge features) 的分子编码器"""
    def __init__(
        self,
        emb_dim: int,
        n_layers: int,
        dropout_rate: float
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # 1. 节点特征嵌入：根据类别数分配较小维度，拼接后投影 (避免维度冗余)
        self.atom_embeddings = nn.ModuleList()
        atom_emb_dim = 0
        for values in ATOM_FEATURES.values():
            n_cat = len(values)
            dim = 4 if n_cat <= 2 else 20
            self.atom_embeddings.append(nn.Embedding(n_cat, dim))
            atom_emb_dim += dim
        
        self.atom_proj = nn.Linear(atom_emb_dim, emb_dim)

        # 2. 边特征嵌入
        self.bond_embeddings = nn.ModuleList()
        total_bond_feat_dim = 0
        for values in BOND_FEATURES.values():
            n_cat = len(values)
            dim = 4 if n_cat <= 2 else 20
            self.bond_embeddings.append(nn.Embedding(n_cat, dim))
            total_bond_feat_dim += dim
            
        self.bond_proj = nn.Linear(total_bond_feat_dim, emb_dim)

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

        # 拼接 Add 和 Mean 池化，因此输入维度为 emb_dim * 2
        self.fc = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. 节点与边特征的拼接与投影
        h_node = torch.cat([
            emb(x[:, i]) for i, emb in enumerate(self.atom_embeddings)
        ], dim=-1)
        h_node = self.atom_proj(h_node)

        h_edge = torch.cat([
            emb(edge_attr[:, i]) for i, emb in enumerate(self.bond_embeddings)
        ], dim=-1)
        h_edge = self.bond_proj(h_edge)

        # 2. GINE 消息传递
        for conv, norm in zip(self.convs, self.norms):
            h_res = h_node # 保存残差
            h_node = conv(h_node, edge_index, edge_attr=h_edge)
            h_node = norm(h_node)
            h_node = F.relu(h_node)
            h_node = h_node + h_res # 残差相加
            h_node = F.dropout(h_node, p=self.dropout_rate, training=self.training)

        # 3. 全局池化 (Graph-level representation): Add + Mean 拼接
        graph_add = global_add_pool(h_node, batch)
        graph_mean = global_mean_pool(h_node, batch)
        graph_repr = torch.cat([graph_add, graph_mean], dim=-1)
        
        return F.relu(self.fc(graph_repr))

class SpecMolAlignModel(nn.Module):
    """质谱-分子对齐模型：双塔结构 (Spec Siamese + Mol GINE)"""
    def __init__(
        self,
        spec_encoder: SiameseModel,
        mol_encoder: GINEEncoder,
        spec_dim: int,
        hidden_dim: int,
        final_dim: int,
        dropout_rate: float,
        tau: float
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

        self.tau = tau

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

        return f_spec, f_mol, 1 / self.tau
