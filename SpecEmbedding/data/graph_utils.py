import torch
from rdkit import Chem
from torch_geometric.data import Data

# 简化后的原子特征 (针对质谱任务优化)
ATOM_FEATURES = {
    'symbol': ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F', 'Br', 'I', 'Si', 'B', 'As', 'Se'],
    'formal_charge': [-1, 0, 1],      # 形式电荷
    'degree': [0, 1, 2, 3, 4, 5, 6],  # 原子度数
    'num_hs': [0, 1, 2, 3, 4],        # 氢原子数
    'is_aromatic': [0, 1],            # 是否芳香性
    'is_in_ring': [0, 1],             # 是否在环内
    'ring_size_3': [0, 1],            # 3元环
    'ring_size_4': [0, 1],            # 4元环
    'ring_size_5': [0, 1],            # 5元环
    'ring_size_6': [0, 1]             # 6元环
}

# 简化后的键特征
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'is_conjugated': [0, 1], # 是否共轭
    'is_in_ring': [0, 1]     # 是否在环内
}

def safe_index(feature_list, value):
    """安全获取索引，若不在列表中则返回列表长度（作为'其他'类别）"""
    try:
        return feature_list.index(value)
    except ValueError:
        return len(feature_list)

def get_atom_features(atom):
    """提取单个原子的特征向量"""
    return [
        safe_index(ATOM_FEATURES['symbol'], atom.GetSymbol()),
        safe_index(ATOM_FEATURES['formal_charge'], atom.GetFormalCharge()),
        safe_index(ATOM_FEATURES['degree'], atom.GetDegree()),
        safe_index(ATOM_FEATURES['num_hs'], atom.GetTotalNumHs()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        int(atom.IsInRingSize(3)),
        int(atom.IsInRingSize(4)),
        int(atom.IsInRingSize(5)),
        int(atom.IsInRingSize(6))
    ]

def get_bond_features(bond):
    """提取单个化学键的特征向量"""
    return [
        safe_index(BOND_FEATURES['bond_type'], bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles: str):
    """将 SMILES 转换为 PyG 图数据对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 提取节点特征
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.long)
    
    # 提取边特征 (无向图)
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        bf = get_bond_features(bond)
        edge_features += [bf, bf]
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.long)
    
    # 处理无边分子 (如单原子)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_FEATURES)), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
