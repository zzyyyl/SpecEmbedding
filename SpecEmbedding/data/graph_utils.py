import torch
from rdkit import Chem
from torch_geometric.data import Data

# 原子特征
ATOM_FEATURES = {
    'symbol': [
        'H', 'C', 'O', 'N', 'P', 'S',
        'Cl', 'F', 'Br', 'I', 'Si', 'B', 'As', 'Se',
        'unknown'
    ],
    'degree': [0, 1, 2, 3, 4, 5, 6, 'unknown'],      # 原子度数
    'num_hs': [0, 1, 2, 3, 4, 'unknown'],            # 氢原子数
    'is_aromatic': [0, 1],                           # 是否芳香性
    'ring_info': [0, 3, 4, 5, 6, '7+'],              # 在几元环内
    'formal_charge': ['<-2', -2, -1, 0, 1, 2, '>2'], # 形式电荷
}

# 键特征
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'unknown'
    ],
    'is_conjugated': [0, 1], # 是否共轭
    'is_in_ring': [0, 1]     # 是否在环内
}

def safe_index(feature_list, value):
    try:
        return feature_list.index(value)
    except ValueError:
        return len(feature_list) - 1

def get_atom_ring_info(atom):
    if not atom.IsInRing(): return 0
    if atom.IsInRingSize(3): return 3
    if atom.IsInRingSize(4): return 4
    if atom.IsInRingSize(5): return 5
    if atom.IsInRingSize(6): return 6
    return '7+'

def get_atom_formal_charge(atom):
    charge = atom.GetFormalCharge()
    if charge < -2: return '<-2'
    if charge > 2: return '>2'
    return charge

def get_atom_features(atom):
    """提取单个原子的特征向量"""

    return [
        safe_index(ATOM_FEATURES['symbol'], atom.GetSymbol()),
        safe_index(ATOM_FEATURES['degree'], atom.GetDegree()),
        safe_index(ATOM_FEATURES['num_hs'], atom.GetTotalNumHs()),
        int(atom.GetIsAromatic()),
        safe_index(ATOM_FEATURES['ring_info'], get_atom_ring_info(atom)),
        safe_index(ATOM_FEATURES['formal_charge'], get_atom_formal_charge(atom)),
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
