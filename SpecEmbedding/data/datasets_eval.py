import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from SpecEmbedding.data.graph_utils import smiles_to_graph


class SpecSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "spec_mz": torch.tensor(seq["mz"], dtype=torch.float32),
            "spec_intensity": torch.tensor(seq["intensity"], dtype=torch.float32),
            "spec_mask": torch.tensor(seq["mask"], dtype=torch.bool),
            "smiles": seq["smiles"],
        }


class MolSmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        try:
            graph = smiles_to_graph(self.smiles_list[idx])
        except Exception:
            graph = None
        return {"graph": graph, "original_idx": idx}


def mol_collate_fn(batch):
    batch = [item for item in batch if item["graph"] is not None]
    if not batch:
        return None
    return {
        "mol_graph": Batch.from_data_list([item["graph"] for item in batch]),
        "indices": [item["original_idx"] for item in batch],
    }


SpecDataset = SpecSequenceDataset
MolDataset = MolSmilesDataset
