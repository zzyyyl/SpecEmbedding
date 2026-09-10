"""Attach observed adducts without changing peak/graph augmentation or candidate sampling."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from SpecEmbedding.utils.adduct_metadata import SpectrumMetadata, require


@dataclass
class AdductAlignmentBatch:
    anchor: tuple
    raw_query_indices: torch.Tensor
    adduct_ids: torch.Tensor


def bind_candidate_adducts(dataset, metadata):
    from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset

    require(isinstance(dataset, CandidateAlignDataset) and isinstance(metadata, SpectrumMetadata)
            and metadata.payload['split'] == 'train', 'Adduct candidate binding requires full training metadata')
    require(not hasattr(dataset, 'spectrum_metadata'), 'Candidate adduct inputs are already bound')
    require(metadata.provenance['source']['dataset_manifest_sha256'] == dataset.provenance['dataset_manifest_sha256'],
            'Candidate/adduct dataset manifests differ')
    mapping, ids = metadata.bind_dataset(dataset.base)
    require(np.array_equal(mapping, dataset.raw_query_indices), 'Candidate/adduct raw query mappings differ')
    ids.setflags(write=False)
    dataset.spectrum_metadata, dataset.adduct_ids_by_dataset = metadata, ids
    dataset.provenance['spectrum_metadata'] = metadata.provenance


class AdductAlignmentDataset(Dataset):
    """The ordinary full validation loss path, preserving the parent's item construction."""

    def __init__(self, base, metadata):
        require(isinstance(metadata, SpectrumMetadata) and metadata.payload['split'] == 'val',
                'Adduct validation requires explicitly bound validation metadata')
        self.base, self.spectrum_metadata = base, metadata
        self.raw_query_indices, self.adduct_ids = metadata.bind_dataset(base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index], int(self.raw_query_indices[index]), int(self.adduct_ids[index])


def adduct_align_collate_fn(items):
    from SpecEmbedding.data.datasets_align import align_collate_fn
    require(bool(items), 'Empty adduct validation batch')
    anchors, raw, ids = zip(*items, strict=True)
    require(all(item[0].shape[0] == 1 for item in anchors), 'Adduct binding requires exactly one view per query')
    return AdductAlignmentBatch(align_collate_fn(anchors), torch.tensor(raw, dtype=torch.long),
                                torch.tensor(ids, dtype=torch.long))


def validate_bound_adducts(metadata, raw, ids):
    require(isinstance(metadata, SpectrumMetadata) and isinstance(raw, torch.Tensor) and isinstance(ids, torch.Tensor)
            and raw.dtype == ids.dtype == torch.long and raw.device.type == ids.device.type == 'cpu'
            and raw.ndim == ids.ndim == 1 and raw.shape == ids.shape, 'Missing or malformed bound adduct batch')
    positions = np.searchsorted(metadata.raw_query_indices, raw.numpy())
    require(np.all(positions < len(metadata.rows)), 'Adduct batch raw query is outside the bound split')
    require(np.array_equal(metadata.raw_query_indices[positions], raw.numpy())
            and np.array_equal(metadata.adduct_ids[positions], ids.numpy()), 'Adduct batch query/ID binding mismatch')


def forward_alignment_batch(model, batch, device, *, adduct_ids=None):
    if isinstance(batch, AdductAlignmentBatch):
        require(adduct_ids is None, 'Duplicate adduct batch input')
        adduct_ids, batch = batch.adduct_ids, batch.anchor
    mz, intensity, mask, molecules, labels = batch
    kwargs = {} if adduct_ids is None else {'adduct_ids': adduct_ids.to(device)}
    result = model(mz.to(device), intensity.to(device), mask.to(device), molecules.to(device), **kwargs)
    return result, labels, mz.shape[0]


class AdductValidationSpectra(Dataset):
    def __init__(self, base, index, metadata):
        require(isinstance(metadata, SpectrumMetadata), 'Validation adduct metadata has not been verified')
        self.base = base
        self.adduct_ids = metadata.bind_index(index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return {**self.base[index], 'adduct_id': int(self.adduct_ids[index])}
