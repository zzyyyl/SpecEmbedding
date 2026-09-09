"""Encode every validation molecule afresh from fixed inputs, using the existing rank protocol."""

import torch
from torch.utils.data import DataLoader, Dataset

from SpecEmbedding.utils.fingerprint_cache import load_fingerprint_cache
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator


class ValidationFingerprints(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        return {'mol_graph': torch.from_numpy(self.cache[index]), 'indices': index}


class FingerprintRetrievalValidator(AlignmentRetrievalValidator):
    def __init__(self, index, settings, output_dir=None, *, fingerprint_root, fingerprint_provenance,
                 index_sha256, spectrum_control=None, control_settings=None):
        if (fingerprint_provenance['dataset_manifest_sha256'] != index['dataset_manifest_sha256']
                or fingerprint_provenance['index_sha256'] != index_sha256):
            raise ValueError('Fingerprint validation input does not match the audited index')
        cache, self.fingerprint_receipt = load_fingerprint_cache(index['mol_smiles'], fingerprint_root,
                                                               fingerprint_provenance)
        super().__init__(index, settings, output_dir, spectrum_control=spectrum_control, control_settings=control_settings)
        self.molecules = ValidationFingerprints(cache)

    def molecule_loader(self, generator):
        return DataLoader(self.molecules, batch_size=self.settings.mol_batch_size,
                          num_workers=self.settings.num_workers, shuffle=False, generator=generator)

    def snapshot_metadata(self):
        return {**super().snapshot_metadata(), 'validation_fingerprint_cache': self.fingerprint_receipt}
