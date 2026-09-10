"""Bind fixed molecular bits to the unchanged full-spectrum graph/negative sampling paths."""

import numpy as np
import torch

from SpecEmbedding.data.datasets_align import AlignGraphDataset
from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset
from SpecEmbedding.data.datasets_fingerprint import bind_target_fingerprints, check_candidate_fingerprint_binding
from SpecEmbedding.utils.training_candidates import _integer


class GraphFingerprintAlignmentDataset(AlignGraphDataset):
    def __init__(self, *args, fingerprint_root, fingerprint_smiles, fingerprint_provenance,
                 dataset_manifest_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprints, self.fingerprint_receipt, self._target_fingerprints = bind_target_fingerprints(
            self, fingerprint_root=fingerprint_root, fingerprint_smiles=fingerprint_smiles,
            fingerprint_provenance=fingerprint_provenance, dataset_manifest_sha256=dataset_manifest_sha256)

    def __getitem__(self, index):
        index = _integer(index, 'dataset_index')
        if index >= len(self):
            raise IndexError(index)
        # Resolve the original target without drawing augmentation RNG a second time.
        identity, offset = self._spectrum_indices[index]
        smiles = self._data[identity][offset]['smiles']
        item = super().__getitem__(index)
        fingerprint = torch.from_numpy(self.fingerprints[self._target_fingerprints[smiles]]).unsqueeze(0)
        # The inherited path returns cloned/augmented graphs; never attach to its cached originals.
        for graph in item[3]:
            graph.fingerprints = fingerprint.clone()
        return item


class CandidateGraphFingerprintDataset(CandidateAlignDataset):
    def __init__(self, base, candidates, *, dataset_manifest_sha256, negative_count, seed, graph_cache_size):
        if not isinstance(base, GraphFingerprintAlignmentDataset):
            raise ValueError('Graph fingerprint candidates require the combined full-spectrum dataset')
        super().__init__(base, candidates, dataset_manifest_sha256=dataset_manifest_sha256,
                         negative_count=negative_count, seed=seed, graph_cache_size=graph_cache_size)
        check_candidate_fingerprint_binding(base, candidates, dataset_manifest_sha256)
        self.provenance.update(molecule_input='graph_with_fixed_morgan_bits', fingerprint_cache=base.fingerprint_receipt,
                               fingerprint_augmentation='None: fixed bits describe original positive and negative molecules')

    def __getitem__(self, index):
        item = super().__getitem__(index)
        rows = self.base.fingerprints.get_many(np.asarray(item.sample.molecule_indices, dtype=np.int64))
        for graph, fingerprint in zip(item.negative_graphs, torch.from_numpy(rows), strict=True):
            graph.fingerprints = fingerprint.unsqueeze(0)
        return item
