"""Joint fixed-input binding; learned molecular embeddings are still rebuilt each validation."""

import torch

from SpecEmbedding.utils.fingerprint_validation import load_validation_fingerprints
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator, StrictValidationMolecules


class GraphFingerprintValidationMolecules(StrictValidationMolecules):
    def __init__(self, smiles, fingerprints, graph_cache=None):
        super().__init__(smiles, graph_cache)
        if len(fingerprints) != len(smiles):
            raise ValueError('Combined validation inputs have different molecule coverage')
        self.fingerprints = fingerprints

    def __getitem__(self, index):
        item = super().__getitem__(index)
        # A private graph also protects callers whose graph cache holds live Data objects.
        item['graph'] = item['graph'].clone()
        item['graph'].fingerprints = torch.from_numpy(self.fingerprints[index]).unsqueeze(0)
        return item


class GraphFingerprintRetrievalValidator(AlignmentRetrievalValidator):
    def __init__(self, index, settings, output_dir=None, *, fingerprint_root, fingerprint_provenance,
                 index_sha256, spectrum_control=None, control_settings=None, graph_cache=None, spectrum_metadata=None):
        if graph_cache is not None and graph_cache.manifest['provenance']['index_sha256'] != index_sha256:
            raise ValueError('Graph and fingerprint validation inputs use different candidate indices')
        cache, self.fingerprint_receipt = load_validation_fingerprints(
            index, fingerprint_root, fingerprint_provenance, index_sha256)
        super().__init__(index, settings, output_dir, spectrum_control=spectrum_control,
                         control_settings=control_settings, graph_cache=graph_cache, spectrum_metadata=spectrum_metadata)
        self.molecules = GraphFingerprintValidationMolecules(index['mol_smiles'], cache, graph_cache)

    def snapshot_metadata(self):
        return {**super().snapshot_metadata(), 'validation_fingerprint_cache': self.fingerprint_receipt}
