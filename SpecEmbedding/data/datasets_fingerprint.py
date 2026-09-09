"""Complete-spectrum alignment datasets backed by audited fixed molecular inputs."""

import numpy as np
import torch

from SpecEmbedding.data.datasets_align import AlignGraphDataset
from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset, CandidateExample, make_candidate_batch
from SpecEmbedding.utils.fingerprint_cache import fingerprint_provenance, load_fingerprint_cache
from SpecEmbedding.utils.training_candidates import _integer


class FingerprintAlignmentDataset(AlignGraphDataset):
    """Share the baseline spectrum ordering and augmentation, without constructing molecular graphs."""

    def __init__(self, *args, fingerprint_root, fingerprint_smiles, fingerprint_provenance,
                 dataset_manifest_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.full_spectra or self.graph_policy != 'rdkit_sanitized':
            raise ValueError('Fingerprint alignment requires the complete audited spectrum/graph-eligibility protocol')
        if any(self.augment_config[key] != 0 for key in ('node_drop_rate', 'edge_mask_rate')):
            raise ValueError('Fixed fingerprint inputs require explicitly disabled molecular graph augmentation')
        if fingerprint_provenance['dataset_manifest_sha256'] != dataset_manifest_sha256:
            raise ValueError('Fingerprint inputs and tokenized spectra have different provenance')
        self.fingerprints, self.fingerprint_receipt = load_fingerprint_cache(
            fingerprint_smiles, fingerprint_root, fingerprint_provenance)
        # Only retain target lookups, rather than a second multi-million-entry dictionary in each worker.
        targets = {row['smiles'] for sequences in self._data.values() for row in sequences}
        self._target_fingerprints = {smiles: i for i, smiles in enumerate(fingerprint_smiles) if smiles in targets}
        if set(self._target_fingerprints) != targets:
            raise ValueError('A paired target is missing from the fixed input inventory; do not substitute an identity alias')

    def __getitem__(self, index):
        index = _integer(index, 'dataset_index')
        if index >= len(self):
            raise IndexError(index)
        sequence, label = self.full_spectrum_item(index)
        fingerprint = torch.from_numpy(self.fingerprints[self._target_fingerprints[sequence['smiles']]])
        return (torch.tensor(sequence['mz'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sequence['intensity'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(sequence['mask'], dtype=torch.bool).unsqueeze(0),
                fingerprint.unsqueeze(0), [label])


def fingerprint_align_collate_fn(examples):
    if not examples:
        raise ValueError('Cannot collate an empty fingerprint batch')
    mz, intensity, mask, fingerprints, labels = zip(*examples, strict=True)
    labels = [label for views in labels for label in views]
    label_ids = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    return (torch.cat(mz), torch.cat(intensity), torch.cat(mask), torch.cat(fingerprints),
            torch.tensor([label_ids[label] for label in labels], dtype=torch.long))


class CandidateFingerprintDataset(CandidateAlignDataset):
    """Keep the existing distinct-2D negative sampler and complete observed-query accounting."""

    def __init__(self, base, candidates, *, dataset_manifest_sha256, negative_count, seed):
        if not isinstance(base, FingerprintAlignmentDataset):
            raise ValueError('Candidate fingerprints require the audited fingerprint spectrum dataset')
        super().__init__(base, candidates, dataset_manifest_sha256=dataset_manifest_sha256,
                         negative_count=negative_count, seed=seed, graph_cache_size=0)
        source = base.fingerprint_receipt['provenance']
        expected = fingerprint_provenance(candidates.metadata['mol_smiles'],
                                          index_sha256=candidates.provenance['sha256'],
                                          dataset_manifest_sha256=dataset_manifest_sha256,
                                          radius=source['options']['radius'], bits=source['options']['bits'])
        if source != expected:
            raise ValueError('Fingerprint index differs from natural training candidates')
        self.provenance.update(molecule_input='fixed_morgan_bits', fingerprint_cache=base.fingerprint_receipt,
                               negative_graph_augmentation='None: fixed fingerprints for both positives and negatives')

    def __getitem__(self, index):
        if self.epoch is None:
            raise RuntimeError('Set the candidate training epoch before iterating')
        index = _integer(index, 'dataset_index')
        if index >= len(self):
            raise IndexError(index)
        raw = int(self.raw_query_indices[index])
        anchor = self.base[index]
        sample = self.candidates.sample(raw, negative_count=self.negative_count, seed=self.seed, epoch=self.epoch)
        negatives = self.base.fingerprints.get_many(np.asarray(sample.molecule_indices, dtype=np.int64))
        return CandidateExample(anchor, list(torch.from_numpy(negatives).unbind()), sample, raw)


def candidate_fingerprint_collate_fn(examples):
    if not examples:
        raise ValueError('Cannot collate an empty candidate fingerprint batch')
    negatives = [row for example in examples for row in example.negative_graphs]
    return make_candidate_batch(examples, fingerprint_align_collate_fn([example.anchor for example in examples]),
                                torch.stack(negatives) if negatives else None)
