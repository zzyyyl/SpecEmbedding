"""Opt-in natural-candidate batches; every full-spectrum alignment query is retained."""

import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.utils.training_candidates import NegativeCandidates, TrainingCandidateIndex, _integer


@dataclass
class CandidateExample:
    anchor: tuple
    negative_graphs: list
    sample: NegativeCandidates
    raw_query_index: int


@dataclass
class CandidateAlignmentBatch:
    anchor: tuple
    negative_graphs: Batch | torch.Tensor | None
    negative_ptr: torch.Tensor
    raw_query_indices: torch.Tensor
    candidate_indices: torch.Tensor
    source_positions: torch.Tensor


class CandidateAlignDataset(Dataset):
    """Wrap fresh, identity-grouped full-spectrum data with its verified source pools.

    ``classified_full_spectra`` preserves raw order *within* each 2D identity.
    Bind those offsets back to the raw query order; Dataset indices are not raw
    query indices. The caller must pass the manifest from the actual tokenization
    audit, and call set_epoch before creating each DataLoader iterator. Persistent
    workers are unsupported unless the caller explicitly rebuilds them per epoch.
    """

    def __init__(self, base, candidates, *, dataset_manifest_sha256, negative_count, seed, graph_cache_size):
        if not isinstance(base, AlignGraphDataset) or not base.full_spectra or not isinstance(candidates, TrainingCandidateIndex):
            raise ValueError("Natural candidates require full-spectrum AlignGraphDataset and a verified candidate index")
        if candidates.provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256 or not dataset_manifest_sha256:
            raise ValueError("Tokenized training data and natural candidates have different provenance")
        if candidates.metadata.get("graph_policy") != "rdkit_sanitized" or base.graph_policy != "rdkit_sanitized":
            raise ValueError("Natural training candidates require the audited sanitized graph policy")
        if len(base) != len(candidates):
            raise ValueError("Training dataset does not cover the complete candidate query mapping")
        self.negative_count = _integer(negative_count, "negative_count", 1)
        self.seed = _integer(seed, "seed")
        self.graph_cache_size = _integer(graph_cache_size, "graph_cache_size")
        self.base, self.candidates = base, candidates
        self.epoch = None
        self._graphs = OrderedDict()
        metadata = candidates.metadata
        by_identity = defaultdict(list)
        for raw_index, row in enumerate(metadata["query_target_rows"]):
            by_identity[metadata["target_identity_2d"][row]].append(raw_index)
        mapping = []
        for identity, offset in base._spectrum_indices:
            if identity not in by_identity or offset >= len(by_identity[identity]):
                raise ValueError("Tokenized identity grouping differs from the complete source query order")
            raw_index = by_identity[identity][offset]
            row = int(metadata["query_target_rows"][raw_index])
            if base._data[identity][offset]["smiles"] != metadata["target_smiles"][row]:
                raise ValueError("Tokenized target SMILES differs from the candidate query mapping")
            mapping.append(raw_index)
        self.raw_query_indices = np.asarray(mapping, dtype=np.int64)
        if not np.array_equal(np.sort(self.raw_query_indices), np.arange(len(candidates))):
            raise ValueError("Tokenized data duplicated or lost raw training queries")
        self.raw_query_indices.setflags(write=False)
        self.provenance = {**candidates.provenance, "negative_count": self.negative_count, "seed": self.seed,
                           "graph_cache_size": self.graph_cache_size,
                           "dataset_to_raw_query_sha256": hashlib.sha256(self.raw_query_indices.astype("<i8").tobytes()).hexdigest(),
                           "negative_graph_augmentation": "Same probability/node/edge settings as training positive graphs"}

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, "epoch", 1)

    def _graph(self, molecule):
        if molecule in self._graphs:
            graph = self._graphs.pop(molecule)
            self._graphs[molecule] = graph
            return graph
        smiles = self.candidates.metadata["mol_smiles"][molecule]
        graph = smiles_to_graph(smiles, graph_policy="rdkit_sanitized")
        if self.graph_cache_size:
            self._graphs[molecule] = graph
            if len(self._graphs) > self.graph_cache_size:
                self._graphs.popitem(last=False)
        return graph

    def __getitem__(self, index):
        if self.epoch is None:
            raise RuntimeError("Set the candidate training epoch before iterating")
        index = _integer(index, "dataset_index")
        if index >= len(self):
            raise IndexError(index)
        raw_query = int(self.raw_query_indices[index])
        anchor = self.base[index]
        sampled = self.candidates.sample(raw_query, negative_count=self.negative_count, seed=self.seed, epoch=self.epoch)
        graphs = []
        for molecule in sampled.molecule_indices:
            graph = self._graph(int(molecule))
            if self.base.is_augment and np.random.random() < self.base.augment_config["prob"]:
                graphs.append(self.base.aug_mol(graph))
            else:
                graphs.append(graph.clone())
        return CandidateExample(anchor, graphs, sampled, raw_query)


def candidate_align_collate_fn(examples):
    if not examples:
        raise ValueError("Cannot collate an empty candidate batch")
    graphs = [graph for example in examples for graph in example.negative_graphs]
    return make_candidate_batch(examples, align_collate_fn([example.anchor for example in examples]),
                                Batch.from_data_list(graphs) if graphs else None)


def make_candidate_batch(examples, anchor, negative_molecules):
    """Bind graph or fingerprint tensors to the same observed candidate sample metadata."""
    if not examples:
        raise ValueError("Cannot collate an empty candidate batch")
    counts = [len(example.negative_graphs) for example in examples]
    if any(count != len(example.sample.molecule_indices) or count != len(example.sample.source_positions)
           or count != len(example.sample.identity_2d) for count, example in zip(counts, examples, strict=True)):
        raise ValueError("Candidate graph and source metadata counts differ")
    return CandidateAlignmentBatch(
        anchor=anchor,
        negative_graphs=negative_molecules,
        negative_ptr=torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long),
        raw_query_indices=torch.tensor([example.raw_query_index for example in examples], dtype=torch.long),
        candidate_indices=torch.tensor([int(i) for example in examples for i in example.sample.molecule_indices], dtype=torch.long),
        source_positions=torch.tensor([int(i) for example in examples for i in example.sample.source_positions], dtype=torch.long),
    )
