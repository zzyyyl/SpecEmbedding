"""Independent graph/fingerprint fusion tower with complete formal construction metadata."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder


def validate_fingerprint_residual(settings):
    if not isinstance(settings, dict) or set(settings) != {'input_bits', 'hidden_dim', 'norm_eps'}:
        raise ValueError('Fingerprint residual requires all explicit construction settings')
    if any(type(settings[key]) is not int or settings[key] <= 0 for key in ('input_bits', 'hidden_dim')):
        raise ValueError('Fingerprint residual dimensions must be positive integers')
    if settings['input_bits'] % 8:
        raise ValueError('Fingerprint width must be a multiple of eight')
    eps = settings['norm_eps']
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Fingerprint residual normalization epsilon must be finite and positive')


class GraphFingerprintEncoder(GINEEncoder):
    """Keep the GINE path and its state names, adding an initially zero molecular residual."""

    def __init__(self, *, fingerprint_config, **graph_config):
        validate_fingerprint_residual(fingerprint_config)
        super().__init__(**graph_config)
        self.fingerprint_config = copy.deepcopy(fingerprint_config)
        self.input_bits = fingerprint_config['input_bits']
        hidden = fingerprint_config['hidden_dim']
        # The added branch must not alter initialization of the existing projection heads.
        # Construction is on CPU; no CUDA generator or context is initialized here.
        with torch.random.fork_rng(devices=[]):
            self.fingerprint_branch = nn.Sequential(
                nn.Linear(self.input_bits, hidden), nn.LayerNorm(hidden, eps=fingerprint_config['norm_eps']),
                nn.GELU(), nn.Linear(hidden, self.emb_dim),
            )
            nn.init.zeros_(self.fingerprint_branch[-1].weight)
            nn.init.zeros_(self.fingerprint_branch[-1].bias)

    def forward(self, x, edge_index, edge_attr, batch, graph_size_features, *, fingerprints):
        # Binary values and source row identities are verified by the fixed-input loader.
        # This structural check avoids a per-forward device synchronization for value scans.
        if (not isinstance(fingerprints, torch.Tensor) or fingerprints.ndim != 2
                or fingerprints.shape != (graph_size_features.numel() // 2, self.input_bits)
                or not fingerprints.is_floating_point() or fingerprints.device != x.device):
            raise ValueError('Graph fusion requires one floating fingerprint of the configured width per graph, on the same device')
        graph = super().forward(x, edge_index, edge_attr, batch, graph_size_features)
        return graph + self.fingerprint_branch(fingerprints)


class GraphFingerprintAlignmentModel(SpecMolAlignModel):
    """Independent molecular fusion tower; spectrum/projection/contrastive paths are inherited."""

    def __init__(self, *, parent_model_config, fingerprint_config):
        from SpecEmbedding.utils.formal_alignment import formal_model_type

        validate_fingerprint_residual(fingerprint_config)
        if formal_model_type(parent_model_config) != 'gine':
            raise ValueError('Graph fingerprint residual requires a GINE parent')
        parent = copy.deepcopy(parent_model_config)
        spec, align = parent['spec_encoder'], parent['align']
        # Fresh GINE training in train_align constructs the molecular tower first.
        # Checkpoint reconstruction has a different order, but then replaces every weight.
        molecule = GraphFingerprintEncoder(
            fingerprint_config=fingerprint_config,
            **{key: value for key, value in parent['mol_encoder'].items() if key != 'graph_policy'},
        )
        spectrum = build_spectrum_encoder(spec)
        super().__init__(spec_encoder=spectrum, mol_encoder=molecule, spec_dim=spec['dim_target'],
                         hidden_dim=align['final_dim'], final_dim=align['final_dim'],
                         dropout_rate=align['dropout_rate'], tau=align['tau'])
        self._construction_config = {**parent, 'type': 'gine_fingerprint',
                                     'fingerprint_residual': copy.deepcopy(fingerprint_config)}

    def construction_config(self):
        return copy.deepcopy(self._construction_config)

    def encode_mol(self, mol_graph, normalize=False):
        result = self.mol_encoder(mol_graph.x, mol_graph.edge_index, mol_graph.edge_attr, mol_graph.batch,
                                  mol_graph.graph_size_features, fingerprints=getattr(mol_graph, 'fingerprints', None))
        result = self.mol_proj(result)
        return F.normalize(result, dim=-1) if normalize else result
