"""Prepared graph-local/global molecular tower; formal runner integration is separate."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool

from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_graph_fingerprint import GraphFingerprintEncoder
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder


def validate_graph_context(settings):
    if (not isinstance(settings, dict)
            or set(settings) != {'hidden_dim', 'pooling', 'normalization', 'initialization', 'placement'}
            or type(settings['hidden_dim']) is not int or settings['hidden_dim'] <= 0
            or settings['pooling'] != 'mean'
            or settings['normalization'] != 'layernorm_no_affine'
            or settings['initialization'] != 'zero_output'
            or settings['placement'] != 'between_local_layers'):
        raise ValueError('Graph context requires explicit mean pooling, non-affine LayerNorm, zero output and between-layer placement')


class GraphGlobalContext(nn.Module):
    """Mean-pool one molecule, transform its context, and broadcast only to its own nodes."""

    def __init__(self, embedding_dim, hidden_dim, norm_eps):
        super().__init__()
        if (any(type(value) is not int or value <= 0 for value in (embedding_dim, hidden_dim))
                or isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float))
                or not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError('Graph context dimensions and normalization epsilon must be positive and finite')
        self.embedding_dim = embedding_dim
        self.norm = nn.LayerNorm(embedding_dim, eps=norm_eps, elementwise_affine=False)
        self.input = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, batch, num_graphs):
        # Graph membership values come from the graph loader. Avoid GPU value scans or
        # batch.max().item() at every local layer; graph count is supplied by size metadata.
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[0] == 0 or features.shape[1] != self.embedding_dim
                or not features.is_floating_point() or not isinstance(batch, torch.Tensor)
                or batch.dtype != torch.long or batch.shape != features.shape[:1]
                or batch.device != features.device or type(num_graphs) is not int or num_graphs <= 0):
            raise ValueError('Graph context requires nonempty node features, int64 membership and an explicit graph count')
        pooled = global_mean_pool(features, batch, size=num_graphs)
        context = self.output(F.relu(self.input(self.norm(pooled))))
        return features + context[batch]


class _GraphContextLayers:
    """Cooperative constructor shared by plain GINE and GINE with fixed-fingerprint fusion."""

    def __init__(self, *, context_config, **graph_config):
        validate_graph_context(context_config)
        layers = graph_config.get('n_layers')
        if type(layers) is not int or layers < 2:
            raise ValueError('Graph context requires at least two local GINE layers')
        super().__init__(**graph_config)
        self.context_config = copy.deepcopy(context_config)
        # Initialize all independent branches in one isolated CPU RNG scope so their
        # hidden weights differ without changing inherited projection/dropout streams.
        with torch.random.fork_rng(devices=[]):
            self.context_branches = nn.ModuleList([
                GraphGlobalContext(self.emb_dim, context_config['hidden_dim'], self.norm_eps)
                for _ in range(layers - 1)
            ])

    def _after_local_update(self, h_node, batch, layer_index, num_graphs):
        return self.context_branches[layer_index](h_node, batch, num_graphs)


class GlobalContextGINEEncoder(_GraphContextLayers, GINEEncoder):
    """GINE with an identity-initialized context branch between each pair of local layers."""


class GlobalContextGraphFingerprintEncoder(_GraphContextLayers, GraphFingerprintEncoder):
    """The same context insertion with the inherited final fingerprint residual intact."""


class GraphGlobalContextAlignmentModel(SpecMolAlignModel):
    """Independent contrastive towers with explicit, not-yet-formal construction metadata."""

    def __init__(self, *, parent_model_config, context_config):
        from SpecEmbedding.utils.formal_alignment import formal_model_type

        validate_graph_context(context_config)
        kind = formal_model_type(parent_model_config)
        if kind not in ('gine', 'gine_fingerprint'):
            raise ValueError('Graph context requires a GINE parent')
        parent = copy.deepcopy(parent_model_config)
        graph_config = {key: value for key, value in parent['mol_encoder'].items() if key != 'graph_policy'}
        self._uses_fingerprint = kind == 'gine_fingerprint'
        if self._uses_fingerprint:
            molecule = GlobalContextGraphFingerprintEncoder(
                context_config=context_config, fingerprint_config=parent['fingerprint_residual'], **graph_config)
        else:
            molecule = GlobalContextGINEEncoder(context_config=context_config, **graph_config)
        spec, align = parent['spec_encoder'], parent['align']
        # Match the fresh training constructor's molecule -> spectrum -> projections order.
        spectrum = build_spectrum_encoder(spec)
        super().__init__(spec_encoder=spectrum, mol_encoder=molecule, spec_dim=spec['dim_target'],
                         hidden_dim=align['final_dim'], final_dim=align['final_dim'],
                         dropout_rate=align['dropout_rate'], tau=align['tau'])
        self._construction_config = {**parent, 'graph_global_context': copy.deepcopy(context_config)}

    def construction_config(self):
        return copy.deepcopy(self._construction_config)

    def encode_mol(self, mol_graph, normalize=False):
        kwargs = {'fingerprints': getattr(mol_graph, 'fingerprints', None)} if self._uses_fingerprint else {}
        result = self.mol_encoder(mol_graph.x, mol_graph.edge_index, mol_graph.edge_attr, mol_graph.batch,
                                  mol_graph.graph_size_features, **kwargs)
        result = self.mol_proj(result)
        return F.normalize(result, dim=-1) if normalize else result
