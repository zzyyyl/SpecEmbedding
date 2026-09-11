"""Synthetic CPU graph-context checks; no formal training or retrieval measurements."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_graph_context import (
    GlobalContextGINEEncoder,
    GraphGlobalContext,
    GraphGlobalContextAlignmentModel,
    validate_graph_context,
)
from SpecEmbedding.models_graph_fingerprint import GraphFingerprintAlignmentModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.utils.formal_alignment import build_formal_alignment
from tests.test_adduct_conditioning import SETTINGS as ADDUCT_SETTINGS
from tests.test_graph_fingerprint_model import parent_config, settings, training_parent
from tests.test_precursor_delta import spectra


def context_config():
    return {'hidden_dim': 8, 'pooling': 'mean', 'normalization': 'layernorm_no_affine',
            'initialization': 'zero_output', 'placement': 'between_local_layers'}


def definition(fingerprint=False, qk=False, pool=False, adduct=False):
    value = parent_config(qk=qk)
    value['mol_encoder'].update(emb_dim=16, n_layers=4)
    if fingerprint:
        value.update(type='gine_fingerprint', fingerprint_residual=settings())
    if pool:
        value['spec_encoder']['attention_pool'] = {'norm_eps': 1e-5}
    if adduct:
        value['spec_encoder']['adduct_conditioning'] = copy.deepcopy(ADDUCT_SETTINGS)
    return value


def fresh_parent(config):
    if config.get('type') == 'gine_fingerprint':
        parent = {k: copy.deepcopy(v) for k, v in config.items() if k != 'fingerprint_residual'}
        parent['type'] = 'gine'
        return GraphFingerprintAlignmentModel(parent_model_config=parent,
                                               fingerprint_config=config['fingerprint_residual'])
    molecule = GINEEncoder(**{k: v for k, v in config['mol_encoder'].items() if k != 'graph_policy'})
    spectrum = build_spectrum_encoder(config['spec_encoder'])
    align = config['align']
    return SpecMolAlignModel(spectrum, molecule, config['spec_encoder']['dim_target'], align['final_dim'],
                             align['final_dim'], align['dropout_rate'], align['tau'])


def graphs():
    result = []
    # Include a single atom, no-edge disconnected input, long path and ring.
    for index, smiles in enumerate(('C', 'C.O', 'CCCCCCCC', 'c1ccccc1')):
        graph = smiles_to_graph(smiles, graph_policy='rdkit_sanitized')
        graph.fingerprints = torch.zeros(1, 32)
        graph.fingerprints[:, index::4] = 1
        result.append(graph)
    return result


def activate_context(model):
    with torch.no_grad():
        for branch in model.mol_encoder.context_branches:
            branch.output.weight.normal_(std=.15)
            branch.output.bias.normal_(std=.03)


@pytest.mark.parametrize('fingerprint', [False, True])
@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('pool', [False, True])
@pytest.mark.parametrize('adduct', [False, True])
def test_zero_context_preserves_every_parent_weight_rng_and_both_tower_outputs(fingerprint, qk, pool, adduct):
    config = definition(fingerprint, qk, pool, adduct)
    torch.manual_seed(42)
    original = fresh_parent(config).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = GraphGlobalContextAlignmentModel(parent_model_config=config, context_config=context_config()).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name]), name
    extra = model.state_dict().keys() - original.state_dict().keys()
    assert extra and all(name.startswith('mol_encoder.context_branches.') for name in extra)
    branches = model.mol_encoder.context_branches
    assert len(branches) == 3 and not torch.equal(branches[0].input.weight, branches[1].input.weight)
    batch = Batch.from_data_list(graphs())
    kwargs = {'adduct_ids': torch.tensor([0, 1])} if adduct else {}
    torch.testing.assert_close(original.encode_mol(batch), model.encode_mol(batch), rtol=0, atol=0)
    torch.testing.assert_close(original.encode_spec(*spectra(), **kwargs), model.encode_spec(*spectra(), **kwargs), rtol=0, atol=0)
    original.train()
    model.train()
    torch.manual_seed(34)
    expected = original(*spectra(), batch, **kwargs)
    state = torch.get_rng_state()
    torch.manual_seed(34)
    actual = model(*spectra(), batch, **kwargs)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_plain_constructor_matches_actual_fresh_training_entry(monkeypatch):
    config = definition()
    torch.manual_seed(42)
    actual = training_parent(monkeypatch, config)
    state = torch.get_rng_state()
    torch.manual_seed(42)
    model = GraphGlobalContextAlignmentModel(parent_model_config=config, context_config=context_config())
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(tensor, model.state_dict()[key]) for key, tensor in actual.state_dict().items())


def scalar_context(layer, features, batch, count):
    """Independent per-graph means and scalar feature normalization, without scatter/pool/modules."""
    result = []
    for graph_id in range(count):
        rows = features[batch == graph_id]
        mean = sum(rows.unbind()) / len(rows)
        centered = mean - sum(mean.unbind()) / mean.numel()
        variance = sum(centered.square().unbind()) / mean.numel()
        normalized = centered / torch.sqrt(variance + layer.norm.eps)
        hidden = torch.clamp(normalized @ layer.input.weight.T + layer.input.bias, min=0)
        result.append(hidden @ layer.output.weight.T + layer.output.bias)
    return torch.stack([row + result[int(graph_id)] for row, graph_id in zip(features, batch)])


def test_context_value_gradient_long_range_graph_isolation_and_node_permutation():
    torch.manual_seed(63)
    layer = GraphGlobalContext(6, 8, 1e-5).double()
    with torch.no_grad():
        layer.output.weight.normal_(std=.2)
        layer.input.bias.fill_(1.)
    # Membership is deliberately interleaved, including one single-node graph.
    batch = torch.tensor([0, 1, 0, 2, 1, 0])
    features = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
    actual = layer(features, batch, 3)
    expected = scalar_context(layer, features, batch, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    weights = torch.randn_like(actual)
    parameters = [features, *layer.parameters()]
    left = torch.autograd.grad((actual * weights).sum(), parameters, retain_graph=True)
    right = torch.autograd.grad((expected * weights).sum(), parameters)
    for observed, reference in zip(left, right):
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)
    gradient = torch.autograd.grad(actual[0, 0], features)[0]
    assert torch.count_nonzero(gradient[5]) > 0  # Same molecule, no local edge required.
    assert torch.count_nonzero(gradient[batch != 0]) == 0
    changed = features.detach().clone()
    changed[5, 2] += 2
    updated = layer(changed, batch, 3)
    assert not torch.allclose(updated[0], actual[0])
    torch.testing.assert_close(updated[batch != 0], actual[batch != 0], rtol=0, atol=0)
    order = torch.tensor([3, 5, 1, 0, 4, 2])
    torch.testing.assert_close(layer(features[order], batch[order], 3), actual[order])


def reference_gine(encoder, graph):
    """Explicit per-edge aggregation checks the complete local/context insertion formula."""
    nodes = torch.cat([F.embedding(graph.x[:, i], emb.weight) for i, emb in enumerate(encoder.atom_embeddings)], -1)
    nodes = nodes @ encoder.atom_proj.weight.T + encoder.atom_proj.bias
    edges = torch.cat([F.embedding(graph.edge_attr[:, i], emb.weight) for i, emb in enumerate(encoder.bond_embeddings)], -1)
    edges = edges @ encoder.bond_proj.weight.T + encoder.bond_proj.bias
    for i, (conv, norm) in enumerate(zip(encoder.convs, encoder.norms)):
        old = nodes
        normalized = F.layer_norm(nodes, (encoder.emb_dim,), norm.weight, norm.bias, norm.eps)
        messages = []
        for target in range(len(nodes)):
            incoming = [F.relu(normalized[int(source)] + edges[e])
                        for e, (source, destination) in enumerate(graph.edge_index.T) if int(destination) == target]
            total = sum(incoming, torch.zeros_like(normalized[target]))
            messages.append((1 + conv.eps) * normalized[target] + total)
        aggregated = torch.stack(messages)
        hidden = torch.clamp(aggregated @ conv.nn[0].weight.T + conv.nn[0].bias, min=0)
        update = torch.clamp(hidden @ conv.nn[2].weight.T + conv.nn[2].bias, min=0)
        nodes = old + update
        if hasattr(encoder, 'context_branches') and i < len(encoder.context_branches):
            nodes = scalar_context(encoder.context_branches[i], nodes, graph.batch, graph.num_graphs)
    pooled = torch.stack([nodes[graph.batch == i].mean(0) for i in range(graph.num_graphs)])
    sizes = graph.graph_size_features.view(graph.num_graphs, 2).to(pooled)
    sizes = torch.clamp(sizes @ encoder.size_proj[0].weight.T + encoder.size_proj[0].bias, min=0)
    sizes = sizes @ encoder.size_proj[2].weight.T + encoder.size_proj[2].bias
    joined = torch.cat([pooled, sizes], -1)
    return torch.clamp(joined @ encoder.fc.weight.T + encoder.fc.bias, min=0)


def test_nonzero_four_layer_gine_matches_independent_edges_context_placement_and_gradients():
    torch.manual_seed(38)
    model = GraphGlobalContextAlignmentModel(parent_model_config=definition(), context_config=context_config()).double().eval()
    activate_context(model)
    graph = Batch.from_data_list(graphs())
    encoder = model.mol_encoder
    actual = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.graph_size_features)
    expected = reference_gine(encoder, graph)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weight = torch.randn_like(actual)
    parameters = list(encoder.parameters())
    left = torch.autograd.grad((actual * weight).sum(), parameters)
    right = torch.autograd.grad((expected * weight).sum(), parameters)
    for observed, reference in zip(left, right):
        torch.testing.assert_close(observed, reference, atol=1e-9, rtol=1e-9)
    base = GINEEncoder(**{k: v for k, v in definition()['mol_encoder'].items() if k != 'graph_policy'}).double()
    base.load_state_dict({k: v for k, v in encoder.state_dict().items() if not k.startswith('context_branches.')})
    base.eval()
    torch.testing.assert_close(base(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.graph_size_features),
                               reference_gine(base, graph), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('fingerprint', [False, True])
def test_nonzero_context_invariant_to_graph_batching_atom_relabeling_and_spectrum_weights(fingerprint):
    torch.manual_seed(13)
    model = GraphGlobalContextAlignmentModel(parent_model_config=definition(fingerprint), context_config=context_config()).double().eval()
    activate_context(model)
    items = graphs()
    for graph in items:
        graph.graph_size_features = graph.graph_size_features.double()
        graph.fingerprints = graph.fingerprints.double()
    with torch.inference_mode():
        expected = model.encode_mol(Batch.from_data_list(items), True)
        individual = torch.cat([model.encode_mol(Batch.from_data_list([graph]), True) for graph in items])
        torch.testing.assert_close(expected, individual, atol=1e-10, rtol=1e-10)
        order = [3, 0, 2, 1]
        torch.testing.assert_close(model.encode_mol(Batch.from_data_list([items[i] for i in order]), True), expected[order])
        relabeled = []
        for graph in items:
            graph = graph.clone()
            permutation = torch.arange(graph.num_nodes).flip(0)
            inverse = permutation.argsort()
            graph.x = graph.x[permutation]
            graph.edge_index = inverse[graph.edge_index]
            relabeled.append(graph)
        torch.testing.assert_close(model.encode_mol(Batch.from_data_list(relabeled), True), expected)
        for parameter in model.spec_encoder.parameters():
            parameter.add_(.15)
        torch.testing.assert_close(model.encode_mol(Batch.from_data_list(items), True), expected, rtol=0, atol=0)


@pytest.mark.parametrize('fingerprint', [False, True])
@pytest.mark.parametrize('adduct', [False, True])
def test_two_step_contrastive_updates_all_branches_and_strict_save_reload(tmp_path, fingerprint, adduct):
    torch.manual_seed(21)
    config = definition(fingerprint, qk=True, pool=True, adduct=adduct)
    settings = context_config()
    model = GraphGlobalContextAlignmentModel(parent_model_config=config, context_config=settings)
    groups = [model.spec_encoder.parameters(), model.mol_encoder.parameters(), model.spec_proj.parameters(),
              model.mol_proj.parameters(), [model.logit_scale]]
    optimizer = torch.optim.AdamW([{'params': group} for group in groups], lr=.001, weight_decay=0)
    grouped = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len(grouped) == len({id(p) for p in grouped}) == len(list(model.parameters()))
    batch = Batch.from_data_list(graphs())
    kwargs = {'adduct_ids': torch.tensor([0, 1])} if adduct else {}
    first_before = [branch.input.weight.detach().clone() for branch in model.mol_encoder.context_branches]
    for step in range(2):
        optimizer.zero_grad()
        spectrum, molecule, scale = model(*spectra(), batch, **kwargs)
        scores = F.normalize(spectrum, dim=-1) @ F.normalize(molecule, dim=-1).T * scale
        loss = F.cross_entropy(scores, torch.tensor([0, 1]))
        loss.backward()
        for component in [model.spec_encoder, model.mol_encoder.convs, model.spec_proj, model.mol_proj]:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0
                       for p in component.parameters())
        for branch in model.mol_encoder.context_branches:
            assert torch.count_nonzero(branch.output.weight.grad) > 0
            assert torch.isfinite(branch.output.weight.grad).all()
            if step == 0:
                assert torch.count_nonzero(branch.input.weight.grad) == 0
            else:
                assert torch.count_nonzero(branch.input.weight.grad) > 0
        if fingerprint:
            assert torch.count_nonzero(model.mol_encoder.fingerprint_branch[-1].weight.grad) > 0
        optimizer.step()
    assert all(not torch.equal(before, branch.input.weight)
               for before, branch in zip(first_before, model.mol_encoder.context_branches))
    path = tmp_path / 'component.pt'
    torch.save({'construction_config': model.construction_config(), 'state_dict': model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    parent = saved['construction_config']
    context = parent.pop('graph_global_context')
    restored = GraphGlobalContextAlignmentModel(parent_model_config=parent, context_config=context).eval()
    restored.load_state_dict(saved['state_dict'], strict=True)
    torch.testing.assert_close(model.eval().encode_mol(batch, True), restored.encode_mol(batch, True), rtol=0, atol=0)
    torch.testing.assert_close(model.encode_spec(*spectra(), **kwargs), restored.encode_spec(*spectra(), **kwargs), rtol=0, atol=0)
    del saved['state_dict']['mol_encoder.context_branches.1.output.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(saved['state_dict'], strict=True)


def test_production_width_parameter_cost_and_metadata_are_explicit_without_formal_activation():
    graph = {'emb_dim': 128, 'n_layers': 4, 'dropout_rate': .2, 'size_feature_dim': 32,
             'norm_type': 'layernorm', 'norm_eps': 1e-5}
    settings = {**context_config(), 'hidden_dim': 64}
    original = GINEEncoder(**graph)
    candidate = GlobalContextGINEEncoder(context_config=settings, **graph)
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters()) == 49728
    assert all(not list(branch.norm.parameters()) for branch in candidate.context_branches)
    model = GraphGlobalContextAlignmentModel(parent_model_config=definition(), context_config=settings)
    settings['hidden_dim'] = 999
    returned = model.construction_config()
    returned['mol_encoder']['n_layers'] = 1
    assert model.construction_config()['mol_encoder']['n_layers'] == 4
    assert model.construction_config()['graph_global_context']['hidden_dim'] == 64
    # Component preparation must not be mistaken for complete formal runner support.
    with pytest.raises(ValueError, match='formal model configuration'):
        build_formal_alignment(model.construction_config())


@pytest.mark.parametrize('damage', ['missing', 'extra', 'bool_hidden', 'zero_hidden', 'placement', 'pooling', 'normalization', 'initialization'])
def test_incomplete_or_unregistered_settings_rejected(damage):
    settings = context_config()
    if damage == 'missing':
        del settings['hidden_dim']
    elif damage == 'extra':
        settings['dropout'] = .1
    elif damage == 'bool_hidden':
        settings['hidden_dim'] = True
    elif damage == 'zero_hidden':
        settings['hidden_dim'] = 0
    else:
        settings[damage] = 'unsupported'
    with pytest.raises(ValueError, match='Graph context requires explicit'):
        validate_graph_context(settings)


@pytest.mark.parametrize('damage', ['no_layers', 'boolean_layers', 'norm_eps', 'single_layer'])
def test_invalid_molecular_construction_rejected(damage):
    graph = {k: v for k, v in definition()['mol_encoder'].items() if k != 'graph_policy'}
    if damage == 'norm_eps':
        graph['norm_eps'] = float('nan')
    else:
        graph['n_layers'] = {'no_layers': 0, 'boolean_layers': True, 'single_layer': 1}[damage]
    with pytest.raises(ValueError):
        GlobalContextGINEEncoder(context_config=context_config(), **graph)


@pytest.mark.parametrize('damage', ['empty', 'wrong_width', 'bad_membership', 'float_membership', 'graph_count'])
def test_structurally_invalid_context_inputs_rejected(damage):
    layer = GraphGlobalContext(4, 8, 1e-5)
    features, batch, count = torch.ones(3, 4), torch.tensor([0, 1, 1]), 2
    if damage == 'empty':
        features, batch = features[:0], batch[:0]
    elif damage == 'wrong_width':
        features = features[:, :3]
    elif damage == 'bad_membership':
        batch = batch[:, None]
    elif damage == 'float_membership':
        batch = batch.float()
    else:
        count = True
    with pytest.raises(ValueError, match='Graph context requires nonempty'):
        layer(features, batch, count)
