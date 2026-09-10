"""Synthetic component checks; no formal model training or retrieval results."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.models_graph_fingerprint import GraphFingerprintAlignmentModel, validate_fingerprint_residual
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, formal_model_type
from tests.test_precursor_delta import delta_config, spectra
from tests.test_qk_norm import definition


def settings():
    return {'input_bits': 32, 'hidden_dim': 12, 'norm_eps': 1e-5}


def molecule_list():
    graphs = []
    for index, smiles in enumerate(('CCO', 'CCC', 'CC', 'c1ccccc1')):
        graph = smiles_to_graph(smiles, graph_policy='rdkit_sanitized')
        graph.fingerprints = torch.zeros(1, 32)
        graph.fingerprints[0, index::4] = 1
        graphs.append(graph)
    return graphs


def parent_config(qk=False, delta=False):
    result = definition(delta=delta)
    if not qk:
        del result['spec_encoder']['qk_norm']
    if delta:
        result['spec_encoder']['precursor_delta'] = delta_config()
    return result


def training_parent(monkeypatch, definition):
    """Capture the actual fresh training entry before optimization, without reproducing its constructor."""
    import train_align
    from SpecEmbedding.config import ConfigObject
    from tests.test_candidate_alignment import candidate_dataset

    data = candidate_dataset(monkeypatch).base._data
    captured = {}
    class CapturedModel(Exception):
        pass
    def capture(model, *args, **kwargs):
        captured['model'] = model
        raise CapturedModel
    with monkeypatch.context() as scope:
        scope.setattr(train_align.config, 'model', ConfigObject(copy.deepcopy(definition)))
        scope.setattr(train_align.config.train.align, 'batching', 'random')
        scope.setattr(train_align.config.train.align.candidate_supervision, 'enabled', False)
        scope.setattr(train_align, 'TrainerAlign', capture)
        with pytest.raises(CapturedModel):
            train_align.train_align(data, list(data), data, list(data), None, batch_size=2, device='cpu',
                                    formal_fulltrain=True, retrieval_validator=object(),
                                    mol_norm_type=definition['mol_encoder']['norm_type'],
                                    mol_norm_eps=definition['mol_encoder']['norm_eps'],
                                    selection_metadata={'fulltrain_audit': {'expected_epoch_counts': {'train': 3, 'val': 3}}})
    return captured['model']


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('delta', [False, True])
def test_preserves_inherited_weights_rng_and_zero_residual_outputs(monkeypatch, qk, delta):
    config = parent_config(qk, delta)
    torch.manual_seed(42)
    baseline = training_parent(monkeypatch, config).eval()
    after = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = GraphFingerprintAlignmentModel(parent_model_config=config, fingerprint_config=settings()).eval()
    assert torch.equal(torch.get_rng_state(), after)
    original = baseline.state_dict()
    current = candidate.state_dict()
    assert all(torch.equal(value, current[name]) for name, value in original.items())
    assert all(name.startswith('mol_encoder.fingerprint_branch.') for name in current.keys() - original.keys())
    molecules = Batch.from_data_list(molecule_list())
    torch.testing.assert_close(candidate.encode_mol(molecules), baseline.encode_mol(molecules), rtol=0, atol=0)
    torch.testing.assert_close(candidate.encode_spec(*spectra()), baseline.encode_spec(*spectra()), rtol=0, atol=0)
    # With no new dropout, the first training forward also consumes exactly the inherited RNG stream.
    baseline.train()
    candidate.train()
    torch.manual_seed(81)
    expected = baseline.encode_mol(molecules)
    state = torch.get_rng_state()
    torch.manual_seed(81)
    torch.testing.assert_close(candidate.encode_mol(molecules), expected, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('delta', [False, True])
def test_original_optimizer_groups_train_both_towers_and_new_branch_then_reload(tmp_path, qk, delta):
    config = parent_config(qk, delta)
    model = GraphFingerprintAlignmentModel(parent_model_config=config, fingerprint_config=settings())
    # Match the production optimizer's module grouping: extra parameters belong to mol_encoder.
    groups = [model.spec_encoder.parameters(), model.mol_encoder.parameters(), model.spec_proj.parameters(),
              model.mol_proj.parameters(), [model.logit_scale]]
    optimizer = torch.optim.AdamW([{'params': parameters} for parameters in groups], lr=.002)
    tracked = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len(tracked) == len({id(p) for p in tracked}) == len(list(model.parameters()))
    molecules = Batch.from_data_list(molecule_list())
    for step in range(2):
        optimizer.zero_grad()
        spectrum, molecule, scale = model(*spectra(), molecules)
        scores = F.normalize(spectrum, dim=-1) @ F.normalize(molecule, dim=-1).T * scale
        loss = F.cross_entropy(scores, torch.tensor([0, 1]))
        loss.backward()
        modules = [model.spec_encoder, model.mol_encoder.convs, model.spec_proj, model.mol_proj,
                   model.mol_encoder.fingerprint_branch[-1]]
        if step:
            modules.extend([model.mol_encoder.fingerprint_branch[0], model.mol_encoder.fingerprint_branch[1]])
        for component in modules:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                       for p in component.parameters())
        optimizer.step()
    path = tmp_path / 'synthetic.pth'
    torch.save(model.state_dict(), path)
    restored = GraphFingerprintAlignmentModel(parent_model_config=config, fingerprint_config=settings()).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(model.eval().encode_mol(molecules, True), restored.encode_mol(molecules, True), rtol=0, atol=0)
    state = torch.load(path, weights_only=True)
    del state['mol_encoder.fingerprint_branch.3.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(state, strict=True)


def test_active_fingerprint_effect_batch_order_chunks_and_spectrum_independence():
    model = GraphFingerprintAlignmentModel(parent_model_config=parent_config(), fingerprint_config=settings()).double().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.mol_encoder.fingerprint_branch[-1].weight, std=.05)
    graphs = [graph.clone() for graph in molecule_list()]
    for graph in graphs:
        graph.fingerprints = graph.fingerprints.double()
        graph.graph_size_features = graph.graph_size_features.double()
    batch = Batch.from_data_list(graphs)
    with torch.inference_mode():
        encoded = model.encode_mol(batch, True)
        order = [2, 0, 3, 1]
        torch.testing.assert_close(model.encode_mol(Batch.from_data_list([graphs[i] for i in order]), True), encoded[order])
        chunks = torch.cat([model.encode_mol(Batch.from_data_list(graphs[:2]), True),
                            model.encode_mol(Batch.from_data_list(graphs[2:]), True)])
        torch.testing.assert_close(chunks, encoded)
        wrong_rows = batch.clone()
        wrong_rows.fingerprints = wrong_rows.fingerprints.roll(1, 0)
        assert not torch.allclose(model.encode_mol(wrong_rows, True), encoded)
        unchanged_graph = model.mol_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                            batch.graph_size_features, fingerprints=batch.fingerprints)
        for parameter in model.spec_encoder.parameters():
            parameter.add_(.1)
        torch.testing.assert_close(model.encode_mol(batch, True), encoded, atol=0, rtol=0)
        # Independent direct residual formula, separate from encode_mol/projection.
        from SpecEmbedding.models_align import GINEEncoder
        graph_only = GINEEncoder.forward(model.mol_encoder, batch.x, batch.edge_index, batch.edge_attr,
                                         batch.batch, batch.graph_size_features)
        branch = model.mol_encoder.fingerprint_branch
        hidden = F.linear(batch.fingerprints, branch[0].weight, branch[0].bias)
        hidden = F.layer_norm(hidden, (12,), branch[1].weight, branch[1].bias, settings()['norm_eps'])
        residual = F.linear(F.gelu(hidden), branch[-1].weight, branch[-1].bias)
        torch.testing.assert_close(unchanged_graph, graph_only + residual, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('damage', ['absent', 'wrong_rows', 'wrong_bits', 'integer', 'one_dimensional'])
def test_missing_or_misaligned_input_shapes_fail(damage):
    model = GraphFingerprintAlignmentModel(parent_model_config=parent_config(), fingerprint_config=settings())
    batch = Batch.from_data_list(molecule_list())
    if damage == 'absent':
        del batch.fingerprints
    elif damage == 'wrong_rows':
        batch.fingerprints = batch.fingerprints[:-1]
    elif damage == 'wrong_bits':
        batch.fingerprints = batch.fingerprints[:, :-1]
    elif damage == 'integer':
        batch.fingerprints = batch.fingerprints.long()
    else:
        batch.fingerprints = batch.fingerprints.flatten()
    with pytest.raises(ValueError, match='one floating fingerprint'):
        model.encode_mol(batch)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'bool_bits', 'odd_bits', 'zero_hidden', 'nan_eps', 'bool_eps'])
def test_invalid_settings_fail_before_model_construction(damage):
    values = settings()
    if damage == 'missing':
        del values['input_bits']
    elif damage == 'extra':
        values['dropout'] = .1
    elif damage == 'bool_bits':
        values['input_bits'] = True
    elif damage == 'odd_bits':
        values['input_bits'] = 31
    elif damage == 'zero_hidden':
        values['hidden_dim'] = 0
    elif damage == 'nan_eps':
        values['norm_eps'] = float('nan')
    else:
        values['norm_eps'] = True
    with pytest.raises(ValueError):
        validate_fingerprint_residual(values)


def test_metadata_is_independent_and_formal_constructor_requires_complete_new_settings():
    parent, fingerprint = parent_config(True, True), settings()
    expected_parent, expected_fingerprint = copy.deepcopy(parent), copy.deepcopy(fingerprint)
    model = GraphFingerprintAlignmentModel(parent_model_config=parent, fingerprint_config=fingerprint)
    parent['align']['tau'] = 999
    fingerprint['hidden_dim'] = 999
    metadata = model.construction_config()
    assert metadata['align'] == expected_parent['align'] and metadata['fingerprint_residual'] == expected_fingerprint
    metadata['fingerprint_residual']['input_bits'] = 999
    assert model.construction_config()['fingerprint_residual'] == expected_fingerprint
    assert formal_model_type(model.construction_config()) == 'gine_fingerprint'
    restored = build_formal_alignment(model.construction_config()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    molecules = Batch.from_data_list(molecule_list())
    torch.testing.assert_close(restored.encode_mol(molecules), model.eval().encode_mol(molecules), rtol=0, atol=0)
    incomplete = model.construction_config()
    del incomplete['fingerprint_residual']
    with pytest.raises(ValueError, match='formal model configuration'):
        formal_model_type(incomplete)
    with pytest.raises(ValueError, match='GINE parent'):
        GraphFingerprintAlignmentModel(parent_model_config=definition('fingerprint'), fingerprint_config=settings())
