"""Synthetic CPU validation of an opt-in attention change, not model performance."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from SpecEmbedding.config import ConfigObject
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.models_qk_norm import QKNormEncoderLayer
from SpecEmbedding.utils.alignment_successor import qk_successor_configuration
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from tests.test_formal_alignment import checkpoint_fixture, model_config
from tests.test_precursor_delta import delta_config, spectra


def definition(kind='gine', delta=False):
    config = model_config(kind)
    config['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    if delta:
        config['spec_encoder']['precursor_delta'] = delta_config()
    return config


def test_attention_matches_independent_per_head_formula_and_eval_cannot_bypass_norm():
    torch.manual_seed(13)
    layer = QKNormEncoderLayer(nn.TransformerEncoderLayer(8, 2, 12, dropout=0., batch_first=True,
                                                        norm_first=True), {'eps': 1e-6}).double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    padding = torch.tensor([[False, False, False, True, True], [False] * 5])
    projected = x @ layer.self_attn.in_proj_weight.T + layer.self_attn.in_proj_bias
    expected = []
    for row in range(2):
        heads = []
        for head in range(2):
            q, k, v = [projected[row, :, offset + 4 * head:offset + 4 * (head + 1)] for offset in (0, 8, 16)]
            q = q / (q.square().mean(-1, keepdim=True) + 1e-6).sqrt() * layer.q_norm.weight
            k = k / (k.square().mean(-1, keepdim=True) + 1e-6).sqrt() * layer.k_norm.weight
            logits = (q @ k.T / 2).masked_fill(padding[row].unsqueeze(0), -torch.inf)
            heads.append(logits.softmax(-1) @ v)
        expected.append(torch.cat(heads, dim=-1))
    expected = torch.stack(expected) @ layer.self_attn.out_proj.weight.T + layer.self_attn.out_proj.bias
    torch.testing.assert_close(layer.attention(x, padding), expected, atol=1e-12, rtol=1e-12)
    encoded = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, batch_first=True,
                                                              norm_first=True), 1, enable_nested_tensor=False)
    encoded.layers = nn.ModuleList([layer])
    encoded.eval()
    # A fused TransformerEncoderLayer forward can silently skip a custom _sa_block.
    with torch.inference_mode(), patch('torch._transformer_encoder_layer_fwd', side_effect=AssertionError('bypass')):
        output = encoded(x, src_key_padding_mask=padding)
        layer.q_norm.weight.mul_(2)
        changed = encoded(x, src_key_padding_mask=padding)
    assert not torch.allclose(output[~padding], changed[~padding])


@pytest.mark.parametrize('delta', [False, True])
def test_inherited_dual_tower_weights_rng_and_non_attention_modules_stay_identical(delta):
    config = definition(delta=delta)
    parent = copy.deepcopy(config)
    del parent['spec_encoder']['qk_norm']
    torch.manual_seed(42)
    original = build_formal_alignment(parent)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = build_formal_alignment(config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in original.state_dict().items())
    extra = set(candidate.state_dict()) - set(original.state_dict())
    assert extra == {'spec_encoder._encoder.layers.0.q_norm.weight', 'spec_encoder._encoder.layers.0.k_norm.weight'}
    assert sum(candidate.state_dict()[key].numel() for key in extra) == 8


@pytest.mark.parametrize('delta', [False, True])
def test_padding_permutation_eval_and_training_attention_semantics(delta):
    torch.manual_seed(7)
    encoder = build_spectrum_encoder(definition(delta=delta)['spec_encoder']).eval()
    mz, intensity, mask = spectra()
    with torch.inference_mode():
        expected = encoder(mz, intensity, mask)
        order = [0, 3, 2, 1, 4]
        torch.testing.assert_close(expected, encoder(mz[:, order], intensity[:, order], mask[:, order]))
        torch.testing.assert_close(expected, encoder(F.pad(mz, (0, 2), value=777),
                                                    F.pad(intensity, (0, 2), value=100), F.pad(mask, (0, 2), value=True)))
    layer = encoder._encoder.layers[0]
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    layer.self_attn.dropout = 0.
    train = encoder.train()(mz, intensity, mask)
    with torch.inference_mode():
        torch.testing.assert_close(train, encoder.eval()(mz, intensity, mask))
    canonical = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, -torch.inf)
    hidden = torch.randn(2, 5, 8)
    torch.testing.assert_close(layer(hidden, src_key_padding_mask=mask), layer(hidden, src_key_padding_mask=canonical))


@pytest.mark.parametrize('kind', ['gine', 'fingerprint'])
@pytest.mark.parametrize('delta', [False, True])
def test_contrastive_training_reaches_norms_and_both_towers_then_strictly_reloads(tmp_path, kind, delta):
    torch.manual_seed(42)
    config = definition(kind, delta)
    model = build_formal_alignment(config)
    molecules = (torch.randint(0, 2, (3, 2048)).float() if kind == 'fingerprint' else
                 Batch.from_data_list([smiles_to_graph(s, graph_policy='rdkit_sanitized') for s in ('CCO', 'CCC', 'CC')]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        spec = model.encode_spec(*spectra(), normalize=True)
        mol = model.encode_mol(molecules, normalize=True)
        loss = F.cross_entropy(spec @ mol.T * model.get_logit_scale(), torch.tensor([0, 1]))
        loss.backward()
        layer = model.spec_encoder._encoder.layers[0]
        for module in (model.spec_encoder.embedding, layer.q_norm, layer.k_norm, model.mol_encoder):
            assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
                       for p in module.parameters())
        optimizer.step()
    path, _, selection, protocol = checkpoint_fixture(tmp_path, kind)
    torch.save(model.state_dict(), path)
    selection['model_config'] = config
    selection['config_snapshot']['model'] = config
    selection['checkpoint_sha256'] = sha256_file(path)
    metadata = tmp_path / 'alignment_selection.json'
    metadata.write_text(json.dumps(selection))
    restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
    assert receipt['model_config'] == config
    torch.testing.assert_close(model.eval().encode_spec(*spectra()), restored.encode_spec(*spectra()), atol=0, rtol=0)
    weights = model.state_dict()
    del weights['spec_encoder._encoder.layers.0.q_norm.weight']
    torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(path)
    metadata.write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match='Missing key'):
        load_formal_alignment(path, torch.device('cpu'), **protocol)


@pytest.mark.parametrize('value', [None, {}, {'eps': True}, {'eps': 0}, {'eps': -1}, {'eps': float('nan')},
                                  {'eps': float('inf')}, {'eps': 1e-100}, {'eps': 1e100}, {'eps': 1e-6, 'unknown': 1}])
def test_invalid_or_incomplete_configuration_is_rejected(value):
    config = definition()['spec_encoder']
    config['qk_norm'] = value
    with pytest.raises(ValueError):
        build_spectrum_encoder(config)


@pytest.mark.parametrize('mode', ['causal', 'mask', 'all_padding', 'shape', 'additive'])
def test_unsupported_masks_fail_explicitly(mode):
    layer = build_spectrum_encoder(definition()['spec_encoder'])._encoder.layers[0]
    x = torch.randn(2, 5, 8)
    args = {'causal': {'is_causal': True}, 'mask': {'src_mask': torch.zeros(5, 5)},
            'all_padding': {'src_key_padding_mask': torch.ones(2, 5, dtype=torch.bool)},
            'shape': {'src_key_padding_mask': torch.zeros(2, 4, dtype=torch.bool)},
            'additive': {'src_key_padding_mask': torch.ones(2, 5)}}[mode]
    with pytest.raises(ValueError):
        layer(x, **args)


@pytest.mark.parametrize('delta', [False, True])
def test_successor_only_adds_qk_norm_and_retains_parent_delta_and_scientific_settings(delta):
    runtime = {'model': model_config(), 'train': {'align': {'batch_size': 128}}, 'augmentation': {'prob': .5},
               'data': {'cache_path': '/old/cache', 'tokenizer': {'max_len': 100}}, 'fulltrain': {}}
    if delta:
        runtime['model']['spec_encoder']['precursor_delta'] = delta_config()
    original = copy.deepcopy(runtime)
    selection = {'model_config': runtime['model'], 'training_config': runtime['train']['align'],
                 'config_snapshot': {'augmentation': runtime['augmentation']}}
    storage = {'storage': {'root': '/tmp/approved'}, 'data': {'cache_path': '/tmp/approved/train_cache'}}
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    result = qk_successor_configuration(runtime, selection, {'eps': 1e-6}, gpu, storage_template=storage)
    expected = copy.deepcopy(runtime)
    expected['model']['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    expected['storage'] = storage['storage']
    expected['data']['cache_path'] = storage['data']['cache_path']
    expected['fulltrain'] = gpu
    assert result == expected and runtime == original
    selection['model_config'] = result['model']
    with pytest.raises(ValueError, match='already'):
        qk_successor_configuration(result, selection, {'eps': 1e-6}, gpu, storage_template=storage)


def test_queue_requires_independent_parent_construction_before_reading_inputs(tmp_path, monkeypatch):
    import run_massspecgym_v15 as runner

    settings = copy.deepcopy(runner.config.to_dict())
    settings['model'] = definition()
    monkeypatch.setattr(runner, 'config', ConfigObject(settings))
    monkeypatch.setattr(runner, 'storage_receipt', lambda _: {'synthetic': True})
    args = SimpleNamespace(output_root=tmp_path, molecule_input='gine', checkpoint_model_config=False)
    with pytest.raises(ValueError, match='independent baseline'):
        runner.preflight(args)
