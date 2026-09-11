"""Independent attention algebra and real synthetic contrastive updates for prepared SDPA gates."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.models_attention_gate import (
    GATE_SETTINGS,
    GatedAttentionEncoderLayer,
    build_gated_spectrum_encoder,
    enable_attention_gate,
    validate_attention_gate,
)
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.models_qk_norm import QKNormEncoderLayer


def original_layer(qk=False, dropout=0.):
    layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=dropout, activation='gelu', batch_first=True, norm_first=True).double()
    return QKNormEncoderLayer(layer, {'eps': 1e-6}) if qk else layer


def parent_config(qk=False, pool=False, adduct=False):
    result = {'embedding_dim': 8, 'n_head': 2, 'n_layer': 2, 'dim_feedward': 16,
              'dim_target': 8, 'feedward_activation': 'selu'}
    if qk:
        result['qk_norm'] = {'eps': 1e-6}
    if pool:
        result['attention_pool'] = {'norm_eps': 1e-5}
    if adduct:
        result['adduct_conditioning'] = {'adducts': ['[M+H]+', '[M+Na]+'], 'unknown_id': 2, 'unknown_policy': 'identity'}
    return result


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('mask_style', ['none', 'bool', 'float'])
@pytest.mark.parametrize('mode', ['train', 'eval', 'inference'])
def test_zero_gate_preserves_weights_rng_and_valid_outputs(qk, mask_style, mode):
    parent = original_layer(qk, dropout=.2)
    inherited = copy.deepcopy(parent)
    rng = torch.get_rng_state()
    gated = GatedAttentionEncoderLayer(inherited, GATE_SETTINGS)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in parent.state_dict().items():
        assert torch.equal(value, gated.state_dict()[key])
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    padding = torch.tensor([[False, False, False, True], [False, True, True, True]])
    mask = None if mask_style == 'none' else padding
    if mask_style == 'float':
        mask = torch.zeros(2, 4, dtype=torch.float64).masked_fill(padding, float('-inf'))
    parent.train(mode == 'train')
    gated.train(mode == 'train')
    with torch.inference_mode() if mode == 'inference' else nullcontext():
        torch.manual_seed(71)
        expected = parent(x, src_key_padding_mask=mask)
        reference_rng = torch.get_rng_state()
        torch.manual_seed(71)
        observed = gated(x, src_key_padding_mask=mask)
        assert torch.equal(reference_rng, torch.get_rng_state())
    relevant = torch.ones_like(padding) if mask is None else ~padding
    torch.testing.assert_close(observed[relevant], expected[relevant], rtol=1e-7, atol=1e-8)
    assert torch.equal(gated.gate_values(gated.norm1(x)), torch.ones(2, 4, 2, dtype=x.dtype))


def independent_reference(layer, x, padding):
    """Loop over each head and query; do not call SDPA, gate_values or the layer forward."""
    clean = x.masked_fill(padding.unsqueeze(-1), 0)
    normalized = F.layer_norm(clean, (8,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
    projections = []
    for block in range(3):
        weight = layer.self_attn.in_proj_weight[block * 8:(block + 1) * 8]
        bias = layer.self_attn.in_proj_bias[block * 8:(block + 1) * 8]
        projections.append(torch.einsum('bld,od->blo', normalized, weight) + bias)
    rows = []
    for batch in range(x.shape[0]):
        heads = []
        for head in range(2):
            q, k, v = [values[batch, :, head * 4:(head + 1) * 4] for values in projections]
            if layer._qk_norm_enabled:
                q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + layer.q_norm.eps) * layer.q_norm.weight
                k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + layer.k_norm.eps) * layer.k_norm.weight
            tokens = []
            for query in range(x.shape[1]):
                logits = (k * q[query]).sum(-1) / 2
                weights = logits.masked_fill(padding[batch], float('-inf')).softmax(-1)
                gate = 2 * torch.sigmoid((normalized[batch, query] * layer.gate_weight[head]).sum())
                tokens.append((weights[:, None] * v).sum(0) * gate)
            heads.append(torch.stack(tokens))
        rows.append(torch.cat(heads, -1))
    attention = torch.stack(rows)
    hidden = clean + F.linear(attention, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
    normalized = F.layer_norm(hidden, (8,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
    feedforward = F.linear(F.gelu(F.linear(normalized, layer.linear1.weight, layer.linear1.bias)),
                           layer.linear2.weight, layer.linear2.bias)
    return (hidden + feedforward).masked_fill(padding.unsqueeze(-1), 0)


@pytest.mark.parametrize('qk', [False, True])
def test_nonzero_gate_forward_and_all_gradients_match_independent_head_formula(qk):
    layer = GatedAttentionEncoderLayer(original_layer(qk), GATE_SETTINGS)
    with torch.no_grad():
        layer.gate_weight.normal_(std=.3)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    observed = layer(x, src_key_padding_mask=padding)
    expected = independent_reference(layer, x, padding)
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-11)
    variables = (x, *layer.parameters())
    actual_gradients = torch.autograd.grad(observed.square().sum(), variables)
    reference_gradients = torch.autograd.grad(expected.square().sum(), variables)
    for actual, reference in zip(actual_gradients, reference_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)


def test_nonzero_gate_is_permutation_equivariant_and_has_no_cross_spectrum_information():
    layer = GatedAttentionEncoderLayer(original_layer(), GATE_SETTINGS).eval()
    with torch.no_grad():
        layer.gate_weight.normal_(std=.7)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    padding = torch.tensor([[False, False, True, True], [False, False, False, True]])
    observed = layer(x, src_key_padding_mask=padding)
    derivative = torch.autograd.grad(observed[0].square().sum(), x)[0]
    assert torch.count_nonzero(derivative[1]) == 0
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        reordered = layer(x[:, permutation], src_key_padding_mask=padding[:, permutation])
        torch.testing.assert_close(reordered, observed[:, permutation], rtol=1e-10, atol=1e-11)
        chunks = torch.cat([layer(x[i:i+1], src_key_padding_mask=padding[i:i+1]) for i in (0, 1)])
        torch.testing.assert_close(chunks, observed, rtol=1e-10, atol=1e-11)
        poisoned = x.detach().masked_fill(padding.unsqueeze(-1), float('nan'))
        torch.testing.assert_close(layer(poisoned, src_key_padding_mask=padding), observed, rtol=0, atol=0)
        single = layer(x[:, :1])
        assert single.shape == (2, 1, 8) and torch.isfinite(single).all()
        gates = layer.gate_values(layer.norm1(x))
        assert torch.all((gates > 0) & (gates < 2))
        assert not torch.equal(gates[:, 0], gates[:, 1])


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('pool', [False, True])
@pytest.mark.parametrize('adduct', [False, True])
def test_parent_spectrum_combinations_preserve_initialization_and_strict_nonzero_reload(tmp_path, qk, pool, adduct):
    config = parent_config(qk, pool, adduct)
    torch.manual_seed(42)
    original = build_spectrum_encoder(config).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    gated = build_gated_spectrum_encoder(config, GATE_SETTINGS).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in original.state_dict().items():
        assert torch.equal(value, gated.state_dict()[key])
    mz = torch.tensor([[200., 55., 70., 0.], [280., 74., 130., 150.], [99., 0., 0., 0.]])
    intensity = torch.tensor([[2., 1., .3, 0.], [2., .1, 1., .4], [2., 0., 0., 0.]])
    mask = mz == 0
    kwargs = {'adduct_ids': torch.tensor([0, 1, 2])} if adduct else {}
    with torch.inference_mode():
        initial = gated(mz, intensity, mask, **kwargs)
        torch.testing.assert_close(initial, original(mz, intensity, mask, **kwargs), rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        layers = [module for module in gated.modules() if isinstance(module, GatedAttentionEncoderLayer)]
        assert len(layers) == config['n_layer']
        for layer in layers:
            layer.gate_weight.normal_(std=.8)
    with torch.inference_mode():
        changed = gated(mz, intensity, mask, **kwargs)
        assert not torch.allclose(changed, initial, rtol=1e-5, atol=1e-6)
    path = tmp_path / 'synthetic_gate.pth'
    torch.save({'parent': config, 'gate': GATE_SETTINGS, 'weights': gated.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = build_gated_spectrum_encoder(saved['parent'], saved['gate']).eval()
    restored.load_state_dict(saved['weights'], strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(mz, intensity, mask, **kwargs), changed, rtol=0, atol=0)
    key = next(name for name in saved['weights'] if name.endswith('_encoder.layers.0.gate_weight'))
    saved['weights'].pop(key)
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(saved['weights'], strict=True)


@pytest.mark.parametrize('qk', [False, True])
def test_two_real_contrastive_steps_update_every_gate_and_both_towers(qk):
    from torch_geometric.data import Batch

    from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
    from tests.test_graph_global_context import graphs

    spectrum = build_gated_spectrum_encoder(parent_config(qk), GATE_SETTINGS)
    molecule = GINEEncoder(emb_dim=8, n_layers=2, dropout_rate=0., size_feature_dim=4, norm_type='layernorm', norm_eps=1e-5)
    model = SpecMolAlignModel(spectrum, molecule, 8, 8, 8, 0., .07)
    batch = Batch.from_data_list(graphs())
    count = batch.num_graphs
    mz = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) * 13 + 40
    intensity = torch.ones_like(mz)
    mask = torch.zeros_like(mz, dtype=torch.bool)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        spec = model.encode_spec(mz, intensity, mask, True)
        mol = model.encode_mol(batch, True)
        logits = model.get_logit_scale() * spec @ mol.T
        target = torch.arange(count)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        loss.backward()
        assert torch.isfinite(loss)
        for layer in spectrum._encoder.layers:
            assert torch.isfinite(layer.gate_weight.grad).all() and layer.gate_weight.grad.abs().sum() > 0
        optimizer.step()
    after = model.state_dict()
    assert all(torch.count_nonzero(layer.gate_weight) > 0 for layer in spectrum._encoder.layers)
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith('mol_encoder.'))
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith('spec_encoder.embedding.'))


@pytest.mark.parametrize('damage', ['missing', 'extra', 'scale', 'bias', 'bool_scale', 'numeric_bias', 'placement'])
def test_incomplete_or_unregistered_settings_refused(damage):
    settings = copy.deepcopy(GATE_SETTINGS)
    if damage == 'missing':
        settings.pop('initialization')
    elif damage == 'extra':
        settings['dropout'] = .1
    elif damage == 'scale':
        settings['scale'] = 1.
    elif damage == 'bias':
        settings['bias'] = True
    elif damage == 'bool_scale':
        settings['scale'] = True
    elif damage == 'numeric_bias':
        settings['bias'] = 0
    else:
        settings['placement'] = 'after_output_projection'
    with pytest.raises(ValueError):
        validate_attention_gate(settings)


def test_actual_parameter_budget_and_no_formal_activation():
    config = parent_config()
    config.update(embedding_dim=512, n_head=16, n_layer=4, dim_feedward=512, dim_target=512)
    original = build_spectrum_encoder(config)
    count = sum(p.numel() for p in original.parameters())
    gated = enable_attention_gate(original, GATE_SETTINGS)
    assert sum(p.numel() for p in gated.parameters()) - count == 32768
    with pytest.raises(ValueError, match='twice'):
        enable_attention_gate(gated, GATE_SETTINGS)
    with pytest.raises(ValueError, match='Incomplete spectral'):
        build_spectrum_encoder({**config, 'attention_output_gate': GATE_SETTINGS})


@pytest.mark.parametrize('damage', ['causal', 'attention_mask', 'shape', 'mask_shape', 'mask_values', 'mask_dtype', 'all_padding'])
def test_unsupported_forward_inputs_rejected(damage):
    layer = GatedAttentionEncoderLayer(original_layer(), GATE_SETTINGS)
    x = torch.ones(2, 3, 8, dtype=torch.float64)
    kwargs = {}
    if damage == 'causal':
        kwargs['is_causal'] = True
    elif damage == 'attention_mask':
        kwargs['src_mask'] = torch.zeros(3, 3)
    elif damage == 'shape':
        x = torch.ones(2, 3, 7)
    else:
        kwargs['src_key_padding_mask'] = {
            'mask_shape': torch.zeros(2, 2, dtype=torch.bool), 'mask_values': torch.ones(2, 3),
            'mask_dtype': torch.zeros(2, 3, dtype=torch.long), 'all_padding': torch.ones(2, 3, dtype=torch.bool),
        }[damage]
    with pytest.raises(ValueError):
        layer(x, **kwargs)
