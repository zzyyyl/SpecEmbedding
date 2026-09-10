"""Synthetic readout/component checks; no formal experiment or full integration claim."""

import copy
import io
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.models_attention_pool import AttentionPoolingEncoder, MaskedAttentionPool, validate_attention_pool
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.utils.formal_alignment import build_formal_alignment
from tests.test_formal_alignment import model_config
from tests.test_precursor_delta import delta_config, spectra


def wrap(model):
    model.spec_encoder = AttentionPoolingEncoder(model.spec_encoder, {'norm_eps': 1e-5})
    return model


def test_nonuniform_pool_matches_scalar_reference_and_padding_has_zero_gradient():
    pool = MaskedAttentionPool(3, norm_eps=1e-5).double()
    with torch.no_grad():
        pool.query.copy_(torch.tensor([.2, -.4, .7], dtype=torch.float64))
    x = torch.tensor([[[1., 3., -2.], [4., 0., 2.], [float('nan')] * 3]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, False, True]])
    output, weights = pool(x, mask, return_weights=True)
    logits = []
    for values in ([1., 3., -2.], [4., 0., 2.]):
        mean = sum(values) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / 3 + 1e-5)
        logits.append(sum(q * (v - mean) / std for q, v in zip((.2, -.4, .7), values)) / math.sqrt(3))
    a = [math.exp(value - max(logits)) for value in logits]
    a = [value / sum(a) for value in a]
    expected = torch.tensor([[a[0] + 4 * a[1], 3 * a[0], -2 * a[0] + 2 * a[1]]], dtype=torch.float64)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights, torch.tensor([[*a, 0.]], dtype=torch.float64))
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.equal(x.grad[mask], torch.zeros_like(x.grad[mask]))
    assert pool.query.grad.abs().sum() > 0 and torch.isfinite(pool.query.grad).all()


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('qk', [False, True])
def test_uniform_initialization_preserves_both_towers_rng_and_eval_outputs(kind, qk):
    config = model_config(kind)
    if qk:
        config['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    torch.manual_seed(42)
    original = build_formal_alignment(config).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    candidate_config = copy.deepcopy(config)
    candidate_config['spec_encoder']['attention_pool'] = {'norm_eps': 1e-5}
    candidate = build_formal_alignment(candidate_config).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in original.state_dict().items():
        target = 'spec_encoder.base.' + key[len('spec_encoder.'):] if key.startswith('spec_encoder.') else key
        assert torch.equal(value, candidate.state_dict()[target])
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters()) == 8
    with torch.inference_mode():
        torch.testing.assert_close(candidate.encode_spec(*spectra()), original.encode_spec(*spectra()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('qk', [False, True])
def test_learned_readout_preserves_permutation_padding_and_train_eval_paths(qk):
    config = model_config()
    if qk:
        config['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    encoder = wrap(build_formal_alignment(config)).spec_encoder
    with torch.no_grad():
        encoder.pool.query.copy_(torch.linspace(-.4, .8, 8))
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0
    mz, intensity, mask = spectra()
    expected = encoder.train()(mz, intensity, mask)
    with torch.inference_mode():
        encoder.eval()
        torch.testing.assert_close(encoder(mz, intensity, mask), expected)
        order = [0, 3, 1, 2, 4]
        torch.testing.assert_close(encoder(mz[:, order], intensity[:, order], mask[:, order]), expected)
        torch.testing.assert_close(encoder(F.pad(mz, (0, 3), value=float('nan')),
                                          F.pad(intensity, (0, 3), value=float('inf')),
                                          F.pad(mask, (0, 3), value=True)), expected)


def test_contrastive_step_updates_readout_and_both_towers_then_strictly_reloads():
    torch.manual_seed(13)
    model = wrap(build_formal_alignment(model_config()))
    molecules = Batch.from_data_list([smiles_to_graph(s, graph_policy='rdkit_sanitized') for s in ('CCO', 'CCC', 'CC')])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss = F.cross_entropy(model.encode_spec(*spectra(), normalize=True) @ model.encode_mol(molecules, True).T
                               * model.get_logit_scale(), torch.tensor([0, 1]))
        loss.backward()
        for part in (model.spec_encoder.pool, model.spec_encoder.base.embedding, model.mol_encoder):
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in part.parameters())
        optimizer.step()
    assert model.spec_encoder.pool.query.detach().abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = wrap(build_formal_alignment(model_config())).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored.encode_spec(*spectra()), model.eval().encode_spec(*spectra()), rtol=0, atol=0)
    weights = model.state_dict()
    del weights['spec_encoder.pool.query']
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(weights, strict=True)


@pytest.mark.parametrize('value', [None, {}, {'norm_eps': True}, {'norm_eps': 0}, {'norm_eps': -1},
                                  {'norm_eps': float('nan')}, {'norm_eps': float('inf')}, {'norm_eps': 1e-100},
                                  {'norm_eps': 1e100}, {'norm_eps': 1e-5, 'unknown': 1}])
def test_incomplete_or_invalid_settings_are_rejected(value):
    with pytest.raises(ValueError):
        validate_attention_pool(value)


@pytest.mark.parametrize('damage', ['all_padding', 'mask_dtype', 'mask_shape', 'width', 'empty'])
def test_invalid_pool_inputs_are_rejected(damage):
    features = torch.ones(2, 4, 3)
    padding = torch.zeros(2, 4, dtype=torch.bool)
    if damage == 'all_padding':
        padding[0] = True
    elif damage == 'mask_dtype':
        padding = padding.float()
    elif damage == 'mask_shape':
        padding = padding[:, :2]
    elif damage == 'width':
        features = features[:, :, :2]
    else:
        features, padding = features[:0], padding[:0]
    with pytest.raises(ValueError):
        MaskedAttentionPool(3, norm_eps=1e-5)(features, padding)


def test_incomplete_formal_configuration_and_other_forward_overrides_are_rejected():
    config = model_config()
    config['spec_encoder']['attention_pool'] = {}
    with pytest.raises(ValueError):
        build_formal_alignment(config)
    config = model_config()['spec_encoder']
    config['precursor_delta'] = delta_config()
    with pytest.raises(ValueError, match='supported SiameseModel'):
        AttentionPoolingEncoder(build_spectrum_encoder(config), {'norm_eps': 1e-5})
    config['attention_pool'] = {'norm_eps': 1e-5}
    with pytest.raises(ValueError, match='does not support precursor delta'):
        build_spectrum_encoder(config)
