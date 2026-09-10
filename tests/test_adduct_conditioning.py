import copy

import pytest
import torch

from SpecEmbedding.models_adduct import AdductConditionedEncoder, AdductFiLM, validate_adduct_settings
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder

SETTINGS = {'adducts': ['[M+H]+', '[M+Na]+'], 'unknown_id': 2, 'unknown_policy': 'identity'}


def parent_config(qk=False, pool=False):
    result = {'embedding_dim': 16, 'n_head': 4, 'n_layer': 2, 'dim_feedward': 24,
              'dim_target': 12, 'feedward_activation': 'selu'}
    if qk:
        result['qk_norm'] = {'eps': 1e-6}
    if pool:
        result['attention_pool'] = {'norm_eps': 1e-5}
    return result


def peaks():
    mz = torch.tensor([[300., 45., 80., 0.], [400., 70., 120., 160.], [250., 50., 0., 0.]])
    intensity = torch.tensor([[2., .4, 1., 0.], [2., .2, 1., .7], [2., 1., 0., 0.]])
    return mz, intensity, intensity == 0, torch.tensor([0, 1, 2])


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('pool', [False, True])
def test_identity_initialization_preserves_weights_rng_and_outputs(qk, pool):
    torch.manual_seed(41)
    parent = build_spectrum_encoder(parent_config(qk, pool)).eval()
    before = copy.deepcopy(parent.state_dict())
    rng = torch.get_rng_state().clone()
    encoder = AdductConditionedEncoder(parent, SETTINGS).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, parent.state_dict()[key]) for key, value in before.items())
    assert sum(p.numel() for p in encoder.parameters()) - sum(p.numel() for p in parent.parameters()) == 64
    mz, intensity, padding, ids = peaks()
    torch.testing.assert_close(encoder(mz, intensity, padding, ids), parent(mz, intensity, padding), rtol=0, atol=1e-6)


def test_independent_affine_reference_unknown_identity_and_padding_gradients():
    layer = AdductFiLM(4, SETTINGS).double()
    with torch.no_grad():
        layer.affine.copy_(torch.arange(16, dtype=torch.double).reshape(2, 2, 4) / 10)
    features = torch.arange(24, dtype=torch.double).reshape(3, 2, 4).requires_grad_()
    padding = torch.tensor([[False, True], [False, False], [False, True]])
    ids = torch.tensor([0, 1, 2])
    expected = torch.zeros_like(features)
    for b in range(3):
        for token in range(2):
            for dim in range(4):
                if not padding[b, token]:
                    scale = 1 if b == 2 else 1 + layer.affine[b, 0, dim]
                    bias = 0 if b == 2 else layer.affine[b, 1, dim]
                    expected[b, token, dim] = features[b, token, dim] * scale + bias
    result = layer(features, padding, ids)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.equal(features.grad[padding], torch.zeros_like(features.grad[padding]))
    torch.testing.assert_close(features.grad[2, 0], torch.ones(4, dtype=torch.double))
    reference = torch.zeros_like(layer.affine)
    reference[0, 0], reference[0, 1] = features.detach()[0, 0], 1
    reference[1, 0], reference[1, 1] = features.detach()[1].sum(0), 2
    torch.testing.assert_close(layer.affine.grad, reference)
    poisoned = features.detach().clone()
    poisoned[padding] = torch.nan
    torch.testing.assert_close(layer(poisoned, padding, ids), result)


def test_unknown_has_no_conditioning_gradient_after_nonzero_parameters():
    layer = AdductFiLM(8, SETTINGS)
    with torch.no_grad():
        layer.affine.fill_(.7)
    features = torch.randn(2, 3, 8, requires_grad=True)
    padding = torch.zeros(2, 3, dtype=torch.bool)
    result = layer(features, padding, torch.tensor([2, 2]))
    assert torch.equal(result, features)
    result.sum().backward()
    assert torch.count_nonzero(layer.affine.grad) == 0
    assert torch.equal(features.grad, torch.ones_like(features))


@pytest.mark.parametrize('qk', [False, True])
@pytest.mark.parametrize('pool', [False, True])
def test_updates_backbone_and_both_adducts_then_preserves_permutation_chunking_and_reload(tmp_path, qk, pool):
    torch.manual_seed(29)
    config = parent_config(qk, pool)
    encoder = AdductConditionedEncoder(build_spectrum_encoder(config), SETTINGS)
    mz, intensity, padding, ids = peaks()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=.001)
    target = torch.randn(3, 12)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (encoder(mz, intensity, padding, ids) - target).square().mean()
        loss.backward()
        assert all(torch.count_nonzero(encoder.conditioning.affine.grad[i]) > 0 for i in (0, 1))
        backbone_grads = [p.grad for p in encoder.encoder.parameters() if p.grad is not None]
        assert backbone_grads and all(torch.isfinite(g).all() for g in backbone_grads)
        assert any(torch.count_nonzero(g) > 0 for g in backbone_grads)
        optimizer.step()
    encoder.eval()
    actual = encoder(mz, intensity, padding, ids)
    swapped_ids = torch.tensor([1, 0, 2])
    swapped = encoder(mz, intensity, padding, swapped_ids)
    assert not torch.allclose(actual[:2], swapped[:2])
    assert torch.equal(actual[2], swapped[2])
    permutation = torch.tensor([0, 3, 1, 2])
    torch.testing.assert_close(encoder(mz[:, permutation], intensity[:, permutation], padding[:, permutation], ids),
                               actual, rtol=1e-5, atol=1e-6)
    chunks = torch.cat([encoder(mz[i:i + 1], intensity[i:i + 1], padding[i:i + 1], ids[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(chunks, actual, rtol=1e-5, atol=1e-6)
    path = tmp_path / 'component.pt'
    torch.save({'model_config': config, 'adduct_settings': SETTINGS, 'state_dict': encoder.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = AdductConditionedEncoder(build_spectrum_encoder(saved['model_config']), saved['adduct_settings']).eval()
    restored.load_state_dict(saved['state_dict'], strict=True)
    torch.testing.assert_close(restored(mz, intensity, padding, ids), actual, rtol=0, atol=0)
    damaged = {key: value for key, value in saved['state_dict'].items() if key != 'conditioning.affine'}
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(damaged, strict=True)


@pytest.mark.parametrize('bad', [None, torch.tensor([0., 1.]), torch.tensor([0]), torch.tensor([-1, 0]),
                               torch.tensor([0, 3]), torch.tensor([[0], [1]])])
def test_invalid_or_missing_ids_fail(bad):
    with pytest.raises(ValueError, match='Adduct IDs'):
        AdductFiLM(4, SETTINGS)(torch.ones(2, 3, 4), torch.zeros(2, 3, dtype=torch.bool), bad)


@pytest.mark.parametrize('damage', ['vocabulary', 'unknown_id', 'unknown_policy', 'extra', 'missing'])
def test_unregistered_settings_fail(damage):
    settings = copy.deepcopy(SETTINGS)
    if damage == 'vocabulary':
        settings['adducts'].reverse()
    elif damage == 'unknown_id':
        settings['unknown_id'] = True
    elif damage == 'unknown_policy':
        settings['unknown_policy'] = 'default_to_H'
    elif damage == 'extra':
        settings['collision_energy'] = True
    else:
        settings.pop('unknown_policy')
    with pytest.raises(ValueError, match='explicit ordered'):
        validate_adduct_settings(settings)


def test_formal_builder_still_refuses_unintegrated_adduct_input():
    with pytest.raises(ValueError, match='Incomplete spectral configuration'):
        build_spectrum_encoder({**parent_config(), 'adduct_conditioning': SETTINGS})


def test_empty_or_nonfinite_valid_spectra_and_unsupported_forward_fail():
    layer = AdductFiLM(4, SETTINGS)
    with pytest.raises(ValueError, match='at least one'):
        layer(torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='Non-finite'):
        layer(torch.full((2, 3, 4), torch.nan), torch.zeros(2, 3, dtype=torch.bool), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='Unsupported parent'):
        AdductConditionedEncoder(torch.nn.Identity(), SETTINGS)


def test_registered_width512_cost_and_no_rng_use():
    state = torch.get_rng_state().clone()
    layer = AdductFiLM(512, SETTINGS)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in layer.parameters()) == 2048
