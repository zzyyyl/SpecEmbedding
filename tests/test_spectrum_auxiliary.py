"""Synthetic auxiliary-target mathematics and gradients, never a formal retrieval trial."""

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

from SpecEmbedding.models_spectrum_aux import MoleculeSpectrumAuxiliaryHead, sparse_spectrum_cosine_loss
from SpecEmbedding.utils.spectrum_targets import sparse_peak_target, validate_target_settings

TARGET = {'bins_per_da': 10, 'max_mz': 5, 'intensity_transform': 'linear_sum_l2',
          'overflow_policy': 'single_bin', 'adducts': ['[M+H]+', '[M+Na]+'],
          'unknown_id': 2, 'unknown_policy': 'zero_condition'}
HEAD = {'embedding_dim': 8, 'hidden_dim': 4, 'normalization_eps': 1e-12, 'initialization_seed': 42}


def test_raw_peaks_sum_before_normalization_with_overflow_and_extreme_intensities():
    bins, values, counts = sparse_peak_target([1.01, 1.09, 2.1, 5., 1e300], [2., 3., 0., 4., 8.], TARGET)
    np.testing.assert_array_equal(bins, [10, 50])
    np.testing.assert_allclose(values, np.array([5., 12.]) / 13, rtol=1e-7)
    assert counts == {'raw_peaks': 5, 'overflow_peaks': 2, 'zero_intensity_peaks': 1, 'float32_underflow_bins': 0}
    large = sparse_peak_target([1., 2.], [1e308, 1e308], TARGET)[1]
    np.testing.assert_allclose(large, np.sqrt(.5))
    permutation = [4, 2, 1, 3, 0]
    permuted = sparse_peak_target(np.array([1.01, 1.09, 2.1, 5., 1e300])[permutation],
                                 np.array([2., 3., 0., 4., 8.])[permutation], TARGET)
    np.testing.assert_array_equal(permuted[0], bins)
    np.testing.assert_array_equal(permuted[1], values)


@pytest.mark.parametrize('mz,values', [([], []), ([1], [0]), ([1], [-1]), ([-1], [1]),
                                     ([float('nan')], [1]), ([1], [float('inf')]), ([1, 2], [1])])
def test_invalid_targets_never_silently_drop_queries(mz, values):
    with pytest.raises(ValueError, match='observed spectrum'):
        sparse_peak_target(mz, values, TARGET)


@pytest.mark.parametrize('change', [{'bins_per_da': True}, {'max_mz': 0}, {'overflow_policy': 'drop'},
                                    {'adducts': ['[M+Na]+', '[M+H]+']}, {'unknown_id': 0}])
def test_target_protocol_is_explicit(change):
    with pytest.raises(ValueError):
        validate_target_settings({**TARGET, **change})


def test_head_preserves_rng_and_parent_weights_and_matches_independent_formula():
    torch.manual_seed(29)
    parent = nn.Linear(3, 8)
    before = copy.deepcopy(parent.state_dict())
    rng = torch.get_rng_state().clone()
    head = MoleculeSpectrumAuxiliaryHead(HEAD, TARGET).double()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, parent.state_dict()[key]) for key, value in before.items())
    embedding = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]] * 3, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 1, 2])
    actual = head(embedding, ids)
    normalized = embedding / torch.sqrt((embedding * embedding).sum(1, keepdim=True))
    condition = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
    x = torch.cat((normalized, condition), dim=1) @ head.input.weight.T + head.input.bias
    logits = (x * torch.sigmoid(x)) @ head.output.weight.T + head.output.bias
    reference = torch.log1p(torch.exp(logits))
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-15)
    assert not torch.equal(actual[0], actual[1]) and not torch.equal(actual[0], actual[2])
    assert torch.equal(rng, torch.get_rng_state())
    for key in head.state_dict():
        assert torch.equal(head.state_dict()[key], MoleculeSpectrumAuxiliaryHead(HEAD, TARGET).double().state_dict()[key])


def test_sparse_cosine_uses_all_prediction_bins_and_matches_dense_loss_and_gradients():
    predicted = torch.tensor([[.2, .8, .7, .1], [.6, .2, .4, .5]], dtype=torch.float64, requires_grad=True)
    ptr, bins = torch.tensor([0, 2, 3]), torch.tensor([0, 2, 1])
    values = torch.tensor([.6, .8, 1.], dtype=torch.float64)
    loss = sparse_spectrum_cosine_loss(predicted, ptr, bins, values, eps=1e-12)
    dense = torch.tensor([[.6, 0., .8, 0.], [0., 1., 0., 0.]], dtype=torch.float64)
    reference = torch.stack([1 - sum(a * b) / torch.sqrt(sum(a * a)) for a, b in zip(predicted, dense)]).mean()
    torch.testing.assert_close(loss, reference, rtol=0, atol=1e-15)
    gradient = torch.autograd.grad(loss, predicted, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(reference, predicted)[0], rtol=0, atol=1e-15)
    assert gradient[0, 1] > 0 and gradient[0, 3] > 0  # Non-target peaks affect the denominator.


@pytest.mark.parametrize('damage', ['empty_row', 'duplicate', 'outside', 'unnormalized', 'nonfinite', 'negative'])
def test_malformed_sparse_loss_inputs_rejected(damage):
    predicted = torch.ones(2, 4)
    ptr, bins, values = torch.tensor([0, 2, 3]), torch.tensor([0, 2, 1]), torch.tensor([.6, .8, 1.])
    if damage == 'empty_row':
        ptr[1] = 0
    elif damage == 'duplicate':
        bins[1] = 0
    elif damage == 'outside':
        bins[1] = 4
    elif damage == 'unnormalized':
        values[-1] = .5
    elif damage == 'nonfinite':
        values[-1] = torch.nan
    else:
        predicted[0, 0] = -1
    with pytest.raises(ValueError):
        sparse_spectrum_cosine_loss(predicted, ptr, bins, values, eps=1e-12)


def test_two_steps_reach_shared_embedding_and_zero_weight_matches_parent():
    torch.manual_seed(11)
    encoder = nn.Linear(3, 8)
    original = copy.deepcopy(encoder)
    head = MoleculeSpectrumAuxiliaryHead(HEAD, TARGET)
    optim = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=.01)
    control = torch.optim.AdamW(original.parameters(), lr=.01)
    inputs = torch.tensor([[1., .5, -.3], [.2, -.8, .4]])
    ptr, bins, values, ids = torch.tensor([0, 1, 2]), torch.tensor([10, 20]), torch.ones(2), torch.tensor([0, 1])
    for _ in range(2):
        optim.zero_grad()
        control.zero_grad()
        embedding, baseline = encoder(inputs), original(inputs)
        extra = sparse_spectrum_cosine_loss(head(embedding, ids), ptr, bins, values, eps=1e-12)
        (embedding.square().mean() + 0 * extra).backward()
        baseline.square().mean().backward()
        optim.step()
        control.step()
        for left, right in zip(encoder.parameters(), original.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    before = copy.deepcopy(encoder.state_dict())
    for _ in range(2):
        optim.zero_grad()
        loss = sparse_spectrum_cosine_loss(head(encoder(inputs), ids), ptr, bins, values, eps=1e-12)
        loss.backward()
        assert encoder.weight.grad.norm() > 0
        optim.step()
    assert not torch.equal(before['weight'], encoder.weight)


def test_strict_roundtrip_batch_order_and_registered_parameter_budget():
    head = MoleculeSpectrumAuxiliaryHead(HEAD, TARGET).eval()
    x, ids = torch.randn(3, 8), torch.tensor([0, 1, 2])
    expected = head(x, ids)
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    restored = MoleculeSpectrumAuxiliaryHead(HEAD, TARGET).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(restored(x, ids), expected)
    torch.testing.assert_close(restored(x[[2, 0, 1]], ids[[2, 0, 1]]), expected[[2, 0, 1]])
    torch.testing.assert_close(torch.cat([restored(x[i:i+1], ids[i:i+1]) for i in range(3)]), expected)
    from SpecEmbedding.config import config
    preset = config.molecule_spectrum_auxiliary.to_dict()
    registered = MoleculeSpectrumAuxiliaryHead(preset['model'], preset['target'])
    assert registered.output_dim == 29551
    assert sum(p.numel() for p in registered.parameters()) == 1953775
    full_embeddings = torch.ones(3, 512, requires_grad=True)
    full_predictions = registered(full_embeddings, ids)
    full_loss = sparse_spectrum_cosine_loss(full_predictions, torch.tensor([0, 1, 2, 3]),
                                          torch.tensor([10, 2048, 29550]), torch.ones(3), eps=1e-12)
    full_loss.backward()
    assert torch.isfinite(full_loss) and full_embeddings.grad.norm() > 0
    assert torch.isfinite(full_embeddings.grad).all()


def test_formal_alignment_still_refuses_unintegrated_auxiliary_feature():
    from SpecEmbedding.utils.formal_alignment import build_formal_alignment
    from tests.test_formal_alignment import model_config
    with pytest.raises(ValueError, match='Incomplete formal model'):
        build_formal_alignment({**model_config(), 'spectrum_auxiliary': {'model': HEAD, 'target': TARGET}})


@pytest.mark.parametrize('ids', [None, torch.tensor([0., 1.]), torch.tensor([0, 3]), torch.tensor([0])])
def test_head_requires_explicit_valid_observed_conditions(ids):
    with pytest.raises(ValueError, match='adduct IDs'):
        MoleculeSpectrumAuxiliaryHead(HEAD, TARGET)(torch.ones(2, 8), ids)
