"""Synthetic CPU checks for the optional downstream tower, never retrieval experiments."""

import copy
import io
import json
import math

import pytest
import torch
from torch.nn import functional as F

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from tests.test_formal_alignment import checkpoint_fixture, model_config


def delta_config():
    return {'fourier_dim': 8, 'hidden_dim': 8, 'min_wavelength': .01, 'max_wavelength': 10000.}


def definition(kind='gine'):
    config = model_config(kind)
    config['spec_encoder']['precursor_delta'] = delta_config()
    return config


def spectra():
    mz = torch.tensor([[200., 70., 210., 200., 0.], [300., 350., 150., 0., 0.]])
    intensity = torch.tensor([[2., 1., .3, .5, 0.], [2., .5, 1., 0., 0.]])
    mask = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    return mz, intensity, mask


def test_original_factory_keeps_all_weights_rng_and_output_identical():
    config = model_config()['spec_encoder']
    torch.manual_seed(42)
    original = SiameseModel(**config).eval()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    built = build_spectrum_encoder(config).eval()
    assert type(built) is SiameseModel
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(v, built.state_dict()[k]) for k, v in original.state_dict().items())
    assert torch.equal(original(*spectra()), built(*spectra()))


def test_signed_features_match_independent_scalar_reference_and_precursor_role():
    encoder = build_spectrum_encoder(definition()['spec_encoder'])
    mz, _, mask = spectra()
    feature = encoder.delta_features(mz, mask)
    # An independent Python scalar reference, including a negative and a zero difference.
    wavelengths = [10 ** (-2 + j * 6 / 3) for j in range(4)]
    for row, col in ((0, 1), (0, 2), (0, 3), (1, 1)):
        delta = float(mz[row, 0] - mz[row, col])
        expected = [math.sin(delta * 2 * math.pi / w) for w in wavelengths]
        expected += [math.cos(delta * 2 * math.pi / w) for w in wavelengths] + [0.]
        torch.testing.assert_close(feature[row, col], torch.tensor(expected), atol=.02, rtol=1e-5)
    assert not torch.equal(feature[0, 0], feature[0, 3])
    assert bool((feature[mask] == 0).all())
    mirrored = mz.clone()
    mirrored[0, 2] = 190.  # +10 rather than -10: preserve the sign in sine coordinates.
    other = encoder.delta_features(mirrored, mask)
    torch.testing.assert_close(feature[0, 2, :4], -other[0, 2, :4])
    torch.testing.assert_close(feature[0, 2, 4:8], other[0, 2, 4:8])


def test_zero_residual_matches_inherited_encoder_and_ignores_padding():
    torch.manual_seed(42)
    original = build_spectrum_encoder(model_config()['spec_encoder']).eval()
    torch.manual_seed(42)
    encoder = build_spectrum_encoder(definition()['spec_encoder']).eval()
    mz, intensity, mask = spectra()
    torch.testing.assert_close(encoder(mz, intensity, mask), original(mz, intensity, mask), atol=0, rtol=0)
    mz[mask], intensity[mask] = float('nan'), float('inf')
    torch.testing.assert_close(encoder(mz, intensity, mask), original(*spectra()), atol=0, rtol=0)
    # Test padding and permutation invariance after the new branch becomes nonzero, too.
    torch.nn.init.normal_(encoder.delta_projection[-1].weight, std=.05)
    output = encoder(mz, intensity, mask)
    order = [0, 3, 2, 1, 4]
    torch.testing.assert_close(output, encoder(mz[:, order], intensity[:, order], mask[:, order]))
    torch.testing.assert_close(output, encoder(F.pad(mz, (0, 3), value=float('nan')),
                                              F.pad(intensity, (0, 3), value=float('inf')),
                                              F.pad(mask, (0, 3), value=True)))


@pytest.mark.parametrize('kind', ['gine', 'fingerprint'])
def test_two_training_steps_reach_delta_spectral_and_molecular_parameters(kind):
    from torch_geometric.data import Batch

    from SpecEmbedding.data.graph_utils import smiles_to_graph

    torch.manual_seed(42)
    model = build_formal_alignment(definition(kind))
    if kind == 'fingerprint':
        molecules = torch.randint(0, 2, (3, 2048)).float()
    else:
        molecules = Batch.from_data_list([smiles_to_graph(s, graph_policy='rdkit_sanitized') for s in ('CCO', 'CCC', 'CC')])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    mz, intensity, mask = spectra()
    for step in range(2):
        optimizer.zero_grad()
        spec = model.encode_spec(mz, intensity, mask, normalize=True)
        mol = model.encode_mol(molecules, normalize=True)
        loss = F.cross_entropy(spec @ mol.T * model.get_logit_scale(), torch.tensor([0, 1]))
        assert torch.isfinite(loss)
        loss.backward()
        for module in (model.spec_encoder.embedding, model.mol_encoder, model.spec_encoder.delta_projection[-1]):
            assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                       and bool(p.grad.abs().sum() > 0) for p in module.parameters())
        if step == 1:
            assert bool(model.spec_encoder.delta_projection[0].weight.grad.abs().sum() > 0)
        optimizer.step()
    state = io.BytesIO()
    torch.save(model.state_dict(), state)
    state.seek(0)
    restored = build_formal_alignment(definition(kind)).eval()
    restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
    torch.testing.assert_close(model.eval().encode_spec(mz, intensity, mask), restored.encode_spec(mz, intensity, mask),
                               atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['gine', 'fingerprint'])
def test_checkpoint_metadata_reconstructs_downstream_tower_and_refuses_old_weights(tmp_path, kind):
    path, _, selection, protocol = checkpoint_fixture(tmp_path, kind)
    original = build_formal_alignment(definition(kind)).eval()
    torch.save(original.state_dict(), path)
    selection['model_config'] = definition(kind)
    selection['config_snapshot']['model'] = definition(kind)
    selection['checkpoint_sha256'] = sha256_file(path)
    metadata = tmp_path / 'alignment_selection.json'
    metadata.write_text(json.dumps(selection))
    restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
    assert receipt['model_config'] == definition(kind)
    assert torch.equal(original.encode_spec(*spectra()), restored.encode_spec(*spectra()))
    weights = original.state_dict()
    del weights['spec_encoder.delta_projection.0.weight']
    torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(path)
    metadata.write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match='Missing key'):
        load_formal_alignment(path, torch.device('cpu'), **protocol)


@pytest.mark.parametrize('change', ['missing', 'odd', 'bool', 'nan', 'reversed', 'unknown', 'null', 'underflow'])
def test_incomplete_or_invalid_delta_configuration_fails(change):
    config = copy.deepcopy(definition()['spec_encoder'])
    delta = config['precursor_delta']
    if change == 'missing':
        del delta['hidden_dim']
    elif change == 'odd':
        delta['fourier_dim'] = 7
    elif change == 'bool':
        delta['hidden_dim'] = True
    elif change == 'nan':
        delta['min_wavelength'] = float('nan')
    elif change == 'reversed':
        delta['min_wavelength'] = delta['max_wavelength']
    elif change == 'unknown':
        config['precursor_dleta'] = config.pop('precursor_delta')
    elif change == 'underflow':
        delta['min_wavelength'] = 1e-100
    else:
        config['precursor_delta'] = None
    with pytest.raises(ValueError):
        build_spectrum_encoder(config)


@pytest.mark.parametrize('change', ['precursor_mask', 'valid_nan', 'empty', 'mask_type'])
def test_invalid_spectra_are_not_silently_reinterpreted(change):
    encoder = build_spectrum_encoder(definition()['spec_encoder'])
    mz, intensity, mask = spectra()
    if change == 'precursor_mask':
        mask[:, 0] = True
    elif change == 'valid_nan':
        mz[0, 1] = float('nan')
    elif change == 'empty':
        mz, intensity, mask = mz[:, :0], intensity[:, :0], mask[:, :0]
    else:
        mask = mask.float()
    with pytest.raises(ValueError):
        encoder(mz, intensity, mask)


def test_queue_refuses_delta_before_reading_data_when_baseline_is_not_independent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import run_massspecgym_v15 as runner
    from SpecEmbedding.config import ConfigObject

    settings = copy.deepcopy(runner.config.to_dict())
    settings['model'] = definition()
    monkeypatch.setattr(runner, 'config', ConfigObject(settings))
    monkeypatch.setattr(runner, 'storage_receipt', lambda _: {'synthetic': True})
    args = SimpleNamespace(output_root=tmp_path, molecule_input='gine', checkpoint_model_config=False)
    with pytest.raises(ValueError, match='independent baseline'):
        runner.preflight(args)
