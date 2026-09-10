"""Synthetic formal-entry and completion checks; no MassSpecGym experiment results."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.utils.alignment_successor import attention_pool_successor_configuration
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.optimization_audit import audit_optimization_run
from tests.test_fingerprint_integration import fingerprint_run as fingerprint_run
from tests.test_formal_alignment import checkpoint_fixture, model_config
from tests.test_graph_fingerprint_integration import combined_run as combined_run
from tests.test_graph_fingerprint_model import training_parent
from tests.test_optimization_audit import completed_run as completed_run
from tests.test_precursor_delta import spectra


def definition(kind='gine', qk=False):
    value = model_config(kind)
    value['spec_encoder']['attention_pool'] = {'norm_eps': 1e-5}
    if qk:
        value['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    return value


@pytest.mark.parametrize('qk', [False, True])
def test_actual_fresh_entry_preserves_initial_weights_and_rng(monkeypatch, qk):
    candidate_config = definition(qk=qk)
    parent_config = copy.deepcopy(candidate_config)
    del parent_config['spec_encoder']['attention_pool']
    torch.manual_seed(42)
    parent = training_parent(monkeypatch, parent_config).eval()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = training_parent(monkeypatch, candidate_config).eval()
    assert torch.equal(torch.get_rng_state(), state)
    for name, value in parent.state_dict().items():
        target = name.replace('spec_encoder.', 'spec_encoder.base.', 1) if name.startswith('spec_encoder.') else name
        assert torch.equal(value, candidate.state_dict()[target])
    with torch.inference_mode():
        torch.testing.assert_close(candidate.encode_spec(*spectra()), parent.encode_spec(*spectra()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
def test_checkpoint_uses_own_pool_config_and_rejects_missing_query_even_when_rehashed(tmp_path, monkeypatch, kind):
    path, _, selection, protocol = checkpoint_fixture(tmp_path, kind)
    settings = definition(kind, qk=True)
    model = build_formal_alignment(settings).eval()
    with torch.no_grad():
        model.spec_encoder.pool.query.copy_(torch.linspace(-.2, .3, 8))
    torch.save(model.state_dict(), path)
    selection.update(model_config=settings, checkpoint_sha256=sha256_file(path))
    selection['config_snapshot']['model'] = settings
    metadata = path.parent / 'alignment_selection.json'
    metadata.write_text(json.dumps(selection))
    monkeypatch.setattr(config, 'model', ConfigObject({'unrelated': 'model'}))
    restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
    assert receipt['model_config'] == settings
    with torch.inference_mode():
        torch.testing.assert_close(model.encode_spec(*spectra()), restored.encode_spec(*spectra()), rtol=0, atol=0)
    weights = model.state_dict()
    del weights['spec_encoder.pool.query']
    torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(path)
    metadata.write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match='Missing key'):
        load_formal_alignment(path, torch.device('cpu'), **protocol)


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('structural', [False, True])
def test_successor_preserves_all_parent_science_and_fixed_cache_bindings(kind, structural):
    parent = definition(kind, qk=True)
    del parent['spec_encoder']['attention_pool']
    sampling = ({'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32,
                 'fingerprint_radius': 2, 'fingerprint_bits': 2048, 'cache_directory': '/tmp/approved/similarity'}
                if structural else {'type': 'uniform'})
    runtime = {'model': parent, 'train': {'align': {'batch_size': 128, 'lr': 1e-4,
               'candidate_supervision': {'negative_count': 16, 'sampling': sampling}}}, 'augmentation': {'prob': .5},
               'data': {'cache_path': '/old/cache', 'tokenizer': {'max_len': 100}}, 'fulltrain': {}}
    original = copy.deepcopy(runtime)
    selection = {'model_config': parent, 'training_config': runtime['train']['align'],
                 'config_snapshot': {'augmentation': runtime['augmentation']}}
    storage = {'storage': {'root': '/tmp/approved'}, 'data': {'cache_path': '/tmp/approved/train_cache'}}
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    result = attention_pool_successor_configuration(runtime, selection, {'norm_eps': 1e-5}, gpu, storage_template=storage)
    expected = copy.deepcopy(runtime)
    expected['model']['spec_encoder']['attention_pool'] = {'norm_eps': 1e-5}
    expected.update(storage=storage['storage'], fulltrain=gpu)
    expected['data']['cache_path'] = storage['data']['cache_path']
    assert result == expected and runtime == original
    selection['model_config'] = result['model']
    with pytest.raises(ValueError, match='already'):
        attention_pool_successor_configuration(result, selection, {'norm_eps': 1e-5}, gpu, storage_template=storage)


@pytest.mark.parametrize('mode', ['not_formal', 'no_validation', 'pretrained', 'incomplete'])
def test_direct_entry_rejects_unsupported_training_before_device_or_data(monkeypatch, mode):
    import train_align as entry

    monkeypatch.setattr(config, 'model', ConfigObject(definition()))
    monkeypatch.setattr(config.train.align.candidate_supervision, 'enabled', False)
    if mode == 'incomplete':
        monkeypatch.setattr(config.model.spec_encoder, 'attention_pool', ConfigObject({}))
    def unexpected(*args, **kwargs):
        raise AssertionError('Reached device resolution before rejecting configuration')
    monkeypatch.setattr(entry, 'resolve_device', unexpected)
    with pytest.raises(ValueError, match='Attention pooling'):
        entry.train_align({}, [], {}, [], object() if mode == 'pretrained' else None,
                          formal_fulltrain=mode != 'not_formal',
                          retrieval_validator=None if mode == 'no_validation' else object())


@pytest.mark.parametrize('arguments', [[], ['--formal-fulltrain'],
    ['--formal-fulltrain', '--validation-index', '/missing/index', '--pretrained_spec', '/missing/weights']])
def test_cli_rejects_incompatible_mode_before_reading_data(tmp_path, arguments):
    settings = copy.deepcopy(config.to_dict())
    settings['model'] = definition()
    path = tmp_path / 'pool.yaml'
    path.write_text(yaml.safe_dump(settings))
    env = {**os.environ, 'SPECEMBEDDING_CONFIG': str(path), 'CUDA_VISIBLE_DEVICES': '',
           'SPECEMBEDDING_REQUIRE_CUDA': '0'}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'train_align.py'), *arguments],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and 'Attention pooling requires fresh formal' in result.stderr


def test_queue_requires_independent_baseline_before_reading_inputs(tmp_path, monkeypatch):
    import run_massspecgym_v15 as runner

    monkeypatch.setattr(config, 'model', ConfigObject(definition()))
    monkeypatch.setattr(runner, 'storage_receipt', lambda _: {'synthetic': True})
    with pytest.raises(ValueError, match='independent baseline'):
        runner.preflight(SimpleNamespace(output_root=tmp_path, molecule_input='gine', checkpoint_model_config=False))


def save_run(run, manifest, selection):
    runtime = manifest['runtime_config']
    selection['config_snapshot'] = copy.deepcopy(runtime)
    (run / 'runtime_params.yaml').write_text(yaml.safe_dump(runtime))
    source = Path(manifest['source_config'])
    source.write_text(yaml.safe_dump(runtime))
    manifest['source_config_sha256'] = sha256_file(source)
    (run / 'inputs_and_commands.json').write_text(json.dumps(manifest))
    (run / 'alignment42_topk256/alignment_selection.json').write_text(json.dumps(selection))
    status = json.loads((run / 'status.json').read_text())
    status['runtime_config_sha256'] = sha256_file(run / 'runtime_params.yaml')
    (run / 'status.json').write_text(json.dumps(status))


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('damage', [None, 'missing_query', 'missing_config', 'incomplete_config', 'unbound_baseline'])
def test_completion_strictly_checks_pool_weights_and_config_with_independent_parent(combined_run, kind, damage):
    run, manifest, selection = combined_run
    directory = run / 'alignment42_topk256'
    manifest['runtime_config']['model'] = definition(kind, qk=True)
    selection['model_config'] = manifest['runtime_config']['model']
    if kind == 'gine':
        manifest.pop('fingerprint_inputs')
        manifest['inputs'] = {k: v for k, v in manifest['inputs'].items() if not k.startswith('fingerprint_')}
        selection.pop('training_fingerprint_cache')
        selection.pop('validation_fingerprint_cache')
        for path in (directory / 'validation_retrieval').glob('*.pt'):
            snapshot = torch.load(path, weights_only=False)
            snapshot.pop('validation_fingerprint_cache')
            torch.save(snapshot, path)
    weights = build_formal_alignment(selection['model_config']).state_dict()
    if damage == 'missing_query':
        del weights['spec_encoder.pool.query']
    # Rehash every selected/candidate tensor set so equality and byte hashing alone cannot pass corruption.
    for path in (directory / 'best_model_stage2.pth', *directory.glob('candidate_stage2_epoch*.pth')):
        torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(directory / 'best_model_stage2.pth')
    if damage == 'missing_config':
        del selection['model_config']['spec_encoder']['attention_pool']
    elif damage == 'incomplete_config':
        selection['model_config']['spec_encoder']['attention_pool'] = {}
    elif damage == 'unbound_baseline':
        manifest.pop('checkpoint_model')
        path = run / 'baseline_validation/metrics.json'
        baseline = json.loads(path.read_text())
        baseline.pop('checkpoint_model')
        path.write_text(json.dumps(baseline))
    save_run(run, manifest, selection)
    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            audit_optimization_run(run)
    else:
        report = audit_optimization_run(run)
        assert report['state'] == 'complete_validation_audit'
        assert report['full_epoch_counts'] == {'train': 7, 'val': 5}
        assert report['selected_metrics']['top1'] == .4  # The empty-positive query remains in the denominator.
        assert ('fingerprint_inputs' in report) == (kind == 'gine_fingerprint')
        assert not report['test_evaluated_by_this_audit']
