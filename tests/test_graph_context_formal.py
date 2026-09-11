"""Formal graph-context configuration, CLI boundaries and checkpoint provenance on CPU fixtures."""

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
from torch_geometric.data import Batch

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.utils.alignment_successor import graph_context_successor_configuration
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from tests.test_formal_alignment import checkpoint_fixture, model_config
from tests.test_graph_global_context import context_config, graphs


def definition(kind='gine'):
    result = model_config(kind)
    result['mol_encoder']['n_layers'] = 4
    result['graph_global_context'] = context_config()
    return result


def test_older_direct_fusion_constructor_cannot_silently_drop_context():
    from SpecEmbedding.models_graph_fingerprint import GraphFingerprintAlignmentModel

    with pytest.raises(ValueError, match='combined formal'):
        GraphFingerprintAlignmentModel(parent_model_config=definition(),
                                       fingerprint_config=model_config('gine_fingerprint')['fingerprint_residual'])


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('damage', [None, 'missing_weight', 'wrong_hidden', 'missing_config', 'incomplete_config'])
def test_strict_reload_uses_own_context_and_rejects_rehashed_structural_damage(tmp_path, monkeypatch, kind, damage):
    path, _, selection, protocol = checkpoint_fixture(tmp_path, kind)
    settings = definition(kind)
    model = build_formal_alignment(settings).eval()
    with torch.no_grad():
        for branch in model.mol_encoder.context_branches:
            branch.output.weight.normal_(std=.1)
    weights = model.state_dict()
    if damage == 'missing_weight':
        del weights['mol_encoder.context_branches.1.output.weight']
    torch.save(weights, path)
    selection.update(model_config=copy.deepcopy(settings), checkpoint_sha256=sha256_file(path))
    if damage == 'wrong_hidden':
        selection['model_config']['graph_global_context']['hidden_dim'] += 1
    elif damage == 'missing_config':
        selection['model_config'].pop('graph_global_context')
    elif damage == 'incomplete_config':
        selection['model_config']['graph_global_context'] = {}
    selection['config_snapshot']['model'] = copy.deepcopy(selection['model_config'])
    (path.parent / 'alignment_selection.json').write_text(json.dumps(selection))
    monkeypatch.setattr(config, 'model', ConfigObject({'unrelated': 'candidate configuration'}))
    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            load_formal_alignment(path, torch.device('cpu'), **protocol)
    else:
        restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
        assert receipt['model_config'] == settings
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in weights.items())
        items = graphs()
        if kind == 'gine_fingerprint':
            for graph in items:
                graph.fingerprints = torch.ones(1, settings['fingerprint_residual']['input_bits'])
        with torch.inference_mode():
            batch = Batch.from_data_list(items)
            torch.testing.assert_close(model.encode_mol(batch, True), restored.encode_mol(batch, True), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('weight', [1., 2.])
@pytest.mark.parametrize('adduct', [False, True])
def test_successor_changes_only_registered_context_and_keeps_all_parent_science(kind, structural, weight, adduct):
    from tests.test_adduct_conditioning import SETTINGS

    parent = model_config(kind)
    parent['mol_encoder'].update(n_layers=4, emb_dim=128)
    parent['spec_encoder'].update(qk_norm={'eps': 1e-6}, attention_pool={'norm_eps': 1e-5})
    if adduct:
        parent['spec_encoder']['adduct_conditioning'] = copy.deepcopy(SETTINGS)
    candidates = config.train.align.candidate_supervision.to_dict()
    candidates.update(enabled=True, negative_count=16, loss_weight=weight)
    if structural:
        candidates['sampling'] = {'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32,
            'fingerprint_radius': 2, 'fingerprint_bits': 2048, 'cache_directory': '/tmp/approved/similarity'}
    else:
        candidates.pop('sampling', None)  # The existing protocol encodes uniform sampling by absence.
    runtime = {'model': parent, 'train': {'align': {'batch_size': 128, 'lr': 1e-4, 'candidate_supervision': candidates}},
               'augmentation': {'prob': .5}, 'data': {'cache_path': '/old/cache', 'tokenizer': {'max_len': 100}},
               'fulltrain': {}}
    original = copy.deepcopy(runtime)
    selection = {'model_config': parent, 'training_config': runtime['train']['align'],
                 'config_snapshot': {'augmentation': runtime['augmentation']}}
    settings = {**context_config(), 'hidden_dim': 64}
    storage = {'storage': {'root': '/tmp/approved'}, 'data': {'cache_path': '/tmp/approved/train_cache'}}
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    result = graph_context_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)
    expected = copy.deepcopy(runtime)
    expected['model']['graph_global_context'] = settings
    expected.update(storage=storage['storage'], fulltrain=gpu)
    expected['data']['cache_path'] = storage['data']['cache_path']
    assert result == expected and runtime == original
    with pytest.raises(ValueError, match='already'):
        graph_context_successor_configuration(result, selection, settings, gpu, storage_template=storage)
    with pytest.raises(ValueError, match='width64'):
        graph_context_successor_configuration(runtime, selection, context_config(), gpu, storage_template=storage)
    runtime['model']['mol_encoder']['n_layers'] = 8
    with pytest.raises(ValueError, match='four-layer'):
        graph_context_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)


@pytest.mark.parametrize('mode', ['not_formal', 'no_validation', 'pretrained', 'disabled', 'incomplete', 'wrong_norm'])
def test_direct_training_rejects_unbound_mode_before_device_or_data(monkeypatch, mode):
    import train_align as entry

    model = definition()
    if mode == 'incomplete':
        model['graph_global_context'] = {}
    monkeypatch.setattr(config, 'model', ConfigObject(model))
    monkeypatch.setattr(config.train.align.candidate_supervision, 'enabled', mode != 'disabled')
    def unexpected(*args, **kwargs):
        raise AssertionError('Reached device/data before rejecting graph-context setup')
    monkeypatch.setattr(entry, 'resolve_device', unexpected)
    with pytest.raises(ValueError, match='[Cc]andidate|Graph context'):
        entry.train_align({}, [], {}, [], object() if mode == 'pretrained' else None,
            formal_fulltrain=mode != 'not_formal', retrieval_validator=None if mode == 'no_validation' else object(),
            training_candidates=None if mode == 'disabled' else object(),
            candidate_input_receipt=None if mode == 'disabled' else {},
            mol_norm_type='rmsnorm' if mode == 'wrong_norm' else model['mol_encoder']['norm_type'],
            mol_norm_eps=model['mol_encoder']['norm_eps'])


@pytest.mark.parametrize('arguments', [[], ['--formal-fulltrain'],
    ['--formal-fulltrain', '--validation-index', '/missing/index'],
    ['--formal-fulltrain', '--validation-index', '/missing/index', '--pretrained_spec', '/missing/weights']])
def test_cli_refuses_incompatible_mode_before_opening_input(tmp_path, arguments):
    settings = copy.deepcopy(config.to_dict())
    settings['model'] = definition()
    settings['train']['align']['candidate_supervision']['enabled'] = True
    source = tmp_path / 'graph_context.yaml'
    source.write_text(yaml.safe_dump(settings))
    env = {**os.environ, 'SPECEMBEDDING_CONFIG': str(source), 'CUDA_VISIBLE_DEVICES': '', 'SPECEMBEDDING_REQUIRE_CUDA': '0'}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'train_align.py'), *arguments],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and 'Graph context requires fresh formal' in result.stderr


@pytest.mark.parametrize('missing', ['independent', 'optimization', 'prepared_data', 'candidate_input'])
def test_preflight_requires_independent_parent_and_prepared_candidate_inputs(tmp_path, monkeypatch, missing):
    import run_massspecgym_v15 as runner

    monkeypatch.setattr(config, 'model', ConfigObject(definition()))
    monkeypatch.setattr(runner, 'storage_receipt', lambda *args: {'synthetic': True})
    args = SimpleNamespace(output_root=tmp_path, molecule_input='gine', checkpoint_model_config=missing != 'independent',
        optimize_alignment=missing != 'optimization', prepared_data=None if missing == 'prepared_data' else tmp_path / 'data',
        alignment_training_candidates=None if missing == 'candidate_input' else tmp_path / 'train.pkl')
    with pytest.raises(ValueError, match='Graph context requires'):
        runner.preflight(args)
