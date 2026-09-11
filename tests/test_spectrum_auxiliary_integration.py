"""The full completion auditor and runner must bind the auxiliary objective explicitly."""

import copy
import json
import shutil
from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
import yaml

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.utils import candidate_training as candidate_io
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, read_formal_alignment_checkpoint
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator
from SpecEmbedding.utils.spectrum_auxiliary_inputs import verify_spectrum_target_files


def verify_complete_auxiliary_fixture(monkeypatch, directory, data, index, selection, input_path, input_receipt, cache, counts):
    import run_massspecgym_v15 as runner
    from SpecEmbedding.utils import optimization_audit as audit

    run = directory.parent
    shutil.copytree(data, run / 'data/MassSpecGym')
    index_path = run / 'validation/mass_val_topk256.pt'
    index_path.parent.mkdir()
    torch.save(index, index_path)
    monkeypatch.setattr(audit, 'load_validation_index', lambda *args: index)
    runtime = copy.deepcopy(selection['config_snapshot'])
    runtime['fulltrain'].update(expected_counts=counts, exclude_val_query_indices=[])
    parent_config = copy.deepcopy(runtime['model'])
    parent_config.pop('spectrum_auxiliary')
    parent = build_formal_alignment(parent_config).eval()
    parent_dir = run / 'synthetic_parent'
    parent_dir.mkdir()
    checkpoint = parent_dir / 'best_model_stage2.pth'
    torch.save(parent.state_dict(), checkpoint)
    parent_selection = copy.deepcopy(selection)
    parent_selection.pop('spectrum_targets')
    parent_selection['training_config'].pop('spectrum_auxiliary')
    parent_selection['config_snapshot']['train']['align'].pop('spectrum_auxiliary')
    parent_selection['stages']['stage2'].pop('spectrum_auxiliary')
    parent_selection.update(model_config=parent_config, checkpoint_sha256=sha256_file(checkpoint))
    parent_selection['config_snapshot']['model'] = parent_config
    (parent_dir / 'alignment_selection.json').write_text(json.dumps(parent_selection))
    _, parent_receipt = read_formal_alignment_checkpoint(checkpoint, dataset_outputs=index['dataset_outputs'],
        dataset_manifest_sha256=index['dataset_manifest_sha256'], tokenizer_config=index['tokenizer_config'],
        expected_counts={'train': 3, 'val': 3}, exclusions=[])
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=(1, 5, 10, 20))
    validator = AlignmentRetrievalValidator(index, settings, run / 'baseline_validation')
    metrics = validator(parent, torch.device('cpu'), 0, 'baseline')
    baseline = {'metrics': metrics, 'checkpoint_sha256': sha256_file(checkpoint), 'index_sha256': sha256_file(index_path),
                'protocol': index['protocol'], 'checkpoint_model': parent_receipt}
    (run / 'baseline_validation/metrics.json').write_text(json.dumps(baseline))
    new_input = run / 'candidate_training_input.json'
    shutil.copyfile(input_path, new_input)
    selection = copy.deepcopy(selection)
    selection.update(validation_index={'path': str(index_path), 'sha256': sha256_file(index_path)},
        candidate_training_input={'path': str(new_input), 'sha256': sha256_file(new_input)}, config_snapshot=runtime,
        device='cuda:0')  # Synthetic audit declaration only; every actual computation here used CPU.
    source = run / 'source.yaml'
    source.write_text(yaml.safe_dump(runtime))
    (run / 'runtime_params.yaml').write_text(source.read_text())
    manifest = {'runtime_config': runtime, 'source_config': str(source), 'source_config_sha256': sha256_file(source),
        'git_commit': 'synthetic-auxiliary-integration', 'runtime_versions': {'torch': version('torch')}, 'device': 'cuda:0',
        'candidate_training_input': input_receipt, 'spectrum_targets': cache.provenance, 'checkpoint_model': parent_receipt,
        'inputs': {**candidate_io.candidate_source_inputs(input_receipt), **verify_spectrum_target_files(cache.provenance),
                   'baseline_checkpoint': {'path': str(checkpoint), 'sha256': sha256_file(checkpoint)}}}
    status = {'state': 'complete', 'runtime_config_sha256': sha256_file(run / 'runtime_params.yaml'),
        'candidate_training_input_sha256': sha256_file(new_input),
        'stages': [{'name': name, 'state': 'complete', 'audit': {'sha256': sha256_file(index_path)}}
                  for name in ('import_v15', 'prepare_validation', 'baseline_validation', 'alignment42')]}
    selection_path = directory / 'alignment_selection.json'
    for path, value in ((run / 'inputs_and_commands.json', manifest), (run / 'status.json', status), (selection_path, selection)):
        path.write_text(json.dumps(value))
    report = audit.audit_optimization_run(run)
    assert report['state'] == 'complete_validation_audit' and not report['test_evaluated_by_this_audit']
    assert report['spectrum_auxiliary']['state'] == 'verified_full_auxiliary_target_replay'
    assert report['candidate_training']['negative_sampling_replayed']
    monkeypatch.setattr(runner, 'verify_dataset', lambda *args: {})
    monkeypatch.setattr(config.fulltrain, 'expected_counts', ConfigObject(counts))
    monkeypatch.setattr(config.fulltrain, 'exclude_val_query_indices', [])
    result = runner.audit_alignment(SimpleNamespace(output_root=run, device='cuda:0'), runtime['train']['align'], runtime['augmentation'])
    assert result['spectrum_auxiliary']['queries'] == 3
    # Full completion cannot silently accept a removed target provenance.
    original = selection_path.read_text()
    selection.pop('spectrum_targets')
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='differs from preflight'):
        audit.audit_optimization_run(run)
    selection_path.write_text(original)


@pytest.mark.parametrize('missing', ['cache', 'model', 'weight', 'baseline', 'candidate_input'])
def test_runner_rejects_unbound_auxiliary_before_reading_data(tmp_path, monkeypatch, missing):
    import run_massspecgym_v15 as runner
    from tests.test_spectrum_auxiliary_training import definition

    monkeypatch.setattr(config, 'model', ConfigObject(definition()))
    monkeypatch.setattr(config.train.align, 'spectrum_auxiliary', ConfigObject({'loss_weight': 1.}), raising=False)
    args = SimpleNamespace(output_root=tmp_path / 'run', molecule_input='gine', checkpoint_model_config=True,
        optimize_alignment=True, prepared_data=tmp_path / 'data', alignment_training_candidates=tmp_path / 'candidates',
        spectrum_target_cache=tmp_path / 'targets')
    if missing == 'cache':
        args.spectrum_target_cache = None
    elif missing == 'model':
        monkeypatch.delattr(config.model, 'spectrum_auxiliary')
    elif missing == 'weight':
        monkeypatch.delattr(config.train.align, 'spectrum_auxiliary')
    elif missing == 'baseline':
        args.checkpoint_model_config = False
    else:
        args.alignment_training_candidates = None
    with pytest.raises(ValueError, match='auxiliary|Auxiliary'):
        runner.preflight(args)
    assert not args.output_root.exists()


def test_runner_passes_targets_only_to_training_command(tmp_path):
    import run_massspecgym_v15 as runner

    args = SimpleNamespace(output_root=tmp_path / 'run', source_dir=tmp_path / 'source', legacy_tsv=tmp_path / 'legacy',
        gpu=0, device='cuda:0', optimize_alignment=True, baseline_checkpoint=tmp_path / 'base.pth',
        spectrum_target_cache=tmp_path / 'targets', checkpoint_model_config=True,
        alignment_training_candidates=tmp_path / 'candidates')
    stages = runner.commands(args)
    assert '--spectrum-target-cache' not in stages[2]['command']
    command = stages[3]['command']
    assert command[command.index('--spectrum-target-cache') + 1] == str(args.spectrum_target_cache)


@pytest.mark.parametrize('damage', [None, 'existing', 'parent_model', 'parent_training', 'disabled', 'width'])
def test_successor_preserves_all_inherited_scientific_settings_and_rejects_wrong_parent(tmp_path, damage):
    from SpecEmbedding.utils.alignment_successor import spectrum_auxiliary_successor_configuration
    from tests.test_spectrum_auxiliary_training import definition

    settings = definition()
    auxiliary = {'weight': 1., **settings.pop('spectrum_auxiliary')}
    parent = config.to_dict()
    parent['model'] = settings
    parent['train']['align']['candidate_supervision']['enabled'] = True
    selection = {'model_config': copy.deepcopy(parent['model']), 'training_config': copy.deepcopy(parent['train']['align']),
                 'config_snapshot': copy.deepcopy(parent)}
    policy = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    if damage == 'existing':
        parent['model']['spectrum_auxiliary'] = {key: auxiliary[key] for key in ('model', 'target')}
    elif damage == 'parent_model':
        selection['model_config']['align']['tau'] *= 2
    elif damage == 'parent_training':
        selection['training_config']['lr'] *= 2
    elif damage == 'disabled':
        parent['train']['align']['candidate_supervision']['enabled'] = False
    elif damage == 'width':
        auxiliary['model']['embedding_dim'] += 1
    template = copy.deepcopy(parent)
    template['storage'] = {'root': str(tmp_path)}
    from SpecEmbedding.config import PATH_FIELDS
    for keys in PATH_FIELDS:
        target = template
        for key in keys[:-1]:
            target = target.get(key, {})
        if keys[-1] in target:
            target[keys[-1]] = str(tmp_path / ('_'.join(keys)))
    if damage:
        with pytest.raises(ValueError):
            spectrum_auxiliary_successor_configuration(parent, selection, auxiliary, policy, storage_template=template)
        return
    result = spectrum_auxiliary_successor_configuration(parent, selection, auxiliary, policy, storage_template=template)
    assert {key: value for key, value in result['model'].items() if key != 'spectrum_auxiliary'} == parent['model']
    assert {key: value for key, value in result['train']['align'].items() if key != 'spectrum_auxiliary'} == parent['train']['align']
    for key in ('augmentation', 'data', 'retrieval_validation'):
        if key != 'data':
            assert result[key] == parent[key]
    assert result['data']['tokenizer'] == parent['data']['tokenizer']
