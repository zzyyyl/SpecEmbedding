import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import run_massspecgym_v15 as runner
from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.mass_batching import MassBlockBatchSampler, molecular_exact_masses


@pytest.mark.parametrize("n", [1, 7, 32, 64, 71, 128, 259])
def test_every_query_is_delivered_once_including_partial_blocks(n):
    sampler = MassBlockBatchSampler(np.linspace(100, 500, n), batch_size=128, block_size=32, seed=42)
    loader = DataLoader(list(range(n)), batch_sampler=sampler, num_workers=0)
    for epoch in (1, 2):
        batches = list(loader)
        seen = torch.cat(batches).tolist()
        assert sorted(seen) == list(range(n))
        assert len(batches) == len(sampler)
        assert all(len(b) == 128 for b in batches[:-1])
        assert len(batches[-1]) == (n % 128 or 128)
        assert sampler.last_audit["epoch"] == epoch
        assert sampler.last_audit["queries"] == sampler.last_audit["unique_queries"] == n


def test_reproducible_epochs_without_consuming_global_rng_or_preserving_tie_order():
    masses = np.repeat([100., 200., 300., 400.], 64)
    left = MassBlockBatchSampler(masses, batch_size=64, block_size=16, seed=42)
    right = MassBlockBatchSampler(masses, batch_size=64, block_size=16, seed=42)
    state = torch.random.get_rng_state()
    first, second = list(left), list(left)
    assert first != second
    assert first == list(right) and second == list(right)
    assert torch.equal(state, torch.random.get_rng_state())
    flatten = [i for batch in first for i in batch]
    tied = [i for i in flatten if masses[i] == 100.]
    assert tied != sorted(tied)


def test_mass_locality_survives_block_mixing():
    masses = np.arange(4096, dtype=float) + 100
    sampler = MassBlockBatchSampler(masses, batch_size=128, block_size=32, seed=42)
    batches = list(sampler)
    # All but the single circular boundary block remain contiguous in mass;
    # the composition of complete batches is still globally mixed.
    spans = [np.ptp(masses[batch[i:i+32]]) for batch in batches for i in range(0, 128, 32)]
    assert sum(span <= 31 for span in spans) >= len(spans)-1
    assert np.median([np.ptp(masses[batch]) for batch in batches]) > 1000


@pytest.mark.parametrize("masses,batch,block,seed", [([],128,32,42), ([float('nan')],128,32,42),
    ([0.],128,32,42), ([100.],128,0,42), ([100.],128,30,42), ([100.],128,32,-1)])
def test_invalid_batching_fails_early(masses, batch, block, seed):
    with pytest.raises(ValueError):
        MassBlockBatchSampler(masses, batch_size=batch, block_size=block, seed=seed)


def test_molecular_masses_are_representation_invariant_and_invalid_graphs_fail():
    values = molecular_exact_masses(["c1ccccc1", "C1=CC=CC=C1", "CCO", "OCC"])
    assert values[0] == values[1] and values[2] == values[3]
    with pytest.raises(ValueError, match="training molecular mass"):
        molecular_exact_masses(["invalid"])


@pytest.mark.parametrize('mol_augmentation', [None, True, False])
@pytest.mark.parametrize('candidate_supervision', [False, True])
@pytest.mark.parametrize('prepared_validation', [False, True])
def test_optimization_preflight_pins_batching_without_changing_other_hyperparameters(tmp_path, mol_augmentation, candidate_supervision, prepared_validation):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'fixture.tsv').write_text('source')
    legacy = tmp_path / 'legacy.tsv'
    legacy.write_text('legacy')
    checkpoint = tmp_path / 'baseline.pth'
    checkpoint.write_text('weights')
    model_config = copy.deepcopy(config.model.to_dict())
    model_config['mol_encoder']['graph_policy'] = 'rdkit_sanitized'
    (tmp_path / 'alignment_selection.json').write_text(json.dumps({
        'checkpoint_sha256': sha256_file(checkpoint), 'seed': 42, 'model_config': model_config,
        'graph_policy': 'rdkit_sanitized', 'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5'}}))
    args = SimpleNamespace(source_dir=source, legacy_tsv=legacy, output_root=tmp_path/'run',
        device='cuda:0', gpu=0, gpus=None, optimize_alignment=True, baseline_checkpoint=checkpoint,
        prepared_data=None, alignment_batching='mass_blocks', alignment_mol_augmentation=mol_augmentation)
    candidate_receipt = {'provenance': {'path': str(tmp_path/'metadata.pkl'), 'sha256': 'a'*64,
                                      'receipt_sha256': 'b'*64, 'verification_sha256': 'c'*64}}
    if candidate_supervision:
        args.prepared_data = tmp_path/'prepared'
        args.alignment_training_candidates = tmp_path/'metadata.pkl'
    validation_receipt = {'path': str(tmp_path/'index.pt'), 'sha256': 'e'*64, 'receipt_sha256': 'f'*64}
    if prepared_validation:
        args.prepared_data = tmp_path/'prepared'
        args.prepared_validation_index = tmp_path/'index.pt'
    with patch.object(config.fulltrain.v15, 'sources', ConfigObject({'fixture.tsv': sha256_file(source/'fixture.tsv')})), \
         patch.object(runner.subprocess, 'check_output', side_effect=lambda command, **kwargs: '' if 'status' in command else 'test-commit'), \
         patch.object(runner, 'prepared_source', return_value={'manifest_sha256': 'd'*64}), \
         patch.object(runner, 'prepared_validation_input', return_value=validation_receipt) as prepare_index, \
         patch.object(runner, 'build_candidate_training_input', return_value=(None, candidate_receipt)) as build:
        manifest = runner.preflight(args)
    assert build.call_count == int(candidate_supervision)
    assert prepare_index.call_count == int(prepared_validation)
    actual = manifest['runtime_config']['train']['align']
    expected = config.train.align.to_dict()
    expected.update(batching='mass_blocks', metric_for_best='validation_top1_then_mrr')
    expected['candidate_supervision']['enabled'] = candidate_supervision
    assert actual == expected
    augmentation = config.augmentation.to_dict()
    if mol_augmentation is False:
        augmentation.update(node_drop_rate=0.0, edge_mask_rate=0.0)
    assert manifest['runtime_config']['augmentation'] == augmentation
    assert config.augmentation.node_drop_rate == config.augmentation.edge_mask_rate == 0.1
    assert config.train.align.batching == 'random'
    assert [s['name'] for s in manifest['stages']] == [
        'import_v15' if candidate_supervision or prepared_validation else 'prepare_v15',
        'import_validation' if prepared_validation else 'prepare_validation', 'baseline_validation', 'alignment42']
    command = manifest['stages'][-1]['command']
    assert ('--candidate-training-input' in command) == candidate_supervision
    assert config.train.align.candidate_supervision.enabled is False
    if candidate_supervision:
        assert command[-2:] == ['--candidate-training-input', str(tmp_path/'run'/'candidate_training_input.json')]
        assert manifest['candidate_training_input'] == candidate_receipt
        assert manifest['inputs']['training_candidates'] == {'path': str(tmp_path/'metadata.pkl'), 'sha256': 'a'*64}
        assert build.call_args.args[2] == expected['candidate_supervision']
    if prepared_validation:
        assert manifest['prepared_validation'] == validation_receipt
        assert manifest['inputs']['prepared_validation_index'] == {'path': str(tmp_path/'index.pt'), 'sha256': 'e'*64}
        assert 'command' not in manifest['stages'][1]


@pytest.mark.parametrize('scheme', ['random', 'mass_blocks'])
def test_queue_completion_audits_actual_batching_and_unique_coverage(tmp_path, scheme):
    data = tmp_path/'data'/'MassSpecGym'
    data.mkdir(parents=True)
    (data/'dataset_manifest.json').write_text('{}')
    alignment = tmp_path/'alignment42_topk256'
    alignment.mkdir()
    checkpoint = alignment/'best_model_stage2.pth'
    checkpoint.write_text('weights')
    settings = config.train.align.to_dict()
    settings.update(batching=scheme, batch_size=2, mass_block_size=1)
    epoch = {'stage': 'stage2', 'epoch': 1, 'train': 3, 'val': 3}
    if scheme == 'mass_blocks':
        sampler = MassBlockBatchSampler([100., 200., 300.], batch_size=2, block_size=1, seed=42)
        list(sampler)
        epoch['batching'] = sampler.last_audit
    selection = {'graph_policy': 'rdkit_sanitized', 'seed': 42, 'device': 'cuda:0',
        'model_config': config.model.to_dict(),
        'checkpoint_sha256': sha256_file(checkpoint), 'exclude_val_query_indices': [],
        'training_config': settings, 'config_snapshot': {'augmentation': config.augmentation.to_dict()},
        'stages': {'stage2': {'stop_epoch': 1, 'best_epoch': 1}},
        'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5',
            'expected_epoch_counts': {'train': 3, 'val': 3}, 'epochs': [epoch],
            'dataset_manifest_sha256': sha256_file(data/'dataset_manifest.json'), 'training_batching': scheme}}
    path = alignment/'alignment_selection.json'
    path.write_text(json.dumps(selection))
    args = SimpleNamespace(output_root=tmp_path, device='cuda:0')
    with patch.object(runner, 'verify_dataset'), \
         patch.object(config.fulltrain, 'expected_counts', ConfigObject({'train':3,'val':3,'test':1})), \
         patch.object(config.fulltrain, 'exclude_val_query_indices', []):
        assert runner.audit_alignment(args, settings, config.augmentation.to_dict())['epochs'] == 1
        wrong_settings = {**settings, 'batching': 'mass_blocks' if scheme == 'random' else 'random'}
        with pytest.raises(ValueError, match='completion audit'):
            runner.audit_alignment(args, wrong_settings, config.augmentation.to_dict())
        with pytest.raises(ValueError, match='completion audit'):
            runner.audit_alignment(args, settings, {**config.augmentation.to_dict(), 'node_drop_rate': 0.0})
        if scheme == 'mass_blocks':
            epoch['batching']['unique_queries'] = 2
        else:
            epoch['train'] = 2
        path.write_text(json.dumps(selection))
        with pytest.raises(ValueError, match='(batching coverage|all train/validation)'):
            runner.audit_alignment(args, settings, config.augmentation.to_dict())


def test_molecular_augmentation_override_cannot_enter_default_queue(tmp_path):
    with patch.object(runner, 'preflight') as preflight:
        with pytest.raises(SystemExit):
            runner.main(['--source-dir', str(tmp_path), '--legacy-tsv', str(tmp_path/'legacy'),
                         '--output-root', str(tmp_path/'new'), '--gpus', '0', '1', '--device', 'cuda:0',
                         '--no-alignment-mol-augmentation', '--dry-run'])
        preflight.assert_not_called()
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('optimization', [False, True])
def test_candidate_supervision_requires_optimization_and_prepared_data(tmp_path, optimization):
    argv = ['--source-dir', str(tmp_path), '--legacy-tsv', str(tmp_path/'legacy'),
            '--output-root', str(tmp_path/'new'), '--gpus', '0', '1', '--device', 'cuda:0',
            '--alignment-training-candidates', str(tmp_path/'metadata.pkl'), '--dry-run']
    if optimization:
        argv += ['--optimize-alignment', '--baseline-checkpoint', str(tmp_path/'baseline.pth')]
    with patch.object(runner, 'preflight') as preflight:
        with pytest.raises(SystemExit):
            runner.main(argv)
        preflight.assert_not_called()
    assert not (tmp_path/'new').exists()


def test_prepared_validation_requires_prepared_data_and_optimization(tmp_path):
    argv = ['--source-dir', str(tmp_path), '--legacy-tsv', str(tmp_path/'legacy'),
            '--output-root', str(tmp_path/'new'), '--gpus', '0', '1', '--device', 'cuda:0',
            '--prepared-validation-index', str(tmp_path/'index.pt'), '--dry-run']
    with patch.object(runner, 'preflight') as preflight:
        with pytest.raises(SystemExit):
            runner.main(argv)
        with pytest.raises(SystemExit):
            runner.main(argv + ['--optimize-alignment', '--baseline-checkpoint', str(tmp_path/'baseline.pth')])
        preflight.assert_not_called()


def test_zero_graph_rates_keep_complete_graph_and_spectrum_augmentation(monkeypatch):
    from SpecEmbedding.data.datasets_align import AlignGraphDataset
    from SpecEmbedding.data.graph_utils import smiles_to_graph

    values = config.augmentation.to_dict()
    values.update(prob=1.0, node_drop_rate=0.0, edge_mask_rate=0.0)
    sequence = {'mz': np.array([100., 40., 0.]), 'intensity': np.array([2., 1., 0.]),
                'mask': np.array([False, False, True]), 'smiles': 'CCO'}
    dataset = AlignGraphDataset(data={'CCO': [sequence]}, keys=['CCO'], n_views=1, is_augment=True,
                                augment_config=values, full_spectra=True)
    calls = []
    def spectrum_augmentation(item):
        calls.append(item)
        output = copy.deepcopy(item)
        output['intensity'][1] = .5
        return output
    monkeypatch.setattr(dataset, 'aug', spectrum_augmentation)
    monkeypatch.setattr(config.model.mol_encoder, 'graph_policy', 'rdkit_sanitized')
    graph = smiles_to_graph('CCO', graph_policy='rdkit_sanitized')
    mz, intensity, mask, graphs, labels = dataset[0]
    assert len(calls) == 1 and intensity[0, 1] == .5
    assert labels == ['CCO'] and mz.shape == mask.shape == (1, 3)
    for key in ('x', 'edge_index', 'edge_attr', 'graph_size_features'):
        assert torch.equal(getattr(graphs[0], key), getattr(graph, key))
