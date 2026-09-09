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


def test_optimization_preflight_pins_batching_without_changing_other_hyperparameters(tmp_path):
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
        prepared_data=None, alignment_batching='mass_blocks')
    with patch.object(config.fulltrain.v15, 'sources', ConfigObject({'fixture.tsv': sha256_file(source/'fixture.tsv')})), \
         patch.object(runner.subprocess, 'check_output', side_effect=lambda command, **kwargs: '' if 'status' in command else 'test-commit'):
        manifest = runner.preflight(args)
    actual = manifest['runtime_config']['train']['align']
    expected = config.train.align.to_dict()
    expected.update(batching='mass_blocks', metric_for_best='validation_top1_then_mrr')
    assert actual == expected
    assert config.train.align.batching == 'random'
    assert [s['name'] for s in manifest['stages']] == ['prepare_v15','prepare_validation','baseline_validation','alignment42']


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
        'checkpoint_sha256': sha256_file(checkpoint), 'exclude_val_query_indices': [],
        'training_config': settings, 'stages': {'stage2': {'stop_epoch': 1, 'best_epoch': 1}},
        'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5',
            'expected_epoch_counts': {'train': 3, 'val': 3}, 'epochs': [epoch],
            'dataset_manifest_sha256': sha256_file(data/'dataset_manifest.json'), 'training_batching': scheme}}
    path = alignment/'alignment_selection.json'
    path.write_text(json.dumps(selection))
    args = SimpleNamespace(output_root=tmp_path, device='cuda:0')
    with patch.object(runner, 'verify_dataset'), \
         patch.object(config.fulltrain, 'expected_counts', ConfigObject({'train':3,'val':3,'test':1})), \
         patch.object(config.fulltrain, 'exclude_val_query_indices', []):
        assert runner.audit_alignment(args, settings)['epochs'] == 1
        wrong_settings = {**settings, 'batching': 'mass_blocks' if scheme == 'random' else 'random'}
        with pytest.raises(ValueError, match='completion audit'):
            runner.audit_alignment(args, wrong_settings)
        if scheme == 'mass_blocks':
            epoch['batching']['unique_queries'] = 2
        else:
            epoch['train'] = 2
        path.write_text(json.dumps(selection))
        with pytest.raises(ValueError, match='(batching coverage|all train/validation)'):
            runner.audit_alignment(args, settings)
