"""Candidate weighting: inherited protocols and an independent optimizer-step reference."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.data.datasets_align import align_collate_fn
from SpecEmbedding.data.datasets_candidates import candidate_align_collate_fn
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils.alignment_successor import candidate_weight_successor_configuration
from tests.test_candidate_alignment import candidate_dataset, pinned_synthetic_candidate_input, small_model
from tests.test_formal_alignment import model_config
from tests.test_graph_fingerprint_inputs import make_dataset
from tests.test_graph_fingerprint_inputs import small_model as combined_model


def parent_inputs(tmp_path, kind='gine', *, attention=False, structural=False):
    runtime = {
        'model': model_config(kind),
        'train': {'align': {'batch_size': 128, 'lr': .0001, 'batching': 'mass_blocks', 'mass_block_size': 32,
                           'candidate_supervision': {'enabled': True, 'negative_count': 16, 'loss_weight': 1.0,
                                                     'pool_cache_size': 128, 'graph_cache_size': 10000}}},
        'augmentation': {'prob': .5, 'node_drop_rate': .1}, 'fulltrain': {'min_free_mib': 20000},
        'data': {'tokenizer': {'max_len': 100}, 'cache_path': '/old/source/train_cache'},
    }
    if attention:
        runtime['model']['spec_encoder'].update(attention_pool={'norm_eps': 1e-5}, qk_norm={'eps': 1e-6})
    if structural:
        runtime['train']['align']['candidate_supervision']['sampling'] = {
            'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32, 'fingerprint_radius': 2,
            'fingerprint_bits': 2048, 'cache_directory': str(tmp_path / 'similarity')}
    selection = copy.deepcopy({'model_config': runtime['model'], 'training_config': runtime['train']['align'],
                               'config_snapshot': {'augmentation': runtime['augmentation']}})
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    storage = {'storage': {'root': str(tmp_path)}, 'data': {'cache_path': str(tmp_path / 'train_cache')}}
    return runtime, selection, config.training_candidate_weight.to_dict(), gpu, storage


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('attention,structural', [(False, False), (True, False), (False, True), (True, True)])
def test_weight_successor_preserves_model_sampling_and_baseline(tmp_path, kind, attention, structural):
    runtime, selection, settings, gpu, storage = parent_inputs(tmp_path, kind, attention=attention, structural=structural)
    originals = copy.deepcopy((runtime, selection, settings, gpu, storage))
    result = candidate_weight_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)
    expected = copy.deepcopy(runtime)
    expected['train']['align']['candidate_supervision']['loss_weight'] = 2.0
    expected['fulltrain'].update(gpu)
    expected['storage'] = storage['storage']
    expected['data']['cache_path'] = storage['data']['cache_path']
    assert result == expected
    assert (runtime, selection, settings, gpu, storage) == originals
    result['model']['spec_encoder']['n_layer'] += 1
    result['train']['align']['candidate_supervision']['negative_count'] += 1
    assert (runtime, selection, settings, gpu, storage) == originals


@pytest.mark.parametrize('damage', ['already_changed', 'disabled', 'missing', 'unknown', 'zero', 'nan', 'bool',
                                   'same', 'bad_expected', 'model', 'training', 'augmentation', 'storage', 'gpu', 'fingerprint'])
def test_weight_successor_refuses_unbound_or_changed_parent(tmp_path, damage):
    runtime, selection, settings, gpu, storage = parent_inputs(tmp_path)
    if damage == 'already_changed':
        runtime['train']['align']['candidate_supervision']['loss_weight'] = 2.0
    elif damage == 'disabled':
        runtime['train']['align']['candidate_supervision']['enabled'] = False
    elif damage == 'missing':
        del settings['expected_parent_weight']
    elif damage == 'unknown':
        settings['negative_count'] = 32
    elif damage in ('zero', 'nan', 'bool', 'same'):
        settings['loss_weight'] = {'zero': 0, 'nan': float('nan'), 'bool': True, 'same': 1.0}[damage]
    elif damage == 'bad_expected':
        settings['expected_parent_weight'] = True
    elif damage == 'model':
        selection['model_config']['spec_encoder']['n_layer'] += 1
    elif damage == 'training':
        selection['training_config']['batch_size'] = 256
    elif damage == 'augmentation':
        selection['config_snapshot']['augmentation']['prob'] = 0
    elif damage == 'storage':
        storage['data']['cache_path'] = '/old/source/train_cache'
    elif damage == 'gpu':
        del gpu['hold_seconds']
    else:
        runtime['model'] = model_config('fingerprint')
    with pytest.raises(ValueError):
        candidate_weight_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)


def independent_losses(model, batch):
    """Scalar log-partitions, without either production loss implementation."""
    mz, intensity, mask, graphs, labels = batch.anchor
    spec, positive, scale = model(mz, intensity, mask, graphs)
    negative = model.encode_mol(batch.negative_graphs)
    spec, positive, negative = [F.normalize(value, dim=-1) for value in (spec, positive, negative)]
    scores = scale * (spec @ positive.T)
    terms = []
    for matrix in (scores, scores.T):
        for i, row in enumerate(matrix):
            positives = [j for j in range(len(labels)) if labels[i] == labels[j]]
            terms.append(torch.logsumexp(row, 0) - torch.stack([row[j] for j in positives]).mean())
    base = torch.stack(terms).mean()
    terms = []
    for i in range(len(spec)):
        start, end = batch.negative_ptr[i:i + 2].tolist()
        target = scale * (spec[i] * positive[i]).sum()
        logits = torch.cat((target[None], scale * (negative[start:end] @ spec[i])))
        terms.append(torch.logsumexp(logits, 0) - target)
    return base, torch.stack(terms).mean()


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
@pytest.mark.parametrize('weight', [1.0, 2.0])
def test_actual_weighted_optimizer_step_matches_independent_loss_and_gradients(tmp_path, monkeypatch, kind, weight):
    # All three synthetic queries include a duplicate identity and a query with no negatives.
    dataset = candidate_dataset(monkeypatch) if kind == 'gine' else make_dataset(tmp_path, monkeypatch)[0]
    dataset.set_epoch(1)
    loader = DataLoader(dataset, batch_size=3, collate_fn=candidate_align_collate_fn)
    val = DataLoader(dataset.base, batch_size=3, collate_fn=align_collate_fn)
    torch.manual_seed(23)
    model = small_model() if kind == 'gine' else combined_model()
    reference = copy.deepcopy(model).train()
    trainer = CandidateTrainerAlign(model, loader, val, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=weight)
    trainer.expected_epoch_counts = {'train': 3, 'val': 3}
    rng = torch.get_rng_state()
    batch = next(iter(loader))
    base, candidate = independent_losses(reference, batch)
    expected_loss = base + weight * candidate
    expected_loss.backward()
    reference_gradients = {name: None if p.grad is None else p.grad.clone() for name, p in reference.named_parameters()}
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    torch.optim.SGD(reference.parameters(), lr=.02).step()
    observed_gradients = {}
    handles = [p.register_hook(lambda gradient, name=name: observed_gradients.__setitem__(name, gradient.clone()))
               for name, p in model.named_parameters() if p.requires_grad]
    torch.set_rng_state(rng)
    loss = trainer.train_epoch(torch.optim.SGD(model.parameters(), lr=.02), 1, 'stage2')
    for handle in handles:
        handle.remove()
    assert loss == pytest.approx(expected_loss.item(), rel=1e-5)
    for name, gradient in reference_gradients.items():
        if gradient is None:
            assert name not in observed_gradients
        else:
            torch.testing.assert_close(observed_gradients[name], gradient, rtol=2e-4, atol=2e-6)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=2e-5, atol=2e-6)
    record = trainer.candidate_epoch_audits[0]
    assert record['queries'] == record['unique_queries'] == 3
    assert record['queries_without_negatives'] == 1 and record['negative_samples'] == 4
    assert record['candidate_loss_query_mean'] == pytest.approx(candidate.item(), rel=1e-5)


@pytest.mark.parametrize('attention,structural', [(False, False), (True, False), (False, True), (True, True)])
def test_combined_formal_entry_saves_weight_two_and_rejects_old_input_or_training_weight(tmp_path, monkeypatch, attention, structural):
    import train_align as entry
    from SpecEmbedding.utils import candidate_training as candidate_io
    from SpecEmbedding.utils.fingerprint_cache import (
        audit_fingerprint_cache,
        build_fingerprint_cache,
        fingerprint_provenance,
        load_fingerprint_cache,
    )
    from SpecEmbedding.utils.formal_alignment import load_formal_alignment

    dataset, settings, input_receipt, input_path = pinned_synthetic_candidate_input(
        monkeypatch, tmp_path, structural=structural, loss_weight=2.0)
    smiles = dataset.candidates.metadata['mol_smiles']
    cache_root = tmp_path / 'combined_bits'
    provenance = fingerprint_provenance(smiles, index_sha256=dataset.candidates.provenance['sha256'],
                                        dataset_manifest_sha256='a' * 64, radius=2, bits=2048)
    build_fingerprint_cache(smiles, cache_root, provenance, workers=1, chunk_size=3)
    audit_fingerprint_cache(smiles, cache_root, provenance, workers=1, chunk_size=3)
    _, cache = load_fingerprint_cache(smiles, cache_root, provenance)
    definition = model_config('gine_fingerprint')
    if attention:
        definition['spec_encoder'].update(attention_pool={'norm_eps': 1e-5}, qk_norm={'eps': 1e-6})
    monkeypatch.setattr(config, 'model', ConfigObject(definition))
    for key, value in {'candidate_supervision': ConfigObject(settings), 'epochs_stage2': 2,
                       'num_workers': 0, 'batching': 'mass_blocks', 'mass_block_size': 1}.items():
        monkeypatch.setattr(config.train.align, key, value)
    monkeypatch.setattr(config.augmentation, 'prob', 0.)
    def cpu_trainer(*args, **kwargs):
        kwargs['record_resources'] = False  # Synthetic CPU integration, no CUDA telemetry or formal trial.
        return CandidateTrainerAlign(*args, **kwargs)
    monkeypatch.setattr(entry, 'CandidateTrainerAlign', cpu_trainer)
    metadata = {'seed': 42, 'exclude_val_query_indices': [],
                'training_fingerprint_cache': cache, 'validation_fingerprint_cache': cache,
                'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5',
                    'expected_epoch_counts': {'train': 3, 'val': 3}, 'dataset_manifest_sha256': 'a' * 64,
                    'input_outputs': {'synthetic': 'complete three-query CPU fixture'}}}
    output = tmp_path / 'combined_run'
    entry.train_align(dataset.base._data, sorted(dataset.base._keys), dataset.base._data, sorted(dataset.base._keys), None,
        batch_size=2, lr=.001, save_dir=str(output), device='cpu', seed=42, formal_fulltrain=True,
        retrieval_validator=lambda *args: {'top1': .5, 'top5': 1., 'top10': 1., 'top20': 1., 'mrr': .75},
        training_candidates=dataset.candidates, candidate_input_receipt=input_receipt, selection_metadata=metadata,
        fingerprint_inputs={split: {'cache': cache} for split in ('train', 'validation')},
        fingerprint_smiles={split: smiles for split in ('train', 'validation')})
    selection = json.loads((output / 'alignment_selection.json').read_text())
    stage = selection['stages']['stage2']
    assert selection['training_config']['candidate_supervision']['loss_weight'] == 2.0
    assert selection['config_snapshot']['train']['align']['candidate_supervision']['loss_weight'] == 2.0
    assert stage['candidate_training']['candidate_loss_weight'] == 2.0
    for row in selection['fulltrain_audit']['epochs']:
        assert row['train'] == row['val'] == row['batching']['unique_queries'] == 3
    kwargs = {'fingerprint_cache': cache, 'graph_fingerprint': True}
    report, _ = candidate_io.audit_candidate_training(output, stage, input_path, settings, 42, 2, tmp_path, {'train': 3}, [], **kwargs)
    assert report['negative_sampling_replayed'] and report['epochs'] == 2
    restored, _, receipt = load_formal_alignment(output / 'best_model_stage2.pth', torch.device('cpu'),
        dataset_outputs=metadata['fulltrain_audit']['input_outputs'], dataset_manifest_sha256='a' * 64,
        tokenizer_config=config.data.tokenizer.to_dict(), expected_counts={'train': 3, 'val': 3}, exclusions=[])
    assert receipt['model_config'] == definition
    saved = torch.load(output / 'best_model_stage2.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in saved.items())
    altered = copy.deepcopy(stage)
    altered['candidate_training']['candidate_loss_weight'] = 1.0
    with pytest.raises(ValueError, match='Candidate loss'):
        candidate_io.audit_candidate_training(output, altered, input_path, settings, 42, 2, tmp_path, {'train': 3}, [], **kwargs)
    with pytest.raises(ValueError, match='configuration'):
        candidate_io.read_candidate_training_input(input_path, tmp_path, {**settings, 'loss_weight': 1.0}, {'train': 3}, [])
