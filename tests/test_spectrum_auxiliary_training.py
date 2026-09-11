"""Real synthetic candidate training, independent dense gradients and auxiliary input audits."""

import copy
import json
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.data.datasets_candidates import candidate_align_collate_fn
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.loss_align import ContrastiveAlignmentLoss
from SpecEmbedding.loss_candidates import candidate_alignment_loss
from SpecEmbedding.models_spectrum_aux import attach_spectrum_auxiliary
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.trainer.trainer_spectrum_aux import SpectrumAuxiliaryTrainer
from SpecEmbedding.utils import spectrum_targets as target_io
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator, build_validation_index
from SpecEmbedding.utils.spectrum_auxiliary_audit import audit_model_spectrum_targets, audit_spectrum_auxiliary_training
from SpecEmbedding.utils.spectrum_auxiliary_inputs import (
    auxiliary_training_weight,
    bind_spectrum_targets,
    spectrum_target_batch,
    verify_spectrum_target_files,
)
from tests.test_adduct_inputs import raw_queries
from tests.test_candidate_alignment import pinned_synthetic_candidate_input
from tests.test_formal_alignment import model_config
from tests.test_spectrum_auxiliary import HEAD, TARGET


def definition(kind='gine'):
    return {**model_config(kind), 'spectrum_auxiliary': {'model': copy.deepcopy(HEAD), 'target': {**TARGET, 'max_mz': 120}}}


def inputs(tmp_path, monkeypatch):
    data, root = tmp_path / 'data', tmp_path / 'targets'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    raw = raw_queries()[:3]
    for split in ('train', 'val'):
        (data / f'{split}.pkl').write_bytes(pickle.dumps(raw))
    counts, tokenizer = {'train': 3, 'val': 3, 'test': 2}, {'max_len': 4, 'show_progress_bar': False}
    monkeypatch.setattr(target_io, 'verify_dataset', lambda *args: {})
    options = (counts, [], tokenizer, definition()['spectrum_auxiliary']['target'])
    target_io.prepare_spectrum_target_cache(data, root, *options)
    target_io.audit_spectrum_target_cache(data, root, *options)
    cache = target_io.load_spectrum_target_cache(data, root, *options)
    dataset, candidate_settings, receipt, path = pinned_synthetic_candidate_input(
        monkeypatch, tmp_path, loss_weight=1., manifest_sha256=sha256_file(data / 'dataset_manifest.json'))
    tokenized = {}
    for row in raw:
        tokenized.setdefault(row.get('identity_2d'), []).append(Tokenizer(**tokenizer).tokenize(row))
    dataset.base._data = tokenized
    return data, cache, dataset, candidate_settings, receipt, path, tokenized, raw, counts, tokenizer


def loader(dataset):
    return DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn,
                      generator=torch.Generator().manual_seed(42))


@pytest.mark.parametrize('kind', ['gine', 'fingerprint', 'gine_fingerprint'])
def test_formal_auxiliary_attachment_preserves_all_parent_parameters_and_rng(kind):
    settings = definition(kind)
    parent_settings = {key: value for key, value in settings.items() if key != 'spectrum_auxiliary'}
    torch.manual_seed(42)
    parent = build_formal_alignment(parent_settings)
    state = torch.get_rng_state()
    torch.manual_seed(42)
    model = build_formal_alignment(settings)
    assert torch.equal(state, torch.get_rng_state())
    for name, value in parent.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    with pytest.raises(ValueError, match='already attached'):
        attach_spectrum_auxiliary(model, settings['spectrum_auxiliary'])


def test_full_target_binding_preserves_query_order_and_rejects_changed_tokens(tmp_path, monkeypatch):
    _, cache, dataset, *_ = inputs(tmp_path, monkeypatch)
    bind_spectrum_targets(dataset, cache)
    actual = spectrum_target_batch(cache, torch.tensor([1, 0, 2]))
    assert actual['raw_query_indices'].tolist() == [1, 0, 2]
    assert actual['adduct_ids'].tolist() == [1, 0, 2]
    identity, offset = dataset.base._spectrum_indices[0]
    dataset.base._data[identity][offset]['intensity'][0] += 1
    with pytest.raises(ValueError, match='query/token'):
        bind_spectrum_targets(dataset, cache)


def reset_rng():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def test_zero_weight_real_optimizer_updates_and_rng_match_candidate_parent(tmp_path, monkeypatch):
    _, cache, dataset, *_ = inputs(tmp_path, monkeypatch)
    base = build_formal_alignment(model_config())
    candidate = copy.deepcopy(base)
    attach_spectrum_auxiliary(candidate, definition()['spectrum_auxiliary'])
    old = CandidateTrainerAlign(base, loader(copy.deepcopy(dataset)), None, torch.device('cpu'),
                                save_dir=str(tmp_path / 'old'), candidate_loss_weight=1.)
    new = SpectrumAuxiliaryTrainer(candidate, loader(dataset), None, torch.device('cpu'),
        save_dir=str(tmp_path / 'new'), candidate_loss_weight=1., spectrum_targets=cache, auxiliary_loss_weight=0.)
    old.expected_epoch_counts = new.expected_epoch_counts = {'train': 3, 'val': 3}
    reset_rng()
    old.train_epoch(torch.optim.AdamW(base.parameters(), lr=.001), 1, 'stage2')
    state, numpy_state, random_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    reset_rng()
    new.train_epoch(torch.optim.AdamW(candidate.parameters(), lr=.001), 1, 'stage2')
    assert torch.equal(state, torch.get_rng_state()) and random_state == random.getstate()
    assert np.array_equal(numpy_state[1], np.random.get_state()[1]) and numpy_state[2:] == np.random.get_state()[2:]
    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items())
    assert all(parameter.grad is None for parameter in candidate.spectrum_auxiliary.parameters())


def test_actual_auxiliary_optimizer_matches_independent_dense_loss_and_keeps_empty_candidate_queries(tmp_path, monkeypatch):
    _, cache, dataset, *_ = inputs(tmp_path, monkeypatch)
    reset_rng()
    model = build_formal_alignment(definition())
    reference = copy.deepcopy(model)
    before = copy.deepcopy(model.state_dict())
    steps = []
    original_step = torch.optim.AdamW.step
    def capture_step(optimizer, *args, **kwargs):
        steps.append([None if p.grad is None else p.grad.detach().clone()
                      for group in optimizer.param_groups for p in group['params']])
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', capture_step)
    trainer = SpectrumAuxiliaryTrainer(model, loader(dataset), None, torch.device('cpu'),
        save_dir=str(tmp_path / 'run'), candidate_loss_weight=1., spectrum_targets=cache, auxiliary_loss_weight=1.)
    reset_rng()
    trainer.train_epoch(torch.optim.AdamW(model.parameters(), lr=.001), 1, 'stage2')
    dataset.set_epoch(1)
    reset_rng()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.001)
    for batch in loader(dataset):
        optimizer.zero_grad()
        spec, positive, scale = reference(*batch.anchor[:4])
        negative = reference.encode_mol(batch.negative_graphs) if batch.negative_graphs is not None else positive[:0]
        main = ContrastiveAlignmentLoss()(spec, positive, scale, batch.anchor[-1])
        ce = candidate_alignment_loss(spec, positive, negative, batch.negative_ptr, scale)
        dense = torch.zeros(len(positive), reference.spectrum_auxiliary.output_dim)
        ids = []
        for row, raw in enumerate(batch.raw_query_indices.tolist()):
            values = cache[raw]
            dense[row, values['bin_indices']] = torch.from_numpy(values['values'])
            ids.append(values['adduct_id'])
        predicted = reference.spectrum_auxiliary(positive, torch.tensor(ids))
        auxiliary = (1 - (predicted * dense).sum(1) / torch.linalg.vector_norm(predicted, dim=1)).mean()
        (main + ce + auxiliary).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        optimizer.step()
    assert len(steps) == 4
    for step in (0, 1):
        for (name, _), actual, expected in zip(model.named_parameters(), steps[step], steps[step + 2], strict=True):
            if actual is None or expected is None:
                assert actual is expected
            else:
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
                if name.endswith('self_attn.in_proj_bias'):
                    width = actual.numel() // 3
                    # A constant key bias cancels in softmax; float32 reductions leave ~1e-11 gradients.
                    # Adam's epsilon can turn their different signs into ~1e-6 updates (see numerical diagnosis).
                    assert actual[width:2 * width].abs().max() < 1e-9
                    assert expected[width:2 * width].abs().max() < 1e-9
    for name, value in reference.state_dict().items():
        actual = model.state_dict()[name]
        if name.endswith('self_attn.in_proj_bias'):
            width = value.numel() // 3
            torch.testing.assert_close(actual[width:2 * width], value[width:2 * width], rtol=0, atol=1e-5)
            torch.testing.assert_close(torch.cat((actual[:width], actual[2 * width:])),
                                      torch.cat((value[:width], value[2 * width:])), rtol=2e-5, atol=1e-7)
        else:
            torch.testing.assert_close(actual, value, rtol=2e-5, atol=1e-7)
    for prefix in ('spectrum_auxiliary.', 'mol_encoder.', 'mol_proj.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(prefix))
    assert trainer.candidate_epoch_audits[0]['queries_without_negatives'] == 1
    record = trainer.auxiliary_epoch_audits[0]
    assert record['queries'] == record['unique_queries'] == 3 and record['optimizer_steps'] == 2
    assert record['gradient_probe']['weighted_auxiliary_norm'] > 0
    stage = {'stop_epoch': 1, 'spectrum_auxiliary': {
        'loss': 'mean_query_cosine_distance_raw_observed_spectrum', 'loss_weight': 1., 'targets': cache.provenance,
        'epochs': trainer.auxiliary_epoch_audits}, 'candidate_training': {
        'candidate_loss_weight': 1., 'epochs': trainer.candidate_epoch_audits}}
    assert audit_spectrum_auxiliary_training(tmp_path / 'run', stage, cache, 1.)[0]['epochs'] == 1
    # Update both summary and record: independent replay still rejects semantic corruption.
    stage['spectrum_auxiliary']['epochs'][0]['observed_target_batch_sha256'] = '0' * 64
    (tmp_path / 'run/spectrum_auxiliary/stage2_epoch001.json').write_text(json.dumps(stage['spectrum_auxiliary']['epochs'][0]))
    with pytest.raises(ValueError, match='target bytes'):
        audit_spectrum_auxiliary_training(tmp_path / 'run', stage, cache, 1.)


def test_real_training_entry_reloads_head_and_validation_ignores_it(tmp_path, monkeypatch):
    import train_align as entry
    data, cache, dataset, candidate_settings, receipt, input_path, tokenized, raw, counts, tokenizer = inputs(tmp_path, monkeypatch)
    definition_config = definition()
    monkeypatch.setattr(config, 'model', ConfigObject(definition_config))
    monkeypatch.setattr(config.data, 'tokenizer', ConfigObject(tokenizer))
    for key, value in {'candidate_supervision': ConfigObject(candidate_settings), 'epochs_stage2': 2,
        'num_workers': 0, 'batching': 'mass_blocks', 'mass_block_size': 1, 'batch_size': 2,
        'spectrum_auxiliary': ConfigObject({'loss_weight': 1.})}.items():
        monkeypatch.setattr(config.train.align, key, value, raising=False)
    monkeypatch.setattr(config.augmentation, 'prob', 0.)
    def cpu_trainer(*args, **kwargs):
        kwargs['record_resources'] = False  # CPU synthetic fixture only; formal device guard remains.
        return SpectrumAuxiliaryTrainer(*args, **kwargs)
    monkeypatch.setattr(entry, 'SpectrumAuxiliaryTrainer', cpu_trainer)
    output = tmp_path / 'run/alignment42_topk256'
    outputs = {f'{split}.pkl': sha256_file(data / f'{split}.pkl') for split in ('train', 'val')}
    index = build_validation_index(raw, {'CCO': ['CC', 'CCC'], 'CC': ['CC', 'CCO']}, [], {}, tokenizer)
    index.update(dataset_manifest_sha256=sha256_file(data / 'dataset_manifest.json'), dataset_outputs=outputs)
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=(1, 5, 10, 20))
    validator = AlignmentRetrievalValidator(index, settings, output / 'validation_retrieval')
    selection_input = {'data_path': str(data), 'seed': 42, 'exclude_val_query_indices': [], 'fulltrain_audit': {
        'formal_fulltrain': True, 'dataset_version': '1.5', 'dataset_manifest_sha256': index['dataset_manifest_sha256'],
        'input_outputs': outputs, 'expected_epoch_counts': {'train': 3, 'val': 3}}}
    model = entry.train_align(tokenized, sorted(tokenized), tokenized, sorted(tokenized), None, batch_size=2, lr=.001,
        save_dir=str(output), device='cpu', seed=42, formal_fulltrain=True, retrieval_validator=validator,
        training_candidates=dataset.candidates, candidate_input_receipt=receipt, selection_metadata=selection_input,
        spectrum_targets=cache)
    selection = json.loads((output / 'alignment_selection.json').read_text())
    assert selection['spectrum_targets'] == cache.provenance
    initial_head = build_formal_alignment(definition_config).spectrum_auxiliary
    assert not torch.equal(model.spectrum_auxiliary.output.weight, initial_head.output.weight)
    assert all(row['train'] == row['val'] == row['batching']['unique_queries'] == 3 for row in selection['fulltrain_audit']['epochs'])
    checkpoint = output / 'best_model_stage2.pth'
    restored, _, load_receipt = load_formal_alignment(checkpoint, torch.device('cpu'), dataset_outputs=outputs,
        dataset_manifest_sha256=index['dataset_manifest_sha256'], tokenizer_config=tokenizer,
        expected_counts={'train': 3, 'val': 3}, exclusions=[])
    assert load_receipt['spectrum_targets'] == cache.provenance
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    validator(restored, torch.device('cpu'), 3, 'with_head')
    del restored.spectrum_auxiliary
    validator(restored, torch.device('cpu'), 3, 'without_head')
    def scores(name):
        return torch.load(output / 'validation_retrieval' / f'{name}_epoch003.pt', weights_only=False)['scores']
    torch.testing.assert_close(scores('with_head'), scores('without_head'), rtol=0, atol=0)
    stage = selection['stages']['stage2']
    report, _ = audit_spectrum_auxiliary_training(output, stage, cache, 1.)
    assert report['epochs'] == 2 and report['queries'] == 3
    runtime = copy.deepcopy(selection['config_snapshot'])
    runtime['fulltrain'].update(expected_counts=counts, exclude_val_query_indices=[])
    manifest = {'runtime_config': runtime, 'spectrum_targets': cache.provenance, 'inputs': verify_spectrum_target_files(cache.provenance)}
    audited_cache, weight, _ = audit_model_spectrum_targets(manifest, selection, data)
    assert len(audited_cache) == 3 and weight == 1.
    manifest['inputs'].pop('spectrum_targets_values.bin')
    with pytest.raises(ValueError, match='not pinned'):
        audit_model_spectrum_targets(manifest, selection, data)
    from tests.test_spectrum_auxiliary_integration import verify_complete_auxiliary_fixture
    verify_complete_auxiliary_fixture(monkeypatch, output, data, index, selection, input_path, receipt, cache, counts)
    weights = torch.load(checkpoint, weights_only=True)
    weights.pop('spectrum_auxiliary.output.weight')
    torch.save(weights, checkpoint)
    selection['checkpoint_sha256'] = sha256_file(checkpoint)
    (output / 'alignment_selection.json').write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match='Missing key'):
        load_formal_alignment(checkpoint, torch.device('cpu'), dataset_outputs=outputs,
            dataset_manifest_sha256=index['dataset_manifest_sha256'], tokenizer_config=tokenizer,
            expected_counts={'train': 3, 'val': 3}, exclusions=[])


@pytest.mark.parametrize('damage', ['weight_missing', 'model_missing', 'weight_zero', 'weight_nan'])
def test_formal_auxiliary_requires_explicit_paired_positive_weight(damage):
    model, training = definition(), {'spectrum_auxiliary': {'loss_weight': 1.}}
    if damage == 'weight_missing':
        training.clear()
    elif damage == 'model_missing':
        model.pop('spectrum_auxiliary')
    else:
        training['spectrum_auxiliary']['loss_weight'] = 0 if damage == 'weight_zero' else float('nan')
    with pytest.raises(ValueError):
        auxiliary_training_weight(model, training)
