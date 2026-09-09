import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_fingerprint import (
    CandidateFingerprintDataset,
    FingerprintAlignmentDataset,
    candidate_fingerprint_collate_fn,
    fingerprint_align_collate_fn,
)
from SpecEmbedding.loss_candidates import candidate_alignment_loss
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_fingerprint import FingerprintAlignmentModel, FingerprintEncoder
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils.fingerprint_cache import (
    audit_fingerprint_cache,
    build_fingerprint_cache,
    fingerprint_provenance,
)
from SpecEmbedding.utils.fingerprint_validation import FingerprintRetrievalValidator
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import build_validation_index
from tests.test_candidate_alignment import candidate_dataset, key
from tests.test_retrieval_validation import build_index, spectrum


def make_cache(root, smiles, *, index_sha='b' * 64):
    provenance = fingerprint_provenance(smiles, index_sha256=index_sha,
                                       dataset_manifest_sha256='a' * 64, radius=2, bits=2048)
    build_fingerprint_cache(smiles, root, provenance, workers=1, chunk_size=2)
    audit_fingerprint_cache(smiles, root, provenance, workers=1, chunk_size=2)
    return provenance


def make_dataset(tmp_path, monkeypatch, *, augment=False):
    original = candidate_dataset(monkeypatch, augment=augment)
    original.candidates.provenance['sha256'] = 'b' * 64
    smiles = original.candidates.metadata['mol_smiles']
    root = tmp_path / 'train_inputs'
    provenance = make_cache(root, smiles)
    options = dict(data=original.base._data, keys=original.base._keys, full_spectra=True, n_views=1,
                   is_augment=augment, graph_policy='rdkit_sanitized',
                   augment_config={**original.base.augment_config, 'node_drop_rate': 0., 'edge_mask_rate': 0.},
                   fingerprint_root=root, fingerprint_smiles=smiles, fingerprint_provenance=provenance,
                   dataset_manifest_sha256='a' * 64)
    base = FingerprintAlignmentDataset(**options)
    dataset = CandidateFingerprintDataset(base, original.candidates, dataset_manifest_sha256='a' * 64,
                                          negative_count=16, seed=42)
    return dataset, original, options


def small_model():
    return FingerprintAlignmentModel(
        spec_config={'embedding_dim': 8, 'n_head': 2, 'n_layer': 1, 'dim_feedward': 8, 'dim_target': 8,
                     'feedward_activation': 'selu'},
        molecule_config={'input_bits': 2048, 'hidden_dim': 16, 'emb_dim': 8, 'dropout_rate': 0., 'norm_eps': 1e-5},
        alignment_config={'final_dim': 8, 'dropout_rate': 0., 'tau': .2})


def validator(tmp_path, *, index=None, workers=0):
    index = build_index() if index is None else index
    index['dataset_manifest_sha256'] = 'a' * 64
    root = tmp_path / 'val_inputs'
    provenance = make_cache(root, index['mol_smiles'], index_sha='c' * 64)
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=workers, top_k=(1, 5, 10, 20))
    result = FingerprintRetrievalValidator(index, settings, tmp_path / 'run' / 'validation_retrieval',
                                           fingerprint_root=root, fingerprint_provenance=provenance,
                                           index_sha256='c' * 64)
    return result, index


def test_fingerprint_samples_preserve_grouped_queries_identity_labels_and_small_pools(tmp_path, monkeypatch):
    dataset, original, _ = make_dataset(tmp_path, monkeypatch)
    assert np.array_equal(dataset.raw_query_indices, original.raw_query_indices)
    with pytest.raises(RuntimeError, match='epoch'):
        dataset[0]
    dataset.set_epoch(7)
    with patch('SpecEmbedding.data.datasets_align.smiles_to_graph', side_effect=AssertionError('Constructed a graph')):
        examples = [dataset[i] for i in range(len(dataset))]
    batch = candidate_fingerprint_collate_fn(examples)
    assert batch.raw_query_indices.tolist() == [1, 0, 2]
    assert batch.anchor[0][:, 0].tolist() == [101, 100, 102]
    assert batch.anchor[-1].tolist() == [0, 1, 1]
    assert batch.negative_ptr.tolist() == [0, 0, 2, 4]
    assert batch.negative_graphs.shape == (4, 2048)
    for example in examples:
        expected = original.candidates.sample(example.raw_query_index, negative_count=16, seed=42, epoch=7)
        assert np.array_equal(example.sample.molecule_indices, expected.molecule_indices)
        assert np.array_equal(example.sample.source_positions, expected.source_positions)
        if example.raw_query_index != 1:
            assert set(example.sample.identity_2d) == {key('CC'), key('CCC')}
    assert candidate_fingerprint_collate_fn([examples[0]]).negative_graphs is None
    expected = dataset.base.fingerprints.get_many(batch.candidate_indices.numpy())
    assert np.array_equal(batch.negative_graphs.numpy(), expected)
    batch.anchor[0][0, 0] = -1
    batch.negative_graphs.zero_()
    assert dataset[0].anchor[0][0, 0] == 101
    assert np.array_equal(dataset.base.fingerprints.get_many(batch.candidate_indices.numpy()), expected)


def test_spectral_augmentation_uses_inherited_path_without_graph_augmentation(tmp_path, monkeypatch):
    dataset, original, _ = make_dataset(tmp_path, monkeypatch, augment=True)
    np.random.seed(11)
    expected, label = original.base.full_spectrum_item(1)
    np.random.seed(11)
    with patch.object(dataset.base, 'get_mol_graph', side_effect=AssertionError('Graph requested')):
        actual = dataset.base[1]
    for position, name in enumerate(('mz', 'intensity', 'mask')):
        assert np.array_equal(actual[position][0].numpy(), expected[name])
    assert actual[-1] == [label]
    assert type(small_model().spec_encoder) is SiameseModel


@pytest.mark.parametrize('damage', ['manifest', 'augmentation', 'partial', 'missing_target', 'source_order', 'source_sha'])
def test_fingerprint_dataset_rejects_wrong_sources_and_implicit_representation_changes(tmp_path, monkeypatch, damage):
    dataset, _, options = make_dataset(tmp_path, monkeypatch)
    if damage == 'manifest':
        options['dataset_manifest_sha256'] = 'd' * 64
    elif damage == 'augmentation':
        options['augment_config'] = {**options['augment_config'], 'node_drop_rate': .1}
    elif damage == 'partial':
        options['full_spectra'] = False
    elif damage == 'missing_target':
        options['data'] = copy.deepcopy(options['data'])
        options['data'][key('CCO')][0]['smiles'] = 'CCCO'
    else:
        if damage == 'source_order':
            dataset.candidates.metadata['mol_smiles'] = list(reversed(dataset.candidates.metadata['mol_smiles']))
        else:
            dataset.candidates.provenance['sha256'] = 'd' * 64
        with pytest.raises(ValueError, match='index differs'):
            CandidateFingerprintDataset(dataset.base, dataset.candidates, dataset_manifest_sha256='a' * 64,
                                        negative_count=16, seed=42)
        return
    with pytest.raises(ValueError):
        FingerprintAlignmentDataset(**options)


def test_spawn_workers_preserve_complete_query_sample_and_fingerprint_binding(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    dataset.set_epoch(2)
    expected = list(DataLoader(dataset, batch_size=2, collate_fn=candidate_fingerprint_collate_fn))
    loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_fingerprint_collate_fn, num_workers=2,
                        multiprocessing_context='spawn', generator=torch.Generator().manual_seed(1), timeout=30)
    for a, b in zip(expected, loader, strict=True):
        for field in ('negative_ptr', 'raw_query_indices', 'candidate_indices', 'source_positions', 'negative_graphs'):
            torch.testing.assert_close(getattr(a, field), getattr(b, field))
        for x, y in zip(a.anchor, b.anchor, strict=True):
            torch.testing.assert_close(x, y)
    dataset.set_epoch(3)
    assert sum(len(batch.raw_query_indices) for batch in loader) == len(dataset)


def test_candidate_loss_backpropagates_through_both_towers_and_negative_encodings(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    dataset.set_epoch(1)
    batch = candidate_fingerprint_collate_fn([dataset[i] for i in range(len(dataset))])
    model = small_model()
    spec, positive, scale = model(*batch.anchor[:4])
    negative = model.encode_mol(batch.negative_graphs)
    negative.retain_grad()
    loss = candidate_alignment_loss(spec, positive, negative, batch.negative_ptr, scale)
    loss.backward()
    assert negative.grad.abs().sum() > 0
    for module in (model.spec_encoder, model.mol_encoder, model.spec_proj, model.mol_proj):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert torch.isfinite(model.logit_scale.grad)


def test_complete_synthetic_training_validation_snapshot_and_checkpoint_reconstruction(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    train = DataLoader(dataset, batch_size=2, collate_fn=candidate_fingerprint_collate_fn)
    val = DataLoader(dataset.base, batch_size=2, collate_fn=fingerprint_align_collate_fn)
    validate, index = validator(tmp_path)
    model = small_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainer = CandidateTrainerAlign(model, train, val, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=1., retrieval_validator=validate)
    trainer.expected_epoch_counts = {'train': 3, 'val': 3}
    trainer.fit(epochs=2, optimizer=torch.optim.AdamW(model.parameters(), lr=.001), stage_name='stage2')
    assert trainer.epoch_counts == [{'stage': 'stage2', 'epoch': e, 'train': 3, 'val': 3} for e in [1, 2]]
    for group in ('spec_encoder.', 'mol_encoder.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(group))
    records = trainer.stage_summaries['stage2']['candidate_training']
    assert records['data']['molecule_input'] == 'fixed_morgan_bits'
    for epoch, record in enumerate(records['epochs'], 1):
        assert record['queries'] == record['unique_queries'] == 3 and record['negative_samples'] == 4
        assert record['queries_without_negatives'] == 1
        path = tmp_path / 'run' / 'validation_retrieval' / f'stage2_epoch{epoch:03}.pt'
        with pytest.raises(ValueError, match='Unexpected fingerprint cache'):
            audit_snapshot(path, index)
        audit = audit_snapshot(path, index, expected_fingerprint_cache=validate.fingerprint_receipt)
        assert audit['queries'] == 3 and audit['positive_queries'] == 1
        assert audit['top1'] <= 1/3 and audit['mrr'] <= 1/3
    # The complete constructor settings must accompany the selected weights; bare dimensions are insufficient.
    settings_path = tmp_path / 'model_settings.json'
    settings_path.write_text(json.dumps(model.construction_config()))
    clone = FingerprintAlignmentModel(**json.loads(settings_path.read_text()))
    clone.load_state_dict(torch.load(tmp_path / 'run' / 'best_model_stage2.pth', weights_only=True), strict=True)
    clone.eval()
    model.eval()
    mz, intensity, mask, molecules, _ = next(iter(val))
    for a, b in zip(model(mz, intensity, mask, molecules), clone(mz, intensity, mask, molecules), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    settings = model.construction_config()
    settings['molecule_config']['hidden_dim'] += 1
    with pytest.raises(RuntimeError):
        FingerprintAlignmentModel(**settings).load_state_dict(model.state_dict(), strict=True)
    assert model.construction_config()['molecule_config']['hidden_dim'] == 16


def test_fingerprint_collisions_remain_distinct_2d_candidates_and_validation_rng_is_unchanged(tmp_path):
    smiles = ['CCCCCCCCCCCC', 'CCCCCCCCCCCCC']
    index = build_validation_index([spectrum(smiles[0]), spectrum(smiles[1])],
                                   {s: smiles for s in smiles}, [], {}, {'max_len': 8, 'show_progress_bar': False})
    validate, index = validator(tmp_path, index=index)
    assert index['positive_mask'][:, :2].tolist() == [[True, False], [False, True]]
    model = small_model()
    before = torch.random.get_rng_state()
    metrics = validate(model, torch.device('cpu'), 1, 'collision')
    assert torch.equal(before, torch.random.get_rng_state())
    snapshot = torch.load(tmp_path / 'run' / 'validation_retrieval' / 'collision_epoch001.pt', weights_only=False)
    assert torch.equal(snapshot['scores'][:, 0], snapshot['scores'][:, 1])
    assert metrics['top1'] == .5 and metrics['mrr'] == .75
    audit_snapshot(tmp_path / 'run' / 'validation_retrieval' / 'collision_epoch001.pt', index,
                   expected_fingerprint_cache=validate.fingerprint_receipt)


def test_validation_reencodes_changed_weights_and_keeps_input_order_with_multiple_workers(tmp_path):
    validate, index = validator(tmp_path, workers=2)
    model = small_model()
    validate(model, torch.device('cpu'), 1, 'fresh')
    first = torch.load(tmp_path / 'run' / 'validation_retrieval' / 'fresh_epoch001.pt', weights_only=False)
    with torch.no_grad():
        model.mol_encoder.layers[0].bias[0].add_(5.)
    validate(model, torch.device('cpu'), 2, 'fresh')
    second = torch.load(tmp_path / 'run' / 'validation_retrieval' / 'fresh_epoch002.pt', weights_only=False)
    valid = index['candidate_indices'] >= 0
    assert not torch.equal(first['scores'][valid], second['scores'][valid])
    assert first['raw_query_indices'] == second['raw_query_indices'] == index['raw_query_indices']
    assert first['validation_fingerprint_cache'] == second['validation_fingerprint_cache'] == validate.fingerprint_receipt
    embeddings = model.encode_mol(torch.tensor(np.stack([validate.molecules.cache[i] for i in range(len(validate.molecules))])),
                                  normalize=True)
    for i, item in enumerate(validate.spectra):
        if not valid[i].any():
            assert second['ranks'][i] == 0
            continue
        spectrum_embedding = model.encode_spec(item['spec_mz'].unsqueeze(0), item['spec_intensity'].unsqueeze(0),
                                                item['spec_mask'].unsqueeze(0), normalize=True)[0]
        expected = torch.stack([spectrum_embedding @ embeddings[j] for j in index['candidate_indices'][i, valid[i]]])
        torch.testing.assert_close(second['scores'][i, valid[i]], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('change', [{'input_bits': 2047}, {'hidden_dim': 0}, {'emb_dim': True},
                                   {'dropout_rate': 1.}, {'dropout_rate': float('nan')}, {'norm_eps': 0}])
def test_invalid_fingerprint_model_settings_fail(change):
    settings = {'input_bits': 2048, 'hidden_dim': 16, 'emb_dim': 8, 'dropout_rate': 0., 'norm_eps': 1e-5}
    with pytest.raises(ValueError):
        FingerprintEncoder(**{**settings, **change})


def test_model_rejects_incomplete_configuration_or_wrong_molecule_input():
    model = small_model()
    for key_name in ('spec_config', 'molecule_config', 'alignment_config'):
        settings = model.construction_config()
        settings[key_name].pop(next(iter(settings[key_name])))
        with pytest.raises(ValueError, match='Incomplete'):
            FingerprintAlignmentModel(**settings)
    for value in (torch.zeros(2, 2047), torch.zeros(2, 2048, dtype=torch.long), torch.zeros(2048), object()):
        with pytest.raises(ValueError, match='configured fingerprint width'):
            model.encode_mol(value)


def test_complete_epoch_rejects_duplicate_queries_even_with_matching_total(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    train = DataLoader(dataset, batch_size=3, sampler=[0, 1, 1], collate_fn=candidate_fingerprint_collate_fn)
    val = DataLoader(dataset.base, batch_size=3, collate_fn=fingerprint_align_collate_fn)
    model = small_model()
    trainer = CandidateTrainerAlign(model, train, val, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=1.)
    with pytest.raises(RuntimeError, match='each original training query'):
        trainer.train_epoch(torch.optim.AdamW(model.parameters(), lr=.001), 1, 'synthetic')
    assert not trainer.candidate_epoch_audits
