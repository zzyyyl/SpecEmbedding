"""Synthetic query-binding and two-tower training checks; no formal model is dispatched."""

import copy
import hashlib
import json
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_adduct import (
    AdductAlignmentDataset,
    adduct_align_collate_fn,
    bind_candidate_adducts,
    validate_bound_adducts,
)
from SpecEmbedding.data.datasets_candidates import candidate_align_collate_fn
from SpecEmbedding.models_adduct import AdductConditionedEncoder
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils import adduct_metadata as module
from SpecEmbedding.utils.adduct_metadata import (
    SpectrumMetadata,
    audit_adduct_cache,
    build_split_metadata,
    load_adduct_cache,
    observed_adduct,
    prepare_adduct_cache,
    sequence_sha256,
)
from SpecEmbedding.utils.graph_fingerprint_validation import GraphFingerprintRetrievalValidator
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator, build_validation_index
from tests.test_adduct_conditioning import SETTINGS
from tests.test_candidate_alignment import candidate_dataset
from tests.test_graph_fingerprint_inputs import make_dataset
from tests.test_retrieval_validation import spectrum


def raw_queries():
    rows = [spectrum(smiles) for smiles in ('CCO', 'CC', 'CCO', 'CCC')]
    for i, (row, adduct) in enumerate(zip(rows, ('[M+H]+', '[M+Na]+', None, '[M+K]+'), strict=True)):
        row.set('identifier', f'q{i}')
        row.set('adduct', adduct)
    return rows


def cache_fixture(tmp_path, monkeypatch):
    data, root = tmp_path / 'data', tmp_path / 'metadata'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    for split in ('train', 'val'):
        with (data / f'{split}.pkl').open('wb') as handle:
            pickle.dump(raw_queries(), handle)
    monkeypatch.setattr(module, 'verify_dataset', lambda *args: {})
    options = ({'train': 4, 'val': 4, 'test': 2}, [1], {'max_len': 8, 'show_progress_bar': False}, SETTINGS)
    prepare_adduct_cache(data, root, *options)
    audit_adduct_cache(data, root, *options)
    return data, root, options


def test_raw_mapping_unknown_retention_and_cache_roundtrip(tmp_path, monkeypatch):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    train = load_adduct_cache(data, root, 'train', *options)
    val = load_adduct_cache(data, root, 'val', *options)
    assert train.raw_query_indices.tolist() == [0, 1, 2, 3]
    assert train.adduct_ids.tolist() == [0, 1, 2, 2]
    assert val.raw_query_indices.tolist() == [0, 2, 3] and val.adduct_ids.tolist() == [0, 2, 2]
    assert train.payload['unknown_values'] == {'null': 1, '"[M+K]+"': 1}
    assert train.payload['id_counts'] == {'0': 1, '1': 1, '2': 2}
    with pytest.raises(FileExistsError):
        prepare_adduct_cache(data, root, *options)
    with pytest.raises(ValueError, match='already exists'):
        audit_adduct_cache(data, root, *options)
    with pytest.raises(ValueError, match='train/val'):
        load_adduct_cache(data, root, 'test', *options)


@pytest.mark.parametrize('damage', ['payload', 'source', 'tokenizer', 'mapping', 'audit', 'manifest'])
def test_cache_rejects_stale_sources_construct_changes_and_partial_artifacts(tmp_path, monkeypatch, damage):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    options = list(copy.deepcopy(options))
    if damage == 'payload':
        (root / 'train.json').write_text('{}')
    elif damage == 'source':
        (data / 'train.pkl').write_bytes(b'changed')
    elif damage == 'tokenizer':
        options[2]['max_len'] += 1
    elif damage == 'mapping':
        options[3]['adducts'].reverse()
    elif damage == 'audit':
        (root / 'audit.json').unlink()
    else:
        (data / 'dataset_manifest.json').write_text('{"changed":true}')
    with pytest.raises((ValueError, FileNotFoundError)):
        load_adduct_cache(data, root, 'train', *options)


def test_metadata_audit_rejects_semantic_tampering_even_with_updated_payload_hash(tmp_path, monkeypatch):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    (root / 'audit.json').unlink()
    payload = json.loads((root / 'train.json').read_text())
    payload['rows'][0]['adduct_id'] = 1
    (root / 'train.json').write_text(json.dumps(payload))
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['files_sha256']['train.json'] = hashlib.sha256((root / 'train.json').read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fresh original'):
        audit_adduct_cache(data, root, *options)
    assert not (root / 'audit.json').exists()


def test_invalid_metadata_and_unregistered_observed_values():
    rows = raw_queries()
    with pytest.raises(ValueError, match='training queries'):
        build_split_metadata(rows, 'train', [0], {'max_len': 8}, SETTINGS)
    rows[1].set('identifier', 'q0')
    with pytest.raises(ValueError, match='duplicate'):
        build_split_metadata(rows, 'train', [], {'max_len': 8}, SETTINGS)
    assert observed_adduct(float('nan'), SETTINGS) == (None, 2)
    assert observed_adduct(' [M+H]+', SETTINGS) == (' [M+H]+', 2)
    with pytest.raises(ValueError, match='string'):
        observed_adduct(17, SETTINGS)


def test_cache_rechecks_files_after_reading_and_fresh_audit(tmp_path, monkeypatch):
    from pathlib import Path

    data, root, options = cache_fixture(tmp_path, monkeypatch)
    read_text = Path.read_text
    def changing_read(path, *args, **kwargs):
        result = read_text(path, *args, **kwargs)
        if path == root / 'train.json':
            path.write_text(result + ' ')
        return result
    with monkeypatch.context() as context:
        context.setattr(Path, 'read_text', changing_read)
        with pytest.raises(ValueError, match='changed while loading'):
            load_adduct_cache(data, root, 'train', *options)
    # Restore only this synthetic test fixture, then simulate a changed manifest during audit.
    payload = (root / 'train.json').read_text()
    (root / 'train.json').write_text(payload[:-1])
    (root / 'audit.json').unlink()
    fresh = module._fresh_splits
    def changing_fresh(*args):
        yield from fresh(*args)
        path = root / 'manifest.json'
        path.write_text(path.read_text() + ' ')
    monkeypatch.setattr(module, '_fresh_splits', changing_fresh)
    with pytest.raises(ValueError, match='changed during audit'):
        audit_adduct_cache(data, root, *options)
    assert not (root / 'audit.json').exists()


def dataset_metadata(dataset, split='train'):
    rows = [None] * len(dataset)
    for position, (identity, offset) in enumerate(dataset.base._spectrum_indices):
        raw = int(dataset.raw_query_indices[position])
        sequence = dataset.base._data[identity][offset]
        rows[raw] = {'raw_query_index': raw, 'identifier': f'q{raw}', 'identity_2d': identity,
                     'smiles': sequence['smiles'], 'sequence_sha256': sequence_sha256(sequence),
                     'adduct_id': [0, 1, 2][raw]}
    return SpectrumMetadata({'rows': rows, 'split': split, 'source_query_count': len(rows)},
                            {'source': {'dataset_manifest_sha256': 'a' * 64, 'settings': SETTINGS}, 'split': split})


@pytest.mark.parametrize('combined', [False, True])
def test_training_binding_preserves_augmentation_sampling_rng_and_grouped_query_order(tmp_path, monkeypatch, combined):
    dataset = make_dataset(tmp_path, monkeypatch, augment=True)[0] if combined else candidate_dataset(monkeypatch, augment=True)
    reference = copy.deepcopy(dataset)
    metadata = dataset_metadata(dataset)
    bind_candidate_adducts(dataset, metadata)
    dataset.set_epoch(2)
    reference.set_epoch(2)
    def draw(source):
        torch.manual_seed(31)
        np.random.seed(31)
        random.seed(31)
        examples = [source[i] for i in range(len(source))]
        return candidate_align_collate_fn(examples), torch.get_rng_state(), np.random.random(), random.random()
    actual, state, n, r = draw(dataset)
    old, old_state, old_n, old_r = draw(reference)
    assert actual.raw_query_indices.tolist() == [1, 0, 2]
    assert actual.adduct_ids.tolist() == [1, 0, 2]
    assert torch.equal(state, old_state) and n == old_n and r == old_r
    for field in ('raw_query_indices', 'candidate_indices', 'source_positions', 'negative_ptr'):
        assert torch.equal(getattr(actual, field), getattr(old, field))
    for a, b in zip(actual.anchor[:3], old.anchor[:3], strict=True):
        assert torch.equal(a, b)
    for name in actual.anchor[3].keys():
        assert torch.equal(actual.anchor[3][name], old.anchor[3][name])
    with pytest.raises(ValueError, match='already bound'):
        bind_candidate_adducts(dataset, metadata)
    broken = copy.deepcopy(reference)
    identity, offset = broken.base._spectrum_indices[0]
    broken.base._data[identity][offset]['mz'][0] += 1
    with pytest.raises(ValueError, match='token binding'):
        bind_candidate_adducts(broken, metadata)
    with pytest.raises(ValueError, match='binding mismatch'):
        validate_bound_adducts(metadata, actual.raw_query_indices, actual.adduct_ids.flip(0))


def test_spawn_workers_preserve_complete_metadata_across_epochs(tmp_path, monkeypatch):
    dataset = candidate_dataset(monkeypatch)
    bind_candidate_adducts(dataset, dataset_metadata(dataset))
    loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn, num_workers=2,
                        multiprocessing_context='spawn', generator=torch.Generator().manual_seed(1), timeout=30)
    for epoch in (1, 2):
        dataset.set_epoch(epoch)
        batches = list(loader)
        assert torch.cat([batch.raw_query_indices for batch in batches]).tolist() == [1, 0, 2]
        assert torch.cat([batch.adduct_ids for batch in batches]).tolist() == [1, 0, 2]


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('attention_pool', [False, True])
def test_two_tower_training_updates_conditioner_and_reencodes_full_validation(tmp_path, monkeypatch, combined, attention_pool):
    from SpecEmbedding.utils.formal_alignment import build_formal_alignment
    from tests.test_attention_pool_integration import definition
    from tests.test_fingerprint_alignment import make_cache

    dataset = make_dataset(tmp_path, monkeypatch, augment=True, structural=True)[0] if combined else candidate_dataset(monkeypatch, augment=True)
    metadata = dataset_metadata(dataset)
    bind_candidate_adducts(dataset, metadata)
    val_base = copy.deepcopy(dataset.base)
    val_base.is_augment = False
    val = AdductAlignmentDataset(val_base, dataset_metadata(dataset, 'val'))
    raw = raw_queries()[:3]
    tokenizer = {'max_len': 8, 'show_progress_bar': False}
    index = build_validation_index(raw, {s: ['CCO', 'CC', 'CCC'] for s in ('CCO', 'CC')}, [], {}, tokenizer)
    index['dataset_manifest_sha256'] = 'a' * 64
    retrieval_metadata = SpectrumMetadata(build_split_metadata(raw, 'val', [], tokenizer, SETTINGS),
        {'source': {'dataset_manifest_sha256': 'a' * 64, 'tokenizer_config': tokenizer, 'settings': SETTINGS}, 'split': 'val'})
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=(1, 5, 10, 20))
    directory = tmp_path / 'run/validation_retrieval'
    if combined:
        root = tmp_path / 'validation_bits'
        source = make_cache(root, index['mol_smiles'], index_sha='c' * 64)
        validate = GraphFingerprintRetrievalValidator(index, settings, directory, fingerprint_root=root,
            fingerprint_provenance=source, index_sha256='c' * 64, spectrum_metadata=retrieval_metadata)
    else:
        validate = AlignmentRetrievalValidator(index, settings, directory, spectrum_metadata=retrieval_metadata)
    model_config = definition('gine_fingerprint' if combined else 'gine', qk=True)
    if not attention_pool:
        model_config['spec_encoder'].pop('attention_pool')
    model = build_formal_alignment(model_config)
    model.spec_encoder = AdductConditionedEncoder(model.spec_encoder, SETTINGS)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn)
    val_loader = DataLoader(val, batch_size=2, collate_fn=adduct_align_collate_fn)
    trainer = CandidateTrainerAlign(model, train_loader, val_loader, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=2., retrieval_validator=validate)
    trainer.expected_epoch_counts = {'train': 3, 'val': 3}
    trainer.fit(2, torch.optim.AdamW(model.parameters(), lr=.001), stage_name='stage2')
    for prefix in ('spec_encoder.conditioning.', 'spec_encoder.encoder.', 'mol_encoder.', 'spec_proj.', 'mol_proj.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(prefix))
    assert all(torch.count_nonzero(model.spec_encoder.conditioning.affine[i]) > 0 for i in (0, 1))
    for record in trainer.candidate_epoch_audits:
        order = np.load(tmp_path / 'run/candidate_training' / record['query_order_file'])
        expected = hashlib.sha256(np.stack((order, metadata.adduct_ids[order]), axis=1).astype('<i8').tobytes()).hexdigest()
        assert record['observed_adduct_order_sha256'] == expected
    for epoch in (1, 2):
        path = directory / f'stage2_epoch{epoch:03d}.pt'
        with pytest.raises(ValueError):
            audit_snapshot(path, index)
        assert audit_snapshot(path, index, expected_spectrum_metadata=retrieval_metadata.provenance,
            expected_fingerprint_cache=getattr(validate, 'fingerprint_receipt', None))['queries'] == 3
    previous = torch.load(directory / 'stage2_epoch002.pt', weights_only=False)['scores']
    with torch.no_grad():
        model.spec_encoder.conditioning.affine[0, 1].add_(torch.linspace(-1, 1, model.spec_encoder.conditioning.embedding_dim))
    validate(model, torch.device('cpu'), 3, 'changed')
    updated = torch.load(directory / 'changed_epoch003.pt', weights_only=False)['scores']
    assert not torch.allclose(updated[0], previous[0])
    # Independent direct cosine calculation for every original candidate row.
    from torch_geometric.data import Batch
    molecules = Batch.from_data_list([validate.molecules[i]['graph'] for i in range(len(validate.molecules))])
    with torch.inference_mode():
        mol = model.encode_mol(molecules, normalize=True)
        for i in range(3):
            item = validate.spectra[i]
            spec = model.encode_spec(*(item[name].unsqueeze(0) for name in ('spec_mz', 'spec_intensity', 'spec_mask')),
                                     normalize=True, adduct_ids=torch.tensor([item['adduct_id']]))[0]
            torch.testing.assert_close(updated[i], mol[index['candidate_indices'][i]] @ spec, rtol=1e-5, atol=1e-6)
    # Missing/wrong metadata must fail even with a constructible component.
    dataset.set_epoch(3)
    broken = candidate_align_collate_fn([dataset[i] for i in range(3)])
    broken.adduct_ids = None
    with pytest.raises(ValueError, match='bound adduct batch'):
        trainer.training_batch_loss(broken)


def test_validation_index_and_control_binding_reject_mismatches(tmp_path, monkeypatch):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    metadata = load_adduct_cache(data, root, 'val', *options)
    raw = raw_queries()
    index = build_validation_index(raw, {s: [s] for s in ('CCO', 'CC', 'CCC')}, [1], {}, options[2])
    index['dataset_manifest_sha256'] = metadata.provenance['source']['dataset_manifest_sha256']
    assert metadata.bind_index(index).tolist() == [0, 2, 2]
    index['sequences'][0]['mz'][0] += 1
    with pytest.raises(ValueError, match='token binding'):
        metadata.bind_index(index)
    index['raw_query_indices'].reverse()
    with pytest.raises(ValueError, match='source/order'):
        metadata.bind_index(index)
