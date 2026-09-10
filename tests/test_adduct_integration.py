"""Observed metadata through real synthetic training, strict reload and full completion gates."""

import copy
import json
import pickle
import shutil
from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
import yaml

from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils import adduct_metadata as metadata_io
from SpecEmbedding.utils import candidate_training as candidate_io
from SpecEmbedding.utils.adduct_alignment_inputs import (
    audit_model_spectrum_metadata,
    load_spectrum_metadata,
    spectrum_metadata_files,
)
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.graph_fingerprint_validation import GraphFingerprintRetrievalValidator
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator, build_validation_index
from tests.test_adduct_conditioning import SETTINGS, peaks
from tests.test_adduct_inputs import cache_fixture, raw_queries
from tests.test_candidate_alignment import pinned_synthetic_candidate_input
from tests.test_formal_alignment import checkpoint_fixture, model_config


def definition(combined=False, pool=False):
    result = model_config('gine_fingerprint' if combined else 'gine')
    result['spec_encoder'].update(adduct_conditioning=copy.deepcopy(SETTINGS), qk_norm={'eps': 1e-6})
    if pool:
        result['spec_encoder']['attention_pool'] = {'norm_eps': 1e-5}
    return result


@pytest.mark.parametrize('combined,pool', [(False, False), (False, True), (True, False), (True, True)])
def test_constructor_preserves_parent_parameters_rng_and_identity_output(combined, pool):
    candidate_config = definition(combined, pool)
    parent_config = copy.deepcopy(candidate_config)
    parent_config['spec_encoder'].pop('adduct_conditioning')
    torch.manual_seed(42)
    parent = build_formal_alignment(parent_config).eval()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = build_formal_alignment(candidate_config).eval()
    assert torch.equal(state, torch.get_rng_state())
    for name, value in parent.state_dict().items():
        target = name.replace('spec_encoder.', 'spec_encoder.encoder.', 1) if name.startswith('spec_encoder.') else name
        assert torch.equal(value, candidate.state_dict()[target])
    mz, intensity, mask, ids = peaks()
    with torch.inference_mode():
        torch.testing.assert_close(parent.encode_spec(mz, intensity, mask),
            candidate.encode_spec(mz, intensity, mask, adduct_ids=ids), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('damage', [None, 'weights', 'missing_receipt', 'mapping', 'raw_input', 'tokenizer', 'inactive'])
def test_strict_checkpoint_uses_own_configuration_and_metadata(tmp_path, monkeypatch, damage):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    _, receipts = load_spectrum_metadata(data, root, counts=options[0], exclusions=options[1],
        tokenizer_config=options[2], settings=SETTINGS)
    path, _, selection, protocol = checkpoint_fixture(tmp_path)
    settings = definition(pool=True)
    model = build_formal_alignment(settings)
    with torch.no_grad():
        model.spec_encoder.conditioning.affine[0, 1].fill_(.2)
    weights = model.state_dict()
    if damage == 'weights':
        del weights['spec_encoder.conditioning.affine']
    torch.save(weights, path)
    protocol.update(dataset_manifest_sha256=sha256_file(data / 'dataset_manifest.json'),
        tokenizer_config=options[2], expected_counts={'train': 4, 'val': 3}, exclusions=[1])
    selection.update(model_config=settings, data_path=str(data), spectrum_metadata=receipts,
                     exclude_val_query_indices=[1], checkpoint_sha256=sha256_file(path))
    selection['config_snapshot'].update(model=settings, data={'tokenizer': options[2]})
    selection['fulltrain_audit'].update(dataset_manifest_sha256=protocol['dataset_manifest_sha256'],
                                       expected_epoch_counts=protocol['expected_counts'])
    if damage == 'missing_receipt':
        selection.pop('spectrum_metadata')
    elif damage == 'mapping':
        selection['spectrum_metadata']['train']['source']['settings']['adducts'].reverse()
    elif damage == 'raw_input':
        (data / 'train.pkl').write_bytes(b'changed raw spectrum data')
    elif damage == 'tokenizer':
        selection['spectrum_metadata']['train']['source']['tokenizer_config']['max_len'] += 1
    elif damage == 'inactive':
        settings['spec_encoder'].pop('adduct_conditioning')
    (path.parent / 'alignment_selection.json').write_text(json.dumps(selection))
    monkeypatch.setattr(config, 'model', ConfigObject({'unrelated': 'candidate architecture'}))
    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            load_formal_alignment(path, torch.device('cpu'), **protocol)
    else:
        restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
        assert receipt['spectrum_metadata'] == receipts and receipt['model_config'] == settings
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in weights.items())


@pytest.mark.parametrize('damage', [None, 'selection', 'manifest', 'file_pin', 'token_order', 'disabled'])
def test_complete_metadata_audit_rejects_stale_or_unbound_inputs(tmp_path, monkeypatch, damage):
    data, root, options = cache_fixture(tmp_path, monkeypatch)
    raw = raw_queries()
    index = build_validation_index(raw, {s: [s] for s in ('CCO', 'CC', 'CCC')}, [1], {}, options[2])
    index['dataset_manifest_sha256'] = sha256_file(data / 'dataset_manifest.json')
    _, receipts = load_spectrum_metadata(data, root, counts=options[0], exclusions=options[1],
        tokenizer_config=options[2], settings=SETTINGS, index=index)
    runtime = {'model': definition(), 'fulltrain': {'expected_counts': options[0], 'exclude_val_query_indices': [1]},
               'data': {'tokenizer': options[2]}}
    manifest = {'runtime_config': runtime, 'spectrum_metadata': receipts, 'inputs': spectrum_metadata_files(receipts)}
    selection = {'model_config': copy.deepcopy(runtime['model']), 'spectrum_metadata': copy.deepcopy(receipts)}
    if damage == 'selection':
        selection.pop('spectrum_metadata')
    elif damage == 'manifest':
        manifest.pop('spectrum_metadata')
    elif damage == 'file_pin':
        manifest['inputs'].pop('spectrum_metadata_train.json')
    elif damage == 'token_order':
        index['sequences'].reverse()
    elif damage == 'disabled':
        runtime['model']['spec_encoder'].pop('adduct_conditioning')
        selection['model_config'] = copy.deepcopy(runtime['model'])
    if damage:
        with pytest.raises(ValueError):
            audit_model_spectrum_metadata(manifest, selection, data, index)
    else:
        metadata, observed, hashes = audit_model_spectrum_metadata(manifest, selection, data, index)
        assert observed == receipts and len(hashes) == 4
        assert metadata['val'].raw_query_indices.tolist() == [0, 2, 3]


def make_bits(root, smiles, index_sha, manifest_sha):
    from SpecEmbedding.utils.fingerprint_cache import (
        audit_fingerprint_cache,
        build_fingerprint_cache,
        fingerprint_provenance,
        load_fingerprint_cache,
    )
    source = fingerprint_provenance(smiles, index_sha256=index_sha, dataset_manifest_sha256=manifest_sha, radius=2, bits=2048)
    build_fingerprint_cache(smiles, root, source, workers=1, chunk_size=3)
    audit_fingerprint_cache(smiles, root, source, workers=1, chunk_size=3)
    return load_fingerprint_cache(smiles, root, source)[1]


@pytest.mark.parametrize('combined,pool,structural', [(False, False, False), (False, True, True),
                                                   (True, False, True), (True, True, False)])
def test_real_fresh_training_replays_adducts_and_strictly_reloads_every_input(tmp_path, monkeypatch, combined, pool, structural):
    import train_align as entry

    data, root = tmp_path / 'data', tmp_path / 'adducts'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    raw = raw_queries()[:3]  # Includes an explicitly missing adduct retained as unknown.
    for split in ('train', 'val'):
        (data / f'{split}.pkl').write_bytes(pickle.dumps(raw))
    counts, tokenizer = {'train': 3, 'val': 3, 'test': 2}, {'max_len': 4, 'show_progress_bar': False}
    monkeypatch.setattr(metadata_io, 'verify_dataset', lambda *args: {})
    metadata_io.prepare_adduct_cache(data, root, counts, [], tokenizer, SETTINGS)
    metadata_io.audit_adduct_cache(data, root, counts, [], tokenizer, SETTINGS)
    manifest_sha = sha256_file(data / 'dataset_manifest.json')
    dataset, candidate_settings, input_receipt, input_path = pinned_synthetic_candidate_input(
        monkeypatch, tmp_path, structural=structural, loss_weight=2., manifest_sha256=manifest_sha)
    # The existing candidate fixture supplies source pools; use actual freshly tokenized peaks here.
    tokenized = {}
    for row in raw:
        tokenized.setdefault(row.get('identity_2d'), []).append(Tokenizer(**tokenizer).tokenize(row))
    dataset.base._data = tokenized
    outputs = {f'{split}.pkl': sha256_file(data / f'{split}.pkl') for split in ('train', 'val')}
    index = build_validation_index(raw, {'CCO': ['CC', 'CCC'], 'CC': ['CC', 'CCO']}, [], {}, tokenizer)
    index.update(dataset_manifest_sha256=manifest_sha, dataset_outputs=outputs)
    index_path = tmp_path / 'val.pt'
    torch.save(index, index_path)
    metadata, receipts = load_spectrum_metadata(data, root, counts=counts, exclusions=[], tokenizer_config=tokenizer,
                                               settings=SETTINGS, index=index)
    definition_config = definition(combined, pool)
    monkeypatch.setattr(config, 'model', ConfigObject(definition_config))
    monkeypatch.setattr(config.data, 'tokenizer', ConfigObject(tokenizer))
    for key, value in {'candidate_supervision': ConfigObject(candidate_settings), 'epochs_stage2': 2,
                      'num_workers': 0, 'batching': 'mass_blocks', 'mass_block_size': 1, 'batch_size': 2}.items():
        monkeypatch.setattr(config.train.align, key, value)
    monkeypatch.setattr(config.augmentation, 'prob', 0.)
    def cpu_trainer(*args, **kwargs):
        kwargs['record_resources'] = False  # Synthetic CPU training only; formal CUDA gate remains unchanged.
        return CandidateTrainerAlign(*args, **kwargs)
    monkeypatch.setattr(entry, 'CandidateTrainerAlign', cpu_trainer)
    output = tmp_path / 'run/alignment42_topk256'
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=(1, 5, 10, 20))
    extra, selection_extra = {}, {}
    if combined:
        smiles = dataset.candidates.metadata['mol_smiles']
        train_cache = make_bits(tmp_path / 'train_bits', smiles, dataset.candidates.provenance['sha256'], manifest_sha)
        val_cache = make_bits(tmp_path / 'val_bits', index['mol_smiles'], sha256_file(index_path), manifest_sha)
        validator = GraphFingerprintRetrievalValidator(index, settings, output / 'validation_retrieval',
            fingerprint_root=val_cache['directory'], fingerprint_provenance=val_cache['provenance'],
            index_sha256=sha256_file(index_path), spectrum_metadata=metadata['val'])
        extra = {'fingerprint_inputs': {'train': {'cache': train_cache}, 'validation': {'cache': val_cache}},
                 'fingerprint_smiles': {'train': smiles, 'validation': index['mol_smiles']}}
        selection_extra = {'training_fingerprint_cache': train_cache, 'validation_fingerprint_cache': val_cache}
    else:
        validator = AlignmentRetrievalValidator(index, settings, output / 'validation_retrieval', spectrum_metadata=metadata['val'])
    selection_input = {'data_path': str(data), 'seed': 42, 'exclude_val_query_indices': [], **selection_extra,
        'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5', 'dataset_manifest_sha256': manifest_sha,
                           'expected_epoch_counts': {'train': 3, 'val': 3}, 'input_outputs': outputs}}
    entry.train_align(tokenized, sorted(tokenized), tokenized, sorted(tokenized), None, batch_size=2, lr=.001,
        save_dir=str(output), device='cpu', seed=42, formal_fulltrain=True, retrieval_validator=validator,
        training_candidates=dataset.candidates, candidate_input_receipt=input_receipt, selection_metadata=selection_input,
        spectrum_metadata=metadata, **extra)
    selection = json.loads((output / 'alignment_selection.json').read_text())
    assert selection['spectrum_metadata'] == receipts
    assert all(row['train'] == row['val'] == row['batching']['unique_queries'] == 3 for row in selection['fulltrain_audit']['epochs'])
    weights = torch.load(output / 'best_model_stage2.pth', weights_only=True)
    assert weights['spec_encoder.conditioning.affine'].abs().sum() > 0
    restored, _, receipt = load_formal_alignment(output / 'best_model_stage2.pth', torch.device('cpu'),
        dataset_outputs=outputs, dataset_manifest_sha256=manifest_sha, tokenizer_config=tokenizer,
        expected_counts={'train': 3, 'val': 3}, exclusions=[])
    assert receipt['spectrum_metadata'] == receipts
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in weights.items())
    replay_options = {'spectrum_metadata': metadata['train'],
                      **({'fingerprint_cache': train_cache, 'graph_fingerprint': True} if combined else {})}
    stage = selection['stages']['stage2']
    report, _ = candidate_io.audit_candidate_training(output, stage, input_path, candidate_settings,
                                                     42, 2, data, counts, [], **replay_options)
    assert report['negative_sampling_replayed'] and report['epochs'] == 2
    for epoch in (1, 2):
        snapshot = output / 'validation_retrieval' / f'stage2_epoch{epoch:03d}.pt'
        metrics = audit_snapshot(snapshot, index, expected_spectrum_metadata=receipts['val'],
                                  expected_fingerprint_cache=val_cache if combined else None)
        assert metrics['queries'] == 3 and metrics['positive_queries'] == 1
    if not combined:
        verify_complete_gine_fixture(monkeypatch, output, data, index, selection, input_path, input_receipt, receipts, counts)
    changed = copy.deepcopy(stage)
    changed['candidate_training']['epochs'][0]['observed_adduct_order_sha256'] = '0' * 64
    (output / 'candidate_training/stage2_epoch001.json').write_text(json.dumps(changed['candidate_training']['epochs'][0]))
    with pytest.raises(ValueError, match='Observed adduct order'):
        candidate_io.audit_candidate_training(output, changed, input_path, candidate_settings,
                                              42, 2, data, counts, [], **replay_options)


def verify_complete_gine_fixture(monkeypatch, directory, data, index, selection, input_path, input_receipt, receipts, counts):
    """The full completion auditor sees actual synthetic weights, scores and candidate/adduct replays."""
    from SpecEmbedding.utils import optimization_audit as audit
    from SpecEmbedding.utils.formal_alignment import read_formal_alignment_checkpoint

    run = directory.parent
    shutil.copytree(data, run / 'data/MassSpecGym')
    index_path = run / 'validation/mass_val_topk256.pt'
    index_path.parent.mkdir()
    torch.save(index, index_path)
    monkeypatch.setattr(audit, 'load_validation_index', lambda *args: index)  # Official source reader has independent coverage.
    runtime = copy.deepcopy(selection['config_snapshot'])
    runtime['fulltrain']['expected_counts'] = counts
    runtime['fulltrain']['exclude_val_query_indices'] = []
    parent_config = copy.deepcopy(runtime['model'])
    parent_config['spec_encoder'].pop('adduct_conditioning')
    parent = build_formal_alignment(parent_config).eval()
    parent_dir = run / 'synthetic_parent'
    parent_dir.mkdir()
    checkpoint = parent_dir / 'best_model_stage2.pth'
    torch.save(parent.state_dict(), checkpoint)
    parent_selection = copy.deepcopy(selection)
    parent_selection.pop('spectrum_metadata')
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
                     candidate_training_input={'path': str(new_input), 'sha256': sha256_file(new_input)},
                     config_snapshot=runtime, device='cuda:0')  # Synthetic completion declaration; no GPU work occurred.
    source = run / 'source.yaml'
    source.write_text(yaml.safe_dump(runtime))
    (run / 'runtime_params.yaml').write_text(source.read_text())
    manifest = {'runtime_config': runtime, 'source_config': str(source), 'source_config_sha256': sha256_file(source),
        'git_commit': 'synthetic-adduct-integration', 'runtime_versions': {'torch': version('torch')}, 'device': 'cuda:0',
        'candidate_training_input': input_receipt, 'spectrum_metadata': receipts, 'checkpoint_model': parent_receipt,
        'inputs': {**candidate_io.candidate_source_inputs(input_receipt), **spectrum_metadata_files(receipts),
                   'baseline_checkpoint': {'path': str(checkpoint), 'sha256': sha256_file(checkpoint)}}}
    status = {'state': 'complete', 'runtime_config_sha256': sha256_file(run / 'runtime_params.yaml'),
        'candidate_training_input_sha256': sha256_file(new_input),
        'stages': [{'name': name, 'state': 'complete', 'audit': {'sha256': sha256_file(index_path)}}
                   for name in ('import_v15', 'prepare_validation', 'baseline_validation', 'alignment42')]}
    for path, value in ((run / 'inputs_and_commands.json', manifest), (run / 'status.json', status),
                        (directory / 'alignment_selection.json', selection)):
        path.write_text(json.dumps(value))
    report = audit.audit_optimization_run(run)
    assert report['state'] == 'complete_validation_audit' and report['full_epoch_counts'] == {'train': 3, 'val': 3}
    assert report['spectrum_metadata'] == receipts and report['candidate_training']['negative_sampling_replayed']
    assert not report['test_evaluated_by_this_audit']
    for damage in ('unbound', 'wrong_snapshot', 'missing_weights'):
        with monkeypatch.context():
            if damage == 'unbound':
                original = (run / 'inputs_and_commands.json').read_bytes()
                changed = copy.deepcopy(manifest)
                changed['inputs'].pop('spectrum_metadata_train.json')
                (run / 'inputs_and_commands.json').write_text(json.dumps(changed))
                target = run / 'inputs_and_commands.json'
            elif damage == 'wrong_snapshot':
                target = directory / 'validation_retrieval/stage2_epoch001.pt'
                original = target.read_bytes()
                changed = torch.load(target, weights_only=False)
                changed.pop('spectrum_metadata')
                torch.save(changed, target)
            else:
                target = directory / 'best_model_stage2.pth'
                original = target.read_bytes()
                changed = torch.load(target, weights_only=True)
                changed.pop('spec_encoder.conditioning.affine')
                torch.save(changed, target)
            with pytest.raises((ValueError, RuntimeError)):
                audit.audit_optimization_run(run)
            target.write_bytes(original)


@pytest.mark.parametrize('parent_has_metadata', [False, True])
def test_stage_commands_keep_candidate_and_parent_metadata_separate(tmp_path, parent_has_metadata):
    import run_massspecgym_v15 as runner
    args = SimpleNamespace(output_root=tmp_path / 'run', source_dir=tmp_path / 'source', legacy_tsv=tmp_path / 'v1.tsv',
        prepared_data=tmp_path / 'data', prepared_validation_index=tmp_path / 'val.pt', gpus=[0, 1], device='cuda:0',
        optimize_alignment=True, baseline_checkpoint=tmp_path / 'parent.pth', molecule_input='gine',
        alignment_training_candidates=tmp_path / 'train.pkl', checkpoint_model_config=True,
        spectrum_metadata_cache=tmp_path / 'candidate_metadata',
        baseline_spectrum_metadata_cache=tmp_path / 'parent_metadata' if parent_has_metadata else None)
    stages = runner.commands(args)
    baseline, training = stages[2]['command'], stages[3]['command']
    assert training[training.index('--spectrum-metadata-cache') + 1] == str(args.spectrum_metadata_cache)
    assert ('--spectrum-metadata-cache' in baseline) == parent_has_metadata
    if parent_has_metadata:
        assert baseline[baseline.index('--spectrum-metadata-cache') + 1] == str(args.baseline_spectrum_metadata_cache)
    assert '--formal-fulltrain' in training and '--checkpoint-model-config' in baseline


@pytest.mark.parametrize('missing', ['cache', 'independent', 'candidate_inputs'])
def test_preflight_rejects_unbound_adduct_mode_before_reading_data(tmp_path, monkeypatch, missing):
    import run_massspecgym_v15 as runner
    monkeypatch.setattr(config, 'model', ConfigObject(definition()))
    monkeypatch.setattr(runner, 'storage_receipt', lambda *args: {'synthetic': True})
    args = SimpleNamespace(output_root=tmp_path, checkpoint_model_config=missing != 'independent', molecule_input='gine',
        spectrum_metadata_cache=None if missing == 'cache' else tmp_path / 'cache', optimize_alignment=True,
        prepared_data=tmp_path / 'data', prepared_validation_index=tmp_path / 'val.pt',
        alignment_training_candidates=None if missing == 'candidate_inputs' else tmp_path / 'train.pkl')
    with pytest.raises(ValueError, match='[Aa]dduct|independent baseline'):
        runner.preflight(args)


@pytest.mark.parametrize('combined,pool', [(False, False), (False, True), (True, False), (True, True)])
def test_successor_only_adds_conditioning_and_keeps_parent_science(combined, pool):
    from SpecEmbedding.utils.alignment_successor import adduct_successor_configuration
    parent = definition(combined, pool)
    parent['spec_encoder'].pop('adduct_conditioning')
    runtime = {'model': parent, 'train': {'align': {'batch_size': 128, 'lr': 1e-4,
        'candidate_supervision': {'enabled': True, 'negative_count': 16, 'loss_weight': 2., 'pool_cache_size': 1,
                                'graph_cache_size': 1}}}, 'augmentation': {'prob': .5},
               'data': {'cache_path': '/old/cache', 'tokenizer': {'max_len': 100}}, 'fulltrain': {}}
    original = copy.deepcopy(runtime)
    selection = {'model_config': parent, 'training_config': runtime['train']['align'],
                 'config_snapshot': {'augmentation': runtime['augmentation']}}
    storage = {'storage': {'root': '/tmp/approved'}, 'data': {'cache_path': '/tmp/approved/train_cache'}}
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    result = adduct_successor_configuration(runtime, selection, SETTINGS, gpu, storage_template=storage)
    expected = copy.deepcopy(runtime)
    expected['model']['spec_encoder']['adduct_conditioning'] = SETTINGS
    expected.update(storage=storage['storage'], fulltrain=gpu)
    expected['data']['cache_path'] = storage['data']['cache_path']
    assert result == expected and runtime == original
