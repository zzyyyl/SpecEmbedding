import copy
import json

import pytest
import torch
import yaml
from rdkit import rdBase

import SpecEmbedding.utils.fingerprint_alignment_inputs as inputs_module
from SpecEmbedding.utils.fingerprint_cache import (
    audit_fingerprint_cache,
    build_fingerprint_cache,
    fingerprint_provenance,
)
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, read_formal_alignment_checkpoint
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.optimization_audit import audit_optimization_run
from tests.test_formal_alignment import model_config
from tests.test_optimization_audit import completed_run as completed_run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fingerprint_run(completed_run, monkeypatch):
    # Source verification has separate real-data coverage; here bind actual bit-cache files,
    # real model tensors and full synthetic trajectory through the completion auditor.
    run, index, selection = completed_run
    manifest_path = run / 'inputs_and_commands.json'
    manifest = json.loads(manifest_path.read_text())
    runtime = manifest['runtime_config']
    runtime['model'] = model_config('fingerprint')
    runtime['augmentation'].update(node_drop_rate=0., edge_mask_rate=0.)
    runtime['molecule_fingerprints'] = {'radius': 2, 'bits': 2048, 'workers': 1, 'chunk_size': 8}
    runtime['train']['align']['candidate_supervision'] = {'enabled': False, 'negative_count': 16,
                                                       'loss_weight': 1., 'pool_cache_size': 1, 'graph_cache_size': 1}
    selection.update(model_config=runtime['model'], training_config=copy.deepcopy(runtime['train']['align']),
                     config_snapshot=copy.deepcopy(runtime),
                     graph_policy='rdkit_sanitized', rdkit_version=rdBase.rdkitVersion)
    index['mol_smiles'] = ['C' * n for n in range(1, 26)]
    index_path = run / 'validation/mass_val_topk256.pt'
    torch.save(index, index_path)
    save(index_path.with_suffix('.json'), {'synthetic': True})
    training_index = run.parent / 'training_source/candidates.pkl'
    training_index.parent.mkdir()
    training_index.write_bytes(b'synthetic source inventory; source reader is mocked')
    for name in ('receipt.json', 'verification.json'):
        save(training_index.parent / name, {'synthetic': True})
    sources = {}
    for kind, path in (('train', training_index), ('validation', index_path)):
        source = {'kind': kind, 'index_path': str(path), 'sha256': sha256_file(path),
                  'dataset_manifest_sha256': index['dataset_manifest_sha256'],
                  'receipt_sha256': sha256_file(path.parent / 'receipt.json' if kind == 'train' else path.with_suffix('.json'))}
        if kind == 'train':
            source['verification_sha256'] = sha256_file(path.parent / 'verification.json')
        sources[kind] = source
    monkeypatch.setattr(inputs_module, 'fingerprint_source', lambda data, path, kind, **kwargs: (index['mol_smiles'], sources[kind]))
    manifest['fingerprint_inputs'] = {}
    for kind, path in (('train', training_index), ('validation', index_path)):
        root = run.parent / f'{kind}_bits'
        provenance = fingerprint_provenance(index['mol_smiles'], index_sha256=sha256_file(path),
                                            dataset_manifest_sha256=index['dataset_manifest_sha256'], radius=2, bits=2048)
        build_fingerprint_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=8)
        audit_fingerprint_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=8)
        _, receipt = inputs_module.load_alignment_fingerprints(
            run / 'data/MassSpecGym', path, kind, root, counts={}, exclusions=[], tokenizer_config={},
            settings=runtime['molecule_fingerprints'], pool_cache_size=1)
        manifest['fingerprint_inputs'][kind] = receipt
        manifest['inputs'].update(inputs_module.fingerprint_input_files(receipt, f'fingerprint_{kind}'))
        selection['training_fingerprint_cache' if kind == 'train' else 'validation_fingerprint_cache'] = receipt['cache']
    directory = run / 'alignment42_topk256'
    weights = build_formal_alignment(runtime['model']).state_dict()
    for path in (directory / 'best_model_stage2.pth', *directory.glob('candidate_stage2_epoch*.pth')):
        torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(directory / 'best_model_stage2.pth')
    selection['validation_index']['sha256'] = sha256_file(index_path)
    for path in (directory / 'validation_retrieval').glob('*.pt'):
        snapshot = torch.load(path, weights_only=False)
        snapshot['validation_fingerprint_cache'] = selection['validation_fingerprint_cache']
        torch.save(snapshot, path)
    baseline_path = run / 'baseline_validation/metrics.json'
    baseline = json.loads(baseline_path.read_text())
    baseline['index_sha256'] = sha256_file(index_path)
    checkpoint = run.parent / 'baseline.pth'
    parent_model = model_config('gine')
    torch.save(build_formal_alignment(parent_model).state_dict(), checkpoint)
    parent = copy.deepcopy(selection)
    parent.update(model_config=parent_model, checkpoint_sha256=sha256_file(checkpoint))
    parent['config_snapshot']['model'] = parent_model
    parent.pop('training_fingerprint_cache')
    parent.pop('validation_fingerprint_cache')
    save(checkpoint.parent / 'alignment_selection.json', parent)
    _, model_receipt = read_formal_alignment_checkpoint(checkpoint, dataset_outputs=index['dataset_outputs'],
        dataset_manifest_sha256=index['dataset_manifest_sha256'], tokenizer_config=runtime['data']['tokenizer'],
        expected_counts={'train': 7, 'val': 5}, exclusions=[2, 5])
    manifest['checkpoint_model'] = baseline['checkpoint_model'] = model_receipt
    manifest['inputs']['baseline_checkpoint']['sha256'] = baseline['checkpoint_sha256'] = sha256_file(checkpoint)
    save(baseline_path, baseline)
    (run / 'runtime_params.yaml').write_text(yaml.safe_dump(runtime))
    source = run.parent / 'source_params.yaml'
    source.write_text((run / 'runtime_params.yaml').read_text())
    manifest['source_config_sha256'] = sha256_file(source)
    save(manifest_path, manifest)
    save(directory / 'alignment_selection.json', selection)
    status = json.loads((run / 'status.json').read_text())
    status['runtime_config_sha256'] = sha256_file(run / 'runtime_params.yaml')
    status['stages'][1]['audit']['sha256'] = sha256_file(index_path)
    save(run / 'status.json', status)
    return run, manifest, selection


def test_complete_fingerprint_model_audit_binds_every_input_and_preserves_missing_queries(fingerprint_run):
    run, _, _ = fingerprint_run
    result = audit_optimization_run(run)
    assert result['fingerprint_inputs']['state'] == 'verified_complete_fingerprint_model_inputs'
    assert result['full_epoch_counts'] == {'train': 7, 'val': 5}
    assert result['selected_metrics']['top1'] == .4  # Missing-positive query stays in denominator.
    assert len([p for p in result['artifact_sha256'] if p.endswith('fingerprints.bin')]) == 2
    assert not result['test_evaluated_by_this_audit']


@pytest.mark.parametrize('damage', ['snapshot', 'selection', 'missing_pin', 'bits', 'baseline_model', 'checkpoint_shape'])
def test_fingerprint_completion_rejects_mixed_unpinned_or_corrupt_evidence(fingerprint_run, damage):
    run, manifest, selection = fingerprint_run
    directory = run / 'alignment42_topk256'
    if damage == 'snapshot':
        path = directory / 'validation_retrieval/stage2_epoch002.pt'
        snapshot = torch.load(path, weights_only=False)
        snapshot.pop('validation_fingerprint_cache')
        torch.save(snapshot, path)
    elif damage == 'selection':
        selection.pop('validation_fingerprint_cache')
        save(directory / 'alignment_selection.json', selection)
    elif damage == 'missing_pin':
        manifest['inputs'].pop('fingerprint_validation_bits')
        save(run / 'inputs_and_commands.json', manifest)
    elif damage == 'bits':
        path = run.parent / 'train_bits/fingerprints.bin'
        with path.open('ab') as handle:
            handle.write(b'changed')
    elif damage == 'baseline_model':
        path = run / 'baseline_validation/metrics.json'
        receipt = json.loads(path.read_text())
        receipt['checkpoint_model']['model_config']['align']['final_dim'] = 16
        save(path, receipt)
    else:
        for path in (directory / 'best_model_stage2.pth', *directory.glob('candidate_stage2_epoch*.pth')):
            weights = torch.load(path, weights_only=True)
            weights.pop('mol_encoder.layers.0.weight')
            torch.save(weights, path)
        selection['checkpoint_sha256'] = sha256_file(directory / 'best_model_stage2.pth')
        save(directory / 'alignment_selection.json', selection)
    messages = {
        'snapshot': 'Unexpected fingerprint cache in validation snapshot',
        'selection': 'Selected fingerprint model/input provenance mismatch',
        'missing_pin': 'Fingerprint source or cache file was not pinned in preflight',
        'bits': 'Input fingerprint mismatch',
        'baseline_model': 'Baseline model construction differs from preflight',
        'checkpoint_shape': 'Missing key',
    }
    with pytest.raises((ValueError, RuntimeError), match=messages[damage]):
        audit_optimization_run(run)
