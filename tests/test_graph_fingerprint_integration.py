"""Formal orchestration/audit checks with complete synthetic inventories, never real GPU trials."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from SpecEmbedding.utils.fingerprint_alignment_inputs import fingerprint_input_files
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, read_formal_alignment_checkpoint
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.molecule_graph_cache import audit_graph_cache, build_graph_cache, graph_cache_provenance
from SpecEmbedding.utils.optimization_audit import audit_optimization_run
from SpecEmbedding.utils.retrieval_validation import load_validation_graph_cache
from tests.test_fingerprint_integration import fingerprint_run as fingerprint_run
from tests.test_fingerprint_integration import save
from tests.test_formal_alignment import model_config
from tests.test_optimization_audit import completed_run as completed_run


@pytest.fixture
def combined_run(fingerprint_run, request):
    run, manifest, selection = fingerprint_run
    parent_kind = getattr(request, 'param', 'gine')
    index_path = run / 'validation/mass_val_topk256.pt'
    index = torch.load(index_path, weights_only=False)
    graph_root = run.parent / 'combined_graphs'
    provenance = graph_cache_provenance(index['mol_smiles'], index_sha256=sha256_file(index_path),
                                        dataset_manifest_sha256=index['dataset_manifest_sha256'])
    build_graph_cache(index['mol_smiles'], graph_root, provenance, workers=1, chunk_size=8)
    audit_graph_cache(index['mol_smiles'], graph_root, provenance, workers=1, chunk_size=8)
    _, graph = load_validation_graph_cache(index_path, index, graph_root)
    manifest['validation_graph_cache'] = graph
    files = {entry['filename']: entry['sha256'] for entry in graph['files'].values()}
    files.update({'manifest.json': graph['manifest_sha256'], 'audit.json': graph['audit_sha256']})
    for name, digest in files.items():
        manifest['inputs'][f'validation_graph_{name}'] = {'path': str(graph_root / name), 'sha256': digest}
    runtime = manifest['runtime_config']
    runtime['model'] = model_config('gine_fingerprint')
    runtime['augmentation'].update(node_drop_rate=.1, edge_mask_rate=.1)
    selection.update(model_config=runtime['model'], validation_graph_cache=graph, config_snapshot=copy.deepcopy(runtime))
    directory = run / 'alignment42_topk256'
    weights = build_formal_alignment(runtime['model']).state_dict()
    for path in (directory / 'best_model_stage2.pth', *directory.glob('candidate_stage2_epoch*.pth')):
        torch.save(weights, path)
    selection['checkpoint_sha256'] = sha256_file(directory / 'best_model_stage2.pth')
    graph_identity = {key: graph[key] for key in ('directory', 'manifest_sha256', 'audit_sha256')}
    for path in (directory / 'validation_retrieval').glob('*.pt'):
        snapshot = torch.load(path, weights_only=False)
        snapshot['validation_graph_cache'] = graph_identity
        torch.save(snapshot, path)
    # Model-specific parent inputs remain independent of candidate runtime/model settings.
    checkpoint = run.parent / 'baseline.pth'
    parent = copy.deepcopy(selection)
    parent['model_config'] = model_config(parent_kind)
    parent['config_snapshot']['model'] = parent['model_config']
    parent_graph = None if parent_kind == 'fingerprint' else graph
    parent['validation_graph_cache'] = parent_graph
    if parent_kind == 'gine':
        parent.pop('training_fingerprint_cache')
        parent.pop('validation_fingerprint_cache')
    torch.save(build_formal_alignment(parent['model_config']).state_dict(), checkpoint)
    parent['checkpoint_sha256'] = sha256_file(checkpoint)
    save(checkpoint.parent / 'alignment_selection.json', parent)
    _, parent_receipt = read_formal_alignment_checkpoint(
        checkpoint, dataset_outputs=index['dataset_outputs'], dataset_manifest_sha256=index['dataset_manifest_sha256'],
        tokenizer_config=runtime['data']['tokenizer'], expected_counts={'train': 7, 'val': 5}, exclusions=[2, 5])
    manifest['checkpoint_model'] = parent_receipt
    manifest['inputs']['baseline_checkpoint']['sha256'] = parent['checkpoint_sha256']
    baseline_path = run / 'baseline_validation/metrics.json'
    baseline = json.loads(baseline_path.read_text())
    baseline.update(checkpoint_model=parent_receipt, checkpoint_sha256=parent['checkpoint_sha256'],
                     validation_graph_cache=parent_graph)
    baseline_snapshot_path = baseline_path.parent / 'baseline_epoch000.pt'
    baseline_snapshot = torch.load(baseline_snapshot_path, weights_only=False)
    baseline_snapshot['validation_graph_cache'] = graph_identity if parent_graph else None
    if parent_kind != 'gine':
        fixed = manifest['fingerprint_inputs']['validation']
        manifest['baseline_fingerprint_input'] = fixed
        manifest['inputs'].update(fingerprint_input_files(fixed, 'baseline_fingerprint'))
        baseline['validation_fingerprint_cache'] = fixed['cache']
        baseline_snapshot['validation_fingerprint_cache'] = fixed['cache']
    torch.save(baseline_snapshot, baseline_snapshot_path)
    save(baseline_path, baseline)
    (run / 'runtime_params.yaml').write_text(yaml.safe_dump(runtime))
    source = run.parent / 'source_params.yaml'
    source.write_text((run / 'runtime_params.yaml').read_text())
    manifest['source_config_sha256'] = sha256_file(source)
    save(run / 'inputs_and_commands.json', manifest)
    save(directory / 'alignment_selection.json', selection)
    status = json.loads((run / 'status.json').read_text())
    status['runtime_config_sha256'] = sha256_file(run / 'runtime_params.yaml')
    save(run / 'status.json', status)
    return run, manifest, selection


@pytest.mark.parametrize('combined_run', ['gine', 'fingerprint', 'gine_fingerprint'], indirect=True)
def test_complete_combined_audit_uses_each_parents_own_inputs_and_keeps_missing_queries(combined_run):
    run, manifest, _ = combined_run
    report = audit_optimization_run(run)
    assert report['state'] == 'complete_validation_audit'
    assert report['full_epoch_counts'] == {'train': 7, 'val': 5}
    assert report['selected_metrics']['top1'] == .4
    assert report['validation_graph_cache'] == manifest['validation_graph_cache']
    assert report['fingerprint_inputs']['state'] == 'verified_complete_fingerprint_model_inputs'
    assert any(path.endswith('x.bin') for path in report['artifact_sha256'])
    assert len([path for path in report['artifact_sha256'] if path.endswith('fingerprints.bin')]) == 2
    assert not report['test_evaluated_by_this_audit']


@pytest.mark.parametrize('damage', ['selection_graph', 'selection_bits', 'snapshot_graph', 'snapshot_bits',
                                   'unbound_graph', 'unbound_bits', 'input_width', 'missing_branch'])
def test_combined_completion_rejects_partial_or_mixed_model_evidence(combined_run, damage):
    run, manifest, selection = combined_run
    directory = run / 'alignment42_topk256'
    if damage.startswith('selection_'):
        selection['validation_graph_cache' if damage.endswith('graph') else 'validation_fingerprint_cache'] = None
    elif damage.startswith('snapshot_'):
        path = directory / 'validation_retrieval/stage2_epoch002.pt'
        snapshot = torch.load(path, weights_only=False)
        snapshot.pop('validation_graph_cache' if damage.endswith('graph') else 'validation_fingerprint_cache')
        torch.save(snapshot, path)
    elif damage.startswith('unbound_'):
        manifest['inputs'].pop('validation_graph_x.bin' if damage.endswith('graph') else 'fingerprint_validation_bits')
        save(run / 'inputs_and_commands.json', manifest)
    elif damage == 'input_width':
        selection['model_config']['fingerprint_residual']['input_bits'] = 1024
    else:
        path = directory / 'best_model_stage2.pth'
        weights = torch.load(path, weights_only=True)
        del weights['mol_encoder.fingerprint_branch.3.weight']
        torch.save(weights, path)
        selection['checkpoint_sha256'] = sha256_file(path)
    save(directory / 'alignment_selection.json', selection)
    with pytest.raises((ValueError, RuntimeError)):
        audit_optimization_run(run)


@pytest.mark.parametrize('parent_kind', ['gine', 'fingerprint', 'gine_fingerprint'])
def test_combined_commands_route_graphs_by_parent_type_and_preserve_all_fingerprint_paths(tmp_path, parent_kind):
    import run_massspecgym_v15 as runner

    args = SimpleNamespace(output_root=tmp_path / 'run', source_dir=tmp_path / 'source', legacy_tsv=tmp_path / 'v1.tsv',
        prepared_data=tmp_path / 'data', prepared_validation_index=tmp_path / 'val.pt', gpus=[0, 1], device='cuda:0',
        optimize_alignment=True, baseline_checkpoint=tmp_path / 'parent.pth', molecule_input='gine_fingerprint',
        validation_graph_cache=tmp_path / 'graphs', fingerprint_training_index=tmp_path / 'train.pkl',
        training_fingerprint_cache=tmp_path / 'train_bits', validation_fingerprint_cache=tmp_path / 'val_bits',
        baseline_fingerprint_cache=None if parent_kind == 'gine' else tmp_path / 'parent_bits')
    stages = runner.commands(args, baseline_model_type=parent_kind)
    assert [stage['name'] for stage in stages] == ['import_v15', 'import_validation', 'baseline_validation', 'alignment42']
    baseline, train = stages[2]['command'], stages[3]['command']
    assert ('--graph-cache' in baseline) == (parent_kind != 'fingerprint')
    assert ('--fingerprint-cache' in baseline) == (parent_kind != 'gine')
    assert '--checkpoint-model-config' in baseline
    assert train[train.index('--validation-graph-cache') + 1] == str(args.validation_graph_cache)
    for field in ('fingerprint_training_index', 'training_fingerprint_cache', 'validation_fingerprint_cache'):
        assert train[train.index('--' + field.replace('_', '-')) + 1] == str(getattr(args, field))
    assert '--formal-fulltrain' in train and '--limit' not in train and '--max-train-queries' not in train
