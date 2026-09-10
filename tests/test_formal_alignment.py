import copy
import json

import pytest
import torch
from rdkit import rdBase

from SpecEmbedding.config import ConfigObject
from SpecEmbedding.utils.formal_alignment import (
    build_formal_alignment,
    formal_model_type,
    load_formal_alignment,
    read_formal_alignment_checkpoint,
)
from SpecEmbedding.utils.fulltrain import sha256_file


def model_config(kind='gine'):
    molecule = ({'emb_dim': 8, 'n_layers': 1, 'dropout_rate': 0., 'size_feature_dim': 4,
                 'norm_type': 'layernorm', 'norm_eps': 1e-5} if kind != 'fingerprint'
                else {'input_bits': 2048, 'hidden_dim': 16, 'emb_dim': 8, 'dropout_rate': 0., 'norm_eps': 1e-5})
    result = {'type': kind,
            'spec_encoder': {'embedding_dim': 8, 'n_head': 2, 'n_layer': 1, 'dim_feedward': 8,
                             'dim_target': 8, 'feedward_activation': 'selu'},
            'mol_encoder': {**molecule, 'graph_policy': 'rdkit_sanitized'},
            'align': {'final_dim': 8, 'dropout_rate': 0., 'tau': .2}}
    if kind == 'gine_fingerprint':
        result['fingerprint_residual'] = {'input_bits': 2048, 'hidden_dim': 12, 'norm_eps': 1e-5}
    return result


def checkpoint_fixture(tmp_path, kind='gine'):
    path = tmp_path / 'best_model_stage2.pth'
    definition = model_config(kind)
    model = build_formal_alignment(definition).eval()
    torch.save(model.state_dict(), path)
    protocol = {'dataset_outputs': {'train': 'synthetic-train'}, 'dataset_manifest_sha256': 'a' * 64,
                'tokenizer_config': {'max_len': 4, 'show_progress_bar': False},
                'expected_counts': {'train': 3, 'val': 4}, 'exclusions': []}
    selection = {'checkpoint_sha256': sha256_file(path), 'seed': 42, 'graph_policy': 'rdkit_sanitized',
                 'rdkit_version': rdBase.rdkitVersion, 'model_config': definition,
                 'config_snapshot': {'model': definition, 'data': {'tokenizer': protocol['tokenizer_config']}},
                 'exclude_val_query_indices': [],
                 'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5',
                                     'input_outputs': protocol['dataset_outputs'],
                                     'dataset_manifest_sha256': protocol['dataset_manifest_sha256'],
                                     'expected_epoch_counts': protocol['expected_counts']}}
    if kind in ('fingerprint', 'gine_fingerprint'):
        selection.update(training_fingerprint_cache={'synthetic': 'not a formal input'},
                         validation_fingerprint_cache={'synthetic': 'not a formal input'})
    (tmp_path / 'alignment_selection.json').write_text(json.dumps(selection))
    return path, model, selection, protocol


@pytest.mark.parametrize('kind', ['gine', 'fingerprint', 'gine_fingerprint'])
def test_saved_model_reconstructs_without_candidate_configuration(tmp_path, monkeypatch, kind):
    path, original, _, protocol = checkpoint_fixture(tmp_path, kind)
    import SpecEmbedding.config as global_configuration
    # An unrelated candidate model is deliberately incompatible in shape and architecture.
    monkeypatch.setattr(global_configuration, 'config', ConfigObject({'model': {'different': 'candidate'}}))
    restored, _, receipt = load_formal_alignment(path, torch.device('cpu'), **protocol)
    assert receipt['model_type'] == kind
    assert receipt['checkpoint_sha256'] == sha256_file(path)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in original.state_dict().items())
    if kind == 'fingerprint':
        inputs = torch.randint(0, 2, (3, 2048)).float()
        assert torch.equal(original.encode_mol(inputs, True), restored.encode_mol(inputs, True))


@pytest.mark.parametrize('field', ['dataset_outputs', 'dataset_manifest_sha256', 'tokenizer_config',
                                  'expected_counts', 'exclusions'])
def test_architecture_independence_never_relaxes_shared_protocol(tmp_path, field):
    path, _, _, protocol = checkpoint_fixture(tmp_path)
    protocol[field] = 'changed'
    with pytest.raises(ValueError, match='shared data/tokenizer/exclusion'):
        read_formal_alignment_checkpoint(path, **protocol)


@pytest.mark.parametrize('damage', ['weights', 'model_type', 'missing_key', 'wrong_shape', 'snapshot', 'version'])
def test_corrupt_or_incomplete_checkpoint_rejected(tmp_path, damage):
    path, _, selection, protocol = checkpoint_fixture(tmp_path)
    if damage == 'weights':
        with path.open('ab') as handle:
            handle.write(b'changed')
    elif damage == 'model_type':
        selection['model_config']['type'] = 'unrecognized'
    elif damage == 'missing_key':
        weights = torch.load(path, weights_only=True)
        del weights['logit_scale']
        torch.save(weights, path)
        selection['checkpoint_sha256'] = sha256_file(path)
    elif damage == 'wrong_shape':
        selection['model_config']['align']['final_dim'] = 16
    elif damage == 'snapshot':
        selection['config_snapshot']['model'] = model_config('fingerprint')
    else:
        selection['rdkit_version'] = 'different'
    (tmp_path / 'alignment_selection.json').write_text(json.dumps(selection))
    with pytest.raises((ValueError, RuntimeError)):
        load_formal_alignment(path, torch.device('cpu'), **protocol)


def test_historical_gine_tag_is_explicitly_recognized_but_fingerprint_requires_provenance(tmp_path):
    definition = model_config()
    del definition['type']
    assert formal_model_type(definition) == 'gine'
    path, _, selection, protocol = checkpoint_fixture(tmp_path, 'fingerprint')
    del selection['training_fingerprint_cache']
    (tmp_path / 'alignment_selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='fixed input provenance'):
        read_formal_alignment_checkpoint(path, **protocol)


@pytest.mark.parametrize('kind', ['gine', 'fingerprint', 'gine_fingerprint'])
def test_baseline_cli_uses_own_configuration_and_binds_receipt(tmp_path, monkeypatch, kind):
    import alignment_validation as entry
    from tests.test_retrieval_validation import build_index

    path, original, selection, protocol = checkpoint_fixture(tmp_path, kind)
    index = build_index()
    index.update(dataset_outputs=protocol['dataset_outputs'], dataset_manifest_sha256=protocol['dataset_manifest_sha256'])
    settings = copy.deepcopy(entry.config.to_dict())
    # The candidate's projection dimension differs; using it to load the baseline would fail.
    settings['model'] = model_config()
    settings['model']['align']['final_dim'] = 16
    settings['data']['tokenizer'] = protocol['tokenizer_config']
    settings['fulltrain']['expected_counts'] = {'train': 3, 'val': 4, 'test': 2}
    settings['fulltrain']['exclude_val_query_indices'] = []
    settings['retrieval_validation'].update(mol_batch_size=2, spec_batch_size=2, num_workers=0)
    monkeypatch.setattr(entry, 'config', ConfigObject(settings))
    monkeypatch.setattr(entry, 'resolve_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(entry, 'load_validation_index', lambda *args: index)
    monkeypatch.setenv('SPECEMBEDDING_REQUIRE_CUDA', '1')
    index_path = tmp_path / 'index.pt'
    torch.save(index, index_path)
    extra = []
    graph_cache = None
    if kind in ('fingerprint', 'gine_fingerprint'):
        from SpecEmbedding.utils.fingerprint_cache import load_fingerprint_cache
        from tests.test_fingerprint_alignment import make_cache
        root = tmp_path / 'bits'
        provenance = make_cache(root, index['mol_smiles'], index_sha=sha256_file(index_path))
        _, cache = load_fingerprint_cache(index['mol_smiles'], root, provenance)
        selection['validation_fingerprint_cache'] = cache
        (tmp_path / 'alignment_selection.json').write_text(json.dumps(selection))
        # Only source inventory is synthetic; the actual validator reads and encodes audited bits.
        monkeypatch.setattr(entry, 'load_alignment_fingerprints', lambda *a, **k: (index['mol_smiles'], {'cache': cache}))
        extra = ['--fingerprint-cache', str(root)]
    if kind == 'gine_fingerprint':
        from SpecEmbedding.utils.molecule_graph_cache import (
            audit_graph_cache,
            build_graph_cache,
            graph_cache_provenance,
        )
        from SpecEmbedding.utils.retrieval_validation import load_validation_graph_cache
        graph_root = tmp_path / 'graphs'
        source = graph_cache_provenance(index['mol_smiles'], index_sha256=sha256_file(index_path),
                                        dataset_manifest_sha256=protocol['dataset_manifest_sha256'])
        build_graph_cache(index['mol_smiles'], graph_root, source, workers=1, chunk_size=2)
        audit_graph_cache(index['mol_smiles'], graph_root, source, workers=1, chunk_size=2)
        graph_cache, graph_receipt = load_validation_graph_cache(index_path, index, graph_root)
        extra += ['--graph-cache', str(graph_root)]
    output = tmp_path / 'validation'
    entry.main(['--data-path', str(tmp_path), '--index', str(index_path), '--checkpoint', str(path),
                '--output', str(output), '--device', 'cuda:0', '--checkpoint-model-config', *extra])
    result = json.loads((output / 'metrics.json').read_text())
    assert result['checkpoint_model']['model_config']['align']['final_dim'] == 8
    assert result['split'] == 'val' and not result['test_evaluated']
    from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator
    if kind in ('fingerprint', 'gine_fingerprint'):
        validator_class = entry.FingerprintRetrievalValidator if kind == 'fingerprint' else entry.GraphFingerprintRetrievalValidator
        reference_validator = validator_class(
            index, entry.config.retrieval_validation, fingerprint_root=root, fingerprint_provenance=provenance,
            index_sha256=sha256_file(index_path), **({'graph_cache': graph_cache} if graph_cache is not None else {}))
        assert result['validation_fingerprint_cache'] == cache
        if graph_cache is not None:
            assert result['validation_graph_cache'] == graph_receipt
    else:
        reference_validator = AlignmentRetrievalValidator(index, entry.config.retrieval_validation)
    reference = reference_validator(original, torch.device('cpu'), 0, 'reference')
    assert all(result['metrics'][name] == reference[name] for name in ('top1', 'top5', 'top10', 'top20', 'mrr'))


@pytest.mark.parametrize('candidate_supervision', [False, True])
@pytest.mark.parametrize('precursor_delta', [False, True])
@pytest.mark.parametrize('kind', ['fingerprint', 'gine_fingerprint'])
def test_training_entry_saves_complete_typed_model_inputs_and_all_queries(
    tmp_path, monkeypatch, candidate_supervision, precursor_delta, kind,
):
    import train_align as entry
    from SpecEmbedding.utils.fingerprint_cache import load_fingerprint_cache
    from SpecEmbedding.utils.fingerprint_validation import FingerprintRetrievalValidator
    from SpecEmbedding.utils.retrieval_validation import build_validation_index
    from tests.test_fingerprint_alignment import make_cache, make_dataset
    from tests.test_retrieval_validation import spectrum

    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    raw = [spectrum(value) for value in ('CCO', 'CC', 'CCO')]
    tokenizer = {'max_len': 4, 'show_progress_bar': False}
    index = build_validation_index(raw, {'CCO': ['CCO', 'CC', 'CCC'], 'CC': ['CC', 'CCC']}, [], {}, tokenizer)
    index.update(dataset_manifest_sha256='a' * 64, dataset_outputs={'synthetic': 'not formal data'})
    index_path = tmp_path / 'val_index.pt'
    torch.save(index, index_path)
    val_root = tmp_path / 'val_bits'
    provenance = make_cache(val_root, index['mol_smiles'], index_sha=sha256_file(index_path))
    _, val_cache = load_fingerprint_cache(index['mol_smiles'], val_root, provenance)
    inputs = {'train': {'cache': dataset.base.fingerprint_receipt}, 'validation': {'cache': val_cache}}
    settings = copy.deepcopy(entry.config.to_dict())
    settings['model'] = model_config(kind)
    if kind == 'gine_fingerprint':
        settings['model']['spec_encoder']['qk_norm'] = {'eps': 1e-6}
    if precursor_delta:
        settings['model']['spec_encoder']['precursor_delta'] = {
            'fourier_dim': 8, 'hidden_dim': 8, 'min_wavelength': .01, 'max_wavelength': 10000.,
        }
    settings['augmentation'] = {**dataset.base.augment_config, 'node_drop_rate': 0., 'edge_mask_rate': 0.}
    if kind == 'gine_fingerprint':
        settings['augmentation'].update(node_drop_rate=.1, edge_mask_rate=.1)
    settings['data']['tokenizer'] = tokenizer
    align = settings['train']['align']
    align.update(batching='mass_blocks', batch_size=2, mass_block_size=1, epochs_stage2=2,
                 num_workers=0, patience=5, metric_for_best='validation_top1_then_mrr')
    align['candidate_supervision']['enabled'] = candidate_supervision
    settings['retrieval_validation'].update(mol_batch_size=2, spec_batch_size=2, num_workers=0)
    monkeypatch.setattr(entry, 'config', ConfigObject(settings))
    output = tmp_path / 'training'
    validator_class = FingerprintRetrievalValidator if kind == 'fingerprint' else entry.GraphFingerprintRetrievalValidator
    validator = validator_class(index, entry.config.retrieval_validation, output / 'validation_retrieval',
                                             fingerprint_root=val_root, fingerprint_provenance=provenance,
                                             index_sha256=sha256_file(index_path))
    metadata = {'seed': 42, 'exclude_val_query_indices': [],
                'training_fingerprint_cache': inputs['train']['cache'], 'validation_fingerprint_cache': val_cache,
                'fulltrain_audit': {'formal_fulltrain': True, 'dataset_version': '1.5',
                                    'dataset_manifest_sha256': 'a' * 64, 'input_outputs': index['dataset_outputs'],
                                    'expected_epoch_counts': {'train': 3, 'val': 3}}}
    monkeypatch.setattr(entry, 'GINEEncoder', lambda **kwargs: pytest.fail('Constructed a GINE tower'))
    entry.set_seed(42)
    entry.train_align(dataset.base._data, dataset.base._keys, dataset.base._data, dataset.base._keys, None,
                       batch_size=2, lr=1e-4, save_dir=output, device='cpu', formal_fulltrain=True,
                       selection_metadata=metadata, seed=42, retrieval_validator=validator,
                       training_candidates=dataset.candidates if candidate_supervision else None,
                       candidate_input_receipt=dataset.provenance if candidate_supervision else None,
                       fingerprint_inputs=inputs,
                       fingerprint_smiles={'train': dataset.candidates.metadata['mol_smiles'], 'validation': index['mol_smiles']})
    selection = json.loads((output / 'alignment_selection.json').read_text())
    assert selection['model_config']['type'] == kind
    assert selection['training_fingerprint_cache'] == inputs['train']['cache']
    assert selection['validation_fingerprint_cache'] == val_cache
    assert [(row['train'], row['val']) for row in selection['fulltrain_audit']['epochs']] == [(3, 3), (3, 3)]
    for row in selection['fulltrain_audit']['epochs']:
        assert row['batching']['queries'] == row['batching']['unique_queries'] == 3
    restored, _, _ = load_formal_alignment(output / 'best_model_stage2.pth', torch.device('cpu'),
                                           dataset_outputs=index['dataset_outputs'], dataset_manifest_sha256='a' * 64,
                                           tokenizer_config=tokenizer, expected_counts={'train': 3, 'val': 3}, exclusions=[])
    assert formal_model_type(selection['model_config']) == kind
    assert restored.mol_encoder.input_bits == 2048
    assert hasattr(restored.spec_encoder, 'delta_projection') == precursor_delta
    assert selection['model_config']['spec_encoder'] == settings['model']['spec_encoder']
    if candidate_supervision:
        molecule_input = 'fixed_morgan_bits' if kind == 'fingerprint' else 'graph_with_fixed_morgan_bits'
        evidence = selection['stages']['stage2']['candidate_training']
        assert evidence['data']['molecule_input'] == molecule_input
        assert [row['queries'] for row in evidence['epochs']] == [3, 3]
        import SpecEmbedding.utils.candidate_training as candidate_io
        marker = tmp_path / 'synthetic_candidate_input.json'
        marker.write_text(json.dumps({'synthetic_source_inventory': True}))
        monkeypatch.setattr(candidate_io, 'read_candidate_training_input', lambda *a, **k: (
            dataset.candidates, dataset.provenance, {'path': str(marker), 'sha256': sha256_file(marker)}))
        monkeypatch.setattr(candidate_io, 'candidate_source_inputs', lambda _: {})
        report, _ = candidate_io.audit_candidate_training(
            output, selection['stages']['stage2'], marker, align['candidate_supervision'], 42, 2,
            tmp_path, {'train': 3}, [], fingerprint_cache=inputs['train']['cache'], graph_fingerprint=kind == 'gine_fingerprint')
        assert report['state'] == 'verified_full_candidate_replay' and report['epochs'] == 2
        assert report['input_provenance']['molecule_input'] == molecule_input
        assert report['queries_per_epoch'] == 3
        damaged = copy.deepcopy(selection['stages']['stage2'])
        damaged['candidate_training']['data']['graph_cache_size'] += 1
        with pytest.raises(ValueError, match='Candidate loss, input or trajectory'):
            candidate_io.audit_candidate_training(output, damaged, marker, align['candidate_supervision'], 42, 2,
                tmp_path, {'train': 3}, [], fingerprint_cache=inputs['train']['cache'], graph_fingerprint=kind == 'gine_fingerprint')


@pytest.mark.parametrize('candidate_supervision', [False, True])
def test_fingerprint_queue_routes_distinct_baseline_inputs_without_enabling_a_loss(tmp_path, candidate_supervision):
    from types import SimpleNamespace

    import run_massspecgym_v15 as runner

    args = SimpleNamespace(output_root=tmp_path / 'run', source_dir=tmp_path / 'source', legacy_tsv=tmp_path / 'v1.tsv',
        prepared_data=tmp_path / 'data', prepared_validation_index=tmp_path / 'val.pt', gpus=[0, 1], device='cuda:0',
        optimize_alignment=True, baseline_checkpoint=tmp_path / 'gine.pth', molecule_input='fingerprint',
        validation_graph_cache=tmp_path / 'graph_cache', fingerprint_training_index=tmp_path / 'training.pkl',
        training_fingerprint_cache=tmp_path / 'train_bits', validation_fingerprint_cache=tmp_path / 'val_bits',
        alignment_training_candidates=tmp_path / 'training.pkl' if candidate_supervision else None)
    stages = runner.commands(args)
    assert [stage['name'] for stage in stages] == ['import_v15', 'import_validation', 'baseline_validation', 'alignment42']
    baseline, training = (stage['command'] for stage in stages[2:])
    assert '--checkpoint-model-config' in baseline and '--graph-cache' in baseline
    assert '--validation-graph-cache' not in training and '--fingerprint-cache' not in baseline
    assert ('--candidate-training-input' in training) == candidate_supervision
    for flag in ('fingerprint_training_index', 'training_fingerprint_cache', 'validation_fingerprint_cache'):
        assert training[training.index('--' + flag.replace('_', '-')) + 1] == str(getattr(args, flag))
