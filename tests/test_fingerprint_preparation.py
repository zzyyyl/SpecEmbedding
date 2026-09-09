import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from SpecEmbedding.utils.fingerprint_preparation import prepare_fingerprint_inputs
from tests.test_retrieval_validation import reusable_index
from tests.test_training_candidates import saved_fixture

SETTINGS = {'radius': 2, 'bits': 2048, 'workers': 1, 'chunk_size': 2}


def node(value):
    return SimpleNamespace(to_dict=lambda: value)


def test_validation_cli_preserves_complete_inventory_with_uncovered_queries(tmp_path, monkeypatch):
    import prepare_candidate_fingerprints as entry

    path, expected, options = reusable_index(tmp_path, monkeypatch)
    data, counts, exclusions, tokenizer = options
    monkeypatch.setattr(entry, 'config', SimpleNamespace(
        fulltrain=SimpleNamespace(expected_counts=node(counts), exclude_val_query_indices=exclusions),
        data=SimpleNamespace(tokenizer=node(tokenizer)), molecule_fingerprints=node(SETTINGS),
        train=SimpleNamespace(align=SimpleNamespace(candidate_supervision=SimpleNamespace(pool_cache_size=1)))))
    root = tmp_path / 'cache'
    args = ['--data-path', str(data), '--validation-index', str(path), '--output', str(root)]
    with patch('torch.cuda.is_available', side_effect=AssertionError('CPU preparation touched CUDA')):
        entry.main(args)
    report = json.loads((root / 'preparation.json').read_text())
    assert report['source']['sha256'] == expected['sha256']
    assert report['source']['queries'] == 3 and report['source']['positive_queries'] == 1
    assert report['audit']['audited_molecules'] == 3
    assert not report['positive_labels_created_from_fingerprints'] and not report['test_evaluated']
    with pytest.raises(FileExistsError):
        entry.main(args)
    with pytest.raises(SystemExit):
        entry.main(args + ['--limit', '1'])
    with pytest.raises(SystemExit):
        entry.main(args + ['--training-candidates', str(path)])


def test_train_source_revalidated_without_sampling_or_merging_aliases(tmp_path):
    path, data, dataset = saved_fixture(tmp_path)
    options = {'expected_counts': {'train': 3}, 'exclusions': [], 'tokenizer_config': {}, 'pool_cache_size': 1}
    with patch('SpecEmbedding.utils.training_candidates.verify_dataset', return_value=dataset) as verify:
        report = prepare_fingerprint_inputs(data, path, 'train', tmp_path / 'cache',
                                             source_options=options, settings=SETTINGS)
    assert verify.call_count == 2
    assert report['source']['observed']['queries'] == 3
    assert report['audit']['audited_molecules'] == 5
    assert not report['positive_labels_created_from_fingerprints']


def test_input_change_during_preparation_preserves_partial_artifacts(tmp_path, monkeypatch):
    import SpecEmbedding.utils.fingerprint_preparation as module

    calls = []
    def source(*args, **kwargs):
        calls.append(1)
        return ['CCO'], {'sha256': ('a' if len(calls) == 1 else 'c') * 64, 'dataset_manifest_sha256': 'b'*64}
    monkeypatch.setattr(module, 'fingerprint_source', source)
    root = tmp_path / 'cache'
    with pytest.raises(ValueError, match='inventory changed'):
        module.prepare_fingerprint_inputs('data', 'index', 'train', root, source_options={}, settings=SETTINGS)
    assert (root / 'audit.json').exists() and not (root / 'preparation.json').exists()


def test_test_split_is_not_a_supported_preparation_source(tmp_path):
    with pytest.raises(ValueError, match='only supports full train or validation'):
        prepare_fingerprint_inputs('data', 'index', 'test', tmp_path / 'cache', source_options={
            'expected_counts': {}, 'exclusions': [], 'tokenizer_config': {}, 'pool_cache_size': 1}, settings=SETTINGS)
    assert not (tmp_path / 'cache').exists()
