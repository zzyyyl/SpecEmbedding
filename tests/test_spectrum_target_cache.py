"""Synthetic full-query target caching, source invalidation and worker isolation."""

import copy
import json
import pickle
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from matchms import Spectrum
from torch.utils.data import DataLoader

from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils import spectrum_targets as module
from SpecEmbedding.utils.spectrum_targets import (
    audit_spectrum_target_cache,
    load_spectrum_target_cache,
    prepare_spectrum_target_cache,
)
from tests.test_spectrum_auxiliary import TARGET


def raw_queries():
    rows = []
    for index, (smiles, adduct) in enumerate(zip(['CCO', 'CC', 'CCO', 'CCC'], ['[M+H]+', '[M+Na]+', None, '[M+K]+'], strict=True)):
        mz = [1.01, 1.09, 2.5, 3.1] if index != 1 else [1.01, 1.09, 2.5, 7.5]
        rows.append(Spectrum(mz=np.array(mz), intensities=np.array([.2, .3, .5, .1]),
            metadata={'identifier': f'q{index}', 'identity_2d': smiles, 'smiles': smiles,
                      'adduct': adduct, 'precursor_mz': 9.9}, metadata_harmonization=False))
    return rows


def prepared(tmp_path, monkeypatch, audit=True):
    data, root = tmp_path / 'data', tmp_path / 'targets'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    with (data / 'train.pkl').open('wb') as stream:
        pickle.dump(raw_queries(), stream)
    monkeypatch.setattr(module, 'verify_dataset', lambda *args: {})
    options = ({'train': 4, 'val': 1, 'test': 1}, [], {'max_len': 3, 'show_progress_bar': False}, copy.deepcopy(TARGET))
    prepare_spectrum_target_cache(data, root, *options)
    if audit:
        audit_spectrum_target_cache(data, root, *options)
    return data, root, options


def rows_collate(rows):
    return rows


class DatasetBinding(SimpleNamespace):
    def __len__(self):
        return len(self._spectrum_indices)


def test_full_query_cache_binding_no_artificial_precursor_and_copy_semantics(tmp_path, monkeypatch):
    data, root, options = prepared(tmp_path, monkeypatch)
    cache = load_spectrum_target_cache(data, root, *options)
    assert len(cache) == 4 and cache.metadata.adduct_ids.tolist() == [0, 1, 2, 2]
    assert cache[0]['bin_indices'].tolist() == [10, 25, 31]  # All raw peaks, despite tokenizer max_len=3.
    assert 50 not in cache[0]['bin_indices']  # The artificial precursor at 9.9 is not a target.
    assert 50 in cache[1]['bin_indices']      # A real observed overflow peak is retained.
    row = cache[0]
    row['values'][:] = 0
    row['bin_indices'][:] = -1
    assert np.all(cache[0]['values'] > 0) and np.all(cache[0]['bin_indices'] >= 0)
    assert not cache._open()['bins'].flags.writeable
    grouped = defaultdict(list)
    tokenizer = Tokenizer(**options[2])
    for raw in raw_queries():
        grouped[raw.get('identity_2d')].append(tokenizer.tokenize(raw))
    order = [('CCO', 0), ('CCO', 1), ('CC', 0), ('CCC', 0)]
    dataset = DatasetBinding(full_spectra=True, _data=grouped, _spectrum_indices=order)
    assert cache.bind_dataset(dataset).tolist() == [0, 2, 1, 3]
    grouped['CCO'][0]['intensity'][1] += .05
    with pytest.raises(ValueError, match='query/token'):
        cache.bind_dataset(dataset)
    with pytest.raises(FileExistsError):
        prepare_spectrum_target_cache(data, root, *options)


def test_real_spawn_workers_reopen_readonly_cache_across_epochs(tmp_path, monkeypatch):
    data, root, options = prepared(tmp_path, monkeypatch)
    cache = load_spectrum_target_cache(data, root, *options)
    cache[0]  # Open parent mappings; these must not be pickled into child workers.
    loader = DataLoader(cache, batch_size=2, num_workers=2, multiprocessing_context='spawn', collate_fn=rows_collate)
    for _ in range(2):
        rows = [item for batch in loader for item in batch]
        assert [x['raw_query_index'] for x in rows] == [0, 1, 2, 3]
        assert [x['adduct_id'] for x in rows] == [0, 1, 2, 2]
        for actual, index in zip(rows, range(4), strict=True):
            np.testing.assert_array_equal(actual['values'], cache[index]['values'])


@pytest.mark.parametrize('kind', ['bins', 'source', 'config', 'code', 'missing_audit'])
def test_old_or_damaged_cache_is_rejected(tmp_path, monkeypatch, kind):
    data, root, options = prepared(tmp_path, monkeypatch)
    if kind == 'bins':
        with (root / 'bins.bin').open('r+b') as stream:
            stream.write(np.array([3], dtype='<i4').tobytes())
    elif kind == 'source':
        with (data / 'train.pkl').open('ab') as stream:
            stream.write(b'changed-source')
    elif kind == 'config':
        options[-1]['bins_per_da'] = 20
    elif kind == 'code':
        original = module.sha256_file
        monkeypatch.setattr(module, 'sha256_file', lambda p: '0'*64 if str(p).endswith('spectrum_targets.py') else original(p))
    else:
        (root / 'audit.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_spectrum_target_cache(data, root, *options)


def test_audit_recomputes_summary_and_rejects_fabricated_counts(tmp_path, monkeypatch):
    data, root, options = prepared(tmp_path, monkeypatch, audit=False)
    path = root / 'manifest.json'
    value = json.loads(path.read_text())
    value['counts']['raw_peaks'] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='summary'):
        audit_spectrum_target_cache(data, root, *options)
    assert not (root / 'audit.json').exists()


def test_source_mutation_during_build_never_publishes_manifest(tmp_path, monkeypatch):
    original = module._row
    def change_source(*args):
        result = original(*args)
        if result['raw_query_index'] == 0:
            with (tmp_path / 'data/train.pkl').open('ab') as stream:
                stream.write(b'changed-during-preparation')
        return result
    monkeypatch.setattr(module, '_row', change_source)
    with pytest.raises(ValueError, match='source changed'):
        prepared(tmp_path, monkeypatch)
    assert (tmp_path / 'targets/inputs.json').exists()
    assert not (tmp_path / 'targets/manifest.json').exists()


def test_mutation_while_loading_is_rejected(tmp_path, monkeypatch):
    data, root, options = prepared(tmp_path, monkeypatch)
    original = module.SpectrumTargetCache._open
    def change_payload(self):
        arrays = original(self)
        with (root / 'values.bin').open('r+b') as stream:
            stream.write(np.array([.125], dtype='<f4').tobytes())
        return arrays
    monkeypatch.setattr(module.SpectrumTargetCache, '_open', change_payload)
    with pytest.raises(ValueError, match='changed while loading'):
        load_spectrum_target_cache(data, root, *options)


def test_real_cpu_entry_requires_external_storage_and_publishes_only_after_full_audit(tmp_path, monkeypatch):
    import prepare_spectrum_target_cache as entry
    from SpecEmbedding.config import ConfigObject
    from SpecEmbedding.utils.storage import storage_environment

    data, _, options = prepared(tmp_path, monkeypatch)
    monkeypatch.delenv('SPECEMBEDDING_RUNTIME_ROOT', raising=False)
    for key, value in storage_environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    settings = copy.deepcopy(entry.config.to_dict())
    settings['fulltrain'].update(expected_counts=options[0], exclude_val_query_indices=[])
    settings['data']['tokenizer'] = options[2]
    settings['molecule_spectrum_auxiliary']['target'] = options[-1]
    monkeypatch.setattr(entry, 'config', ConfigObject(settings))
    output = tmp_path / 'entry_targets'
    entry.main(['--data-path', str(data), '--output', str(output)])
    report = json.loads((output / 'preparation.json').read_text())
    assert report['queries'] == 4 and report['cpu_only'] and not report['model_encoding_performed']
    assert json.loads((output / 'audit.json').read_text())['queries'] == 4
    with pytest.raises(SystemExit):
        entry.main(['--data-path', str(data), '--output', str(output)])
    outside = tmp_path.parent / 'not_in_storage'
    with pytest.raises(ValueError, match='escapes'):
        entry.main(['--data-path', str(data), '--output', str(outside)])
    assert not outside.exists()
