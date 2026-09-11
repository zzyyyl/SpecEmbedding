"""Versioned, train-only sparse spectrum targets for a downstream auxiliary task."""

import copy
import hashlib
import json
import logging
import math
import pickle
from collections import Counter, defaultdict
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np

from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.adduct_metadata import SpectrumMetadata, sequence_sha256
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import verify_dataset


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_target_settings(settings):
    fields = {'bins_per_da', 'max_mz', 'intensity_transform', 'overflow_policy',
              'adducts', 'unknown_id', 'unknown_policy'}
    require(isinstance(settings, dict) and set(settings) == fields, 'Incomplete spectrum target settings')
    require(all(type(settings[k]) is int and settings[k] > 0 for k in ('bins_per_da', 'max_mz')),
            'Spectrum target range and resolution must be positive integers')
    dimensions = settings['bins_per_da'] * settings['max_mz'] + 1
    require(dimensions < np.iinfo(np.int32).max
            and settings['intensity_transform'] == 'linear_sum_l2'
            and settings['overflow_policy'] == 'single_bin', 'Unsupported spectrum target construction')
    require(settings['adducts'] == ['[M+H]+', '[M+Na]+']
            and type(settings['unknown_id']) is int and settings['unknown_id'] == 2
            and settings['unknown_policy'] == 'zero_condition', 'Invalid auxiliary adduct vocabulary or policy')
    return dimensions


def observed_condition(value, settings):
    validate_target_settings(settings)
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        return None, settings['unknown_id']
    require(isinstance(value, str), 'Observed adduct must be a string or explicitly missing')
    return value, settings['adducts'].index(value) if value in settings['adducts'] else settings['unknown_id']


def sparse_peak_target(mz, intensity, settings):
    """Sum raw observed intensities by bin; do not add or remove precursor peaks."""
    dimensions = validate_target_settings(settings)
    mz, intensity = np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    require(mz.ndim == 1 and len(mz) > 0 and mz.shape == intensity.shape
            and np.isfinite(mz).all() and np.isfinite(intensity).all()
            and (mz >= 0).all() and (intensity >= 0).all() and intensity.max() > 0,
            'Invalid or zero-norm observed spectrum target')
    bins = np.full(len(mz), dimensions - 1, dtype=np.int32)
    inside = mz < settings['max_mz']
    bins[inside] = np.floor(mz[inside] * settings['bins_per_da']).astype(np.int32)
    positive = intensity > 0
    unique, reverse = np.unique(bins[positive], return_inverse=True)
    # Scaling first preserves the linear target and avoids overflowing its squared norm.
    total = np.bincount(reverse, weights=intensity[positive] / intensity.max())
    values = (total / np.linalg.norm(total)).astype('<f4')
    keep = values > 0
    return unique[keep].astype('<i4'), values[keep], {
        'raw_peaks': len(mz), 'overflow_peaks': int((~inside).sum()),
        'zero_intensity_peaks': int((~positive).sum()), 'float32_underflow_bins': int((~keep).sum())}


def _scalar_target(mz, intensity, settings):
    """Independent scalar arithmetic for a full source replay during cache audit."""
    overflow = settings['bins_per_da'] * settings['max_mz']
    grouped = defaultdict(list)
    scale = max(map(float, intensity))
    for mass, value in zip(mz, intensity, strict=True):
        if value > 0:
            index = math.floor(float(mass) * settings['bins_per_da']) if mass < settings['max_mz'] else overflow
            grouped[index].append(float(value) / scale)
    keys = sorted(grouped)
    totals = [math.fsum(grouped[k]) for k in keys]
    norm = math.sqrt(math.fsum(x * x for x in totals))
    values = np.asarray([x / norm for x in totals], dtype='<f4')
    keep = values > 0
    return np.asarray(keys, dtype='<i4')[keep], values[keep], {
        'raw_peaks': len(mz), 'overflow_peaks': sum(float(x) >= settings['max_mz'] for x in mz),
        'zero_intensity_peaks': sum(float(x) == 0 for x in intensity), 'float32_underflow_bins': int((~keep).sum())}


def _row(spectrum, index, tokenizer, settings):
    identifier, identity, smiles = (spectrum.get(k) for k in ('identifier', 'identity_2d', 'smiles'))
    require(all(isinstance(x, str) and x for x in (identifier, identity, smiles)), 'Missing source query binding')
    observed, adduct_id = observed_condition(spectrum.get('adduct'), settings)
    digest = hashlib.sha256()
    for values in (spectrum.peaks.mz, spectrum.peaks.intensities):
        digest.update(len(values).to_bytes(8, 'little'))
        digest.update(np.asarray(values, dtype='<f8').tobytes())
    sequence = tokenizer.tokenize(spectrum)
    require(sequence['smiles'] == smiles, 'Tokenization changed the auxiliary query binding')
    return {'raw_query_index': index, 'identifier': identifier, 'identity_2d': identity, 'smiles': smiles,
            'observed_adduct': observed, 'adduct_id': adduct_id, 'original_peaks_sha256': digest.hexdigest(),
            'sequence_sha256': sequence_sha256(sequence)}


def target_source(data_path, expected_counts, exclusions, tokenizer_config, settings):
    validate_target_settings(settings)
    require(type(expected_counts.get('train')) is int and expected_counts['train'] > 0, 'Invalid full train count')
    base = Path(__file__).parents[1]
    return {'schema_version': 1, 'split': 'train', 'queries': expected_counts['train'],
            'expected_counts': copy.deepcopy(expected_counts), 'exclude_val_query_indices': list(exclusions),
            'tokenizer_config': copy.deepcopy(tokenizer_config), 'settings': copy.deepcopy(settings),
            'dataset_manifest_sha256': sha256_file(Path(data_path) / 'dataset_manifest.json'),
            'train_sha256': sha256_file(Path(data_path) / 'train.pkl'),
            'construction_sha256': {name: sha256_file(base / name) for name in (
                'utils/spectrum_targets.py', 'utils/adduct_metadata.py', 'data/tokenizer.py', 'data/const.py')},
            'versions': {name: version(name) for name in ('numpy', 'matchms')}}


def _write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')


def _raw_train(data_path, source):
    with (Path(data_path) / 'train.pkl').open('rb') as stream:
        raw = pickle.load(stream)
    require(len(raw) == source['queries'], 'Spectrum targets require the entire training split')
    return raw


def prepare_spectrum_target_cache(data_path, root, expected_counts, exclusions, tokenizer_config, settings):
    verify_dataset(data_path, expected_counts, exclusions)
    source = target_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _write_new(root / 'inputs.json', source)
    raw = _raw_train(data_path, source)
    tokenizer, seen, ptr, counts = Tokenizer(**tokenizer_config), set(), [0], Counter()
    with ExitStack() as stack:
        bins_file = stack.enter_context((root / 'bins.bin').open('xb'))
        values_file = stack.enter_context((root / 'values.bin').open('xb'))
        rows_file = stack.enter_context((root / 'rows.jsonl').open('x'))
        for index, spectrum in enumerate(raw):
            try:
                bins, values, stats = sparse_peak_target(spectrum.peaks.mz, spectrum.peaks.intensities, settings)
                row = _row(spectrum, index, tokenizer, settings)
                require(row['identifier'] not in seen, 'Duplicate source query identifier')
                seen.add(row['identifier'])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'Invalid spectrum target query {index} ({spectrum.get("identifier")}): {error}') from error
            bins.tofile(bins_file)
            values.tofile(values_file)
            ptr.append(ptr[-1] + len(bins))
            rows_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            counts.update(stats)
            counts[f'adduct_id_{row["adduct_id"]}'] += 1
            if (index + 1) % 16384 == 0:
                logging.info('Spectrum targets: %s/%s train queries', index + 1, len(raw))
    with (root / 'row_ptr.bin').open('xb') as stream:
        np.asarray(ptr, dtype='<i8').tofile(stream)
    files = {name: {'sha256': sha256_file(root / name), 'bytes': (root / name).stat().st_size}
             for name in ('bins.bin', 'values.bin', 'row_ptr.bin', 'rows.jsonl', 'inputs.json')}
    require(source == target_source(data_path, expected_counts, exclusions, tokenizer_config, settings),
            'Spectrum target source changed during preparation')
    manifest = {'schema_version': 1, 'state': 'built_pending_full_audit', 'source': source,
                'queries': len(raw), 'nonzero_bins': ptr[-1], 'counts': dict(counts), 'files': files}
    _write_new(root / 'manifest.json', manifest)
    return manifest


class SpectrumTargetCache:
    """Read-only sparse payloads; returned rows are copies, including in spawned workers."""

    def __init__(self, root, manifest, rows, provenance):
        self.root, self.manifest = Path(root), copy.deepcopy(manifest)
        self.metadata = SpectrumMetadata({'rows': rows}, provenance)
        self.provenance = copy.deepcopy(provenance)
        self._arrays = None

    def __len__(self):
        return self.manifest['queries']

    def __getstate__(self):
        return {**self.__dict__, '_arrays': None}

    def _open(self):
        if self._arrays is None:
            count = self.manifest['nonzero_bins']
            self._arrays = {
                'bins': np.memmap(self.root / 'bins.bin', mode='r', dtype='<i4', shape=(count,)),
                'values': np.memmap(self.root / 'values.bin', mode='r', dtype='<f4', shape=(count,)),
                'ptr': np.memmap(self.root / 'row_ptr.bin', mode='r', dtype='<i8', shape=(len(self) + 1,))}
        return self._arrays

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(self):
            raise IndexError(index)
        arrays = self._open()
        start, end = map(int, arrays['ptr'][index:index + 2])
        return {'bin_indices': arrays['bins'][start:end].astype(np.int64, copy=True),
                'values': arrays['values'][start:end].copy(), 'adduct_id': int(self.metadata.adduct_ids[index]),
                'raw_query_index': int(index)}

    def bind_dataset(self, dataset):
        return self.metadata.bind_dataset(dataset)[0]


def _read_prepared_cache(data_path, root, options):
    root = Path(root)
    manifest_sha = sha256_file(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    source = target_source(data_path, *options)
    require(manifest['schema_version'] == 1 and manifest['state'] == 'built_pending_full_audit'
            and manifest['source'] == source and manifest['queries'] == source['queries'],
            'Spectrum target source/configuration mismatch')
    require(type(manifest['nonzero_bins']) is int and manifest['nonzero_bins'] >= manifest['queries']
            and set(manifest['files']) == {'bins.bin', 'values.bin', 'row_ptr.bin', 'rows.jsonl', 'inputs.json'},
            'Invalid spectrum target inventory')
    for name, entry in manifest['files'].items():
        require((root / name).stat().st_size == entry['bytes'] and sha256_file(root / name) == entry['sha256'],
                'Spectrum target payload changed')
    for name, size in {'bins.bin': manifest['nonzero_bins'] * 4, 'values.bin': manifest['nonzero_bins'] * 4,
                       'row_ptr.bin': (manifest['queries'] + 1) * 8}.items():
        require(manifest['files'][name]['bytes'] == size, 'Spectrum target array shape mismatch')
    require(json.loads((root / 'inputs.json').read_text()) == source, 'Spectrum target input receipt mismatch')
    with (root / 'rows.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == source['queries'] and [row['raw_query_index'] for row in rows] == list(range(source['queries']))
            and len({row['identifier'] for row in rows}) == len(rows), 'Incomplete or duplicate target query binding')
    for row in rows:
        require(type(row['adduct_id']) is int
                and row['adduct_id'] == observed_condition(row['observed_adduct'], source['settings'])[1],
                'Invalid auxiliary target adduct ID')
    cache = SpectrumTargetCache(root, manifest, rows, {'source': source, 'manifest_sha256': manifest_sha})
    arrays = cache._open()
    dimensions = validate_target_settings(source['settings'])
    require(arrays['ptr'][0] == 0 and arrays['ptr'][-1] == manifest['nonzero_bins']
            and (np.diff(arrays['ptr']) > 0).all() and (arrays['bins'] >= 0).all()
            and (arrays['bins'] < dimensions).all() and np.isfinite(arrays['values']).all()
            and (arrays['values'] > 0).all(), 'Invalid spectrum target sparse arrays')
    require(source == target_source(data_path, *options) and sha256_file(root / 'manifest.json') == manifest_sha
            and all(sha256_file(root / name) == entry['sha256'] for name, entry in manifest['files'].items()),
            'Spectrum target source/payload changed while loading')
    return cache


def audit_spectrum_target_cache(data_path, root, expected_counts, exclusions, tokenizer_config, settings):
    root = Path(root)
    require(not (root / 'audit.json').exists(), 'Spectrum target audit already exists')
    options = (expected_counts, exclusions, tokenizer_config, settings)
    cache = _read_prepared_cache(data_path, root, options)
    source, raw = cache.provenance['source'], _raw_train(data_path, cache.provenance['source'])
    tokenizer = Tokenizer(**tokenizer_config)
    counts = Counter()
    for index, spectrum in enumerate(raw):
        actual = cache[index]
        bins, values, stats = _scalar_target(spectrum.peaks.mz, spectrum.peaks.intensities, settings)
        require(cache.metadata.rows[index] == _row(spectrum, index, tokenizer, settings)
                and np.array_equal(bins, actual['bin_indices'])
                and np.allclose(values, actual['values'], rtol=1e-6, atol=1e-8),
                f'Spectrum target differs from independent scalar source replay: query {index}')
        counts.update(stats)
        counts[f'adduct_id_{actual["adduct_id"]}'] += 1
    require(dict(counts) == cache.manifest['counts'], 'Spectrum target summary differs from complete source replay')
    require(source == target_source(data_path, *options)
            and sha256_file(root / 'manifest.json') == cache.provenance['manifest_sha256']
            and all(sha256_file(root / name) == entry['sha256'] for name, entry in cache.manifest['files'].items()),
            'Spectrum target source changed during full audit')
    result = {'state': 'complete_full_train_target_audit', 'queries': len(raw),
              'manifest_sha256': cache.provenance['manifest_sha256'], 'files': cache.manifest['files'],
              'scalar_value_comparison': {'rtol': 1e-6, 'atol': 1e-8}, 'model_encoding_performed': False}
    _write_new(root / 'audit.json', result)
    return result


def load_spectrum_target_cache(data_path, root, expected_counts, exclusions, tokenizer_config, settings):
    root = Path(root).resolve()
    audit_sha = sha256_file(root / 'audit.json')
    audit = json.loads((root / 'audit.json').read_text())
    cache = _read_prepared_cache(data_path, root, (expected_counts, exclusions, tokenizer_config, settings))
    require(audit['state'] == 'complete_full_train_target_audit' and audit['queries'] == len(cache)
            and audit['manifest_sha256'] == cache.provenance['manifest_sha256'] and audit['files'] == cache.manifest['files']
            and sha256_file(root / 'audit.json') == audit_sha, 'Incomplete or changed spectrum target audit')
    cache.provenance.update(directory=str(root), audit_sha256=audit_sha)
    cache.metadata.provenance = copy.deepcopy(cache.provenance)
    cache._arrays = None
    return cache
