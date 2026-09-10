"""Versioned observed-adduct inputs, bound to complete raw queries and unchanged peak tokens."""

import copy
import hashlib
import json
import logging
import math
import pickle
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.models_adduct import validate_adduct_settings
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import verify_dataset


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sequence_sha256(sequence):
    """Match the float32/bool values actually passed to the spectral tower."""
    digest = hashlib.sha256()
    shape = None
    for name, dtype in (('mz', '<f4'), ('intensity', '<f4'), ('mask', '?')):
        value = np.asarray(sequence[name])
        require(value.ndim == 1 and len(value) > 0 and (shape is None or value.shape == shape),
                'Metadata binding requires matching nonempty token vectors')
        shape = value.shape
        if name == 'mask':
            require(value.dtype == np.bool_, 'Metadata binding requires boolean padding')
        else:
            require(np.isfinite(value).all(), 'Metadata binding found non-finite peak tokens')
        digest.update(len(value).to_bytes(8, 'little'))
        digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes())
    return digest.hexdigest()


def observed_adduct(value, settings):
    validate_adduct_settings(settings)
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        return None, settings['unknown_id']
    require(isinstance(value, str), 'Observed adduct must be a string or explicitly missing')
    # Do not infer, strip, or canonicalize an unregistered observed value.
    return value, settings['adducts'].index(value) if value in settings['adducts'] else settings['unknown_id']


def build_split_metadata(raw, split, exclusions, tokenizer_config, settings):
    validate_adduct_settings(settings)
    require(split in ('train', 'val') and len(raw) > 0, 'Metadata preparation is limited to complete train/val')
    require(isinstance(exclusions, list) and all(type(i) is int and 0 <= i < len(raw) for i in exclusions)
            and exclusions == sorted(set(exclusions)) and (split != 'train' or not exclusions),
            'Invalid metadata exclusions; training queries cannot be excluded')
    tokenizer = Tokenizer(**tokenizer_config)
    excluded = set(exclusions)
    rows, identifiers, counts, unknown = [], set(), Counter(), Counter()
    for index, spectrum in enumerate(raw):
        if index in excluded:
            continue
        identifier, identity, smiles = (spectrum.get(key) for key in ('identifier', 'identity_2d', 'smiles'))
        require(all(isinstance(value, str) and value for value in (identifier, identity, smiles))
                and identifier not in identifiers, 'Missing or duplicate raw query identity')
        identifiers.add(identifier)
        observed, adduct_id = observed_adduct(spectrum.get('adduct'), settings)
        counts[str(adduct_id)] += 1
        if adduct_id == settings['unknown_id']:
            unknown[json.dumps(observed, ensure_ascii=False)] += 1
        sequence = tokenizer.tokenize(spectrum)
        require(sequence['smiles'] == smiles, 'Tokenization changed the query target binding')
        rows.append({'raw_query_index': index, 'identifier': identifier, 'identity_2d': identity,
                     'smiles': smiles, 'observed_adduct': observed, 'adduct_id': adduct_id,
                     'sequence_sha256': sequence_sha256(sequence)})
        if len(rows) % 16384 == 0:
            logging.info('Adduct metadata %s: %s queries', split, len(rows))
    require(bool(rows), 'Metadata exclusions removed the entire split')
    return {'split': split, 'source_query_count': len(raw), 'excluded_query_indices': exclusions,
            'queries': len(rows), 'id_counts': dict(counts), 'unknown_values': dict(unknown), 'rows': rows}


def metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings):
    validate_adduct_settings(settings)
    data_path = Path(data_path)
    return {'schema_version': 1, 'expected_counts': expected_counts, 'exclude_val_query_indices': exclusions,
            'tokenizer_config': tokenizer_config, 'settings': settings,
            'dataset_manifest_sha256': sha256_file(data_path / 'dataset_manifest.json'),
            'split_sha256': {split: sha256_file(data_path / f'{split}.pkl') for split in ('train', 'val')},
            'construction_sha256': {name: sha256_file(Path(__file__).parents[1] / name) for name in (
                'utils/adduct_metadata.py', 'models_adduct.py', 'data/tokenizer.py', 'data/const.py')},
            'versions': {name: version(name) for name in ('numpy', 'matchms')}}


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def _fresh_splits(data_path, source):
    for split in ('train', 'val'):
        with (Path(data_path) / f'{split}.pkl').open('rb') as handle:
            raw = pickle.load(handle)
        require(len(raw) == source['expected_counts'][split], 'Metadata raw split count changed')
        yield split, build_split_metadata(raw, split, source['exclude_val_query_indices'] if split == 'val' else [],
                                         source['tokenizer_config'], source['settings'])


def prepare_adduct_cache(data_path, root, expected_counts, exclusions, tokenizer_config, settings):
    verify_dataset(data_path, expected_counts, exclusions)
    source = metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, payload in _fresh_splits(data_path, source):
        path = root / f'{split}.json'
        write_new(path, payload)
        files[path.name] = sha256_file(path)
    require(source == metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings),
            'Metadata source changed during preparation')
    write_new(root / 'manifest.json', {'state': 'prepared', 'source': source, 'files_sha256': files})


def audit_adduct_cache(data_path, root, expected_counts, exclusions, tokenizer_config, settings):
    """Freshly read every original query again before publishing a usable cache receipt."""
    root = Path(root)
    require(not (root / 'audit.json').exists(), 'Adduct cache audit already exists')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest_sha = sha256_file(root / 'manifest.json')
    source = metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
    require(manifest['state'] == 'prepared' and manifest['source'] == source, 'Adduct source/configuration changed')
    require(set(manifest['files_sha256']) == {'train.json', 'val.json'}, 'Incomplete adduct file inventory')
    summaries = {}
    for split, expected in _fresh_splits(data_path, source):
        path = root / f'{split}.json'
        require(sha256_file(path) == manifest['files_sha256'][path.name]
                and json.loads(path.read_text()) == expected, 'Adduct payload differs from fresh original queries')
        summaries[split] = {key: value for key, value in expected.items() if key != 'rows'}
    require(source == metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
            and sha256_file(root / 'manifest.json') == manifest_sha
            and all(sha256_file(root / name) == digest for name, digest in manifest['files_sha256'].items()),
            'Metadata source changed during audit')
    write_new(root / 'audit.json', {'state': 'complete', 'manifest_sha256': manifest_sha,
                                   'files_sha256': manifest['files_sha256'], 'splits': summaries})


class SpectrumMetadata:
    """Validated row binding; molecular identities are used for indexing checks, never as features."""

    def __init__(self, payload, provenance):
        self.payload, self.provenance = copy.deepcopy(payload), copy.deepcopy(provenance)
        self.rows = self.payload['rows']
        self.raw_query_indices = np.asarray([row['raw_query_index'] for row in self.rows], dtype=np.int64)
        self.adduct_ids = np.asarray([row['adduct_id'] for row in self.rows], dtype=np.int64)
        self.raw_query_indices.setflags(write=False)
        self.adduct_ids.setflags(write=False)

    def bind_dataset(self, dataset):
        require(dataset.full_spectra and len(dataset) == len(self.rows), 'Adduct inputs require the complete spectrum dataset')
        grouped = defaultdict(list)
        for row in self.rows:
            grouped[row['identity_2d']].append(row)
        require(set(grouped) == set(dataset._data), 'Adduct identities differ from tokenized queries')
        mapping, ids = [], []
        for identity, offset in dataset._spectrum_indices:
            require(offset < len(grouped[identity]), 'Adduct identity group lost a query')
            row, sequence = grouped[identity][offset], dataset._data[identity][offset]
            require(row['smiles'] == sequence['smiles'] and row['sequence_sha256'] == sequence_sha256(sequence),
                    'Adduct query/token binding mismatch')
            mapping.append(row['raw_query_index'])
            ids.append(row['adduct_id'])
        require(sorted(mapping) == self.raw_query_indices.tolist(), 'Adduct dataset has duplicate or missing queries')
        return np.asarray(mapping, dtype=np.int64), np.asarray(ids, dtype=np.int64)

    def bind_index(self, index):
        source = self.provenance['source']
        require(self.payload['split'] == 'val' and index['raw_query_indices'] == self.raw_query_indices.tolist()
                and index['dataset_manifest_sha256'] == source['dataset_manifest_sha256']
                and index['tokenizer_config'] == source['tokenizer_config'], 'Adduct validation source/order mismatch')
        require(len(index['sequences']) == len(self.rows), 'Incomplete adduct validation queries')
        for row, sequence in zip(self.rows, index['sequences'], strict=True):
            require(sequence['smiles'] == row['smiles'] and sequence_sha256(sequence) == row['sequence_sha256'],
                    'Adduct validation token binding mismatch')
        return self.adduct_ids.copy()


def load_adduct_cache(data_path, root, split, expected_counts, exclusions, tokenizer_config, settings):
    require(split in ('train', 'val'), 'Adduct optimization cache supports train/val only')
    root = Path(root).resolve()
    manifest_path, audit_path = root / 'manifest.json', root / 'audit.json'
    manifest_sha, audit_sha = sha256_file(manifest_path), sha256_file(audit_path)
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    source = metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
    require(manifest['state'] == 'prepared' and manifest['source'] == source
            and audit['state'] == 'complete' and audit['manifest_sha256'] == manifest_sha
            and audit['files_sha256'] == manifest['files_sha256']
            and set(manifest['files_sha256']) == {'train.json', 'val.json'}, 'Adduct cache provenance mismatch')
    for name, digest in manifest['files_sha256'].items():
        require(sha256_file(root / name) == digest, 'Adduct cache payload changed')
    payload = json.loads((root / f'{split}.json').read_text())
    expected_indices = [i for i in range(expected_counts[split]) if split != 'val' or i not in set(exclusions)]
    require(payload['split'] == split and payload['source_query_count'] == expected_counts[split]
            and [row['raw_query_index'] for row in payload['rows']] == expected_indices
            and {key: value for key, value in payload.items() if key != 'rows'} == audit['splits'][split],
            'Adduct cache does not cover every eligible query')
    for row in payload['rows']:
        require(type(row['adduct_id']) is int and row['adduct_id'] == observed_adduct(row['observed_adduct'], settings)[1],
                'Adduct cache contains an invalid mapped ID')
    require(source == metadata_source(data_path, expected_counts, exclusions, tokenizer_config, settings)
            and sha256_file(manifest_path) == manifest_sha and sha256_file(audit_path) == audit_sha
            and all(sha256_file(root / name) == digest for name, digest in manifest['files_sha256'].items()),
            'Adduct cache/source changed while loading')
    return SpectrumMetadata(payload, {'directory': str(root), 'split': split, 'source': source,
        'manifest_sha256': manifest_sha, 'audit_sha256': audit_sha,
        'payload_sha256': manifest['files_sha256'][f'{split}.json']})
