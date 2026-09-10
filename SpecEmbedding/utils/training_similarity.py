"""Audited, fixed training-candidate similarities; no learned embeddings or held-out scoring."""

import hashlib
import json
import logging
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from SpecEmbedding.utils.fingerprint_cache import fingerprint_provenance, load_fingerprint_cache
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.training_candidates import TrainingCandidateIndex, _integer, load_training_candidates

SCHEMA = 1
POLICY = ('Exact rational mean of source-entry Morgan Tanimotos to the original exact target SMILES; '
          'negative 2D groups ordered by descending mean then identity key; all positive identities excluded; '
          'canonical groups by identity key, entries by original source position; zero union has similarity zero')
DTYPES = {'intersection': '<u2', 'union': '<u2', 'group_order': '<i2'}
POP = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def metadata_digest(metadata):
    """Bind actual in-memory identities, target strings, source positions and every query mapping."""
    _require(metadata['split'] == 'train' and metadata['candidate_type'] == 'mass'
             and metadata['forcing'] is False, 'Similarity cache requires natural Mass training candidates')
    digest = hashlib.sha256()
    for name in ('target_smiles', 'target_identity_2d', 'mol_smiles', 'mol_identity_2d'):
        digest.update(name.encode())
        digest.update(len(metadata[name]).to_bytes(8, 'little'))
        for value in metadata[name]:
            encoded = value.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
    for name in ('candidate_indices', 'source_positions', 'positive_mask', 'query_target_rows', 'raw_query_indices'):
        value = metadata[name]
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def canonical_groups(metadata, row):
    """Each group's entries are column indices into this original, unmodified source row."""
    groups = {}
    target = metadata['target_identity_2d'][row]
    for col, mol in enumerate(metadata['candidate_indices'][row]):
        if mol < 0:
            continue
        key = metadata['mol_identity_2d'][int(mol)]
        _require(bool(metadata['positive_mask'][row, col]) == (key == target), 'Candidate identity/label mismatch')
        if key != target:
            groups.setdefault(key, []).append(col)
    return tuple((key, tuple(sorted(groups[key], key=lambda col: int(metadata['source_positions'][row, col]))))
                 for key in sorted(groups))


def exact_group_order(metadata, row, intersection, union):
    groups = canonical_groups(metadata, row)
    values = []
    for key, columns in groups:
        score = sum((Fraction(int(intersection[col]), int(union[col])) if union[col] else Fraction(0)
                     for col in columns), Fraction(0)) / len(columns)
        values.append((score, key))
    return np.asarray(sorted(range(len(groups)), key=lambda i: (-values[i][0], values[i][1])), dtype=np.int16)


def _fingerprints(index, directory, *, radius, bits):
    _require(isinstance(index, TrainingCandidateIndex), 'A verified complete training index is required')
    _require(type(bits) is int and 0 < bits < 65535, 'Similarity counters require fewer than 65535 bits')
    source = fingerprint_provenance(index.metadata['mol_smiles'], index_sha256=index.provenance['sha256'],
                                    dataset_manifest_sha256=index.provenance['dataset_manifest_sha256'],
                                    radius=radius, bits=bits)
    return load_fingerprint_cache(index.metadata['mol_smiles'], directory, source)


def _source(index, fingerprint_receipt):
    return {'metadata_sha256': index.provenance['sha256'],
            'dataset_manifest_sha256': index.provenance['dataset_manifest_sha256'],
            'metadata_content_sha256': metadata_digest(index.metadata),
            'queries': len(index), 'target_rows': len(index.metadata['target_smiles']),
            'candidate_entries': int((index.metadata['candidate_indices'] >= 0).sum()),
            'fingerprint_cache': fingerprint_receipt, 'policy': POLICY,
            'implementation_sha256': sha256_file(Path(__file__))}


def _targets(metadata):
    targets = set(metadata['target_smiles'])
    mapping = {s: i for i, s in enumerate(metadata['mol_smiles']) if s in targets}
    _require(len(mapping) == len(targets), 'Missing original exact target fingerprint')
    for row, text in enumerate(metadata['target_smiles']):
        i = mapping[text]
        _require(i in metadata['candidate_indices'][row]
                 and metadata['mol_identity_2d'][i] == metadata['target_identity_2d'][row],
                 'Exact target fingerprint does not match the original candidate pool')
    return mapping


def _verify_files(root, manifest, shape):
    _require(manifest['schema_version'] == SCHEMA and manifest['state'] == 'built_pending_full_audit'
             and set(manifest['files']) == set(DTYPES), 'Unsupported similarity cache schema')
    arrays = {}
    for name, dtype in DTYPES.items():
        entry = manifest['files'][name]
        _require(entry['filename'] == f'{name}.npy' and entry['dtype'] == dtype and entry['shape'] == list(shape),
                 'Similarity cache file layout changed')
        path = root / entry['filename']
        _require(path.stat().st_size == entry['bytes'] and sha256_file(path) == entry['sha256'],
                 'Similarity cache payload changed')
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        _require(a.shape == shape and a.dtype.str == dtype, 'Similarity cache array shape/dtype mismatch')
        arrays[name] = a
    return arrays


def build_training_similarity_cache(index, fingerprint_directory, output, *, radius, bits):
    root = Path(output).resolve()
    _require(not root.exists(), 'Refusing an existing similarity cache directory')
    store, fp_receipt = _fingerprints(index, fingerprint_directory, radius=radius, bits=bits)
    source = _source(index, fp_receipt)
    metadata = index.metadata
    targets = _targets(metadata)
    shape = metadata['candidate_indices'].shape
    root.mkdir(parents=True, exist_ok=False)
    _save(root / 'inputs.json', source)
    started = time.monotonic()
    intersections = np.full(shape, 65535, dtype='<u2')
    unions = np.full(shape, 65535, dtype='<u2')
    order = np.full(shape, -1, dtype='<i2')
    packed = store._open()
    for row, text in enumerate(metadata['target_smiles']):
        valid = metadata['candidate_indices'][row] >= 0
        candidates = packed[metadata['candidate_indices'][row, valid]]
        target = packed[targets[text]]
        intersections[row, valid] = POP[candidates & target].sum(axis=1)
        unions[row, valid] = POP[candidates | target].sum(axis=1)
        ranked = exact_group_order(metadata, row, intersections[row], unions[row])
        order[row, :len(ranked)] = ranked
        if (row + 1) % 5000 == 0 or row + 1 == len(targets):
            logging.info('Built exact training similarity groups: %s/%s', row + 1, len(targets))
    files = {}
    for name, values in (('intersection', intersections), ('union', unions), ('group_order', order)):
        path = root / f'{name}.npy'
        with path.open('xb') as handle:
            np.save(handle, values, allow_pickle=False)
        files[name] = {'filename': path.name, 'dtype': values.dtype.str, 'shape': list(values.shape),
                       'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    _, after_fp = _fingerprints(index, fingerprint_directory, radius=radius, bits=bits)
    _require(_source(index, after_fp) == source, 'Similarity inputs or implementation changed during build')
    manifest = {'schema_version': SCHEMA, 'state': 'built_pending_full_audit', 'source': source,
                'files': files, 'build_seconds': time.monotonic() - started}
    _save(root / 'manifest.json', manifest)
    return manifest


def audit_training_similarity_cache(index, output, *, radius, bits):
    """Recompute every bit count and group order using unpacked bits and independent grouping."""
    root = Path(output).resolve()
    _require(not (root / 'audit.json').exists(), 'Refusing an existing similarity cache audit')
    manifest_sha = sha256_file(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    fp_directory = manifest['source']['fingerprint_cache']['directory']
    store, fp_receipt = _fingerprints(index, fp_directory, radius=radius, bits=bits)
    source = _source(index, fp_receipt)
    _require(manifest['source'] == source, 'Similarity cache source changed')
    metadata = index.metadata
    arrays = _verify_files(root, manifest, metadata['candidate_indices'].shape)
    targets = _targets(metadata)
    started = time.monotonic()
    groups_seen = 0
    for row, text in enumerate(metadata['target_smiles']):
        valid = metadata['candidate_indices'][row] >= 0
        ids = metadata['candidate_indices'][row, valid]
        candidate_bits = np.unpackbits(store._open()[ids], axis=1, bitorder='big')
        target_bits = np.unpackbits(store._open()[targets[text]], bitorder='big')
        common = candidate_bits[:, target_bits.astype(bool)].sum(axis=1, dtype=np.uint16)
        union = candidate_bits.sum(axis=1, dtype=np.uint16) + int(target_bits.sum()) - common
        _require(np.array_equal(common, arrays['intersection'][row, valid])
                 and np.array_equal(union, arrays['union'][row, valid])
                 and (arrays['intersection'][row, ~valid] == 65535).all()
                 and (arrays['union'][row, ~valid] == 65535).all(), 'Incorrect full candidate intersection/union')
        groups = {}
        for i, j, mol in zip(common, union, ids, strict=True):
            key = metadata['mol_identity_2d'][int(mol)]
            if key != metadata['target_identity_2d'][row]:
                groups.setdefault(key, []).append(Fraction(int(i), int(j)) if j else Fraction(0))
        keys = sorted(groups)
        means = {key: sum(groups[key], Fraction(0)) / len(groups[key]) for key in keys}
        ordinals = {key: i for i, key in enumerate(keys)}
        expected = [ordinals[key] for key in sorted(keys, key=lambda key: (-means[key], key))]
        _require(arrays['group_order'][row].tolist() == expected + [-1] * (len(valid) - len(keys)),
                 'Incorrect complete negative group order')
        groups_seen += len(keys)
        if (row + 1) % 5000 == 0 or row + 1 == len(targets):
            logging.info('Independently audited training similarity groups: %s/%s', row + 1, len(targets))
    _, after_fp = _fingerprints(index, fp_directory, radius=radius, bits=bits)
    _require(_source(index, after_fp) == source and sha256_file(root / 'manifest.json') == manifest_sha,
             'Similarity source/manifest changed during independent audit')
    _verify_files(root, manifest, metadata['candidate_indices'].shape)
    audit = {'state': 'verified_complete_training_similarity', 'source': source, 'manifest_sha256': manifest_sha,
             'target_rows': len(targets), 'candidate_entries': source['candidate_entries'],
             'negative_groups': groups_seen, 'audit_seconds': time.monotonic() - started}
    _save(root / 'audit.json', audit)
    return audit


class TrainingSimilarityCache:
    """Only the small immutable group order is opened by training workers; no fingerprint payload."""

    def __init__(self, root, shape):
        self.root, self.shape, self._order = Path(root), tuple(shape), None

    def __getstate__(self):
        return {**self.__dict__, '_order': None}

    def row(self, row):
        row = _integer(row, 'target_row')
        _require(row < self.shape[0], 'Similarity target row outside complete training index')
        if self._order is None:
            self._order = np.load(self.root / 'group_order.npy', mmap_mode='r', allow_pickle=False)
            _require(self._order.shape == self.shape and self._order.dtype.str == '<i2', 'Similarity worker layout changed')
        return self._order[row]


def load_training_similarity_cache(index, directory, *, radius, bits):
    root = Path(directory).resolve()
    manifest_sha, audit_sha = sha256_file(root / 'manifest.json'), sha256_file(root / 'audit.json')
    manifest, audit = (json.loads((root / name).read_text()) for name in ('manifest.json', 'audit.json'))
    _, fp_receipt = _fingerprints(index, manifest['source']['fingerprint_cache']['directory'], radius=radius, bits=bits)
    source = _source(index, fp_receipt)
    _require(manifest['source'] == audit['source'] == source
             and audit['state'] == 'verified_complete_training_similarity' and audit['manifest_sha256'] == manifest_sha
             and audit['target_rows'] == source['target_rows'] and audit['candidate_entries'] == source['candidate_entries'],
             'Missing, incomplete or stale training similarity audit')
    shape = index.metadata['candidate_indices'].shape
    arrays = _verify_files(root, manifest, shape)
    groups_seen = 0
    for row in range(shape[0]):
        n = len(canonical_groups(index.metadata, row))
        order = arrays['group_order'][row]
        _require(np.array_equal(np.sort(order[:n]), np.arange(n)) and (order[n:] == -1).all(),
                 'Similarity order lost or duplicated negative identities')
        groups_seen += n
    _require(audit['negative_groups'] == groups_seen and sha256_file(root / 'manifest.json') == manifest_sha
             and sha256_file(root / 'audit.json') == audit_sha, 'Similarity audit coverage or receipts changed')
    return TrainingSimilarityCache(root, shape), {'directory': str(root), 'manifest_sha256': manifest_sha,
                                                  'audit_sha256': audit_sha, 'source': source, 'files': manifest['files']}


def similarity_source_inputs(receipt):
    root = Path(receipt['directory'])
    inputs = {f'training_similarity_{name}': {'path': str(root / name), 'sha256': value}
              for name, value in (('manifest.json', receipt['manifest_sha256']), ('audit.json', receipt['audit_sha256']))}
    for entry in receipt['files'].values():
        inputs[f'training_similarity_{entry["filename"]}'] = {'path': str(root / entry['filename']), 'sha256': entry['sha256']}
    fp = receipt['source']['fingerprint_cache']
    fp_root = Path(fp['directory'])
    for name, value in (('manifest.json', fp['manifest_sha256']), ('audit.json', fp['audit_sha256']),
                        (fp['file']['filename'], fp['file']['sha256'])):
        inputs[f'training_sampling_fingerprint_{name}'] = {'path': str(fp_root / name), 'sha256': value}
    return inputs


def prepare_training_similarity(data_path, metadata_path, fingerprint_directory, output, *, counts, exclusions,
                                pool_cache_size, radius, bits):
    _require(not Path(output).exists(), 'Refusing an existing training similarity preparation')
    index = load_training_candidates(metadata_path, data_path, counts, exclusions, pool_cache_size=pool_cache_size)
    build_training_similarity_cache(index, fingerprint_directory, output, radius=radius, bits=bits)
    audit_training_similarity_cache(index, output, radius=radius, bits=bits)
    refreshed = load_training_candidates(metadata_path, data_path, counts, exclusions, pool_cache_size=pool_cache_size)
    _require(refreshed.provenance == index.provenance, 'Training sources changed during similarity preparation')
    _, receipt = load_training_similarity_cache(refreshed, output, radius=radius, bits=bits)
    return receipt
