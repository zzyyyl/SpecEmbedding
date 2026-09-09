"""Complete, audited, input-only Morgan fingerprints; never use bits as identity labels."""

import hashlib
import json
import logging
import multiprocessing
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from SpecEmbedding.utils.fulltrain import sha256_file

SCHEMA = 1
_GENERATOR = None
_OPTIONS = None
_AUDIT_BITS = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def fingerprint_options(radius, bits):
    _require(type(radius) is int and radius >= 0, 'Fingerprint radius must be a nonnegative integer')
    _require(type(bits) is int and bits > 0 and bits % 8 == 0, 'Fingerprint bits must be a positive multiple of eight')
    return {'radius': radius, 'bits': bits, 'count_simulation': False, 'include_chirality': False,
            'use_bond_types': True, 'include_ring_membership': True, 'include_redundant_environments': False,
            'bit_order': 'big', 'graph_policy': 'rdkit_sanitized'}


def fingerprint_provenance(smiles, *, index_sha256, dataset_manifest_sha256, radius, bits):
    _require(len(smiles) > 0, 'Fingerprints require the complete nonempty molecule index')
    options = fingerprint_options(radius, bits)
    for value in (index_sha256, dataset_manifest_sha256):
        _require(isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'),
                 'Invalid fingerprint source SHA-256')
    digest = hashlib.sha256()
    for text in smiles:
        _require(isinstance(text, str) and bool(text), 'Fingerprint source contains an empty SMILES')
        encoded = text.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return {'molecules': len(smiles), 'molecule_order_sha256': digest.hexdigest(), 'options': options,
            'index_sha256': index_sha256, 'dataset_manifest_sha256': dataset_manifest_sha256,
            'versions': {name: version(name) for name in ('rdkit', 'numpy')}}


def _check_source(smiles, provenance):
    _require(provenance == fingerprint_provenance(smiles, index_sha256=provenance['index_sha256'],
             dataset_manifest_sha256=provenance['dataset_manifest_sha256'],
             radius=provenance['options']['radius'], bits=provenance['options']['bits']),
             'Fingerprint source, order, settings or software version changed')


def _check_workers(workers, chunk_size):
    _require(type(workers) is int and workers > 0 and type(chunk_size) is int and chunk_size > 0,
             'Fingerprint workers/chunk_size must be positive integers')


def _chunks(smiles, size):
    for start in range(0, len(smiles), size):
        yield start, smiles[start:start + size]


def _init_worker(options, audit_file=None, count=None):
    global _GENERATOR, _OPTIONS, _AUDIT_BITS
    _OPTIONS = options
    _GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=options['radius'], fpSize=options['bits'], countSimulation=False, includeChirality=False,
        useBondTypes=True, includeRingMembership=True, includeRedundantEnvironments=False)
    _AUDIT_BITS = (np.memmap(audit_file, mode='r', dtype=np.uint8, shape=(count, options['bits'] // 8))
                   if audit_file is not None else None)


def _molecule(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    _require(mol is not None and mol.GetNumAtoms() > 0, f'Invalid fingerprint molecule: {smiles}')
    return mol


def _encode_chunk(item):
    start, smiles = item
    array = np.empty((len(smiles), _OPTIONS['bits'] // 8), dtype=np.uint8)
    for i, value in enumerate(smiles):
        bits = _GENERATOR.GetFingerprintAsNumPy(_molecule(value))
        _require(bits.shape == (_OPTIONS['bits'],) and np.isin(bits, [0, 1]).all(),
                 'Unexpected Morgan bit vector')
        array[i] = np.packbits(bits, bitorder='big')
    return start, array


def build_fingerprint_cache(smiles, root, provenance, *, workers, chunk_size):
    root = Path(root).resolve()
    source_sha = sha256_file(Path(__file__))
    _check_workers(workers, chunk_size)
    _check_source(smiles, provenance)
    root.mkdir(parents=True, exist_ok=False)
    _save(root / 'inputs.json', {'provenance': provenance, 'workers': workers, 'chunk_size': chunk_size})
    started, seen = time.monotonic(), 0
    with ExitStack() as stack:
        handle = stack.enter_context((root / 'fingerprints.bin').open('xb'))
        chunks = _chunks(smiles, chunk_size)
        if workers == 1:
            _init_worker(provenance['options'])
            results = map(_encode_chunk, chunks)
        else:
            pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(
                workers, initializer=_init_worker, initargs=(provenance['options'],)))
            results = pool.imap(_encode_chunk, chunks, chunksize=1)
        for start, bits in results:
            _require(start == seen and bits.dtype == np.uint8 and bits.ndim == 2
                     and bits.shape[1] == provenance['options']['bits'] // 8, 'Fingerprint chunk order/shape changed')
            bits.tofile(handle)
            seen += len(bits)
            if seen % 65536 == 0 or seen == len(smiles):
                logging.info('Built fixed fingerprints: %s/%s', seen, len(smiles))
    _require(seen == len(smiles), 'Incomplete fingerprint input coverage')
    _check_source(smiles, provenance)
    _require(sha256_file(Path(__file__)) == source_sha, 'Fingerprint implementation changed during build')
    data = root / 'fingerprints.bin'
    manifest = {'schema_version': SCHEMA, 'state': 'built_pending_full_audit', 'provenance': provenance,
                'file': {'filename': data.name, 'dtype': '|u1', 'shape': [len(smiles), provenance['options']['bits'] // 8],
                         'bytes': data.stat().st_size, 'sha256': sha256_file(data)},
                'cache_source_sha256': source_sha, 'build_seconds': time.monotonic() - started}
    _save(root / 'manifest.json', manifest)
    return manifest


class FingerprintCache:
    """Memory map packed input bits, returning fresh float32 arrays for requested rows only."""

    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self._bits = None

    def __len__(self):
        return self.manifest['provenance']['molecules']

    def __getstate__(self):
        return {**self.__dict__, '_bits': None}

    def _open(self):
        if self._bits is None:
            self._bits = np.memmap(self.root / 'fingerprints.bin', mode='r', dtype=np.uint8,
                                   shape=tuple(self.manifest['file']['shape']))
        return self._bits

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(self):
            raise IndexError(index)
        return self.get_many(np.asarray([index], dtype=np.int64))[0]

    def get_many(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError('Fingerprint indices must be a one-dimensional integer array within bounds')
        return np.unpackbits(self._open()[indices], axis=1, bitorder='big').astype(np.float32)


def _verify_files(root, manifest):
    _require(manifest['schema_version'] == SCHEMA and manifest['state'] == 'built_pending_full_audit'
             and manifest['cache_source_sha256'] == sha256_file(Path(__file__)), 'Fingerprint cache schema/source changed')
    p = manifest['provenance']
    entry = manifest['file']
    shape = [p['molecules'], p['options']['bits'] // 8]
    _require(entry['filename'] == 'fingerprints.bin' and entry['dtype'] == '|u1'
             and entry['shape'] == shape and all(type(n) is int and n > 0 for n in shape),
             'Invalid fingerprint cache file schema')
    data = root / 'fingerprints.bin'
    _require(data.stat().st_size == entry['bytes'] == shape[0] * shape[1]
             and sha256_file(data) == entry['sha256'], 'Fingerprint cache bytes/size mismatch')
    return FingerprintCache(root, manifest)


def _audit_chunk(item):
    start, smiles = item
    # Use the independent legacy bit-vector API and compare unpacked bits, not a checksum alone.
    with rdBase.BlockLogs():
        for offset, text in enumerate(smiles):
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                _molecule(text), _OPTIONS['radius'], nBits=_OPTIONS['bits'], useChirality=False,
                useBondTypes=True, useFeatures=False, includeRedundantEnvironments=False)
            expected = np.empty(_OPTIONS['bits'], dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, expected)
            observed = np.unpackbits(_AUDIT_BITS[start + offset], bitorder='big')
            _require(np.array_equal(observed, expected), f'Cached fingerprint differs from fresh bits at {start + offset}')
    return start, len(smiles)


def audit_fingerprint_cache(smiles, root, provenance, *, workers, chunk_size):
    root = Path(root).resolve()
    _check_workers(workers, chunk_size)
    _check_source(smiles, provenance)
    _require(not (root / 'audit.json').exists(), 'Refusing an existing fingerprint audit receipt')
    manifest_sha = sha256_file(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    _require(manifest['provenance'] == provenance, 'Fingerprint input provenance differs')
    _verify_files(root, manifest)
    started, seen = time.monotonic(), 0
    initargs = (provenance['options'], str(root / 'fingerprints.bin'), len(smiles))
    with ExitStack() as stack:
        chunks = _chunks(smiles, chunk_size)
        if workers == 1:
            _init_worker(*initargs)
            results = map(_audit_chunk, chunks)
        else:
            pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(
                workers, initializer=_init_worker, initargs=initargs))
            results = pool.imap(_audit_chunk, chunks, chunksize=1)
        for start, count in results:
            _require(start == seen, 'Fingerprint audit order changed')
            seen += count
            if seen % 65536 == 0 or seen == len(smiles):
                logging.info('Independently audited complete fingerprint bits: %s/%s', seen, len(smiles))
    _require(seen == len(smiles) and sha256_file(root / 'manifest.json') == manifest_sha,
             'Fingerprint audit coverage or input changed')
    _check_source(smiles, provenance)
    _verify_files(root, manifest)
    result = {'state': 'complete_full_fingerprint_bit_audit', 'provenance': provenance,
              'manifest_sha256': manifest_sha, 'audited_molecules': seen, 'audit_seconds': time.monotonic() - started,
              'comparison': 'Every unpacked bit versus fresh legacy Morgan API; fingerprint equality is not an identity label'}
    _save(root / 'audit.json', result)
    return result


def load_fingerprint_cache(smiles, root, provenance):
    root = Path(root).resolve()
    _check_source(smiles, provenance)
    manifest_sha = sha256_file(root / 'manifest.json')
    audit_sha = sha256_file(root / 'audit.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    _require(manifest['provenance'] == audit['provenance'] == provenance
             and audit['manifest_sha256'] == manifest_sha and audit['audited_molecules'] == len(smiles)
             and audit['state'] == 'complete_full_fingerprint_bit_audit', 'Missing, stale or incomplete fingerprint audit')
    store = _verify_files(root, manifest)
    _require(sha256_file(root / 'manifest.json') == manifest_sha and sha256_file(root / 'audit.json') == audit_sha,
             'Fingerprint receipts changed during loading')
    return store, {'directory': str(root), 'manifest_sha256': manifest_sha, 'audit_sha256': audit_sha,
                   'provenance': provenance, 'file': manifest['file']}
