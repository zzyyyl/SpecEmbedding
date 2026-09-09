"""Fingerprint-bound, lossless CPU molecule graph storage; never stores model embeddings."""

import hashlib
import json
import logging
import multiprocessing
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from SpecEmbedding.data import graph_utils
from SpecEmbedding.utils.fulltrain import sha256_file

SCHEMA = 1
DTYPES = {'x': '|u1', 'edge_index': '<i4', 'edge_attr': '|u1', 'graph_size_features': '<f4',
          'node_ptr': '<i8', 'edge_ptr': '<i8'}
_AUDIT_STORE = None


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _save(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def molecule_order_sha256(smiles):
    digest = hashlib.sha256()
    for value in smiles:
        _require(isinstance(value, str) and bool(value), 'Graph cache requires nonempty SMILES strings')
        encoded = value.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return digest.hexdigest()


def graph_cache_provenance(smiles, *, index_sha256, dataset_manifest_sha256):
    _require(len(smiles) > 0, 'Graph cache requires the complete nonempty molecule index')
    for value in (index_sha256, dataset_manifest_sha256):
        _require(isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'),
                 'Invalid graph cache source SHA-256')
    return {'molecules': len(smiles), 'molecule_order_sha256': molecule_order_sha256(smiles),
            'index_sha256': index_sha256, 'dataset_manifest_sha256': dataset_manifest_sha256,
            'graph_policy': 'rdkit_sanitized', 'graph_source_sha256': sha256_file(Path(graph_utils.__file__)),
            'versions': {name: version(name) for name in ('rdkit', 'torch', 'torch-geometric')}}


def _check_options(workers, chunk_size):
    _require(type(workers) is int and workers > 0 and type(chunk_size) is int and chunk_size > 0,
             'Graph cache workers/chunk_size must be positive integers')


def _check_source(smiles, provenance):
    _require(provenance == graph_cache_provenance(smiles, index_sha256=provenance['index_sha256'],
             dataset_manifest_sha256=provenance['dataset_manifest_sha256']), 'Graph cache source/order/version changed')


def _chunks(smiles, chunk_size):
    for start in range(0, len(smiles), chunk_size):
        yield start, smiles[start:start + chunk_size]


def _encode_chunk(item):
    start, smiles = item
    graphs = [graph_utils.smiles_to_graph(s, graph_policy='rdkit_sanitized') for s in smiles]
    nodes = np.array([g.x.shape[0] for g in graphs], dtype='<i8')
    edges = np.array([g.edge_index.shape[1] for g in graphs], dtype='<i8')
    x = torch.cat([g.x for g in graphs]).numpy()
    ei = torch.cat([g.edge_index.t() for g in graphs]).numpy()
    ea = torch.cat([g.edge_attr for g in graphs]).numpy()
    sizes = torch.stack([g.graph_size_features for g in graphs]).numpy()
    _require(x.dtype == np.int64 and x.shape[1] == 6 and (x >= 0).all() and (x <= 255).all(),
             'Graph node features cannot be losslessly packed')
    _require(ea.dtype == np.int64 and ea.shape[1] == 3 and (ea >= 0).all() and (ea <= 255).all(),
             'Graph edge features cannot be losslessly packed')
    _require(ei.dtype == np.int64 and ei.shape[1] == 2 and (ei >= 0).all()
             and (ei <= np.iinfo(np.int32).max).all() and sizes.dtype == np.float32,
             'Graph indices/sizes cannot be losslessly packed')
    return start, nodes, edges, {'x': x.astype('|u1'), 'edge_index': ei.astype('<i4'),
                                'edge_attr': ea.astype('|u1'), 'graph_size_features': sizes}


def build_graph_cache(smiles, root, provenance, *, workers, chunk_size):
    """Build every input graph in source order; partial failures remain unusable and preserved."""
    root = Path(root).resolve()
    _check_options(workers, chunk_size)
    _check_source(smiles, provenance)
    root.mkdir(parents=True, exist_ok=False)
    _save(root / 'inputs.json', {'provenance': provenance, 'workers': workers, 'chunk_size': chunk_size})
    started = time.monotonic()
    node_ptr, edge_ptr = [0], [0]
    with ExitStack() as stack:
        handles = {name: stack.enter_context((root / f'{name}.bin').open('xb')) for name in DTYPES if not name.endswith('_ptr')}
        chunks = _chunks(smiles, chunk_size)
        if workers == 1:
            results = map(_encode_chunk, chunks)
        else:
            pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(workers))
            results = pool.imap(_encode_chunk, chunks, chunksize=1)
        for start, nodes, edges, arrays in results:
            _require(start == len(node_ptr) - 1 and len(nodes) == len(edges), 'Graph cache chunk order/coverage mismatch')
            node_ptr.extend((nodes.cumsum() + node_ptr[-1]).tolist())
            edge_ptr.extend((edges.cumsum() + edge_ptr[-1]).tolist())
            for name, array in arrays.items():
                array.tofile(handles[name])
            if (len(node_ptr) - 1) % 16384 == 0 or len(node_ptr) - 1 == len(smiles):
                logging.info('Built lossless molecule graphs: %s/%s', len(node_ptr) - 1, len(smiles))
    _require(len(node_ptr) == len(smiles) + 1, 'Incomplete molecule graph cache')
    shapes = {'x': [node_ptr[-1], 6], 'edge_index': [edge_ptr[-1], 2], 'edge_attr': [edge_ptr[-1], 3],
              'graph_size_features': [len(smiles), 2], 'node_ptr': [len(node_ptr)], 'edge_ptr': [len(edge_ptr)]}
    for name, array in (('node_ptr', node_ptr), ('edge_ptr', edge_ptr)):
        with (root / f'{name}.bin').open('xb') as handle:
            np.asarray(array, dtype='<i8').tofile(handle)
    files = {name: {'filename': f'{name}.bin', 'dtype': DTYPES[name], 'shape': shapes[name],
                    'bytes': (root / f'{name}.bin').stat().st_size, 'sha256': sha256_file(root / f'{name}.bin')}
             for name in DTYPES}
    manifest = {'schema_version': SCHEMA, 'state': 'built_pending_full_audit', 'provenance': provenance,
                'files': files, 'build_seconds': time.monotonic() - started,
                'cache_source_sha256': sha256_file(Path(__file__))}
    _save(root / 'manifest.json', manifest)
    return manifest


class MoleculeGraphCache:
    """Read verified fixed inputs; each graph is copied before callers can augment/mutate it."""

    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self._arrays = None

    def __len__(self):
        return self.manifest['provenance']['molecules']

    def __getstate__(self):
        return {**self.__dict__, '_arrays': None}

    def _open(self):
        if self._arrays is None:
            self._arrays = {}
            for name, entry in self.manifest['files'].items():
                shape = tuple(entry['shape'])
                self._arrays[name] = (np.memmap(self.root / entry['filename'], dtype=entry['dtype'], mode='r', shape=shape)
                                      if entry['bytes'] else np.empty(shape, dtype=entry['dtype']))
        return self._arrays

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(self):
            raise IndexError(index)
        a = self._open()
        n0, n1 = a['node_ptr'][index:index + 2]
        e0, e1 = a['edge_ptr'][index:index + 2]
        return Data(x=torch.tensor(a['x'][n0:n1], dtype=torch.long),
                    edge_index=torch.tensor(a['edge_index'][e0:e1], dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(a['edge_attr'][e0:e1], dtype=torch.long),
                    graph_size_features=torch.tensor(a['graph_size_features'][index], dtype=torch.float32))


def _verify_files(root, manifest):
    _require(manifest['schema_version'] == SCHEMA and manifest['state'] == 'built_pending_full_audit'
             and manifest['cache_source_sha256'] == sha256_file(Path(__file__)), 'Graph cache schema/source changed')
    _require(set(manifest['files']) == set(DTYPES), 'Graph cache file set changed')
    for name, entry in manifest['files'].items():
        _require(entry['filename'] == f'{name}.bin' and entry['dtype'] == DTYPES[name]
                 and all(type(n) is int and n >= 0 for n in entry['shape']), 'Invalid graph cache file schema')
        path = root / entry['filename']
        _require(path.stat().st_size == entry['bytes'] == int(np.prod(entry['shape'])) * np.dtype(entry['dtype']).itemsize
                 and sha256_file(path) == entry['sha256'], 'Graph cache file fingerprint/size mismatch')
    store = MoleculeGraphCache(root, manifest)
    a = store._open()
    count = len(store)
    _require(a['node_ptr'].shape == a['edge_ptr'].shape == (count + 1,)
             and a['graph_size_features'].shape == (count, 2), 'Graph cache query dimensions changed')
    _require(a['x'].shape[1:] == (6,) and a['edge_index'].shape[1:] == (2,) and a['edge_attr'].shape[1:] == (3,),
             'Graph cache feature dimensions changed')
    _require(a['node_ptr'][0] == a['edge_ptr'][0] == 0 and (np.diff(a['node_ptr']) > 0).all()
             and (np.diff(a['edge_ptr']) >= 0).all() and a['node_ptr'][-1] == len(a['x'])
             and a['edge_ptr'][-1] == len(a['edge_index']) == len(a['edge_attr']), 'Invalid graph cache offsets')
    return store


def _init_audit(root, manifest):
    global _AUDIT_STORE
    _AUDIT_STORE = MoleculeGraphCache(root, manifest)


def _audit_chunk(item):
    start, smiles = item
    for offset, value in enumerate(smiles):
        observed = _AUDIT_STORE[start + offset]
        expected = graph_utils.smiles_to_graph(value, graph_policy='rdkit_sanitized')
        for name in ('x', 'edge_index', 'edge_attr', 'graph_size_features'):
            _require(torch.equal(observed[name], expected[name]) and observed[name].dtype == expected[name].dtype,
                     f'Cached graph differs from fresh graph at molecule {start + offset}: {name}')
    return start, len(smiles)


def audit_graph_cache(smiles, root, provenance, *, workers, chunk_size):
    """Independently rebuild and compare every graph, after reopening the saved cache."""
    root = Path(root).resolve()
    _check_options(workers, chunk_size)
    _check_source(smiles, provenance)
    _require(not (root / 'audit.json').exists(), 'Refusing an existing graph audit receipt')
    manifest_sha = sha256_file(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    _require(manifest['provenance'] == provenance, 'Graph cache input changed')
    _verify_files(root, manifest)
    begin, seen = time.monotonic(), 0
    with ExitStack() as stack:
        if workers == 1:
            _init_audit(root, manifest)
            results = map(_audit_chunk, _chunks(smiles, chunk_size))
        else:
            pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(workers, initializer=_init_audit,
                                                                              initargs=(root, manifest)))
            results = pool.imap(_audit_chunk, _chunks(smiles, chunk_size), chunksize=1)
        for start, count in results:
            _require(start == seen, 'Graph audit order changed')
            seen += count
            if seen % 16384 == 0 or seen == len(smiles):
                logging.info('Audited every tensor against fresh molecule graphs: %s/%s', seen, len(smiles))
    _require(seen == len(smiles) and sha256_file(root / 'manifest.json') == manifest_sha, 'Graph audit input/coverage changed')
    _verify_files(root, manifest)
    result = {'state': 'complete_full_graph_tensor_audit', 'manifest_sha256': manifest_sha,
              'provenance': provenance, 'audited_molecules': seen, 'audit_seconds': time.monotonic() - begin,
              'tensor_comparison': 'Exact dtype, shape and torch.equal for every molecule and all four tensors'}
    _save(root / 'audit.json', result)
    return result


def load_graph_cache(smiles, root, provenance):
    root = Path(root).resolve()
    _check_source(smiles, provenance)
    manifest_sha, audit_sha = sha256_file(root / 'manifest.json'), sha256_file(root / 'audit.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    _require(manifest['provenance'] == audit['provenance'] == provenance and audit['manifest_sha256'] == manifest_sha
             and audit['state'] == 'complete_full_graph_tensor_audit' and audit['audited_molecules'] == len(smiles),
             'Missing, stale or incomplete molecule graph audit')
    store = _verify_files(root, manifest)
    _require(sha256_file(root / 'manifest.json') == manifest_sha and sha256_file(root / 'audit.json') == audit_sha,
             'Graph cache receipts changed during verification')
    return store, {'directory': str(root), 'manifest_sha256': manifest_sha, 'audit_sha256': audit_sha,
                   'provenance': provenance, 'files': manifest['files']}
