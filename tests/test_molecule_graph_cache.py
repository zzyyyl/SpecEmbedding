import json
import pickle

import numpy as np
import pytest
import torch

from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.molecule_graph_cache import (
    audit_graph_cache,
    build_graph_cache,
    graph_cache_provenance,
    load_graph_cache,
)

SMILES = ['CCO', 'OCC', 'c1ccccc1', 'C.C', '[Na+]', 'C[N+](C)(C)C', 'C[C@H](O)C(=O)O']


def prepare(tmp_path, *, workers=1, smiles=SMILES, audit=True):
    root = tmp_path / 'graphs'
    source = graph_cache_provenance(smiles, index_sha256='a'*64, dataset_manifest_sha256='b'*64)
    build_graph_cache(smiles, root, source, workers=workers, chunk_size=2)
    if audit:
        audit_graph_cache(smiles, root, source, workers=workers, chunk_size=3)
    return root, source


@pytest.mark.parametrize('workers', [1, 2])
def test_exact_all_tensor_roundtrip_and_worker_reopen(tmp_path, workers):
    root, source = prepare(tmp_path, workers=workers)
    cache, receipt = load_graph_cache(SMILES, root, source)
    assert len(cache) == len(SMILES)
    assert receipt['provenance'] == source
    for i, smiles in enumerate(SMILES):
        fresh = smiles_to_graph(smiles, graph_policy='rdkit_sanitized')
        for name in ('x', 'edge_index', 'edge_attr', 'graph_size_features'):
            assert cache[i][name].dtype == fresh[name].dtype
            assert torch.equal(cache[i][name], fresh[name])
    cache[0].x.fill_(200)
    assert torch.equal(cache[0].x, smiles_to_graph('CCO', graph_policy='rdkit_sanitized').x)
    reopened = pickle.loads(pickle.dumps(cache))
    assert reopened._arrays is None
    assert torch.equal(reopened[2].edge_index, cache[2].edge_index)
    with pytest.raises(FileExistsError):
        build_graph_cache(SMILES, root, source, workers=1, chunk_size=2)
    with pytest.raises(ValueError, match='existing graph audit'):
        audit_graph_cache(SMILES, root, source, workers=1, chunk_size=2)
    for index in [-1, len(cache), 1.0, True]:
        with pytest.raises(IndexError):
            cache[index]


def test_all_edgeless_molecules_and_compact_storage(tmp_path):
    smiles = ['[Na+]', 'C', 'C.C']
    root, source = prepare(tmp_path, smiles=smiles)
    cache, receipt = load_graph_cache(smiles, root, source)
    assert receipt['files']['edge_index']['bytes'] == receipt['files']['edge_attr']['bytes'] == 0
    for graph in (cache[i] for i in range(len(cache))):
        assert graph.edge_index.shape == (2, 0) and graph.edge_attr.shape == (0, 3)
    assert cache.manifest['files']['x']['dtype'] == '|u1'
    assert cache.manifest['files']['edge_index']['dtype'] == '<i4'


def test_no_loading_before_independent_audit(tmp_path):
    root, source = prepare(tmp_path, audit=False)
    with pytest.raises(FileNotFoundError):
        load_graph_cache(SMILES, root, source)


@pytest.mark.parametrize('change', ['order', 'index', 'dataset', 'graph_version', 'audit_count', 'audit_manifest'])
def test_changed_source_or_receipt_rejected(tmp_path, change):
    root, source = prepare(tmp_path)
    smiles = SMILES
    if change == 'order':
        smiles = list(reversed(SMILES))
        source = graph_cache_provenance(smiles, index_sha256='a'*64, dataset_manifest_sha256='b'*64)
    elif change == 'index':
        source['index_sha256'] = 'c'*64
    elif change == 'dataset':
        source['dataset_manifest_sha256'] = 'c'*64
    elif change == 'graph_version':
        source['versions']['rdkit'] = 'changed'
    else:
        path = root / 'audit.json'
        receipt = json.loads(path.read_text())
        if change == 'audit_count':
            receipt['audited_molecules'] -= 1
        else:
            receipt['manifest_sha256'] = 'c'*64
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_graph_cache(smiles, root, source)


@pytest.mark.parametrize('change', ['append', 'truncate', 'tensor_value'])
def test_changed_array_bytes_rejected(tmp_path, change):
    root, source = prepare(tmp_path)
    path = root / 'x.bin'
    raw = path.read_bytes()
    path.write_bytes(raw + b'changed' if change == 'append' else raw[:-1] if change == 'truncate' else bytes([255]) + raw[1:])
    with pytest.raises(ValueError, match='fingerprint/size'):
        load_graph_cache(SMILES, root, source)


@pytest.mark.parametrize('change', ['tensor_value', 'offsets', 'filename', 'dtype'])
def test_independent_audit_rejects_bad_data_even_with_updated_file_fingerprints(tmp_path, change):
    root, source = prepare(tmp_path, audit=False)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if change == 'tensor_value':
        path = root / 'x.bin'
        raw = path.read_bytes()
        path.write_bytes(bytes([255]) + raw[1:])
        manifest['files']['x']['sha256'] = sha256_file(path)
    elif change == 'offsets':
        path = root / 'node_ptr.bin'
        offsets = np.fromfile(path, dtype='<i8')
        offsets[1] = 0
        offsets.tofile(path)
        manifest['files']['node_ptr']['sha256'] = sha256_file(path)
    elif change == 'filename':
        manifest['files']['x']['filename'] = '../x.bin'
    else:
        manifest['files']['x']['dtype'] = '<i8'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit_graph_cache(SMILES, root, source, workers=1, chunk_size=2)
    assert not (root / 'audit.json').exists()


def test_invalid_graph_preserves_partial_files_without_completion_manifest(tmp_path):
    smiles = ['CCO', 'this is not a molecule']
    source = graph_cache_provenance(smiles, index_sha256='a'*64, dataset_manifest_sha256='b'*64)
    root = tmp_path / 'bad'
    with pytest.raises(ValueError):
        build_graph_cache(smiles, root, source, workers=1, chunk_size=1)
    assert (root / 'inputs.json').exists() and not (root / 'manifest.json').exists()


def test_invalid_preparation_options_rejected_before_outputs(tmp_path):
    source = graph_cache_provenance(SMILES, index_sha256='a'*64, dataset_manifest_sha256='b'*64)
    for workers, chunk_size in [(0, 2), (1, 0), (True, 2), (1, 1.5)]:
        with pytest.raises(ValueError):
            build_graph_cache(SMILES, tmp_path/'unused', source, workers=workers, chunk_size=chunk_size)
        assert not (tmp_path/'unused').exists()
