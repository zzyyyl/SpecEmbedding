import json
import pickle

import numpy as np
import pytest
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader

from SpecEmbedding.utils.fingerprint_cache import (
    audit_fingerprint_cache,
    build_fingerprint_cache,
    fingerprint_options,
    fingerprint_provenance,
    load_fingerprint_cache,
)
from SpecEmbedding.utils.fulltrain import sha256_file

SMILES = ['CCO', 'OCC', 'C.C', '[Na+]', 'c1ccccc1', 'C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O']


def prepare(tmp_path, *, workers=1, audit=True, smiles=SMILES):
    root = tmp_path / 'fingerprints'
    source = fingerprint_provenance(smiles, index_sha256='a'*64, dataset_manifest_sha256='b'*64, radius=2, bits=2048)
    build_fingerprint_cache(smiles, root, source, workers=workers, chunk_size=2)
    if audit:
        audit_fingerprint_cache(smiles, root, source, workers=workers, chunk_size=3)
    return root, source


@pytest.mark.parametrize('workers', [1, 2])
def test_all_bits_and_dataloader_roundtrip(tmp_path, workers):
    root, source = prepare(tmp_path, workers=workers)
    cache, receipt = load_fingerprint_cache(SMILES, root, source)
    assert receipt['file']['bytes'] == len(SMILES) * 256
    expected = []
    with rdBase.BlockLogs():
        for smiles in SMILES:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048, useChirality=False)
            value = np.empty(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, value)
            expected.append(value)
    expected = np.stack(expected).astype(np.float32)
    batches = list(DataLoader(cache, batch_size=3, num_workers=workers - 1))
    assert np.array_equal(np.concatenate(batches), expected)
    assert np.array_equal(cache.get_many(np.array([5, 1, 5])), expected[[5, 1, 5]])
    assert cache.get_many(np.array([], dtype=np.int64)).shape == (0, 2048)
    assert cache[0].dtype == np.float32
    cache[0].fill(3)
    assert np.array_equal(cache[0], expected[0])
    reopened = pickle.loads(pickle.dumps(cache))
    assert reopened._bits is None and np.array_equal(reopened[0], expected[0])
    assert np.array_equal(cache[0], cache[1])  # SMILES aliases remain separate inventory rows.
    assert np.array_equal(cache[5], cache[6])  # Explicitly achiral representation, no labels created here.
    with pytest.raises(FileExistsError):
        build_fingerprint_cache(SMILES, root, source, workers=1, chunk_size=2)
    with pytest.raises(ValueError, match='existing fingerprint audit'):
        audit_fingerprint_cache(SMILES, root, source, workers=1, chunk_size=2)


@pytest.mark.parametrize('bad', [-1, 7, 1.5, True, np.bool_(True)])
def test_invalid_indices_rejected(tmp_path, bad):
    root, source = prepare(tmp_path)
    cache, _ = load_fingerprint_cache(SMILES, root, source)
    with pytest.raises(IndexError):
        cache[bad]
    with pytest.raises(IndexError):
        cache.get_many(np.asarray([bad]))


@pytest.mark.parametrize('radius,bits', [(-1, 2048), (True, 2048), (2, 0), (2, 2049), (2, True)])
def test_invalid_parameters_rejected(radius, bits):
    with pytest.raises(ValueError):
        fingerprint_options(radius, bits)


def test_no_load_until_independent_audit(tmp_path):
    root, source = prepare(tmp_path, audit=False)
    with pytest.raises(FileNotFoundError):
        load_fingerprint_cache(SMILES, root, source)


@pytest.mark.parametrize('change', ['order', 'index', 'dataset', 'version', 'radius', 'count', 'manifest', 'source'])
def test_source_and_audit_changes_rejected(tmp_path, change):
    root, source = prepare(tmp_path)
    smiles = SMILES
    if change == 'order':
        smiles = list(reversed(SMILES))
    elif change in ['index', 'dataset']:
        source['index_sha256' if change == 'index' else 'dataset_manifest_sha256'] = 'c'*64
    elif change == 'version':
        source['versions']['rdkit'] = 'changed'
    elif change == 'radius':
        source['options']['radius'] = 3
    elif change == 'source':
        path = root / 'manifest.json'
        saved = json.loads(path.read_text())
        saved['cache_source_sha256'] = 'c'*64
        path.write_text(json.dumps(saved))
    else:
        path = root / 'audit.json'
        saved = json.loads(path.read_text())
        if change == 'count':
            saved['audited_molecules'] -= 1
        else:
            saved['manifest_sha256'] = 'c'*64
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_fingerprint_cache(smiles, root, source)


@pytest.mark.parametrize('change', ['flip', 'truncate', 'append'])
def test_damaged_packed_data_rejected(tmp_path, change):
    root, source = prepare(tmp_path)
    path = root / 'fingerprints.bin'
    value = path.read_bytes()
    path.write_bytes(bytes([value[0] ^ 1]) + value[1:] if change == 'flip' else value[:-1] if change == 'truncate' else value + b'0')
    with pytest.raises(ValueError, match='bytes/size'):
        load_fingerprint_cache(SMILES, root, source)


def test_full_bit_reconstruction_catches_corruption_with_updated_hash(tmp_path):
    root, source = prepare(tmp_path, audit=False)
    path = root / 'fingerprints.bin'
    value = path.read_bytes()
    path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['file']['sha256'] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='differs from fresh bits'):
        audit_fingerprint_cache(SMILES, root, source, workers=1, chunk_size=2)
    assert not (root / 'audit.json').exists()


def test_invalid_molecule_retains_unusable_partial_output(tmp_path):
    smiles = ['CCO', 'not-a-smiles']
    source = fingerprint_provenance(smiles, index_sha256='a'*64, dataset_manifest_sha256='b'*64, radius=2, bits=2048)
    root = tmp_path / 'bad'
    with pytest.raises(ValueError, match='Invalid fingerprint molecule'):
        build_fingerprint_cache(smiles, root, source, workers=1, chunk_size=1)
    assert (root / 'inputs.json').exists() and (root / 'fingerprints.bin').exists()
    assert not (root / 'manifest.json').exists()


def test_fingerprint_collision_does_not_merge_distinct_molecules(tmp_path):
    smiles = ['CCCCCCCCCCCC', 'CCCCCCCCCCCCC']
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles[0]))[:14] != Chem.MolToInchiKey(Chem.MolFromSmiles(smiles[1]))[:14]
    root, source = prepare(tmp_path, smiles=smiles)
    cache, receipt = load_fingerprint_cache(smiles, root, source)
    assert len(cache) == 2 and np.array_equal(cache[0], cache[1])
    assert 'positive_mask' not in receipt and 'labels' not in receipt


def test_preparation_preserves_random_streams(tmp_path):
    import random

    import torch

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    root, source = prepare(tmp_path)
    cache, _ = load_fingerprint_cache(SMILES, root, source)
    cache.get_many(np.arange(len(cache)))
    after = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
