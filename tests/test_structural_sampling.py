import copy
import json
import pickle
import random
from fractions import Fraction

import numpy as np
import pytest
import torch
from rdkit import Chem, DataStructs

from SpecEmbedding.utils.candidate_training import validate_candidate_sampling_binding, validate_candidate_settings
from SpecEmbedding.utils.fingerprint_cache import (
    audit_fingerprint_cache,
    build_fingerprint_cache,
    fingerprint_provenance,
)
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.structural_sampling import (
    StructuralTrainingCandidateIndex,
    load_structural_training_candidates,
    reference_structural_sample,
)
from SpecEmbedding.utils.training_candidates import TrainingCandidateIndex, validate_training_candidate_metadata
from SpecEmbedding.utils.training_similarity import (
    audit_training_similarity_cache,
    build_training_similarity_cache,
    canonical_groups,
    exact_group_order,
    load_training_similarity_cache,
)


def source_index():
    smiles = ['CCO', 'OCC', 'C(C)', *['C' * n for n in range(2, 44)]]
    keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split('-')[0] for s in smiles]
    targets = ['CCO', 'CC', 'CCC', 'CCCCCCCC']
    pools = [list(range(len(smiles))), [3, 2, 0, 1, 4], [4], [9, 10, 11]]
    ids = np.full((4, len(smiles)), -1, dtype=np.int32)
    positions = np.full_like(ids, -1, dtype=np.int16)
    labels = np.zeros_like(ids, dtype=np.bool_)
    target_keys = [keys[smiles.index(s)] for s in targets]
    for row, pool in enumerate(pools):
        ids[row, :len(pool)] = pool
        positions[row, :len(pool)] = np.arange(len(pool))
        labels[row, :len(pool)] = [keys[i] == target_keys[row] for i in pool]
    rows = np.array([0, 1, 0, 2, 0, 1, 3], dtype=np.int32)
    metadata = {'schema_version': 1, 'kind': 'diagnostic_candidate_metadata', 'split': 'train',
                'candidate_type': 'mass', 'forcing': False, 'graph_policy': 'rdkit_sanitized',
                'mol_smiles': smiles, 'mol_identity_2d': keys, 'target_smiles': targets,
                'target_identity_2d': target_keys, 'candidate_indices': ids, 'source_positions': positions,
                'positive_mask': labels, 'query_target_rows': rows, 'raw_query_indices': np.arange(len(rows), dtype=np.int32),
                'graph_rejections': []}
    raw = [{'smiles': targets[row], 'identity_2d': target_keys[row]} for row in rows]
    source = {target: [smiles[i] for i in pool] for target, pool in zip(targets, pools, strict=True)}
    observed = validate_training_candidate_metadata(metadata, raw, source, {}, len(raw))
    return TrainingCandidateIndex(metadata, {'sha256': 'a' * 64, 'dataset_manifest_sha256': 'b' * 64,
                                             'observed': observed}, pool_cache_size=1)


def prepare(tmp_path, *, audit=True):
    index = source_index()
    fp = tmp_path / 'fp'
    provenance = fingerprint_provenance(index.metadata['mol_smiles'], index_sha256='a'*64,
                                        dataset_manifest_sha256='b'*64, radius=2, bits=2048)
    build_fingerprint_cache(index.metadata['mol_smiles'], fp, provenance, workers=1, chunk_size=8)
    audit_fingerprint_cache(index.metadata['mol_smiles'], fp, provenance, workers=1, chunk_size=7)
    root = tmp_path / 'similarity'
    build_training_similarity_cache(index, fp, root, radius=2, bits=2048)
    if audit:
        audit_training_similarity_cache(index, root, radius=2, bits=2048)
    settings = {'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32,
                'fingerprint_radius': 2, 'fingerprint_bits': 2048, 'cache_directory': str(root)}
    return index, settings, root, fp


def test_complete_cache_bits_group_order_and_original_labels(tmp_path):
    base, settings, root, fp = prepare(tmp_path)
    cache, receipt = load_training_similarity_cache(base, root, radius=2, bits=2048)
    assert receipt['source']['queries'] == 7 and receipt['source']['target_rows'] == 4
    raw = np.fromfile(fp / 'fingerprints.bin', dtype=np.uint8).reshape(-1, 256)
    intersection = np.load(root / 'intersection.npy')
    union = np.load(root / 'union.npy')
    meta = base.metadata
    for row, target in enumerate(meta['target_smiles']):
        target_bits = DataStructs.CreateFromBitString(''.join(map(str, np.unpackbits(raw[meta['mol_smiles'].index(target)]))))
        for col, mol in enumerate(meta['candidate_indices'][row]):
            if mol < 0:
                assert intersection[row, col] == union[row, col] == 65535
                continue
            bits = DataStructs.CreateFromBitString(''.join(map(str, np.unpackbits(raw[mol]))))
            observed = int(intersection[row, col]) / int(union[row, col]) if union[row, col] else 0
            assert observed == DataStructs.TanimotoSimilarity(target_bits, bits)
        assert sorted(cache.row(row)[cache.row(row) >= 0].tolist()) == list(range(len(canonical_groups(meta, row))))
    assert (cache.row(2) == -1).all()  # Query retained despite no negatives.
    np.testing.assert_array_equal(intersection[3, :3], union[3, :3])
    assert len(canonical_groups(meta, 3)) == 2  # Identical fingerprints remain distinct negative labels.
    restored = pickle.loads(pickle.dumps(cache))
    assert restored._order is None and np.array_equal(restored.row(0), cache.row(0))
    assert not restored.row(0).flags.writeable
    with pytest.raises(ValueError, match='existing'):
        build_training_similarity_cache(base, fp, root, radius=2, bits=2048)
    with pytest.raises(ValueError, match='existing'):
        audit_training_similarity_cache(base, root, radius=2, bits=2048)


@pytest.mark.parametrize('count', [6, 7, 23])
def test_exact_ties_ignore_alias_count_and_source_order(count):
    # The full D22 audit exposed this case with floating-point means.
    keys = ['positive'] + [f'H{i:02}' for i in range(31)] + ['A'] + ['Z'] * count
    meta = {'target_identity_2d': ['positive'], 'mol_identity_2d': keys,
            'candidate_indices': np.arange(len(keys), dtype=np.int32)[None],
            'source_positions': np.arange(len(keys), dtype=np.int16)[None],
            'positive_mask': np.array([[True] + [False] * (len(keys)-1)])}
    numer = np.array([18] * 32 + [17] * (count+1))
    denom = np.full(len(keys), 18)
    groups = canonical_groups(meta, 0)
    near = [groups[i][0] for i in exact_group_order(meta, 0, numer, denom)[:32]]
    assert 'A' in near and 'Z' not in near
    assert Fraction(17, 18) == sum([Fraction(17, 18)] * count) / count
    permutation = np.arange(len(keys))[::-1]
    permuted = {**meta, **{name: meta[name][:, permutation] for name in
                         ('candidate_indices', 'source_positions', 'positive_mask')}}
    reordered_groups = canonical_groups(permuted, 0)
    reordered = exact_group_order(permuted, 0, numer[permutation], denom[permutation])
    assert [reordered_groups[i][0] for i in reordered[:32]] == near


def test_mixed_sampler_matches_independent_replay_and_preserves_rng(tmp_path):
    base, settings, _, _ = prepare(tmp_path)
    original = copy.deepcopy(base.metadata)
    index = load_structural_training_candidates(base, settings)
    py, numpy, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    varied = set()
    for epoch in (1, 2, 3, 17):
        for query in (6, 4, 2, 1, 3, 5, 0):
            got = index.sample(query, negative_count=16, seed=42, epoch=epoch)
            expected = reference_structural_sample(index, query, negative_count=16, seed=42, epoch=epoch)
            assert got.identity_2d == expected.identity_2d
            np.testing.assert_array_equal(got.molecule_indices, expected.molecule_indices)
            np.testing.assert_array_equal(got.source_positions, expected.source_positions)
            row = int(base.metadata['query_target_rows'][query])
            groups = canonical_groups(base.metadata, row)
            assert len(got.identity_2d) == len(set(got.identity_2d)) == min(16, len(groups))
            assert base.metadata['target_identity_2d'][row] not in got.identity_2d
            near = {groups[i][0] for i in index.cache.row(row)[:min(32, len(groups))]}
            assert len(set(got.identity_2d) & near) >= min(8, len(groups))
            if query == 0:
                varied.add(tuple(got.molecule_indices))
            assert not got.molecule_indices.flags.writeable
    assert len(varied) == 4 and len(index._structural_pool_cache) == 1
    assert py == random.getstate() and torch.equal(torch_rng, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(numpy, np.random.get_state(), strict=True))
    for name in ('candidate_indices', 'source_positions', 'positive_mask', 'query_target_rows', 'raw_query_indices'):
        np.testing.assert_array_equal(base.metadata[name], original[name])
    # The legacy instance still uses the exact original uniform sampler.
    untouched = source_index()
    for query in range(len(base)):
        np.testing.assert_array_equal(base.sample(query, negative_count=16, seed=42, epoch=1).molecule_indices,
                                      untouched.sample(query, negative_count=16, seed=42, epoch=1).molecule_indices)
    # Calling the inherited helper cannot corrupt the structural sampler's different cached representation.
    index._negative_groups(0)
    np.testing.assert_array_equal(index.sample(0, negative_count=16, seed=42, epoch=1).molecule_indices,
                                  reference_structural_sample(index, 0, negative_count=16, seed=42, epoch=1).molecule_indices)


@pytest.mark.parametrize('damage', ['metadata', 'labels', 'query_order', 'radius', 'bits', 'payload', 'missing_audit'])
def test_loader_rejects_stale_missing_or_mismatched_sources(tmp_path, damage):
    base, _, root, _ = prepare(tmp_path, audit=damage != 'missing_audit')
    radius, bits = 2, 2048
    if damage == 'metadata':
        base.metadata['mol_identity_2d'][5] = 'X' * 14
    elif damage in ('labels', 'query_order'):
        key = 'positive_mask' if damage == 'labels' else 'query_target_rows'
        value = base.metadata[key].copy()
        value.flat[0] = not value.flat[0] if damage == 'labels' else 1
        base.metadata[key] = value
    elif damage == 'radius':
        radius = 1
    elif damage == 'bits':
        bits = 1024
    elif damage == 'payload':
        with (root / 'group_order.npy').open('ab') as handle:
            handle.write(b'changed')
    with pytest.raises((ValueError, FileNotFoundError)):
        load_training_similarity_cache(base, root, radius=radius, bits=bits)


@pytest.mark.parametrize('name', ['intersection', 'group_order'])
def test_independent_audit_rejects_rehashed_wrong_payload(tmp_path, name):
    base, _, root, _ = prepare(tmp_path, audit=False)
    path = root / f'{name}.npy'
    array = np.load(path)
    array[0, 0] += 1
    np.save(path, array, allow_pickle=False)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['files'][name]['sha256'] = sha256_file(path)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incorrect'):
        audit_training_similarity_cache(base, root, radius=2, bits=2048)
    assert not (root / 'audit.json').exists()


@pytest.mark.parametrize('change', [{'type': 'uniform'}, {'near_count': 17}, {'near_pool_size': 7},
                                     {'near_pool_size': 256}, {'near_count': True}, {'fingerprint_radius': -1},
                                     {'cache_directory': ''}, {'fingerprint_bits': 2049}, {'unexpected': 1}])
def test_explicit_configuration_rejects_unknown_or_incomplete_options(change):
    sampling = {'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32, 'fingerprint_radius': 2,
                'fingerprint_bits': 2048, 'cache_directory': '/synthetic/cache', **change}
    settings = {'enabled': True, 'negative_count': 16, 'loss_weight': 1., 'pool_cache_size': 1,
                'graph_cache_size': 1, 'sampling': sampling}
    with pytest.raises(ValueError):
        validate_candidate_settings(settings)


def test_direct_training_cannot_silently_use_uniform_for_structural_configuration(tmp_path):
    base, sampling, _, _ = prepare(tmp_path)
    settings = {'enabled': True, 'negative_count': 16, 'loss_weight': 1., 'pool_cache_size': 1,
                'graph_cache_size': 1, 'sampling': sampling}
    with pytest.raises(ValueError, match='strategy'):
        validate_candidate_sampling_binding(base, settings, {})
    mixed = load_structural_training_candidates(base, sampling)
    assert isinstance(mixed, StructuralTrainingCandidateIndex)
    with pytest.raises(ValueError, match='strategy'):
        validate_candidate_sampling_binding(mixed, {k: v for k, v in settings.items() if k != 'sampling'}, {})
    with pytest.raises(ValueError, match='explicitly enabled'):
        validate_candidate_settings({**settings, 'enabled': False})


def test_preparation_entry_refuses_missing_external_storage_before_loading(monkeypatch, tmp_path):
    import prepare_training_similarity as entry
    monkeypatch.setattr(entry, 'storage_receipt', lambda output: None)
    monkeypatch.setattr(entry, 'prepare_training_similarity', lambda *a, **k: pytest.fail('should fail before preparation'))
    with pytest.raises(SystemExit) as error:
        entry.main(['--data-path', str(tmp_path / 'data'), '--training-candidates', str(tmp_path / 'metadata'),
                    '--fingerprint-cache', str(tmp_path / 'fp'), '--output', str(tmp_path / 'output')])
    assert error.value.code == 2 and not (tmp_path / 'output').exists()
