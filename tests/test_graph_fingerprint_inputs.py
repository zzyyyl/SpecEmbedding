"""Synthetic combined-input checks; these do not authorize a formal A09 launch."""

import copy
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from SpecEmbedding.data.datasets_align import align_collate_fn
from SpecEmbedding.data.datasets_candidates import candidate_align_collate_fn
from SpecEmbedding.data.datasets_graph_fingerprint import (
    CandidateGraphFingerprintDataset,
    GraphFingerprintAlignmentDataset,
)
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.models_graph_fingerprint import GraphFingerprintAlignmentModel
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils.graph_fingerprint_validation import GraphFingerprintRetrievalValidator
from SpecEmbedding.utils.molecule_graph_cache import (
    audit_graph_cache,
    build_graph_cache,
    graph_cache_provenance,
    load_graph_cache,
)
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import build_validation_index
from tests.test_candidate_alignment import candidate_dataset, key
from tests.test_fingerprint_alignment import make_cache
from tests.test_graph_fingerprint_model import parent_config
from tests.test_retrieval_validation import build_index, spectrum


def fixed_bits(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    return torch.from_numpy(generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.float32))


def make_dataset(tmp_path, monkeypatch, *, augment=False):
    original = candidate_dataset(monkeypatch, augment=augment)
    original.candidates.provenance['sha256'] = 'b' * 64
    smiles = original.candidates.metadata['mol_smiles']
    root = tmp_path / 'train_bits'
    provenance = make_cache(root, smiles)
    options = dict(data=original.base._data, keys=original.base._keys, full_spectra=True, n_views=1,
                   is_augment=augment, graph_policy='rdkit_sanitized', graph_cache_size=2,
                   augment_config=copy.deepcopy(original.base.augment_config),
                   fingerprint_root=root, fingerprint_smiles=smiles, fingerprint_provenance=provenance,
                   dataset_manifest_sha256='a' * 64)
    base = GraphFingerprintAlignmentDataset(**options)
    dataset = CandidateGraphFingerprintDataset(base, original.candidates, dataset_manifest_sha256='a' * 64,
                                               negative_count=16, seed=42, graph_cache_size=1)
    return dataset, original, options


def small_model():
    return GraphFingerprintAlignmentModel(parent_model_config=parent_config(),
                                           fingerprint_config={'input_bits': 2048, 'hidden_dim': 12, 'norm_eps': 1e-5})


def same_graph(actual, expected):
    assert set(actual.keys()) - {'fingerprints'} == set(expected.keys()) - {'fingerprints'}
    for field in set(expected.keys()) - {'fingerprints'}:
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)


@pytest.mark.parametrize('augment', [False, True])
def test_combined_samples_keep_graph_spectrum_augmentation_rng_and_exact_molecule_rows(tmp_path, monkeypatch, augment):
    dataset, original, _ = make_dataset(tmp_path, monkeypatch, augment=augment)
    with pytest.raises(RuntimeError, match='epoch'):
        dataset[0]
    dataset.set_epoch(7)
    original.set_epoch(7)
    def draw(source):
        torch.manual_seed(11)
        np.random.seed(11)
        random.seed(11)
        examples = [source[i] for i in range(len(source))]
        return examples, torch.get_rng_state(), np.random.random(4), random.random()
    before, torch_state, np_next, py_next = draw(original)
    actual, torch_after, np_after, py_after = draw(dataset)
    assert torch.equal(torch_state, torch_after) and np.array_equal(np_next, np_after) and py_next == py_after
    for a, b in zip(actual, before, strict=True):
        assert a.raw_query_index == b.raw_query_index
        assert np.array_equal(a.sample.molecule_indices, b.sample.molecule_indices)
        assert np.array_equal(a.sample.source_positions, b.sample.source_positions)
        for left, right in zip(a.anchor[:3], b.anchor[:3], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert a.anchor[-1] == b.anchor[-1]
        row = int(dataset.candidates.metadata['query_target_rows'][a.raw_query_index])
        target = dataset.candidates.metadata['target_smiles'][row]
        same_graph(a.anchor[3][0], b.anchor[3][0])
        torch.testing.assert_close(a.anchor[3][0].fingerprints[0], fixed_bits(target), rtol=0, atol=0)
        for graph, reference, molecule in zip(a.negative_graphs, b.negative_graphs, a.sample.molecule_indices, strict=True):
            same_graph(graph, reference)
            smiles = dataset.candidates.metadata['mol_smiles'][molecule]
            torch.testing.assert_close(graph.fingerprints[0], fixed_bits(smiles), rtol=0, atol=0)
    batch = candidate_align_collate_fn(actual)
    assert batch.raw_query_indices.tolist() == [1, 0, 2]
    assert batch.anchor[-1].tolist() == [0, 1, 1]
    assert batch.negative_ptr.tolist() == [0, 0, 2, 4]
    assert batch.anchor[3].fingerprints.shape == (3, 2048)
    assert batch.negative_graphs.fingerprints.shape == (4, 2048)
    assert candidate_align_collate_fn([actual[0]]).negative_graphs is None
    assert dataset.provenance['graph_cache_size'] == original.provenance['graph_cache_size'] == 1
    assert dataset.provenance['negative_graph_augmentation'] == original.provenance['negative_graph_augmentation']
    # Returned tensors and graph augmentations must never contaminate either source cache.
    for example in actual:
        for graph in example.anchor[3] + example.negative_graphs:
            graph.fingerprints.zero_()
            graph.x.zero_()
    assert all('fingerprints' not in graph for graph in dataset.base._mol_cache.values())
    assert all('fingerprints' not in graph for graph in dataset._graphs.values())
    for i, smiles in enumerate(dataset.candidates.metadata['mol_smiles']):
        torch.testing.assert_close(torch.from_numpy(dataset.base.fingerprints[i]), fixed_bits(smiles))


def test_target_row_binding_uses_each_exact_smiles_and_follows_validation_permutation(tmp_path, monkeypatch):
    _, _, options = make_dataset(tmp_path, monkeypatch)
    options['data'] = copy.deepcopy(options['data'])
    options['data'][key('CCO')][1]['smiles'] = 'OCC'
    base = GraphFingerprintAlignmentDataset(**options)
    assert set(base._target_fingerprints) == {'CC', 'CCO', 'OCC'}
    base._spectrum_indices.reverse()
    # Equal-identity aliases happen to have equal Morgan bits; record requested row numbers too.
    requested = []
    store = base.fingerprints
    class ObservedRows:
        def __getitem__(self, index):
            requested.append(index)
            return store[index]
    base.fingerprints = ObservedRows()
    for i, (identity, offset) in enumerate(base._spectrum_indices):
        smiles = base._data[identity][offset]['smiles']
        item = base[i]
        assert requested[-1] == options['fingerprint_smiles'].index(smiles)
        same_graph(item[3][0], smiles_to_graph(smiles, graph_policy='rdkit_sanitized'))
    assert len(requested) == len(base) == 3


@pytest.mark.parametrize('damage', ['manifest', 'partial', 'missing_target', 'cache_order', 'candidate_order', 'candidate_sha'])
def test_combined_training_rejects_unbound_or_incomplete_inputs(tmp_path, monkeypatch, damage):
    dataset, _, options = make_dataset(tmp_path, monkeypatch)
    if damage == 'manifest':
        options['dataset_manifest_sha256'] = 'd' * 64
    elif damage == 'partial':
        options['full_spectra'] = False
    elif damage == 'missing_target':
        options['data'] = copy.deepcopy(options['data'])
        options['data'][key('CCO')][0]['smiles'] = 'CCCO'
    elif damage == 'cache_order':
        options['fingerprint_smiles'] = list(reversed(options['fingerprint_smiles']))
    else:
        if damage == 'candidate_order':
            dataset.candidates.metadata['mol_smiles'] = list(reversed(dataset.candidates.metadata['mol_smiles']))
        else:
            dataset.candidates.provenance['sha256'] = 'd' * 64
        with pytest.raises(ValueError, match='index differs'):
            CandidateGraphFingerprintDataset(dataset.base, dataset.candidates, dataset_manifest_sha256='a' * 64,
                                               negative_count=16, seed=42, graph_cache_size=1)
        return
    with pytest.raises(ValueError):
        GraphFingerprintAlignmentDataset(**options)


def test_spawn_workers_keep_every_query_graph_and_bit_row_across_epochs(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    dataset.set_epoch(2)
    expected = list(DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn))
    loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn, num_workers=2,
                        multiprocessing_context='spawn', generator=torch.Generator().manual_seed(1), timeout=30)
    for a, b in zip(expected, loader, strict=True):
        for field in ('negative_ptr', 'raw_query_indices', 'candidate_indices', 'source_positions'):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
        for position in (0, 1, 2, 4):
            torch.testing.assert_close(a.anchor[position], b.anchor[position], rtol=0, atol=0)
        for x, y in ((a.anchor[3], b.anchor[3]), (a.negative_graphs, b.negative_graphs)):
            same_graph(x, y)
            torch.testing.assert_close(x.fingerprints, y.fingerprints, rtol=0, atol=0)
    dataset.set_epoch(3)
    assert sum(len(batch.raw_query_indices) for batch in loader) == len(dataset)


def make_validator(tmp_path, *, graph_cache=True, workers=0, index=None):
    index = build_index() if index is None else index
    index['dataset_manifest_sha256'] = 'a' * 64
    root = tmp_path / 'validation_bits'
    provenance = make_cache(root, index['mol_smiles'], index_sha='c' * 64)
    graphs = None
    if graph_cache:
        graph_root = tmp_path / 'validation_graphs'
        source = graph_cache_provenance(index['mol_smiles'], index_sha256='c' * 64, dataset_manifest_sha256='a' * 64)
        build_graph_cache(index['mol_smiles'], graph_root, source, workers=1, chunk_size=2)
        audit_graph_cache(index['mol_smiles'], graph_root, source, workers=1, chunk_size=2)
        graphs, _ = load_graph_cache(index['mol_smiles'], graph_root, source)
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=workers, top_k=(1, 5, 10, 20))
    kwargs = dict(fingerprint_root=root, fingerprint_provenance=provenance, index_sha256='c' * 64, graph_cache=graphs)
    return GraphFingerprintRetrievalValidator(index, settings, tmp_path / 'run/validation_retrieval', **kwargs), kwargs


@pytest.mark.parametrize('cached', [False, True])
def test_validation_binds_both_inputs_reencodes_weights_and_matches_direct_cosines(tmp_path, cached):
    validate, _ = make_validator(tmp_path, graph_cache=cached, workers=2)
    index = validate.index
    model = small_model().eval()
    graphs = []
    for smiles in index['mol_smiles']:
        graph = smiles_to_graph(smiles, graph_policy='rdkit_sanitized')
        graph.fingerprints = fixed_bits(smiles).unsqueeze(0)
        graphs.append(graph)
    # A reused returned graph must not change fixed inputs subsequently read from disk.
    first_item = validate.molecules[0]
    first_item['graph'].x.zero_()
    first_item['graph'].fingerprints.zero_()
    same_graph(validate.molecules[0]['graph'], graphs[0])
    torch.testing.assert_close(validate.molecules[0]['graph'].fingerprints, graphs[0].fingerprints)
    scores = []
    for epoch in (1, 2):
        if epoch == 2:
            with torch.no_grad():
                torch.nn.init.normal_(model.mol_encoder.fingerprint_branch[-1].weight, std=.1)
        state = torch.get_rng_state()
        validate(model, torch.device('cpu'), epoch, 'fresh')
        assert torch.equal(state, torch.get_rng_state())
        path = tmp_path / 'run/validation_retrieval' / f'fresh_epoch{epoch:03}.pt'
        snapshot = torch.load(path, weights_only=False)
        with pytest.raises(ValueError):
            audit_snapshot(path, index)
        result = audit_snapshot(path, index, expected_graph_cache=validate.graph_cache_fingerprint,
                                expected_fingerprint_cache=validate.fingerprint_receipt)
        assert result['queries'] == 3 and result['positive_queries'] == 1
        assert snapshot['raw_query_indices'] == index['raw_query_indices']
        with torch.inference_mode():
            embeddings = model.encode_mol(Batch.from_data_list(graphs), True)
            for i, spectrum_item in enumerate(validate.spectra):
                valid = index['candidate_indices'][i] >= 0
                if not valid.any():
                    assert snapshot['ranks'][i] == 0
                    continue
                spec = model.encode_spec(*(spectrum_item[key].unsqueeze(0)
                    for key in ('spec_mz', 'spec_intensity', 'spec_mask')), normalize=True)[0]
                expected = torch.stack([spec @ embeddings[j] for j in index['candidate_indices'][i, valid]])
                torch.testing.assert_close(snapshot['scores'][i, valid], expected, rtol=1e-5, atol=1e-6)
        scores.append(snapshot['scores'])
    valid = index['candidate_indices'] >= 0
    assert not torch.equal(scores[0][valid], scores[1][valid])


@pytest.mark.parametrize('damage', ['graph_index', 'graph_dataset', 'graph_order', 'bit_index', 'bit_dataset', 'bit_order'])
def test_validation_rejects_mixed_graph_and_fingerprint_provenance(tmp_path, damage):
    validate, kwargs = make_validator(tmp_path)
    if damage.startswith('graph_'):
        key_name = {'graph_index': 'index_sha256', 'graph_dataset': 'dataset_manifest_sha256',
                    'graph_order': 'molecule_order_sha256'}[damage]
        kwargs['graph_cache'].manifest['provenance'][key_name] = 'd' * 64
    else:
        key_name = {'bit_index': 'index_sha256', 'bit_dataset': 'dataset_manifest_sha256',
                    'bit_order': 'molecule_order_sha256'}[damage]
        kwargs['fingerprint_provenance'] = {**kwargs['fingerprint_provenance'], key_name: 'd' * 64}
    with pytest.raises(ValueError):
        GraphFingerprintRetrievalValidator(validate.index, validate.settings, **kwargs)


def test_fingerprint_collision_does_not_merge_2d_positives_or_candidate_rows(tmp_path):
    smiles = ['CCCCCCCCCCCC', 'CCCCCCCCCCCCC']
    torch.testing.assert_close(fixed_bits(smiles[0]), fixed_bits(smiles[1]), rtol=0, atol=0)
    index = build_validation_index([spectrum(s) for s in smiles], {s: smiles for s in smiles}, [], {},
                                   {'max_len': 8, 'show_progress_bar': False})
    validate, _ = make_validator(tmp_path, index=index)
    model = small_model()
    validate(model, torch.device('cpu'), 1, 'collision')
    assert index['candidate_indices'].tolist() == [[0, 1], [0, 1]]
    assert index['positive_mask'].tolist() == [[True, False], [False, True]]
    result = audit_snapshot(tmp_path / 'run/validation_retrieval/collision_epoch001.pt', index,
                            expected_graph_cache=validate.graph_cache_fingerprint,
                            expected_fingerprint_cache=validate.fingerprint_receipt)
    assert result['queries'] == result['positive_queries'] == 2


def test_synthetic_training_audits_full_query_coverage_and_reaches_both_molecule_branches(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch, augment=True)
    train = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn)
    validation_base = copy.copy(dataset.base)
    validation_base.is_augment = False
    val = DataLoader(validation_base, batch_size=2, collate_fn=align_collate_fn)
    validate, _ = make_validator(tmp_path)
    model = small_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    trainer = CandidateTrainerAlign(model, train, val, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=1., retrieval_validator=validate)
    trainer.expected_epoch_counts = {'train': 3, 'val': 3}
    trainer.fit(epochs=2, optimizer=torch.optim.AdamW(model.parameters(), lr=.001), stage_name='stage2')
    assert trainer.epoch_counts == [{'stage': 'stage2', 'epoch': e, 'train': 3, 'val': 3} for e in (1, 2)]
    for prefix in ('spec_encoder.', 'mol_encoder.convs.', 'mol_encoder.fingerprint_branch.0.',
                   'mol_encoder.fingerprint_branch.3.', 'spec_proj.', 'mol_proj.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith(prefix))
    records = trainer.stage_summaries['stage2']['candidate_training']
    assert records['data']['molecule_input'] == 'graph_with_fixed_morgan_bits'
    assert records['data']['graph_cache_size'] == 1
    assert records['data']['negative_graph_augmentation'] == dataset.provenance['negative_graph_augmentation']
    for epoch, record in enumerate(records['epochs'], 1):
        assert record['queries'] == record['unique_queries'] == 3 and record['negative_samples'] == 4
        assert record['queries_without_negatives'] == 1
        audit_snapshot(tmp_path / 'run/validation_retrieval' / f'stage2_epoch{epoch:03}.pt', validate.index,
                        expected_graph_cache=validate.graph_cache_fingerprint,
                        expected_fingerprint_cache=validate.fingerprint_receipt)
    clone = small_model().eval()
    clone.load_state_dict(torch.load(tmp_path / 'run/best_model_stage2.pth', weights_only=True), strict=True)
    # Loading the selected weight set and encoding actual combined DataLoader tensors succeeds.
    batch = next(iter(val))
    assert torch.isfinite(clone(*batch[:4])[1]).all()


def test_duplicate_query_iteration_is_rejected_even_with_matching_combined_batch_count(tmp_path, monkeypatch):
    dataset, _, _ = make_dataset(tmp_path, monkeypatch)
    train = DataLoader(dataset, batch_size=3, sampler=[0, 1, 1], collate_fn=candidate_align_collate_fn)
    val = DataLoader(dataset.base, batch_size=3, collate_fn=align_collate_fn)
    model = small_model()
    trainer = CandidateTrainerAlign(model, train, val, torch.device('cpu'), save_dir=str(tmp_path / 'run'),
                                    candidate_loss_weight=1.)
    with pytest.raises(RuntimeError, match='each original training query'):
        trainer.train_epoch(torch.optim.AdamW(model.parameters(), lr=.001), 1, 'synthetic')
    assert not trainer.candidate_epoch_audits


def test_cached_validation_never_constructs_new_graphs(tmp_path):
    validate, _ = make_validator(tmp_path)
    with patch('SpecEmbedding.utils.retrieval_validation.smiles_to_graph', side_effect=AssertionError('new graph')):
        for i, smiles in enumerate(validate.index['mol_smiles']):
            item = validate.molecules[i]
            assert item['original_idx'] == i
            torch.testing.assert_close(item['graph'].fingerprints[0], fixed_bits(smiles))
