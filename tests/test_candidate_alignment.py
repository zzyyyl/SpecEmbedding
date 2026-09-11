import copy
import hashlib
import json
import math
import pickle

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset, candidate_align_collate_fn
from SpecEmbedding.loss_align import ContrastiveAlignmentLoss
from SpecEmbedding.loss_candidates import candidate_alignment_loss
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils import candidate_training as candidate_io
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.training_candidates import TrainingCandidateIndex, validate_training_candidate_metadata


def key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]


def candidate_dataset(monkeypatch, *, augment=False):
    monkeypatch.setattr(config.model.mol_encoder, "graph_policy", "rdkit_sanitized")
    smiles = ["CCO", "OCC", "CC", "C(C)", "CCC"]
    targets = ["CCO", "CC"]
    raw = [{"smiles": value, "identity_2d": key(value)} for value in ("CCO", "CC", "CCO")]
    source = {"CCO": ["CCO", "OCC", "CC", "C(C)", "CCC"], "CC": ["CC", "C(C)"]}
    metadata = {
        "schema_version": 1, "kind": "diagnostic_candidate_metadata", "split": "train",
        "candidate_type": "mass", "forcing": False, "graph_policy": "rdkit_sanitized",
        "target_smiles": targets, "target_identity_2d": [key(s) for s in targets],
        "mol_smiles": smiles, "mol_identity_2d": [key(s) for s in smiles],
        "candidate_indices": np.array([[0, 1, 2, 3, 4], [2, 3, -1, -1, -1]], dtype=np.int32),
        "source_positions": np.array([[0, 1, 2, 3, 4], [0, 1, -1, -1, -1]], dtype=np.int16),
        "positive_mask": np.array([[True, True, False, False, False], [True, True, False, False, False]]),
        "query_target_rows": np.array([0, 1, 0], dtype=np.int32), "raw_query_indices": np.arange(3, dtype=np.int32),
        "graph_rejections": [],
    }
    observed = validate_training_candidate_metadata(metadata, raw, source, {}, 3)
    candidates = TrainingCandidateIndex(metadata, {"dataset_manifest_sha256": "a" * 64, "observed": observed}, pool_cache_size=1)
    data = {}
    for i, record in enumerate(raw):
        data.setdefault(record["identity_2d"], []).append({
            "mz": np.array([100 + i, 20, 30, 0], dtype=np.float32),
            "intensity": np.array([2, 1, 0.5, 0], dtype=np.float32),
            "mask": np.array([False, False, False, True]), "smiles": record["smiles"]})
    base = AlignGraphDataset(data=data, keys=[key("CC"), key("CCO")], full_spectra=True, n_views=1,
                            is_augment=augment, graph_cache_size=2, graph_policy="rdkit_sanitized",
                            augment_config={"prob": 1.0, "node_drop_rate": 0.1, "edge_mask_rate": 0.1,
                                            "removal_max": 0.2, "removal_intensity": 0.3, "rate_intensity": 0.15})
    return CandidateAlignDataset(base, candidates, dataset_manifest_sha256="a" * 64,
                                 negative_count=16, seed=42, graph_cache_size=1)


def test_candidate_loss_matches_independent_per_query_reference_and_retains_empty_query():
    spec = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.double, requires_grad=True)
    pos = torch.tensor([[1., 0.], [1., 1.], [1., 0.]], dtype=torch.double, requires_grad=True)
    neg = torch.tensor([[0., 1.], [-1., 0.], [0., 1.]], dtype=torch.double, requires_grad=True)
    scale = torch.tensor(2., dtype=torch.double, requires_grad=True)
    ptr = torch.tensor([0, 2, 2, 3])
    loss = candidate_alignment_loss(spec, pos, neg, ptr, scale)
    # Query 1 has zero negatives and remains in denominator 3.
    expected = (math.log(1 + math.exp(-2) + math.exp(-4)) + 0 + math.log(2)) / 3
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    assert torch.autograd.gradcheck(lambda a, b, c, s: candidate_alignment_loss(a, b, c, ptr, s), (spec, pos, neg, scale))
    loss.backward()
    assert torch.count_nonzero(spec.grad[1]) == torch.count_nonzero(pos.grad[1]) == 0
    assert neg.grad.abs().sum() > 0 and scale.grad.abs() > 0


def test_empty_negative_batch_and_half_precision_have_finite_gradients():
    features = torch.tensor([[1., 1.], [2., -1.]], dtype=torch.float16, requires_grad=True)
    loss = candidate_alignment_loss(features, features, features[:0], torch.tensor([0, 0, 0]), 100.)
    assert loss.dtype == torch.float32 and loss.item() == 0
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.count_nonzero() == 0


@pytest.mark.parametrize("change", ["ptr_length", "ptr_start", "ptr_end", "ptr_reverse", "ptr_float", "nan", "scale_zero", "scale_vector"])
def test_candidate_loss_rejects_invalid_input(change):
    spec, pos, neg = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    ptr, scale = torch.tensor([0, 1, 2, 3]), torch.tensor(2.)
    if change == "ptr_length":
        ptr = ptr[:3]
    elif change == "ptr_start":
        ptr[0] = 1
    elif change == "ptr_end":
        ptr[-1] = 4
    elif change == "ptr_reverse":
        ptr = torch.tensor([0, 2, 1, 3])
    elif change == "ptr_float":
        ptr = ptr.float()
    elif change == "nan":
        neg[0, 0] = torch.nan
    elif change == "scale_zero":
        scale = torch.tensor(0.)
    elif change == "scale_vector":
        scale = torch.ones(3)
    with pytest.raises(ValueError):
        candidate_alignment_loss(spec, pos, neg, ptr, scale)


def test_grouped_dataset_maps_back_to_every_raw_query_and_keeps_small_pools(monkeypatch):
    dataset = candidate_dataset(monkeypatch)
    assert dataset.raw_query_indices.tolist() == [1, 0, 2]
    with pytest.raises(RuntimeError, match="epoch"):
        dataset[0]
    dataset.set_epoch(1)
    examples = [dataset[i] for i in range(3)]
    assert [e.raw_query_index for e in examples] == [1, 0, 2]
    assert [len(e.negative_graphs) for e in examples] == [0, 2, 2]
    assert examples[0].anchor[0][0, 0] == 101
    assert examples[1].anchor[0][0, 0] == 100
    assert examples[2].anchor[0][0, 0] == 102
    for example in examples[1:]:
        assert set(example.sample.identity_2d) == {key("CC"), key("CCC")}
    batch = candidate_align_collate_fn(examples)
    assert batch.anchor[0].shape == (3, 4) and batch.negative_graphs.num_graphs == 4
    assert batch.negative_ptr.tolist() == [0, 0, 2, 4]
    assert batch.raw_query_indices.tolist() == [1, 0, 2]
    empty = candidate_align_collate_fn([examples[0]])
    assert empty.negative_graphs is None and empty.negative_ptr.tolist() == [0, 0]
    assert len(dataset._graphs) == 1


@pytest.mark.parametrize("damage", ["manifest", "smiles", "duplicate_query", "missing_query", "graph_policy"])
def test_candidate_dataset_refuses_wrong_binding(monkeypatch, damage):
    dataset = candidate_dataset(monkeypatch)
    base, index = dataset.base, dataset.candidates
    manifest = "a" * 64
    if damage == "manifest":
        manifest = "b" * 64
    elif damage == "smiles":
        base._data[key("CCO")][0]["smiles"] = "OCC"
    elif damage == "duplicate_query":
        base._spectrum_indices[2] = base._spectrum_indices[1]
    elif damage == "missing_query":
        base._spectrum_indices.pop()
    elif damage == "graph_policy":
        index.metadata["graph_policy"] = "legacy_raw"
    with pytest.raises(ValueError):
        CandidateAlignDataset(base, index, dataset_manifest_sha256=manifest, negative_count=16, seed=42, graph_cache_size=1)


def test_worker_count_and_epoch_restart_preserve_query_candidate_binding(monkeypatch):
    dataset = candidate_dataset(monkeypatch)
    dataset.set_epoch(7)
    expected = list(DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn, num_workers=0))
    loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn, num_workers=2,
                        multiprocessing_context="spawn", generator=torch.Generator().manual_seed(3), timeout=30)
    actual = list(loader)
    for left, right in zip(expected, actual, strict=True):
        for field in ("raw_query_indices", "candidate_indices", "source_positions", "negative_ptr"):
            assert torch.equal(getattr(left, field), getattr(right, field))
        torch.testing.assert_close(left.anchor[0], right.anchor[0])
    dataset.set_epoch(8)
    for batch in loader:
        assert batch.anchor[0].shape[0] == len(batch.raw_query_indices)


def test_negative_graph_augmentation_cannot_mutate_cached_graph(monkeypatch):
    dataset = candidate_dataset(monkeypatch, augment=True)
    dataset.set_epoch(1)
    dataset.graph_cache_size = 5
    original = copy.deepcopy(dataset._graph(4))
    for _ in range(5):
        dataset[1]
    current = dataset._graph(4)
    assert torch.equal(original.x, current.x) and torch.equal(original.edge_index, current.edge_index)


def small_model():
    return SpecMolAlignModel(SiameseModel(embedding_dim=8, n_head=2, n_layer=1, dim_feedward=8, dim_target=8,
                                        feedward_activation="selu"),
                            GINEEncoder(emb_dim=8, n_layers=2, dropout_rate=0., size_feature_dim=4),
                            spec_dim=8, hidden_dim=8, final_dim=8, dropout_rate=0., tau=0.2)


def test_actual_small_two_tower_model_receives_candidate_gradients(monkeypatch):
    dataset = candidate_dataset(monkeypatch)
    dataset.set_epoch(1)
    batch = candidate_align_collate_fn([dataset[i] for i in range(len(dataset))])
    model = small_model()
    mz, intensity, mask, graphs, labels = batch.anchor
    spec, positive, scale = model(mz, intensity, mask, graphs)
    negative = model.encode_mol(batch.negative_graphs)
    candidate_loss = candidate_alignment_loss(spec, positive, negative, batch.negative_ptr, scale)
    candidate_loss.backward(retain_graph=True)
    for module in (model.spec_encoder, model.mol_encoder, model.spec_proj, model.mol_proj):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.logit_scale.grad is not None and torch.isfinite(model.logit_scale.grad)
    model.zero_grad()
    combined = ContrastiveAlignmentLoss()(spec, positive, scale, labels) + candidate_loss
    combined.backward()
    assert torch.isfinite(combined) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_explicit_graph_policy_survives_serialization_and_worker_global_config(monkeypatch):
    dataset = candidate_dataset(monkeypatch)
    base = pickle.loads(pickle.dumps(dataset.base))
    monkeypatch.setattr(config.model.mol_encoder, "graph_policy", "legacy_raw")
    graph = base.get_mol_graph("new-aromatic", "C1=CC=CC=C1")
    assert graph.x[:, 3].sum() == 6
    assert base.graph_policy == "rdkit_sanitized"


def test_candidate_trainer_full_epoch_counts_checkpoint_and_empty_pool(monkeypatch, tmp_path):
    dataset = candidate_dataset(monkeypatch)
    loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn)
    val_loader = DataLoader(dataset.base, batch_size=2, collate_fn=align_collate_fn)
    model = small_model()
    trainer = CandidateTrainerAlign(model, loader, val_loader, torch.device("cpu"), save_dir=str(tmp_path),
                                    candidate_loss_weight=0.5)
    trainer.expected_epoch_counts = {"train": 3, "val": 3}
    trainer.fit(epochs=2, optimizer=torch.optim.AdamW(model.parameters(), lr=0.001), stage_name="stage2")
    summary = trainer.stage_summaries["stage2"]["candidate_training"]
    assert summary["candidate_loss_weight"] == 0.5
    assert summary["data"]["dataset_manifest_sha256"] == "a" * 64
    assert len(summary["epochs"]) == 2
    for epoch, record in enumerate(summary["epochs"], 1):
        assert record["epoch"] == epoch and record["queries"] == record["unique_queries"] == 3
        assert record["negative_samples"] == 4 and record["minimum_negatives"] == 0
        assert record["maximum_negatives"] == 2 and record["queries_without_negatives"] == 1
        assert math.isfinite(record["candidate_loss_query_mean"]) and len(record["observed_query_sample_order_sha256"]) == 64
    assert trainer.epoch_counts == [{"stage": "stage2", "epoch": e, "train": 3, "val": 3} for e in (1, 2)]
    weights = torch.load(tmp_path / "best_model_stage2.pth", weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(weights[name], value)


def test_candidate_trainer_detects_duplicate_query_despite_correct_total(monkeypatch, tmp_path):
    dataset = candidate_dataset(monkeypatch)
    loader = DataLoader(dataset, batch_size=3, sampler=[0, 1, 1], collate_fn=candidate_align_collate_fn)
    val_loader = DataLoader(dataset.base, batch_size=3, collate_fn=align_collate_fn)
    model = small_model()
    trainer = CandidateTrainerAlign(model, loader, val_loader, torch.device("cpu"), save_dir=str(tmp_path),
                                    candidate_loss_weight=1.0)
    with pytest.raises(RuntimeError, match="each original training query"):
        trainer.train_epoch(torch.optim.AdamW(model.parameters(), lr=0.001), 1, "synthetic")
    assert trainer.candidate_epoch_audits == []


def test_candidate_trainer_refuses_persistent_workers_and_query_dropping(monkeypatch, tmp_path):
    dataset = candidate_dataset(monkeypatch)
    val = DataLoader(dataset.base, batch_size=3, collate_fn=align_collate_fn)
    for settings in ({"num_workers": 1, "persistent_workers": True}, {"drop_last": True}):
        loader = DataLoader(dataset, batch_size=2, collate_fn=candidate_align_collate_fn, **settings)
        with pytest.raises(ValueError):
            CandidateTrainerAlign(small_model(), loader, val, torch.device("cpu"), save_dir=str(tmp_path),
                                  candidate_loss_weight=1.0)


def pinned_synthetic_candidate_input(monkeypatch, tmp_path, *, structural=False, loss_weight=0.5, manifest_sha256=None):
    dataset = candidate_dataset(monkeypatch)
    index = dataset.candidates
    if manifest_sha256 is not None:
        index.provenance['dataset_manifest_sha256'] = manifest_sha256
        dataset.provenance['dataset_manifest_sha256'] = manifest_sha256
    metadata = tmp_path / "metadata.pkl"
    metadata.write_bytes(pickle.dumps(index.metadata))
    for name in ("receipt.json", "verification.json"):
        (tmp_path / name).write_text('{}')
    index.provenance.update(path=str(metadata), sha256=sha256_file(metadata),
                            receipt_sha256=sha256_file(tmp_path / "receipt.json"),
                            verification_sha256=sha256_file(tmp_path / "verification.json"))
    # Full metadata/source validation has separate real-file tests. Here the
    # verified synthetic index connects input pinning to actual model training.
    monkeypatch.setattr(candidate_io, "load_training_candidates", lambda *a, **k: index)
    settings = {"enabled": True, "negative_count": 16, "loss_weight": loss_weight, "pool_cache_size": 1, "graph_cache_size": 1}
    if structural:
        from SpecEmbedding.utils.fingerprint_cache import (
            audit_fingerprint_cache,
            build_fingerprint_cache,
            fingerprint_provenance,
        )
        from SpecEmbedding.utils.training_similarity import (
            audit_training_similarity_cache,
            build_training_similarity_cache,
        )
        fp = tmp_path / 'sampling_fp'
        source = fingerprint_provenance(index.metadata['mol_smiles'], index_sha256=index.provenance['sha256'],
                                        dataset_manifest_sha256=index.provenance['dataset_manifest_sha256'], radius=2, bits=2048)
        build_fingerprint_cache(index.metadata['mol_smiles'], fp, source, workers=1, chunk_size=2)
        audit_fingerprint_cache(index.metadata['mol_smiles'], fp, source, workers=1, chunk_size=3)
        root = tmp_path / 'sampling_cache'
        build_training_similarity_cache(index, fp, root, radius=2, bits=2048)
        audit_training_similarity_cache(index, root, radius=2, bits=2048)
        settings['sampling'] = {'type': 'tanimoto_mixed', 'near_count': 8, 'near_pool_size': 32,
                                'fingerprint_radius': 2, 'fingerprint_bits': 2048, 'cache_directory': str(root)}
    prepared_index, receipt = candidate_io.build_candidate_training_input(metadata, tmp_path, settings, {"train": 3}, [])
    path = tmp_path / "candidate_training_input.json"
    path.write_text(json.dumps(receipt))
    dataset.candidates = prepared_index
    dataset.provenance.update(prepared_index.provenance)
    return dataset, settings, receipt, path


def test_candidate_input_checks_actual_grouped_order_and_refuses_changed_receipt(monkeypatch, tmp_path):
    dataset, settings, receipt, path = pinned_synthetic_candidate_input(monkeypatch, tmp_path)
    _, actual, fingerprint = candidate_io.read_candidate_training_input(path, tmp_path, settings, {"train": 3}, [])
    assert actual == receipt and fingerprint == {"path": str(path), "sha256": sha256_file(path)}
    assert candidate_io.grouped_query_order(dataset.candidates).tolist() == [0, 2, 1]
    receipt["dataset_to_raw_query_sha256"] = hashlib.sha256(np.arange(3, dtype='<i8').tobytes()).hexdigest()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="provenance"):
        candidate_io.read_candidate_training_input(path, tmp_path, settings, {"train": 3}, [])
    with pytest.raises(ValueError, match="configuration"):
        candidate_io.read_candidate_training_input(path, tmp_path, {**settings, "loss_weight": 2.0}, {"train": 3}, [])


@pytest.mark.parametrize('spectrum_variant', [None, 'precursor_delta', 'attention_pool', 'graph_global_context'])
@pytest.mark.parametrize('qk_norm', [False, True])
@pytest.mark.parametrize('structural', [False, True])
@pytest.mark.parametrize('loss_weight', [0.5, 2.0])
def test_formal_training_wiring_saves_replayable_full_candidate_trajectory(monkeypatch, tmp_path, spectrum_variant, qk_norm, structural, loss_weight):
    import train_align as entry
    from SpecEmbedding.config import ConfigObject

    dataset, settings, receipt, path = pinned_synthetic_candidate_input(monkeypatch, tmp_path, structural=structural, loss_weight=loss_weight)
    monkeypatch.setattr(config.train.align, "candidate_supervision", ConfigObject(settings))
    monkeypatch.setattr(config.train.align, "epochs_stage2", 2)
    monkeypatch.setattr(config.train.align, "num_workers", 0)
    monkeypatch.setattr(config.train.align, "batching", "mass_blocks")
    monkeypatch.setattr(config.train.align, "mass_block_size", 1)
    monkeypatch.setattr(config.model, "spec_encoder", ConfigObject({
        "embedding_dim": 8, "n_head": 2, "n_layer": 1, "dim_feedward": 8, "dim_target": 8,
        "feedward_activation": "selu"}))
    if spectrum_variant == 'precursor_delta':
        monkeypatch.setattr(config.model.spec_encoder, 'precursor_delta', ConfigObject({
            'fourier_dim': 8, 'hidden_dim': 8, 'min_wavelength': .01, 'max_wavelength': 10000.,
        }), raising=False)
    elif spectrum_variant == 'attention_pool':
        monkeypatch.setattr(config.model.spec_encoder, 'attention_pool', ConfigObject({'norm_eps': 1e-5}), raising=False)
    if qk_norm:
        monkeypatch.setattr(config.model.spec_encoder, 'qk_norm', ConfigObject({'eps': 1e-6}), raising=False)
    monkeypatch.setattr(config.model, "mol_encoder", ConfigObject({
        "emb_dim": 8, "n_layers": 2, "dropout_rate": 0., "size_feature_dim": 4,
        "norm_type": "layernorm", "norm_eps": 1e-5, "graph_policy": "rdkit_sanitized"}))
    if spectrum_variant == 'graph_global_context':
        from tests.test_graph_global_context import context_config

        monkeypatch.setattr(config.model, 'graph_global_context', ConfigObject(context_config()), raising=False)
        monkeypatch.setattr(config.model.mol_encoder, 'n_layers', 4)
    monkeypatch.setattr(config.model.align, "final_dim", 8)
    monkeypatch.setattr(config.augmentation, "prob", 0.)
    # Synthetic CPU wiring test: keep the actual trainer and optimizer; only
    # omit formal CUDA telemetry, which is independently tested with fixtures.
    def cpu_trainer(*args, **kwargs):
        kwargs["record_resources"] = False
        return CandidateTrainerAlign(*args, **kwargs)
    monkeypatch.setattr(entry, "CandidateTrainerAlign", cpu_trainer)
    output = tmp_path / "run"
    def validator(*args):
        return {"top1": 0.5, "top5": 1., "top10": 1., "top20": 1., "mrr": 0.75}
    metadata = {"fulltrain_audit": {"expected_epoch_counts": {"train": 3, "val": 3},
                                   "dataset_manifest_sha256": "a" * 64}}
    if spectrum_variant in ('attention_pool', 'graph_global_context'):
        from types import SimpleNamespace

        from SpecEmbedding.utils.retrieval_validation import AlignmentRetrievalValidator
        from tests.test_retrieval_validation import build_index

        index = build_index()  # One positive query and two queries without a positive, including an empty pool.
        index.update(dataset_manifest_sha256='a' * 64, dataset_outputs={'synthetic': 'CPU wiring only'})
        validator = AlignmentRetrievalValidator(index, SimpleNamespace(
            mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=(1, 5, 10, 20)),
            output / 'validation_retrieval')
        monkeypatch.setattr(config.data, 'tokenizer', ConfigObject(index['tokenizer_config']))
        metadata.update(seed=42, exclude_val_query_indices=[])
        metadata['fulltrain_audit'].update(formal_fulltrain=True, dataset_version='1.5',
                                           input_outputs=index['dataset_outputs'])
    entry.train_align(
        dataset.base._data, sorted(dataset.base._keys), dataset.base._data, sorted(dataset.base._keys), None,
        batch_size=2, lr=0.001, save_dir=str(output), device="cpu", seed=42, formal_fulltrain=True,
        retrieval_validator=validator,
        training_candidates=dataset.candidates, candidate_input_receipt=receipt,
        selection_metadata=metadata,
    )
    selection = json.loads((output / "alignment_selection.json").read_text())
    assert ('precursor_delta' in selection['model_config']['spec_encoder']) == (spectrum_variant == 'precursor_delta')
    assert ('attention_pool' in selection['model_config']['spec_encoder']) == (spectrum_variant == 'attention_pool')
    assert ('qk_norm' in selection['model_config']['spec_encoder']) == qk_norm
    assert ('graph_global_context' in selection['model_config']) == (spectrum_variant == 'graph_global_context')
    weights = torch.load(output / 'best_model_stage2.pth', map_location='cpu', weights_only=True)
    if spectrum_variant in ('attention_pool', 'graph_global_context'):
        from SpecEmbedding.utils.formal_alignment import load_formal_alignment
        from SpecEmbedding.utils.optimization_audit import audit_snapshot

        if spectrum_variant == 'attention_pool':
            assert torch.isfinite(weights['spec_encoder.pool.query']).all()
            assert weights['spec_encoder.pool.query'].abs().sum() > 0
        else:
            for layer in range(3):
                value = weights[f'mol_encoder.context_branches.{layer}.output.weight']
                assert torch.isfinite(value).all() and value.abs().sum() > 0
        restored, _, receipt_model = load_formal_alignment(output / 'best_model_stage2.pth', torch.device('cpu'),
            dataset_outputs=index['dataset_outputs'], dataset_manifest_sha256='a' * 64,
            tokenizer_config=index['tokenizer_config'], expected_counts={'train': 3, 'val': 3}, exclusions=[])
        assert receipt_model['model_config'] == selection['model_config']
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in weights.items())
        for epoch in (1, 2):
            result = audit_snapshot(output / 'validation_retrieval' / f'stage2_epoch{epoch:03d}.pt', index)
            assert result['queries'] == 3 and result['positive_queries'] == 1
    norm_keys = [key for key in weights if '.q_norm.' in key or '.k_norm.' in key]
    assert bool(norm_keys) == qk_norm
    assert all(bool(torch.isfinite(weights[key]).all()) for key in norm_keys)
    stage = selection["stages"]["stage2"]
    assert selection['training_config']['candidate_supervision']['loss_weight'] == loss_weight
    assert selection['config_snapshot']['train']['align']['candidate_supervision']['loss_weight'] == loss_weight
    assert stage['candidate_training']['candidate_loss_weight'] == loss_weight
    report, hashes = candidate_io.audit_candidate_training(output, stage, path, settings, 42, 2, tmp_path, {"train": 3}, [])
    assert report["state"] == "verified_full_candidate_replay" and report["epochs"] == 2
    assert report["queries_per_epoch"] == 3 and len(hashes) == (16 if structural else 8)
    for row in selection["fulltrain_audit"]["epochs"]:
        assert row["train"] == row["val"] == 3 and row["batching"]["unique_queries"] == 3
    record = stage["candidate_training"]["epochs"][0]
    saved_path = output / "candidate_training" / "stage2_epoch001.json"
    order_path = saved_path.with_suffix('.npy')
    original = copy.deepcopy(record)
    original_order_bytes = order_path.read_bytes()
    for damage in ("sample_hash", "count", "batch_sizes", "weight", "missing_epochs", "order", "duplicate"):
        altered = copy.deepcopy(stage)
        if damage == "weight":
            altered["candidate_training"]["candidate_loss_weight"] = loss_weight + 1.0
        elif damage == "missing_epochs":
            altered["candidate_training"]["epochs"].pop()
        else:
            row = altered["candidate_training"]["epochs"][0]
            if damage == "sample_hash":
                row["observed_query_sample_order_sha256"] = '0' * 64
            elif damage == "count":
                row["negative_samples"] += 1
            elif damage == "batch_sizes":
                row["batch_sizes"] = [1, 2]
            else:
                order = np.load(order_path)
                if damage == "order":
                    order = np.roll(order, 1)
                else:
                    order[0] = order[1]
                with order_path.open('wb') as handle:
                    np.save(handle, order)
                row["query_order_sha256"] = sha256_file(order_path)
            saved_path.write_text(json.dumps(row))
        with pytest.raises(ValueError):
            candidate_io.audit_candidate_training(output, altered, path, settings, 42, 2, tmp_path, {"train": 3}, [])
        saved_path.write_text(json.dumps(original))
        order_path.write_bytes(original_order_bytes)


@pytest.mark.parametrize('change', [
    {'enabled': 1}, {'negative_count': 0}, {'negative_count': 256}, {'negative_count': 2.5},
    {'loss_weight': 0}, {'loss_weight': float('nan')}, {'loss_weight': True},
    {'pool_cache_size': 0}, {'graph_cache_size': -1}, {'unexpected': 1},
])
def test_candidate_settings_reject_invalid_or_undeclared_parameters(change):
    settings = {'enabled': True, 'negative_count': 16, 'loss_weight': 1., 'pool_cache_size': 1, 'graph_cache_size': 1}
    with pytest.raises(ValueError):
        candidate_io.validate_candidate_settings({**settings, **change})


def test_inactive_or_legacy_audit_refuses_candidate_artifacts(tmp_path):
    args = (None, 42, 128, tmp_path, {'train': 3}, [])
    assert candidate_io.audit_candidate_training(tmp_path, {}, None, *args)[0]['state'] == 'disabled'
    with pytest.raises(ValueError, match='Unexpected'):
        candidate_io.audit_candidate_training(tmp_path, {'candidate_training': {}}, None, *args)
    (tmp_path / 'candidate_training').mkdir()
    with pytest.raises(ValueError, match='Unexpected'):
        candidate_io.audit_candidate_training(tmp_path, {}, None, *args)
