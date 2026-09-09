import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import run_massspecgym_v15 as runner
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import identity
from SpecEmbedding.utils.retrieval_validation import (
    AlignmentRetrievalValidator,
    build_validation_index,
    import_validation_index,
    prepared_validation_input,
    retrieval_metrics,
    validation_index_receipt,
)


def spectrum(smiles):
    from matchms import Spectrum

    return Spectrum(mz=np.array([30., 40.]), intensities=np.array([.3, 1.]),
                    metadata={"smiles": smiles, "precursor_mz": 80., "identity_2d": identity(smiles)},
                    metadata_harmonization=False)


def build_index():
    raw = [spectrum(s) for s in ("CCO", "CCC", "CCN", "CCCC")]
    candidates = {"CCO": ["CCC", "OCC", "CCO"], "CCN": ["CCC"], "CCCC": []}
    return build_validation_index(raw, candidates, [1], {}, {"max_len": 8, "show_progress_bar": False})


def reusable_index(tmp_path, monkeypatch):
    import SpecEmbedding.utils.retrieval_validation as module

    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    report = {'outputs': {'synthetic': 'CPU fixture'}}
    monkeypatch.setattr(module, 'verify_dataset', lambda *a: report)
    index = build_index()
    index.update(dataset_manifest_sha256=sha256_file(data / 'dataset_manifest.json'),
                 dataset_outputs=report['outputs'], graph_policy='rdkit_sanitized')
    source = tmp_path / 'original.pt'
    torch.save(index, source)
    source.with_suffix('.json').write_text(json.dumps(validation_index_receipt(index, source)))
    options = (data, {'val': 4}, [1], index['tokenizer_config'])
    expected = prepared_validation_input(source, *options)
    return source, expected, options


def test_prepared_index_import_preserves_every_byte_and_rejects_existing_outputs(tmp_path, monkeypatch):
    source, expected, options = reusable_index(tmp_path, monkeypatch)
    destination = tmp_path / 'new' / 'mass_val_topk256.pt'
    result = import_validation_index(source, destination, expected, *options)
    assert result == {**expected, 'path': str(destination)}
    assert result['queries'] == 3 and result['molecules'] == 3
    assert result['positive_queries'] == 1 and result['multi_positive_queries'] == 1
    assert destination.read_bytes() == source.read_bytes()
    assert destination.with_suffix('.json').read_bytes() == source.with_suffix('.json').read_bytes()
    with pytest.raises(FileExistsError):
        import_validation_index(source, destination, expected, *options)


@pytest.mark.parametrize('damage', ['index_bytes', 'receipt_count', 'receipt_bytes', 'tokenizer', 'exclusions', 'manifest'])
def test_prepared_index_rejects_changed_inputs_before_copy(tmp_path, monkeypatch, damage):
    source, expected, options = reusable_index(tmp_path, monkeypatch)
    data, counts, exclusions, tokenizer = options
    if damage == 'index_bytes':
        source.write_bytes(source.read_bytes() + b'changed')
    elif damage == 'receipt_count':
        receipt = json.loads(source.with_suffix('.json').read_text())
        receipt['queries'] = 2
        source.with_suffix('.json').write_text(json.dumps(receipt))
    elif damage == 'receipt_bytes':
        source.with_suffix('.json').write_text(source.with_suffix('.json').read_text() + ' ')
    elif damage == 'tokenizer':
        tokenizer = {**tokenizer, 'max_len': 9}
    elif damage == 'exclusions':
        exclusions = []
    else:
        (data / 'dataset_manifest.json').write_text('{"changed": true}')
    destination = tmp_path / 'new' / 'mass_val_topk256.pt'
    with pytest.raises(ValueError):
        import_validation_index(source, destination, expected, data, counts, exclusions, tokenizer)
    assert not destination.exists()


def test_prepared_index_detects_source_changes_during_copy(tmp_path, monkeypatch):
    import SpecEmbedding.utils.retrieval_validation as module

    source, expected, options = reusable_index(tmp_path, monkeypatch)
    original = module.shutil.copyfileobj
    def change_source(reader, writer):
        original(reader, writer)
        if reader.name == str(source):
            source.write_bytes(source.read_bytes() + b'changed')
    monkeypatch.setattr(module.shutil, 'copyfileobj', change_source)
    with pytest.raises(ValueError, match='changed during import'):
        import_validation_index(source, tmp_path / 'new.pt', expected, *options)


def test_index_keeps_source_order_multiple_2d_positives_and_unlabeled_queries():
    index = build_index()
    assert index["raw_query_indices"] == [0, 2, 3]
    assert index["positive_mask"].tolist() == [[False, True, True], [False, False, False], [False, False, False]]
    assert index["source_positions"].tolist() == [[0, 1, 2], [0, -1, -1], [-1, -1, -1]]
    assert index["candidate_indices"].tolist() == [[0, 1, 2], [0, -1, -1], [-1, -1, -1]]
    assert index["mol_smiles"] == ["CCC", "OCC", "CCO"]


def test_index_rejects_missing_mapping_and_unrecorded_invalid_molecule():
    raw = [spectrum("CCO")]
    options = {"max_len": 8, "show_progress_bar": False}
    with pytest.raises(ValueError, match="Missing validation"):
        build_validation_index(raw, {}, [], {}, options)
    with pytest.raises(ValueError):
        build_validation_index(raw, {"CCO": ["CCO", "not-a-smiles"]}, [], {}, options)
    result = build_validation_index(raw, {"CCO": ["CCO", "not-a-smiles"]}, [],
                                    {"CCO": [{"candidate_index": 1, "smiles": "not-a-smiles"}]}, options)
    assert result["source_positions"].tolist() == [[0]]
    assert len(result["graph_rejections"]) == 1
    with pytest.raises(ValueError, match="exclusion"):
        build_validation_index(raw, {"CCO": ["CCO"]}, [],
                               {"CCO": [{"candidate_index": 0, "smiles": "CCO"}]}, options)


def test_metrics_full_denominator_ties_multiple_positives_and_official_argsort():
    from torchmetrics.functional.retrieval import retrieval_hit_rate, retrieval_reciprocal_rank

    scores = torch.tensor([[.9, .7, .7], [.4, .1, .2], [0., 0., 0.]])
    positives = torch.tensor([[False, True, True], [False, False, False], [False, False, False]])
    valid = torch.tensor([[True, True, True], [True, True, True], [False, False, False]])
    values, ranks = retrieval_metrics(scores, positives, valid, (1, 2, 5))
    assert ranks.tolist() == [2, 0, 0]
    assert values["queries"] == 3 and values["mrr"] == pytest.approx(1 / 6)
    assert values["top1"] == 0 and values["top2"] == pytest.approx(1 / 3)
    for k in (1, 2, 5):
        expected = sum(float(retrieval_hit_rate(s[v], p[v], top_k=k)) if v.any() else 0
                       for s, p, v in zip(scores, positives, valid, strict=True)) / 3
        assert values[f"top{k}"] == pytest.approx(expected)
    expected_mrr = sum(float(retrieval_reciprocal_rank(s[v], p[v])) if v.any() else 0
                       for s, p, v in zip(scores, positives, valid, strict=True)) / 3
    assert values["mrr"] == pytest.approx(expected_mrr)
    tied, ranks = retrieval_metrics(torch.ones((1, 3)), torch.tensor([[False, False, True]]), torch.ones((1, 3), dtype=torch.bool))
    assert ranks.tolist() == [3] and tied["top1"] == 0
    # Long equal-score arrays exercise the reference's non-stable tie behavior.
    for width in (17, 65, 256):
        tied_scores = torch.ones((1, width))
        tied_labels = torch.zeros((1, width), dtype=torch.bool)
        tied_labels[0, 0] = True
        actual, _ = retrieval_metrics(tied_scores, tied_labels, torch.ones_like(tied_labels))
        assert actual["stable_top1"] == 1
        for k in (1, 5, 10, 20):
            assert actual[f"top{k}"] == float(retrieval_hit_rate(tied_scores[0], tied_labels[0], top_k=k))
        if width == 17:
            # The pinned CPU reference uses topk for RR but argsort for HitRate.
            # Our MRR deliberately shares the HitRate ranking across all metrics.
            assert actual["mrr"] == pytest.approx(.1)
            assert float(retrieval_reciprocal_rank(tied_scores[0], tied_labels[0])) == pytest.approx(.0625)
    with pytest.raises(ValueError, match="Non-finite"):
        retrieval_metrics(torch.full((1, 1), float("nan")), torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 1), dtype=torch.bool))


def test_non_tied_metrics_are_candidate_permutation_invariant():
    scores = torch.tensor([[.9, .8, .7, .1]])
    labels = torch.tensor([[False, False, True, False]])
    mask = torch.ones_like(labels)
    expected, _ = retrieval_metrics(scores, labels, mask)
    order = torch.tensor([2, 0, 3, 1])
    actual, _ = retrieval_metrics(scores[:, order], labels[:, order], mask[:, order])
    assert expected == actual


def test_mrr_keeps_nonpositive_cosine_scores_in_the_same_ranking():
    from torchmetrics.functional.retrieval import retrieval_reciprocal_rank

    scores = torch.tensor([[.9, .8, .7]])
    labels = torch.tensor([[False, True, False]])
    actual, ranks = retrieval_metrics(scores, labels, torch.ones_like(labels))
    shifted, shifted_ranks = retrieval_metrics(scores - 1, labels, torch.ones_like(labels))
    assert actual == shifted and torch.equal(ranks, shifted_ranks)
    assert actual['mrr'] == .5 and ranks.tolist() == [2]
    # TorchMetrics 1.8.2 RR suppresses targets at scores <= 0; raw cosine
    # scores do not have probability semantics, so that filter is not used.
    assert float(retrieval_reciprocal_rank(scores[0], labels[0])) == .5
    assert float(retrieval_reciprocal_rank(scores[0] - 1, labels[0])) == 0.


def test_real_small_model_validation_preserves_rng_and_responds_to_weight_changes(tmp_path):
    index = build_index()
    model = SpecMolAlignModel(SiameseModel(embedding_dim=16, n_head=2, n_layer=1, dim_feedward=16, dim_target=16),
                              GINEEncoder(emb_dim=8, n_layers=1, size_feature_dim=4, dropout_rate=0.),
                              spec_dim=16, hidden_dim=16, final_dim=16, dropout_rate=0., tau=.07)
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=[1, 5, 10, 20])
    validator = AlignmentRetrievalValidator(index, settings, tmp_path)
    rng = torch.get_rng_state().clone()
    before = validator(model, torch.device("cpu"), 1, "synthetic")
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        for parameter in model.mol_proj.parameters():
            parameter.zero_()
    after = validator(model, torch.device("cpu"), 2, "synthetic")
    old = torch.load(tmp_path / "synthetic_epoch001.pt", weights_only=False)
    new = torch.load(tmp_path / "synthetic_epoch002.pt", weights_only=False)
    assert not torch.equal(old["scores"], new["scores"])
    assert before["queries"] == after["queries"] == 3
    with pytest.raises(FileExistsError):
        validator(model, torch.device("cpu"), 2, "synthetic")


@pytest.mark.parametrize('workers', [0, 2])
def test_cached_validation_matches_fresh_scores_and_reencodes_changed_weights(tmp_path, monkeypatch, workers):
    from SpecEmbedding.utils.molecule_graph_cache import audit_graph_cache, build_graph_cache, graph_cache_provenance
    from SpecEmbedding.utils.optimization_audit import audit_snapshot
    from SpecEmbedding.utils.retrieval_validation import load_validation_graph_cache

    source, _, _ = reusable_index(tmp_path, monkeypatch)
    index = torch.load(source, weights_only=False)
    provenance = graph_cache_provenance(index['mol_smiles'], index_sha256=sha256_file(source),
                                        dataset_manifest_sha256=index['dataset_manifest_sha256'])
    root = tmp_path / 'graphs'
    build_graph_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=2)
    audit_graph_cache(index['mol_smiles'], root, provenance, workers=1, chunk_size=2)
    store, _ = load_validation_graph_cache(source, index, root)
    model = SpecMolAlignModel(SiameseModel(embedding_dim=16, n_head=2, n_layer=1, dim_feedward=16, dim_target=16),
                              GINEEncoder(emb_dim=8, n_layers=1, size_feature_dim=4, dropout_rate=0.),
                              spec_dim=16, hidden_dim=16, final_dim=16, dropout_rate=0., tau=.07)
    settings = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=workers, top_k=[1, 5, 10, 20])
    fresh = AlignmentRetrievalValidator(index, settings, tmp_path / 'fresh')
    cached = AlignmentRetrievalValidator(index, settings, tmp_path / 'cached', graph_cache=store)
    rng = torch.get_rng_state().clone()
    for epoch in (1, 2):
        fresh(model, torch.device('cpu'), epoch, 'synthetic')
        with patch('SpecEmbedding.utils.retrieval_validation.smiles_to_graph', side_effect=AssertionError('Cache bypassed')):
            cached(model, torch.device('cpu'), epoch, 'synthetic')
        a = torch.load(tmp_path / 'fresh' / f'synthetic_epoch{epoch:03d}.pt', weights_only=False)
        b = torch.load(tmp_path / 'cached' / f'synthetic_epoch{epoch:03d}.pt', weights_only=False)
        assert torch.equal(a['scores'], b['scores']) and torch.equal(a['ranks'], b['ranks'])
        assert {k: v for k, v in a['metrics'].items() if k != 'seconds'} == {k: v for k, v in b['metrics'].items() if k != 'seconds'}
        assert b['raw_query_indices'] == [0, 2, 3] and b['metrics']['queries'] == 3
        audit_snapshot(tmp_path / 'cached' / f'synthetic_epoch{epoch:03d}.pt', index,
                       expected_graph_cache=cached.graph_cache_fingerprint)
        with pytest.raises(ValueError, match='graph cache'):
            audit_snapshot(tmp_path / 'cached' / f'synthetic_epoch{epoch:03d}.pt', index)
        if epoch == 1:
            old_scores = b['scores'].clone()
            with torch.no_grad():
                for parameter in model.mol_proj.parameters():
                    parameter.zero_()
        else:
            assert not torch.equal(old_scores, b['scores'])
    assert torch.equal(rng, torch.get_rng_state())
    for changed in ({**index, 'mol_smiles': list(reversed(index['mol_smiles']))},
                    {**index, 'dataset_manifest_sha256': 'f' * 64}):
        with pytest.raises(ValueError, match='cache'):
            AlignmentRetrievalValidator(changed, settings, graph_cache=store)


def test_graph_preparation_cli_builds_full_audited_inputs_without_gpu(tmp_path, monkeypatch):
    import prepare_validation_graph_cache as entry
    from SpecEmbedding.utils.retrieval_validation import load_validation_graph_cache

    source, _, options = reusable_index(tmp_path, monkeypatch)
    data, counts, exclusions, tokenizer = options
    def node(value):
        return SimpleNamespace(to_dict=lambda: value)
    monkeypatch.setattr(entry, 'config', SimpleNamespace(
        fulltrain=SimpleNamespace(expected_counts=node(counts), exclude_val_query_indices=exclusions),
        data=SimpleNamespace(tokenizer=node(tokenizer)),
        retrieval_validation=SimpleNamespace(graph_cache_preparation=node({'workers': 1, 'chunk_size': 2}))))
    root = tmp_path / 'cache'
    args = ['--data-path', str(data), '--index', str(source), '--output', str(root)]
    with patch('torch.cuda.is_available', side_effect=AssertionError('CPU preparation touched CUDA')):
        entry.main(args)
    report = json.loads((root / 'preparation.json').read_text())
    assert report['state'] == 'complete_validation_graph_cache_preparation'
    assert report['audit']['audited_molecules'] == 3 and not report['model_encoding_performed']
    _, receipt = load_validation_graph_cache(source, torch.load(source, weights_only=False), root)
    assert receipt == report['graph_cache']
    with pytest.raises(SystemExit):
        entry.main(args)


def test_graph_cache_queue_binds_same_cache_to_both_gpu_stages(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, source_dir=tmp_path / 'source', legacy_tsv=tmp_path / 'old',
                           gpu=None, gpus=[0, 1], device='cuda:0', optimize_alignment=True,
                           baseline_checkpoint=tmp_path / 'baseline.pth', prepared_data=tmp_path / 'prepared',
                           prepared_validation_index=tmp_path / 'index.pt', validation_graph_cache=tmp_path / 'graphs')
    stages = runner.commands(args)
    assert [s['name'] for s in stages] == ['import_v15', 'import_validation', 'baseline_validation', 'alignment42']
    for stage, flag in ((stages[2], '--graph-cache'), (stages[3], '--validation-graph-cache')):
        assert stage['command'][stage['command'].index(flag) + 1] == str(args.validation_graph_cache)


def test_retrieval_selection_can_prefer_higher_loss_and_preserves_topk_frontier(tmp_path):
    model = torch.nn.Linear(1, 1)
    trajectory = [{"top1": .1, "mrr": .2, "top5": .5, "top10": .6, "top20": .8},
                  {"top1": .2, "mrr": .3, "top5": .4, "top10": .5, "top20": .7},
                  {"top1": .2, "mrr": .25, "top5": .3, "top10": .4, "top20": .6}]
    def validator(model, device, epoch, stage):
        return trajectory[epoch - 1]
    trainer = TrainerAlign(model, [], [], torch.device("cpu"), str(tmp_path), retrieval_validator=validator)
    def train(optimizer, epoch, stage):
        with torch.no_grad():
            model.weight.fill_(epoch)
        return 1.
    trainer.train_epoch = train
    trainer.validate = lambda epoch, stage: float(epoch)
    trainer.fit(10, torch.optim.SGD(model.parameters(), lr=.1), stage_name="stage2", patience=1)
    summary = trainer.stage_summaries["stage2"]
    assert summary["best_epoch"] == 2 and summary["best_val_loss"] == 2.
    assert model.weight.item() == 2 and summary["stop_epoch"] == 3
    assert {record["epoch"] for record in summary["pareto_frontier"]} == {1, 2}
    assert (tmp_path / "candidate_stage2_epoch001.pth").exists()
    assert not (tmp_path / "candidate_stage2_epoch003.pth").exists()


def test_optimization_queue_never_dispatches_test_or_rerank(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, source_dir=tmp_path / "source", legacy_tsv=tmp_path / "old",
                           gpu=None, gpus=[0, 1], device="cuda:0", optimize_alignment=True,
                           baseline_checkpoint=tmp_path / "baseline.pth", prepared_data=tmp_path / "prepared")
    stages = runner.commands(args)
    assert [step["name"] for step in stages] == ["import_v15", "prepare_validation", "baseline_validation", "alignment42"]
    assert [step["gpu"] for step in stages] == [False, False, True, True]
    assert "--validation-index" in stages[-1]["command"]
    assert all("rerank" not in " ".join(step.get("command", [])) for step in stages)


def test_index_receipt_prevents_mutated_labels(tmp_path):
    import json

    from SpecEmbedding.utils.fulltrain import sha256_file
    from SpecEmbedding.utils.retrieval_validation import load_validation_index
    index = build_index()
    path = tmp_path / "index.pt"
    torch.save(index, path)
    path.with_suffix(".json").write_text(json.dumps({"sha256": sha256_file(path)}))
    mutated = copy.deepcopy(index)
    mutated["positive_mask"].fill_(True)
    torch.save(mutated, path)
    with patch("SpecEmbedding.utils.retrieval_validation.verify_dataset") as verify:
        with pytest.raises(ValueError, match="fingerprint"):
            load_validation_index(path, tmp_path, {"val": 4}, [1], {})
        verify.assert_not_called()
