import copy
import json
import pickle
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.training_candidates import (
    TrainingCandidateIndex,
    load_training_candidates,
    validate_training_candidate_metadata,
)


def fixture_data():
    key_a, key_b, key_c = "A" * 14, "B" * 14, "C" * 14
    raw = [{"smiles": "CCO", "identity_2d": key_a}, {"smiles": "CC", "identity_2d": key_b},
           {"smiles": "CCO", "identity_2d": key_a}]
    source = {"CCO": ["CCO", "OCC", "bad", "CC", "C(C)", "CCC"], "CC": ["CC", "C(C)", "CCC"]}
    rejections = {"CCO": [{"candidate_index": 2, "smiles": "bad"}]}
    metadata = {"schema_version": 1, "kind": "diagnostic_candidate_metadata", "split": "train",
                "candidate_type": "mass", "forcing": False, "target_smiles": ["CCO", "CC"],
                "target_identity_2d": [key_a, key_b], "mol_smiles": ["CCO", "OCC", "CC", "C(C)", "CCC"],
                "mol_identity_2d": [key_a, key_a, key_b, key_b, key_c],
                "candidate_indices": np.asarray([[0, 1, 2, 3, 4], [2, 3, 4, -1, -1]], dtype=np.int32),
                "source_positions": np.asarray([[0, 1, 3, 4, 5], [0, 1, 2, -1, -1]], dtype=np.int16),
                "positive_mask": np.asarray([[True, True, False, False, False], [True, True, False, False, False]]),
                "query_target_rows": np.asarray([0, 1, 0], dtype=np.int32),
                "raw_query_indices": np.arange(3, dtype=np.int32),
                "graph_rejections": [{"target": "CCO", "source_position": 2, "smiles": "bad"}]}
    return metadata, raw, source, rejections


def validated_index(*, cache_size=1):
    metadata, raw, source, rejections = fixture_data()
    observed = validate_training_candidate_metadata(metadata, raw, source, rejections, 3)
    return TrainingCandidateIndex(metadata, {"observed": observed}, pool_cache_size=cache_size)


def test_sampling_excludes_all_positives_and_duplicate_negative_identities():
    index = validated_index()
    before = copy.deepcopy(index.metadata)
    result = index.sample(0, negative_count=16, seed=42, epoch=1)
    assert set(result.identity_2d) == {"B" * 14, "C" * 14}
    assert len(result.molecule_indices) == 2
    assert len(set(result.identity_2d)) == len(result.identity_2d)
    assert all(value not in {0, 1} for value in result.molecule_indices)
    for molecule, position in zip(result.molecule_indices, result.source_positions, strict=True):
        assert fixture_data()[2]["CCO"][position] == index.metadata["mol_smiles"][molecule]
    assert len(index) == 3
    for name in ("candidate_indices", "source_positions", "positive_mask", "query_target_rows", "raw_query_indices"):
        assert np.array_equal(index.metadata[name], before[name])
        assert not index.metadata[name].flags.writeable
    assert not result.molecule_indices.flags.writeable


def test_sampling_is_order_independent_and_preserves_all_global_rng_streams():
    index = validated_index()
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = index.sample(0, negative_count=1, seed=42, epoch=1)
    index.sample(1, negative_count=1, seed=42, epoch=1)  # Evicts the first target's pool.
    again = index.sample(0, negative_count=1, seed=42, epoch=1)
    assert np.array_equal(first.molecule_indices, again.molecule_indices)
    assert np.array_equal(first.source_positions, again.source_positions)
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_rng, strict=True))
    choices = {tuple(index.sample(0, negative_count=1, seed=42, epoch=epoch).molecule_indices) for epoch in range(1, 11)}
    assert len(choices) > 1 and len(index._pool_cache) == 1


def test_query_with_only_positive_aliases_is_retained_with_empty_negative_sample():
    metadata, raw, source, rejections = fixture_data()
    source["CC"] = ["CC", "C(C)"]
    metadata["candidate_indices"][1, 2] = -1
    metadata["source_positions"][1, 2] = -1
    observed = validate_training_candidate_metadata(metadata, raw, source, rejections, 3)
    index = TrainingCandidateIndex(metadata, {}, pool_cache_size=1)
    assert observed["queries_with_no_distinct_negative"] == 1
    assert len(index) == 3 and len(index.sample(1, negative_count=16, seed=42, epoch=1).molecule_indices) == 0


@pytest.mark.parametrize("changes", [
    {"raw_query_index": 3}, {"raw_query_index": -1}, {"raw_query_index": True},
    {"negative_count": 0}, {"negative_count": 2.5}, {"epoch": 0}, {"seed": -1},
])
def test_invalid_sampling_configuration_fails(changes):
    settings = {"raw_query_index": 0, "negative_count": 1, "seed": 42, "epoch": 1, **changes}
    with pytest.raises(ValueError):
        validated_index().sample(**settings)


@pytest.mark.parametrize("damage", ["query_order", "target_mapping", "label", "position", "candidate", "padding", "forcing", "extra_molecule"])
def test_source_validation_rejects_incomplete_or_changed_inputs(damage):
    metadata, raw, source, rejections = fixture_data()
    if damage == "query_order":
        metadata["raw_query_indices"] = metadata["raw_query_indices"][::-1]
    elif damage == "target_mapping":
        metadata["query_target_rows"][0] = 1
    elif damage == "label":
        metadata["positive_mask"][0, 1] = False
    elif damage == "position":
        metadata["source_positions"][0, 2] = 2
    elif damage == "candidate":
        metadata["candidate_indices"][0, 2] = 4
    elif damage == "padding":
        metadata["positive_mask"][1, 4] = True
    elif damage == "forcing":
        metadata["forcing"] = True
    else:
        metadata["mol_smiles"].append("CCCC")
        metadata["mol_identity_2d"].append("D" * 14)
    with pytest.raises(ValueError):
        validate_training_candidate_metadata(metadata, raw, source, rejections, 3)


def saved_fixture(tmp_path):
    metadata, raw, source, rejections = fixture_data()
    directory, prepared = tmp_path / "audit", tmp_path / "data"
    directory.mkdir()
    prepared.mkdir()
    for name, value in (("train.pkl", raw), ("candidates_mass.pkl", source)):
        (prepared / name).write_bytes(pickle.dumps(value))
    graph_path = prepared / "invalid_graph_mass.jsonl"
    graph_path.write_text(json.dumps({"target": "CCO", "rejected": rejections["CCO"]}) + "\n")
    report = {"outputs": {name: {"sha256": sha256_file(prepared / name)} for name in ("train.pkl", "candidates_mass.pkl")},
              "identity_policy": "synthetic_2d", "graph_policy": "synthetic_graph",
              "candidate_audits": {"mass": {"graph_rejections": {"sha256": sha256_file(graph_path)}}}}
    (prepared / "dataset_manifest.json").write_text(json.dumps(report))
    manifest_sha = sha256_file(prepared / "dataset_manifest.json")
    metadata.update(source_data_manifest_sha256=manifest_sha, source_train_sha256=report["outputs"]["train.pkl"]["sha256"],
                    source_candidates_sha256=report["outputs"]["candidates_mass.pkl"]["sha256"],
                    identity_policy=report["identity_policy"], graph_policy=report["graph_policy"])
    path = directory / "metadata.pkl"
    path.write_bytes(pickle.dumps(metadata))
    digest = sha256_file(path)
    receipt = {"state": "complete", "metadata_file": path.name, "metadata_bytes": path.stat().st_size,
               "metadata_sha256": digest, "dataset_manifest_sha256": manifest_sha}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    verification = {"state": "verified", "metadata_sha256": digest, "receipt_sha256": sha256_file(directory / "receipt.json"),
                    "raw_queries_compared": 3, "target_rows_compared": 2, "candidate_entries_compared": 8,
                    "positive_mask_matches_identity_and_exact_source_target": True,
                    "candidate_source_order_and_graph_exclusions_match": True, "heldout_labels_in_training_metadata": False}
    (directory / "verification.json").write_text(json.dumps(verification))
    return path, prepared, report


def test_loader_binds_receipts_and_revalidates_every_source_entry(tmp_path):
    path, prepared, report = saved_fixture(tmp_path)
    with patch("SpecEmbedding.utils.training_candidates.verify_dataset", return_value=report) as verify:
        index = load_training_candidates(path, prepared, {"train": 3}, [], pool_cache_size=1)
    verify.assert_called_once_with(prepared, {"train": 3}, [])
    assert len(index) == 3 and index.provenance["sha256"] == sha256_file(path)
    assert index.provenance["observed"]["candidate_entries"] == 8


@pytest.mark.parametrize("damage", ["metadata", "receipt", "verification_scope", "source_during_load"])
def test_loader_rejects_stale_fingerprints_and_insufficient_verification(tmp_path, damage):
    path, prepared, report = saved_fixture(tmp_path)
    if damage == "metadata":
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage == "receipt":
        receipt_path = path.parent / "receipt.json"
        receipt_path.write_text(receipt_path.read_text() + " ")
    elif damage == "verification_scope":
        verification_path = path.parent / "verification.json"
        value = json.loads(verification_path.read_text())
        value["raw_queries_compared"] = 2
        verification_path.write_text(json.dumps(value))
    def source_check(*args):
        if damage == "source_during_load":
            source_path = prepared / "train.pkl"
            source_path.write_bytes(source_path.read_bytes() + b"changed")
        return report
    with patch("SpecEmbedding.utils.training_candidates.verify_dataset", side_effect=source_check), pytest.raises(ValueError):
        load_training_candidates(path, prepared, {"train": 3}, [], pool_cache_size=1)
