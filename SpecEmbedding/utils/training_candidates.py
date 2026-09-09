"""Verified natural training candidates and deterministic identity-aware sampling."""

import hashlib
import json
import operator
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import verify_dataset

SAMPLING_POLICY = (
    "Uniform distinct negative 2D identities without replacement; uniform source entry within each selected identity; "
    "min(budget, available); private NumPy SeedSequence(seed, epoch, raw_query_index)"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    _require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def validate_training_candidate_metadata(metadata, raw, source_candidates, rejections, expected_queries):
    """Compare all serialized query mappings, candidate entries and labels with their source."""
    _require(metadata["schema_version"] == 1 and metadata["kind"] == "diagnostic_candidate_metadata"
             and metadata["split"] == "train" and metadata["candidate_type"] == "mass"
             and metadata["forcing"] is False, "Unsupported training candidate metadata protocol")
    _require(not {"val", "test", "heldout_structure_overlap"} & metadata.keys(),
             "Held-out supervision fields in training metadata")
    targets, target_keys = metadata["target_smiles"], metadata["target_identity_2d"]
    smiles, keys = metadata["mol_smiles"], metadata["mol_identity_2d"]
    _require(len(targets) > 0 and len(targets) == len(target_keys) == len(set(targets)),
             "Invalid training target inventory")
    _require(len(smiles) > 0 and len(smiles) == len(keys) == len(set(smiles)),
             "Invalid training candidate molecule inventory")
    _require(all(isinstance(key, str) and len(key) == 14 for key in [*target_keys, *keys]),
             "Invalid training candidate 2D identities")
    ids, positions, positive = (metadata[name] for name in ("candidate_indices", "source_positions", "positive_mask"))
    query_rows, raw_indices = metadata["query_target_rows"], metadata["raw_query_indices"]
    _require(all(isinstance(value, np.ndarray) for value in (ids, positions, positive, query_rows, raw_indices)),
             "Training candidate arrays must be NumPy arrays")
    _require(ids.ndim == 2 and ids.shape == positions.shape == positive.shape and ids.shape[0] == len(targets)
             and 0 < ids.shape[1] <= 256 and ids.dtype.kind == "i" and positions.dtype.kind == "i"
             and positive.dtype == np.bool_, "Invalid training candidate matrix shape or dtype")
    _require(ids.min() >= -1 and ids.max() < len(smiles) and not positive[ids < 0].any()
             and np.all(positions[ids < 0] == -1), "Invalid candidate index, padding or positive label")
    _require(query_rows.ndim == raw_indices.ndim == 1 and query_rows.dtype.kind in "iu"
             and raw_indices.dtype.kind in "iu" and len(raw) == len(query_rows) == expected_queries
             and np.array_equal(raw_indices, np.arange(expected_queries))
             and query_rows.min() >= 0 and query_rows.max() < len(targets),
             "Training candidate metadata does not cover every raw query in order")
    observed_targets = set()
    for query, row in zip(raw, query_rows, strict=True):
        _require(targets[row] == query.get("smiles") and target_keys[row] == query.get("identity_2d"),
                 "Training query target/identity mapping differs from the source")
        observed_targets.add(int(row))
    _require(len(observed_targets) == len(targets), "Unused target rows in training candidate metadata")
    seen = np.zeros(len(smiles), dtype=np.bool_)
    observed_rejections = []
    entries, positive_queries, no_negative_queries = 0, 0, 0
    multiplicity = np.bincount(query_rows, minlength=len(targets))
    for row, target in enumerate(targets):
        source = source_candidates[target]
        _require(0 < len(source) <= 256, "Source training pool is outside the 256-candidate protocol")
        rejected = {item["candidate_index"]: item["smiles"] for item in rejections.get(target, [])}
        for position, value in rejected.items():
            _require(0 <= position < len(source) and source[position] == value and value != target,
                     "Training graph exclusion differs from the source or excludes the target")
            observed_rejections.append({"target": target, "source_position": position, "smiles": value})
        valid = ids[row] >= 0
        expected_positions = [j for j in range(len(source)) if j not in rejected]
        _require(positions[row, valid].tolist() == expected_positions,
                 "Training source order or graph exclusions differ")
        row_ids = ids[row, valid]
        row_smiles = [smiles[int(j)] for j in row_ids]
        row_keys = [keys[int(j)] for j in row_ids]
        _require(row_smiles == [source[j] for j in expected_positions], "Training candidate entries differ from source")
        _require(target in row_smiles and positive[row, valid].tolist() == [key == target_keys[row] for key in row_keys],
                 "Training positive mask differs from the source identities")
        _require(positive[row].any(), "Training source pool lost its natural positive")
        seen[row_ids] = True
        entries += len(row_ids)
        positive_queries += int(multiplicity[row])
        no_negative_queries += int(multiplicity[row]) * int(all(key == target_keys[row] for key in row_keys))
    _require(seen.all(), "Unused molecules in training candidate metadata")
    _require(metadata["graph_rejections"] == observed_rejections, "Training graph rejection inventory differs")
    return {"queries": expected_queries, "target_rows": len(targets), "candidate_strings": len(smiles),
            "candidate_entries": entries, "positive_queries": positive_queries,
            "queries_with_no_distinct_negative": no_negative_queries}


@dataclass(frozen=True)
class NegativeCandidates:
    molecule_indices: np.ndarray
    source_positions: np.ndarray
    identity_2d: tuple[str, ...]


class TrainingCandidateIndex:
    """Only construct after full source validation; sampling never modifies the source pool."""

    def __init__(self, metadata, provenance, *, pool_cache_size):
        self.pool_cache_size = _integer(pool_cache_size, "pool_cache_size", 1)
        self.metadata = metadata
        self.provenance = {**provenance, "sampling_policy": SAMPLING_POLICY}
        self._pool_cache = OrderedDict()
        for name in ("candidate_indices", "source_positions", "positive_mask", "query_target_rows", "raw_query_indices"):
            metadata[name].setflags(write=False)

    def __len__(self):
        return len(self.metadata["query_target_rows"])

    def _negative_groups(self, row):
        if row in self._pool_cache:
            groups = self._pool_cache.pop(row)
            self._pool_cache[row] = groups
            return groups
        metadata = self.metadata
        target_key = metadata["target_identity_2d"][row]
        by_identity = {}
        for mol, position in zip(metadata["candidate_indices"][row], metadata["source_positions"][row], strict=True):
            if mol < 0:
                continue
            key = metadata["mol_identity_2d"][int(mol)]
            if key != target_key:
                by_identity.setdefault(key, []).append((int(mol), int(position)))
        groups = tuple((key, tuple(entries)) for key, entries in by_identity.items())
        self._pool_cache[row] = groups
        if len(self._pool_cache) > self.pool_cache_size:
            self._pool_cache.popitem(last=False)
        return groups

    def sample(self, raw_query_index, *, negative_count, seed, epoch):
        """Return at most the budget, excluding every positive identity; retain queries with small/empty pools."""
        query = _integer(raw_query_index, "raw_query_index")
        _require(query < len(self), "Training query index is outside the complete split")
        budget = _integer(negative_count, "negative_count", 1)
        seed, epoch = _integer(seed, "seed"), _integer(epoch, "epoch", 1)
        row = int(self.metadata["query_target_rows"][query])
        groups = self._negative_groups(row)
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, query]))
        chosen = generator.choice(len(groups), size=min(budget, len(groups)), replace=False)
        molecules, positions, keys = [], [], []
        for index in chosen:
            key, entries = groups[int(index)]
            molecule, position = entries[int(generator.integers(len(entries)))]
            molecules.append(molecule)
            positions.append(position)
            keys.append(key)
        molecule_array = np.asarray(molecules, dtype=np.int32)
        position_array = np.asarray(positions, dtype=np.int16)
        molecule_array.setflags(write=False)
        position_array.setflags(write=False)
        return NegativeCandidates(molecule_array, position_array, tuple(keys))


def load_training_candidates(path, data_path, expected_counts, exclusions, *, pool_cache_size):
    """Require the completed preparation and independent verification, then recheck every source entry."""
    path, data_path = Path(path), Path(data_path)
    pool_cache_size = _integer(pool_cache_size, "pool_cache_size", 1)
    report = verify_dataset(data_path, expected_counts, exclusions)
    receipt_path, verification_path = path.parent / "receipt.json", path.parent / "verification.json"
    receipt_bytes, verification_bytes = receipt_path.read_bytes(), verification_path.read_bytes()
    receipt, verification = json.loads(receipt_bytes), json.loads(verification_bytes)
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    verification_digest = hashlib.sha256(verification_bytes).hexdigest()
    digest = sha256_file(path)
    _require(receipt["state"] == "complete" and verification["state"] == "verified"
             and receipt["metadata_file"] == path.name and receipt["metadata_bytes"] == path.stat().st_size
             and receipt["metadata_sha256"] == verification["metadata_sha256"] == digest
             and verification["receipt_sha256"] == receipt_digest,
             "Training candidate preparation or verification fingerprint mismatch")
    manifest_digest = sha256_file(data_path / "dataset_manifest.json")
    _require(receipt["dataset_manifest_sha256"] == manifest_digest, "Training candidate dataset manifest mismatch")
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    _require(metadata["source_data_manifest_sha256"] == manifest_digest
             and metadata["source_train_sha256"] == report["outputs"]["train.pkl"]["sha256"]
             and metadata["source_candidates_sha256"] == report["outputs"]["candidates_mass.pkl"]["sha256"]
             and metadata["identity_policy"] == report["identity_policy"]
             and metadata["graph_policy"] == report["graph_policy"], "Training candidate source or policy mismatch")
    with (data_path / "train.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    with (data_path / "candidates_mass.pkl").open("rb") as handle:
        source_candidates = pickle.load(handle)
    rejections = {}
    for line in (data_path / "invalid_graph_mass.jsonl").read_text().splitlines():
        record = json.loads(line)
        rejections[record["target"]] = record["rejected"]
    observed = validate_training_candidate_metadata(metadata, raw, source_candidates, rejections, expected_counts["train"])
    _require(verification["raw_queries_compared"] == observed["queries"]
             and verification["target_rows_compared"] == observed["target_rows"]
             and verification["candidate_entries_compared"] == observed["candidate_entries"]
             and verification["positive_mask_matches_identity_and_exact_source_target"] is True
             and verification["candidate_source_order_and_graph_exclusions_match"] is True
             and verification["heldout_labels_in_training_metadata"] is False,
             "Training candidate verification scope differs from observed data")
    _require(sha256_file(path) == digest and sha256_file(receipt_path) == receipt_digest
             and sha256_file(verification_path) == verification_digest
             and sha256_file(data_path / "dataset_manifest.json") == manifest_digest,
             "Training candidate input changed during loading")
    for name in ("train.pkl", "candidates_mass.pkl"):
        _require(sha256_file(data_path / name) == report["outputs"][name]["sha256"],
                 "Training candidate source changed during loading")
    _require(sha256_file(data_path / "invalid_graph_mass.jsonl")
             == report["candidate_audits"]["mass"]["graph_rejections"]["sha256"],
             "Training graph exclusions changed during loading")
    return TrainingCandidateIndex(metadata, {"path": str(path.resolve()), "sha256": digest,
                                  "receipt_sha256": receipt_digest, "verification_sha256": verification_digest,
                                  "dataset_manifest_sha256": manifest_digest, "observed": observed},
                                  pool_cache_size=pool_cache_size)
