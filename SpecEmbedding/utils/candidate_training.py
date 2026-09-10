"""Pinned inputs and complete observed-sampling audits for candidate alignment."""

import copy
import hashlib
import json
import logging
import math
from numbers import Real
from pathlib import Path

import numpy as np

from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.training_candidates import _integer, load_training_candidates

LOSS_NAME = "baseline_bidirectional_inbatch_plus_per_query_candidate_ce"
GRAPH_AUGMENTATION = "Same probability/node/edge settings as training positive graphs"


def validate_candidate_settings(settings):
    required = {"enabled", "negative_count", "loss_weight", "pool_cache_size", "graph_cache_size"}
    if (not isinstance(settings, dict) or set(settings) not in (required, required | {"sampling"})
            or not isinstance(settings["enabled"], bool)):
        raise ValueError("Incomplete candidate-supervision configuration")
    if _integer(settings["negative_count"], "negative_count", 1) > 255:
        raise ValueError("Natural top-256 supervision supports at most 255 distinct negative identities")
    _integer(settings["pool_cache_size"], "pool_cache_size", 1)
    _integer(settings["graph_cache_size"], "graph_cache_size")
    weight = settings["loss_weight"]
    if not isinstance(weight, Real) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Candidate loss weight must be finite and positive")
    if 'sampling' in settings:
        from SpecEmbedding.utils.structural_sampling import validate_structural_sampling
        if not settings['enabled']:
            raise ValueError('Structural sampling requires explicitly enabled candidate supervision')
        validate_structural_sampling(settings['sampling'], settings['negative_count'])


def validate_candidate_sampling_binding(index, settings, receipt):
    from SpecEmbedding.utils.structural_sampling import StructuralTrainingCandidateIndex
    structural = 'sampling' in settings
    if structural != isinstance(index, StructuralTrainingCandidateIndex):
        raise ValueError('Candidate sampling strategy differs from the explicitly configured index')
    if structural and (index.sampling_settings != settings['sampling'] or receipt['settings'] != settings
                       or receipt['provenance'] != index.provenance):
        raise ValueError('Structural sampling settings/provenance do not match the pinned receipt')


def grouped_query_order(index):
    metadata = index.metadata
    keys = np.asarray([metadata["target_identity_2d"][row] for row in metadata["query_target_rows"]])
    return np.argsort(keys, kind="stable").astype(np.int64)


def build_candidate_training_input(metadata_path, data_path, settings, expected_counts, exclusions):
    validate_candidate_settings(settings)
    if not settings["enabled"]:
        raise ValueError("Candidate input preparation requires explicit activation")
    index = load_training_candidates(metadata_path, data_path, expected_counts, exclusions,
                                     pool_cache_size=settings["pool_cache_size"])
    if 'sampling' in settings:
        from SpecEmbedding.utils.structural_sampling import load_structural_training_candidates
        index = load_structural_training_candidates(index, settings['sampling'])
    order = grouped_query_order(index)
    receipt = {"schema_version": 2 if 'sampling' in settings else 1, "state": "verified_training_candidates",
               "settings": copy.deepcopy(settings),
               "provenance": index.provenance,
               "dataset_to_raw_query_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()}
    return index, receipt


def candidate_source_inputs(receipt):
    provenance = receipt["provenance"]
    path = Path(provenance["path"])
    result = {
        "training_candidates": {"path": str(path), "sha256": provenance["sha256"]},
        "training_candidate_preparation": {"path": str(path.parent / "receipt.json"), "sha256": provenance["receipt_sha256"]},
        "training_candidate_verification": {"path": str(path.parent / "verification.json"), "sha256": provenance["verification_sha256"]},
    }
    if 'structural_sampling' in provenance:
        from SpecEmbedding.utils.training_similarity import similarity_source_inputs
        result.update(similarity_source_inputs(provenance['structural_sampling']['cache']))
    return result


def read_candidate_training_input(path, data_path, settings, expected_counts, exclusions):
    path = Path(path)
    before = sha256_file(path)
    receipt = json.loads(path.read_text())
    if receipt["settings"] != settings:
        raise ValueError("Candidate input configuration differs from the pinned training settings")
    index, observed = build_candidate_training_input(receipt["provenance"]["path"], data_path, settings, expected_counts, exclusions)
    if observed != receipt or sha256_file(path) != before:
        raise ValueError("Candidate input provenance changed or differs from full source verification")
    return index, receipt, {"path": str(path.resolve()), "sha256": before}


def audit_candidate_training(directory, stage, input_path, settings, seed, batch_size, data_path, counts, exclusions,
                             *, fingerprint_cache=None, graph_fingerprint=False, spectrum_metadata=None):
    """Replay every observed query's negative sample; never run a model or use test labels."""
    record = stage.get("candidate_training")
    if settings is not None:
        validate_candidate_settings(settings)
    if settings is None or not settings["enabled"]:
        if (record is not None or input_path is not None or spectrum_metadata is not None
                or (Path(directory) / "candidate_training").exists()):
            raise ValueError("Unexpected candidate supervision in an inactive/legacy run")
        return {"state": "disabled"}, {}
    if record is None or input_path is None:
        raise ValueError("Missing candidate training completion evidence")
    if graph_fingerprint and fingerprint_cache is None:
        raise ValueError('Graph fingerprint candidate audit requires both input representations')
    index, input_receipt, input_fingerprint = read_candidate_training_input(input_path, data_path, settings, counts, exclusions)
    validate_candidate_sampling_binding(index, settings, input_receipt)
    if 'sampling' in settings:
        from SpecEmbedding.utils.structural_sampling import reference_structural_sample
    expected_provenance = {**index.provenance, "negative_count": settings["negative_count"], "seed": seed,
                           "graph_cache_size": settings["graph_cache_size"],
                           "dataset_to_raw_query_sha256": input_receipt["dataset_to_raw_query_sha256"],
                           "negative_graph_augmentation": GRAPH_AUGMENTATION}
    if fingerprint_cache is not None:
        from SpecEmbedding.utils.fingerprint_alignment_inputs import fingerprint_training_provenance
        from SpecEmbedding.utils.fingerprint_cache import fingerprint_provenance, load_fingerprint_cache
        source = fingerprint_cache['provenance']
        expected_source = fingerprint_provenance(index.metadata['mol_smiles'], index_sha256=index.provenance['sha256'],
                                                 dataset_manifest_sha256=index.provenance['dataset_manifest_sha256'],
                                                 radius=source['options']['radius'], bits=source['options']['bits'])
        _, verified = load_fingerprint_cache(index.metadata['mol_smiles'], fingerprint_cache['directory'], expected_source)
        if verified != fingerprint_cache:
            raise ValueError('Candidate fingerprint source differs from the declared model inputs')
        expected_provenance = fingerprint_training_provenance(expected_provenance, verified,
                                                              graph_fingerprint=graph_fingerprint)
    if spectrum_metadata is not None:
        from SpecEmbedding.utils.adduct_metadata import SpectrumMetadata
        if (not isinstance(spectrum_metadata, SpectrumMetadata) or spectrum_metadata.payload['split'] != 'train'
                or spectrum_metadata.provenance['source']['dataset_manifest_sha256'] != index.provenance['dataset_manifest_sha256']
                or not np.array_equal(spectrum_metadata.raw_query_indices, np.arange(len(index)))):
            raise ValueError('Candidate audit requires the complete bound training adduct input')
        expected_provenance['spectrum_metadata'] = spectrum_metadata.provenance
    if (record["loss"] != LOSS_NAME or record["candidate_loss_weight"] != settings["loss_weight"]
            or record["data"] != expected_provenance or len(record["epochs"]) != stage["stop_epoch"]):
        raise ValueError("Candidate loss, input or trajectory differs from the pinned experiment")
    hashes = {input_fingerprint["path"]: input_fingerprint["sha256"]}
    inputs = candidate_source_inputs(input_receipt)
    hashes.update({item["path"]: item["sha256"] for item in inputs.values()})
    for epoch, saved in enumerate(record["epochs"], 1):
        stem = f"stage2_epoch{epoch:03d}"
        path = Path(directory) / "candidate_training" / f"{stem}.json"
        if json.loads(path.read_text()) != saved:
            raise ValueError("Candidate epoch file differs from checkpoint selection metadata")
        hashes[str(path)] = sha256_file(path)
        order_path = path.with_suffix(".npy")
        order = np.load(order_path, allow_pickle=False)
        order_sha = sha256_file(order_path)
        if (saved["query_order_file"] != order_path.name or saved["query_order_sha256"] != order_sha
                or order.dtype != np.int64 or order.shape != (len(index),)
                or not np.array_equal(np.sort(order), np.arange(len(index)))):
            raise ValueError("Candidate observed order lost, duplicated or changed a raw query")
        hashes[str(order_path)] = order_sha
        if spectrum_metadata is None:
            if 'observed_adduct_order_sha256' in saved:
                raise ValueError('Unexpected adduct order in an unconditioned candidate run')
        else:
            # Independently reconstruct from original query IDs, not the training Dataset mapping.
            pairs = np.column_stack((order, spectrum_metadata.adduct_ids[order])).astype('<i8')
            if saved.get('observed_adduct_order_sha256') != hashlib.sha256(pairs.tobytes()).hexdigest():
                raise ValueError('Observed adduct order differs from the complete raw-query binding')
        observed_hash = hashlib.sha256()
        negative_counts = np.zeros(len(index), dtype=np.int64)
        batch_sizes = []
        for start in range(0, len(order), batch_size):
            queries = order[start:start + batch_size]
            batch_sizes.append(len(queries))
            molecules, positions, ptr = [], [], [0]
            for query in queries:
                if 'sampling' in settings:
                    sample = reference_structural_sample(index, int(query), negative_count=settings["negative_count"],
                                                         seed=seed, epoch=epoch)
                else:
                    sample = index.sample(int(query), negative_count=settings["negative_count"], seed=seed, epoch=epoch)
                negative_counts[query] = len(sample.molecule_indices)
                molecules.extend(sample.molecule_indices)
                positions.extend(sample.source_positions)
                ptr.append(len(molecules))
            for values in (queries, ptr, molecules, positions):
                observed_hash.update(len(values).to_bytes(8, "little"))
                observed_hash.update(np.asarray(values, dtype="<i8").tobytes())
        expected = {"stage": "stage2", "epoch": epoch, "seed": seed, "queries": len(index),
                    "unique_queries": len(index), "negative_samples": int(negative_counts.sum()),
                    "minimum_negatives": int(negative_counts.min()), "maximum_negatives": int(negative_counts.max()),
                    "queries_without_negatives": int((negative_counts == 0).sum()), "batch_sizes": batch_sizes,
                    "observed_query_sample_order_sha256": observed_hash.hexdigest()}
        if (any(saved[key] != value for key, value in expected.items())
                or not math.isfinite(saved["candidate_loss_query_mean"]) or saved["candidate_loss_query_mean"] < 0):
            raise ValueError("Candidate source sampling, coverage or statistics failed complete replay")
        logging.info("Replayed all %s training candidate queries for epoch %s", len(index), epoch)
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError("Candidate training input/artifact changed during replay")
    return {"state": "verified_full_candidate_replay", "epochs": stage["stop_epoch"], "queries_per_epoch": len(index),
            "input_provenance": expected_provenance, "negative_sampling_replayed": True}, hashes
