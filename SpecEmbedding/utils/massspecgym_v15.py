"""Version-pinned, read-only source audits and isolated MassSpecGym 1.5 preparation."""

import json
import logging
import multiprocessing
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from SpecEmbedding.utils.fulltrain import sha256_file


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecule: {smiles}")
    key = Chem.MolToInchiKey(mol)
    if not key or len(key.split("-")[0]) != 14:
        raise ValueError(f"Missing 2D InChIKey: {smiles}")
    return key.split("-")[0]


def audit_candidate_list(item):
    """Inspect every source candidate; preserve the original list and order."""
    target, values = item
    key = identity(target)
    keys = [identity(value) for value in values]
    if not values or target not in values or key not in keys:
        raise ValueError(f"Official candidate list does not cover target: {target}")
    return {
        "lists": 1,
        "entries": len(values),
        "duplicate_strings": len(values) - len(set(values)),
        "duplicate_2d_identities": len(keys) - len(set(keys)),
        "lists_with_duplicate_2d_identities": int(len(keys) != len(set(keys))),
        "lists_with_multiple_2d_positives": int(keys.count(key) > 1),
    }


def audit_targets(frame, legacy, exclusions):
    if frame.identifier.tolist() != legacy.identifier.tolist() or frame.fold.tolist() != legacy.fold.tolist():
        raise ValueError("v1.5 changed the source ID order or split")
    if not frame.identifier.is_unique:
        raise ValueError("Duplicate spectrum identifiers")
    pairs = dict.fromkeys(zip(legacy.smiles, frame.smiles))
    keys = {smiles: identity(smiles) for smiles in frame.smiles.unique()}
    graph_changes = 0
    stereo_changes = []
    for old, new in pairs:
        if identity(old) != keys[new]:
            raise ValueError("v1 to v1.5 target connectivity changed")
        old_mol, new_mol = Chem.MolFromSmiles(old), Chem.MolFromSmiles(new)
        if Chem.MolToSmiles(old_mol) != Chem.MolToSmiles(new_mol):
            stereo_changes.append({"v1": old, "v15": new, "identity_2d": keys[new]})
        raw_old = Chem.MolFromSmiles(old, sanitize=False)
        raw_new = Chem.MolFromSmiles(new, sanitize=False)
        graph_changes += (
            sum(a.GetIsAromatic() for a in raw_old.GetAtoms()) != sum(a.GetIsAromatic() for a in raw_new.GetAtoms())
            or Counter(str(b.GetBondType()) for b in raw_old.GetBonds()) != Counter(str(b.GetBondType()) for b in raw_new.GetBonds())
        )
    split_keys = {split: [keys[s] for s in frame.loc[frame.fold == split, "smiles"]] for split in ("train", "val", "test")}
    train_test = set(split_keys["train"]) & set(split_keys["test"])
    train_val = set(split_keys["train"]) & set(split_keys["val"])
    test_keys = set(split_keys["test"])
    val_test_indices = [i for i, key in enumerate(split_keys["val"]) if key in test_keys]
    if sorted(set(exclusions)) != exclusions or any(i < 0 or i >= len(split_keys["val"]) for i in exclusions):
        raise ValueError("Invalid frozen validation exclusions")
    if train_test or train_val or set(val_test_indices) - set(exclusions):
        raise ValueError(f"Unexpected split overlap: train/test={len(train_test)}, train/val={len(train_val)}, val/test indices={val_test_indices}")
    return keys, {
        "records": len(frame), "unique_target_strings": len(keys),
        "smiles_changed_records": int((legacy.smiles != frame.smiles).sum()),
        "raw_graph_feature_changed_unique_targets": graph_changes,
        "canonical_stereo_differences_with_same_2d_identity": stereo_changes,
        "split_counts": dict(Counter(frame.fold)), "train_test_2d_overlap": len(train_test),
        "train_val_2d_overlap": len(train_val), "observed_val_test_2d_overlap_indices": val_test_indices,
        "excluded_val_query_indices": exclusions,
        "exclusion_policy": "Preserve previously authorized validation indices; do not infer new exclusions or relax old ones",
        "excluded_val_identifiers": frame.loc[frame.fold == "val", "identifier"].iloc[exclusions].tolist(),
    }


def prepare_dataset(source_dir, legacy_tsv, output_dir, settings, expected_counts, exclusions):
    """CPU preparation; output directory must be new. Failure never publishes a complete manifest."""
    from matchms import Spectrum

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    sources = {}
    for name, expected in settings.sources.to_dict().items():
        path = source_dir / name
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"Pinned v1.5 source hash mismatch: {name}")
        sources[name] = {"path": str(path), "sha256": digest,
                         "url": "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/" +
                         ("data/" if name.endswith(".tsv") else "data/molecules/") + name}
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"state": "running", "dataset_version": "1.5", "rdkit_version": rdBase.rdkitVersion,
              "graph_policy": settings.graph_policy, "sources": sources,
              "legacy_tsv": {"path": str(legacy_tsv), "sha256": sha256_file(legacy_tsv)},
              "candidate_audits": {}, "outputs": {}}
    write_json(output_dir / "dataset_manifest.json", report)
    try:
        columns = ["identifier", "smiles", "fold"]
        frame = pd.read_csv(source_dir / "MassSpecGym1.5.tsv", sep="\t", usecols=columns)
        legacy = pd.read_csv(legacy_tsv, sep="\t", usecols=columns)
        keys, report["target_audit"] = audit_targets(frame, legacy, exclusions)
        if report["target_audit"]["split_counts"] != expected_counts:
            raise ValueError("Unexpected v1.5 split counts")
        write_json(output_dir / "dataset_manifest.json", report)
        logging.info("v1.5 target audit: %s", report["target_audit"])
        # Check spectral columns verbatim; harmless float serialization changes in metadata are recorded.
        spectral_columns = ["mzs", "intensities", "precursor_mz", "parent_mass", "collision_energy"]
        old_chunks = pd.read_csv(legacy_tsv, sep="\t", usecols=spectral_columns, chunksize=4096)
        data = defaultdict(list)
        metadata_diffs = Counter()
        for chunk, old_chunk in zip(pd.read_csv(source_dir / "MassSpecGym1.5.tsv", sep="\t", chunksize=4096), old_chunks, strict=True):
            for name in ("mzs", "intensities"):
                if chunk[name].tolist() != old_chunk[name].tolist():
                    raise ValueError(f"Changed spectral peaks: {name}")
            for name in ("precursor_mz", "parent_mass", "collision_energy"):
                x, y = chunk[name].to_numpy(), old_chunk[name].to_numpy()
                if not np.allclose(x, y, rtol=0, atol=1e-10, equal_nan=True):
                    raise ValueError(f"Material spectral metadata change: {name}")
                metadata_diffs[name] += int((~(np.equal(x, y) | (np.isnan(x) & np.isnan(y)))).sum())
            for row in chunk.to_dict("records"):
                mz = np.fromstring(row.pop("mzs"), sep=",")
                intensities = np.fromstring(row.pop("intensities"), sep=",")
                if not len(mz) or len(mz) != len(intensities) or not np.isfinite(mz).all() or not np.isfinite(intensities).all() or intensities.max() <= 0:
                    raise ValueError(f"Invalid spectrum: {row['identifier']}")
                row["identity_2d"] = keys[row["smiles"]]
                data[row["fold"]].append(Spectrum(mz=mz, intensities=intensities, metadata=row, metadata_harmonization=False))
        report["spectral_audit"] = {"peaks_identical": True, "numeric_metadata_tolerance": 1e-10,
                                     "numeric_metadata_differences": dict(metadata_diffs)}
        for split, count in expected_counts.items():
            if len(data[split]) != count:
                raise ValueError(f"Lost spectra in {split}")
            path = output_dir / f"{split}.pkl"
            with path.open("xb") as handle:
                pickle.dump(data[split], handle, protocol=pickle.HIGHEST_PROTOCOL)
            report["outputs"][path.name] = {"sha256": sha256_file(path), "records": count}
        del data
        write_json(output_dir / "dataset_manifest.json", report)
        for candidate in ("mass", "formula"):
            source = source_dir / f"MassSpecGym1.5_retrieval_candidates_{candidate}.json"
            candidates = json.loads(source.read_text())
            if set(keys) - set(candidates):
                raise ValueError("Official candidate mapping misses target SMILES")
            summary = Counter()
            # Fixed bounded CPU workers; no CUDA and no filtering/subsampling of the source lists.
            context = multiprocessing.get_context("spawn")
            with context.Pool(settings.audit_workers) as pool:
                for count, counts in enumerate(pool.imap_unordered(audit_candidate_list, candidates.items(), chunksize=8), 1):
                    summary.update(counts)
                    if count % 256 == 0 or count == len(candidates):
                        report["candidate_audits"][candidate] = {**summary, "expected_lists": len(candidates), "state": "running"}
                        write_json(output_dir / "dataset_manifest.json", report)
                        logging.info("%s candidate identity audit %s/%s lists, %s entries", candidate, count, len(candidates), summary["entries"])
            path = output_dir / f"candidates_{candidate}.pkl"
            with path.open("xb") as handle:
                pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report["outputs"][path.name] = {"sha256": sha256_file(path), "lists": len(candidates)}
            report["candidate_audits"][candidate] = {**summary, "expected_lists": len(candidates), "state": "complete",
                                                       "source_order_preserved": True, "forcing": False,
                                                       "unused_source_keys": len(set(candidates) - set(keys))}
            del candidates
            write_json(output_dir / "dataset_manifest.json", report)
        # Recheck the source fingerprints after the potentially long audit.
        for value in sources.values():
            if sha256_file(value["path"]) != value["sha256"]:
                raise ValueError("Source changed during v1.5 preparation")
        report["state"] = "complete"
    except BaseException as error:
        report.update(state="failed_or_interrupted", error=repr(error))
        raise
    finally:
        write_json(output_dir / "dataset_manifest.json", report)
    return report


def verify_dataset(directory, expected_counts, exclusions):
    directory = Path(directory)
    path = directory / "dataset_manifest.json"
    report = json.loads(path.read_text())
    if report["state"] != "complete" or report["dataset_version"] != "1.5" or report["graph_policy"] != "rdkit_sanitized":
        raise ValueError("v1.5 preparation is incomplete or uses the wrong graph policy")
    if report["rdkit_version"] != rdBase.rdkitVersion:
        raise ValueError("RDKit version differs from the audited preprocessing environment")
    if report["target_audit"]["split_counts"] != expected_counts or report["target_audit"]["excluded_val_query_indices"] != exclusions:
        raise ValueError("v1.5 split/exclusion audit mismatch")
    required = {f"{s}.pkl" for s in ("train", "val", "test", "candidates_mass", "candidates_formula")}
    if set(report["outputs"]) != required:
        raise ValueError("Incomplete v1.5 output inventory")
    for name, value in report["outputs"].items():
        if sha256_file(directory / name) != value["sha256"]:
            raise ValueError(f"Changed v1.5 prepared input: {name}")
    for candidate in ("mass", "formula"):
        audit = report["candidate_audits"][candidate]
        if audit["state"] != "complete" or audit["lists"] != audit["expected_lists"] or not audit["source_order_preserved"] or audit["forcing"]:
            raise ValueError("Incomplete candidate audit")
    return report


def classified_full_spectra(directory, expected_counts, exclusions, tokenizer):
    """Fresh tokenization, with one record per raw spectrum and 2D multi-positive labels."""
    report = verify_dataset(directory, expected_counts, exclusions)
    classified = {}
    counts = {}
    for split in ("train", "val"):
        with (Path(directory) / f"{split}.pkl").open("rb") as handle:
            raw = pickle.load(handle)
        if len(raw) != expected_counts[split]:
            raise ValueError("Raw split count mismatch")
        if split == "val":
            raw = [value for i, value in enumerate(raw) if i not in set(exclusions)]
        sequences = tokenizer.tokenize_sequence(raw)
        if len(sequences) != len(raw):
            raise ValueError("Tokenization lost spectra")
        data = defaultdict(list)
        for spectrum, sequence in zip(raw, sequences, strict=True):
            key = spectrum.get("identity_2d")
            if not key:
                raise ValueError("v1.5 spectrum has no audited 2D identity")
            data[key].append(sequence)
        classified[f"{split}_data"] = dict(data)
        classified[f"{split}_keys"] = sorted(data)
        counts[split] = len(sequences)
    return classified, {"dataset_version": "1.5", "dataset_manifest_sha256": sha256_file(Path(directory) / "dataset_manifest.json"),
                        "input_outputs": report["outputs"], "expected_epoch_counts": counts,
                        "validation_exclusion_report": {"query_indices": exclusions,
                            "identifiers": report["target_audit"]["excluded_val_identifiers"],
                            "policy": "frozen raw validation indices, applied before 2D identity grouping"},
                        "label_identity": "2D InChIKey connectivity", "formal_fulltrain": True}
