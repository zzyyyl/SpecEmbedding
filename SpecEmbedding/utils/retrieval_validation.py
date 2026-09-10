"""Full validation retrieval with explicit candidate order, 2D labels and fresh embeddings."""

import json
import logging
import multiprocessing
import pickle
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from SpecEmbedding.data.datasets_eval import SpecSequenceDataset, mol_collate_fn
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import IDENTITY_POLICY, identity, verify_dataset
from SpecEmbedding.utils.molecule_graph_cache import graph_cache_provenance, load_graph_cache, molecule_order_sha256
from SpecEmbedding.utils.spectrum_controls import ControlledValidationSpectra

PROTOCOL = "v1.5 source candidate order; 2D InChIKey; audited graph exclusions; torchmetrics 1.8.2 CPU argsort descending"
SCHEMA_VERSION = 1


def build_validation_index(raw, candidates, exclusions, rejections, tokenizer_config, *, workers=1):
    """CPU only. Preserve every eligible query and all graph-eligible source entries, including duplicates."""
    exclusions = set(exclusions)
    if any(i < 0 or i >= len(raw) for i in exclusions):
        raise ValueError("Validation exclusion index out of bounds")
    indices = [i for i in range(len(raw)) if i not in exclusions]
    rows = [raw[i] for i in indices]
    if not rows:
        raise ValueError("Empty validation split")
    pools = {}
    unique = {}
    rejected = []
    for row in rows:
        target = row.get("smiles")
        if target in pools:
            continue
        if target not in candidates:
            raise ValueError(f"Missing validation candidate mapping: {target}")
        source = candidates[target]
        if len(source) > 256:
            raise ValueError("Validation source exceeds the fixed 256-candidate protocol")
        rejected_positions = {entry["candidate_index"]: entry["smiles"] for entry in rejections.get(target, [])}
        for position, smiles in rejected_positions.items():
            if position >= len(source) or position < 0 or source[position] != smiles or smiles == target:
                raise ValueError("Invalid graph exclusion does not match the source candidate list")
            rejected.append({"target": target, "source_position": position, "smiles": smiles})
        pool = []
        for position, smiles in enumerate(source):
            if position in rejected_positions:
                continue
            if smiles not in unique:
                unique[smiles] = len(unique)
            pool.append((unique[smiles], position))
        pools[target] = pool
    smiles = list(unique)
    if workers < 1:
        raise ValueError("CPU identity workers must be positive")
    if workers == 1:
        keys = list(map(identity, smiles))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            keys = []
            for count, key in enumerate(pool.imap(identity, smiles, chunksize=128), 1):
                keys.append(key)
                if count % 16384 == 0 or count == len(smiles):
                    logging.info("Validation candidate identities: %s/%s", count, len(smiles))
    logging.info("Validation identity index: %s queries, %s unique molecules", len(rows), len(smiles))
    width = max(1, max(map(len, pools.values())))
    ids = torch.full((len(rows), width), -1, dtype=torch.long)
    positions = torch.full_like(ids, -1)
    positive = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        key = identity(row.get("smiles"))
        if key != row.get("identity_2d"):
            raise ValueError("Validation target identity differs from audited raw data")
        for j, (mol_index, position) in enumerate(pools[row.get("smiles")]):
            ids[i, j], positions[i, j] = mol_index, position
            positive[i, j] = keys[mol_index] == key
    sequences = Tokenizer(**tokenizer_config).tokenize_sequence(rows)
    if len(sequences) != len(rows) or any(seq["smiles"] != row.get("smiles") for seq, row in zip(sequences, rows, strict=True)):
        raise ValueError("Validation tokenization dropped or reordered queries")
    return {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "identity_policy": IDENTITY_POLICY,
            "sequences": sequences, "raw_query_indices": indices, "mol_smiles": smiles, "mol_identity_2d": keys,
            "candidate_indices": ids, "source_positions": positions, "positive_mask": positive,
            "graph_rejections": rejected, "tokenizer_config": tokenizer_config,
            "excluded_query_indices": sorted(exclusions), "source_query_count": len(raw)}


def prepare_validation_index(data_path, expected_counts, exclusions, tokenizer_config, *, workers):
    data_path = Path(data_path)
    report = verify_dataset(data_path, expected_counts, exclusions)
    with (data_path / "val.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    with (data_path / "candidates_mass.pkl").open("rb") as handle:
        candidates = pickle.load(handle)
    rejections = {}
    for line in (data_path / "invalid_graph_mass.jsonl").read_text().splitlines():
        record = json.loads(line)
        rejections[record["target"]] = record["rejected"]
    result = build_validation_index(raw, candidates, exclusions, rejections, tokenizer_config, workers=workers)
    result.update(dataset_manifest_sha256=sha256_file(data_path / "dataset_manifest.json"),
                  dataset_outputs=report["outputs"], graph_policy="rdkit_sanitized")
    return result


def load_validation_index(path, data_path, expected_counts, exclusions, tokenizer_config):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["sha256"] != sha256_file(path):
        raise ValueError("Validation index fingerprint changed")
    index = torch.load(path, map_location="cpu", weights_only=False)
    report = verify_dataset(data_path, expected_counts, exclusions)
    if (index["schema_version"] != SCHEMA_VERSION or index["protocol"] != PROTOCOL
            or index["identity_policy"] != IDENTITY_POLICY or index["graph_policy"] != "rdkit_sanitized"
            or index["dataset_manifest_sha256"] != sha256_file(Path(data_path) / "dataset_manifest.json")
            or index["dataset_outputs"] != report["outputs"] or index["tokenizer_config"] != tokenizer_config
            or index["excluded_query_indices"] != exclusions):
        raise ValueError("Validation index provenance/configuration mismatch")
    expected_indices = [i for i in range(expected_counts["val"]) if i not in set(exclusions)]
    if index["raw_query_indices"] != expected_indices or len(index["sequences"]) != len(expected_indices):
        raise ValueError("Validation index does not cover the full eligible split in order")
    ids, labels = index["candidate_indices"], index["positive_mask"]
    if (ids.shape != labels.shape or ids.shape[0] != len(expected_indices) or ids.shape[1] > 256
            or ids.min() < -1 or ids.max() >= len(index["mol_smiles"]) or (labels & (ids < 0)).any()):
        raise ValueError("Invalid candidate/label matrix in validation index")
    return index


def validation_index_receipt(index, path):
    return {"sha256": sha256_file(path), "protocol": index["protocol"],
            "dataset_manifest_sha256": index["dataset_manifest_sha256"], "queries": len(index["sequences"]),
            "molecules": len(index["mol_smiles"]), "graph_rejections": len(index["graph_rejections"]),
            "positive_queries": int(index["positive_mask"].any(dim=1).sum()),
            "multi_positive_queries": int((index["positive_mask"].sum(dim=1) > 1).sum())}


def prepared_validation_input(path, data_path, expected_counts, exclusions, tokenizer_config):
    """Verify a reusable CPU index; no checkpoint or embeddings are included."""
    path = Path(path).resolve()
    receipt_path = path.with_suffix('.json')
    index_sha, receipt_sha = sha256_file(path), sha256_file(receipt_path)
    index = load_validation_index(path, data_path, expected_counts, exclusions, tokenizer_config)
    expected = validation_index_receipt(index, path)
    if (json.loads(receipt_path.read_text()) != expected or expected['sha256'] != index_sha
            or sha256_file(receipt_path) != receipt_sha):
        raise ValueError('Prepared validation index receipt is stale or inconsistent')
    return {'path': str(path), 'receipt_sha256': receipt_sha, **expected}


def import_validation_index(source, destination, expected, data_path, expected_counts, exclusions, tokenizer_config):
    """Copy verified CPU index bytes into a new run; leave all model encoding fresh."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError('Refusing existing validation index artifacts')
    observed = prepared_validation_input(source, data_path, expected_counts, exclusions, tokenizer_config)
    if observed != expected:
        raise ValueError('Prepared validation source changed since preflight')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for original, copied in ((source, destination), (source.with_suffix('.json'), destination.with_suffix('.json'))):
        with original.open('rb') as reader, copied.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
    imported = prepared_validation_input(destination, data_path, expected_counts, exclusions, tokenizer_config)
    if (imported != {**expected, 'path': str(destination)} or sha256_file(source) != expected['sha256']
            or sha256_file(source.with_suffix('.json')) != expected['receipt_sha256']):
        raise ValueError('Validation source or imported copy changed during import')
    return imported


def load_validation_graph_cache(index_path, index, directory):
    provenance = graph_cache_provenance(index['mol_smiles'], index_sha256=sha256_file(index_path),
                                        dataset_manifest_sha256=index['dataset_manifest_sha256'])
    return load_graph_cache(index['mol_smiles'], directory, provenance)


class StrictValidationMolecules(Dataset):
    def __init__(self, smiles, graph_cache=None):
        self.smiles = smiles
        self.graph_cache = graph_cache
        if graph_cache is not None:
            source = graph_cache.manifest['provenance']
            if len(graph_cache) != len(smiles) or source['molecule_order_sha256'] != molecule_order_sha256(smiles):
                raise ValueError('Validation graph cache changed molecule order or coverage')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        graph = (smiles_to_graph(self.smiles[index], graph_policy="rdkit_sanitized")
                 if self.graph_cache is None else self.graph_cache[index])
        if graph is None:
            raise ValueError(f"Previously eligible validation molecule failed to encode: {self.smiles[index]}")
        return {"graph": graph, "original_idx": index}


def retrieval_metrics(scores, positive, valid, top_k=(1, 5, 10, 20)):
    """Match torchmetrics 1.8.2 per-query argsort; retain stable-sort tie sensitivity."""
    scores, positive, valid = scores.cpu(), positive.cpu(), valid.cpu()
    if scores.ndim != 2 or scores.shape != positive.shape or scores.shape != valid.shape or not len(scores):
        raise ValueError("Mismatched or empty retrieval arrays")
    if not torch.isfinite(scores[valid]).all() or (positive & ~valid).any():
        raise ValueError("Non-finite retrieval scores or positive label on padding")
    scores = scores.masked_fill(~valid, -torch.inf)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_positive = positive.gather(1, order)
    covered = positive.any(dim=1)
    stable_ranks = torch.where(covered, sorted_positive.long().argmax(dim=1) + 1, 0)
    ranks = torch.zeros(len(scores), dtype=torch.long)
    for i in range(len(scores)):
        # Trim padding before sorting: unstable tie ordering can depend on array length.
        row_scores, row_positive = scores[i, valid[i]], positive[i, valid[i]]
        if row_positive.any():
            row_order = torch.argsort(row_scores, descending=True)
            ranks[i] = row_positive[row_order].long().argmax() + 1
    values = {"queries": len(scores), "positive_queries": int(covered.sum()),
              "coverage": float(covered.float().mean()),
              "mrr": float(torch.where(covered, ranks.double().clamp_min(1).reciprocal(), 0).mean())}
    for k in top_k:
        if k < 1:
            raise ValueError("Top-k must be positive")
        values[f"top{k}"] = float(((ranks > 0) & (ranks <= k)).double().mean())
        values[f"stable_top{k}"] = float(((stable_ranks > 0) & (stable_ranks <= k)).double().mean())
    values["stable_mrr"] = float(torch.where(covered, stable_ranks.double().clamp_min(1).reciprocal(), 0).mean())
    return values, ranks


class AlignmentRetrievalValidator:
    def __init__(self, index, settings, output_dir=None, *, spectrum_control=None, control_settings=None, graph_cache=None,
                 spectrum_metadata=None):
        self.index = index
        self.settings = settings
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if graph_cache is not None and graph_cache.manifest['provenance']['dataset_manifest_sha256'] != index.get('dataset_manifest_sha256'):
            raise ValueError('Validation graph cache belongs to a different dataset')
        self.molecules = StrictValidationMolecules(index['mol_smiles'], graph_cache)
        self.graph_cache_fingerprint = (None if graph_cache is None else {
            'directory': str(graph_cache.root), 'manifest_sha256': sha256_file(graph_cache.root / 'manifest.json'),
            'audit_sha256': sha256_file(graph_cache.root / 'audit.json')})
        if spectrum_control is None:
            if control_settings is not None:
                raise ValueError("Control settings require an explicit spectrum control")
            self.spectra = SpecSequenceDataset(index["sequences"])
            self.control_metadata = None
        else:
            if control_settings is None:
                raise ValueError("Spectrum controls require explicit settings")
            self.spectra = ControlledValidationSpectra(index, spectrum_control, control_settings)
            self.control_metadata = self.spectra.metadata
        self.spectrum_metadata_fingerprint = None
        if spectrum_metadata is not None:
            from SpecEmbedding.data.datasets_adduct import AdductValidationSpectra
            if spectrum_control is not None:
                raise ValueError('Adduct spectrum controls require a separately registered metadata policy')
            self.spectra = AdductValidationSpectra(self.spectra, index, spectrum_metadata)
            self.spectrum_metadata_fingerprint = spectrum_metadata.provenance

    def molecule_loader(self, generator):
        return DataLoader(self.molecules, batch_size=self.settings.mol_batch_size,
                          num_workers=self.settings.num_workers, shuffle=False, collate_fn=mol_collate_fn,
                          generator=generator)

    def snapshot_metadata(self):
        extra = {"spectrum_control": self.control_metadata} if self.control_metadata is not None else {}
        if self.graph_cache_fingerprint is not None:
            extra['validation_graph_cache'] = self.graph_cache_fingerprint
        if self.spectrum_metadata_fingerprint is not None:
            extra['spectrum_metadata'] = self.spectrum_metadata_fingerprint
        return extra

    @torch.inference_mode()
    def __call__(self, model, device, epoch, stage):
        started = time.monotonic()
        model.eval()
        index, settings = self.index, self.settings
        # This generator is private: validation must not change the training RNG stream.
        generator = torch.Generator().manual_seed(0)
        mol_loader = self.molecule_loader(generator)
        embeddings = None
        seen = torch.zeros(len(index["mol_smiles"]), dtype=torch.bool)
        for batch_number, batch in enumerate(mol_loader, 1):
            indices = torch.as_tensor(batch["indices"], dtype=torch.long, device='cpu')
            encoded = model.encode_mol(batch["mol_graph"].to(device), normalize=True).float().cpu()
            if not torch.isfinite(encoded).all() or seen[indices].any():
                raise ValueError("Non-finite or repeated validation molecule embeddings")
            if embeddings is None:
                embeddings = torch.empty((len(seen), encoded.shape[1]), dtype=torch.float32)
            embeddings[indices] = encoded
            seen[indices] = True
            if batch_number % 128 == 0:
                logging.info("Fresh validation molecule encoding: %s/%s batches", batch_number, len(mol_loader))
        if not seen.all() or embeddings is None:
            raise ValueError("Incomplete validation molecule encoding")
        logging.info("Encoded all %s validation molecules afresh", len(seen))
        spec_loader = DataLoader(self.spectra, batch_size=settings.spec_batch_size,
                                 shuffle=False, num_workers=0, generator=generator)
        all_scores = torch.empty(index["candidate_indices"].shape, dtype=torch.float32)
        offset = 0
        for batch in spec_loader:
            kwargs = {'adduct_ids': batch['adduct_id'].to(device)} if 'adduct_id' in batch else {}
            spec = model.encode_spec(batch["spec_mz"].to(device), batch["spec_intensity"].to(device),
                                     batch["spec_mask"].to(device), normalize=True, **kwargs).float()
            if not torch.isfinite(spec).all():
                raise ValueError("Non-finite validation spectrum embeddings")
            n = len(spec)
            ids = index["candidate_indices"][offset:offset + n]
            candidates = embeddings[ids.clamp_min(0)].to(device)
            all_scores[offset:offset + n] = torch.einsum("bd,bkd->bk", spec, candidates).cpu()
            offset += n
        if offset != len(index["sequences"]):
            raise ValueError("Incomplete validation spectrum encoding")
        metrics, ranks = retrieval_metrics(all_scores, index["positive_mask"], index["candidate_indices"] >= 0,
                                          settings.top_k)
        metrics["seconds"] = time.monotonic() - started
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{stage}_epoch{epoch:03d}.pt"
            extra = self.snapshot_metadata()
            with path.open("xb") as handle:
                torch.save({"metrics": metrics, "ranks": ranks, "scores": all_scores,
                            "raw_query_indices": index["raw_query_indices"], "protocol": PROTOCOL, **extra}, handle)
        logging.info("[%s] epoch=%s full validation retrieval: %s", stage, epoch, metrics)
        return metrics
