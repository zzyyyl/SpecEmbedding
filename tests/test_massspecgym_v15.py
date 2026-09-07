import json
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import run_massspecgym_v15 as runner
import train_align
from SpecEmbedding.config import ConfigObject, config
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.models_align import GINEEncoder
from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.utils.align import load_align_model
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import (
    audit_candidate_list,
    audit_targets,
    classified_full_spectra,
    molecule_identity,
    prepare_dataset,
    verify_dataset,
)

# Real v1.5 Mass list 4163, candidate 180 (zero-based): sanitization succeeds,
# but RDKit 2026.03.1's canonical Kekule search during InChI conversion fails.
FUSED_RING_CANDIDATE = (
    "CN1C(=O)c2ccc3c4cc5c6ccccc6c6cc7c8ccc9c%10c(ccc(c%11cc%12c%13ccccc%13c%13cc"
    "(c%14ccc(c2c3%14)C1=O)c4c1c5c6c(c7%11)c%12c%131)c%108)C(=O)N(C)C9=O"
)


def test_real_fused_ring_identity_keeps_graph_and_source_candidate():
    mol = Chem.MolFromSmiles(FUSED_RING_CANDIDATE)
    before = mol.ToBinary()
    key, retried = molecule_identity(mol)
    assert key == "XPHPEGPJWUASIV"
    assert mol.ToBinary() == before
    # Standard InChI identity must survive atom reordering; the embedding graph
    # retains its sanitized aromatic features and is never replaced by the copy.
    reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert molecule_identity(reverse)[0] == key
    values = ["CCO", FUSED_RING_CANDIDATE]
    summary = audit_candidate_list(("CCO", values))
    assert values == ["CCO", FUSED_RING_CANDIDATE]
    assert summary["graph_eligible_entries"] == 2
    assert summary["invalid_graph_entries"] == 0
    assert summary["identity_retry_entries"] == int(retried)
    if retried:
        assert summary["identity_retry_records"] == [{"candidate_index": 1, "smiles": values[1], "identity_2d": key}]


def test_identity_conversion_still_fails_closed_with_candidate_context():
    with patch("SpecEmbedding.utils.massspecgym_v15.Chem.MolToInchiKey", return_value=""):
        with pytest.raises(ValueError, match="Missing 2D InChIKey"):
            molecule_identity(Chem.MolFromSmiles("CCO"))
    original = Chem.MolToInchiKey

    def fail_candidate(mol):
        if mol.GetNumAtoms() > 3:
            raise Chem.KekulizeException("synthetic search failure")
        return original(mol)

    with patch("SpecEmbedding.utils.massspecgym_v15.Chem.MolToInchiKey", side_effect=fail_candidate):
        with pytest.raises(ValueError, match="candidate_index=1.*CCCC"):
            audit_candidate_list(("CCO", ["CCO", "CCCC"]))


def test_sanitized_graphs_remove_kekule_shortcut_in_gine_forward():
    pairs = [("CC1=CC=C(O)C=C1", "Cc1ccc(O)cc1"), ("C1=CC=CC=C1", "c1ccccc1")]
    model = GINEEncoder(emb_dim=8, n_layers=2, dropout_rate=0, size_feature_dim=4).eval()
    for old, new in pairs:
        raw = [smiles_to_graph(s, graph_policy="legacy_raw") for s in (old, new)]
        assert raw[0].x[:, 3].sum() != raw[1].x[:, 3].sum()
        graphs = Batch.from_data_list([smiles_to_graph(s, graph_policy="rdkit_sanitized") for s in (old, new)])
        with torch.no_grad():
            embeddings = model(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch, graphs.graph_size_features)
        torch.testing.assert_close(embeddings[0], embeddings[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("smiles", ["", "not a molecule", "C(C)(C)(C)(C)C"])
def test_sanitized_graph_rejects_invalid_molecules(smiles):
    with pytest.raises(ValueError):
        smiles_to_graph(smiles, graph_policy="rdkit_sanitized")


def test_candidate_audit_counts_identity_duplicates_without_mutation():
    candidates = ["OCC", "CCO", "CCN"]
    summary = audit_candidate_list(("CCO", candidates))
    assert summary["duplicate_2d_identities"] == 1
    assert summary["duplicate_strings"] == 0
    assert summary["lists_with_multiple_2d_positives"] == 1
    assert candidates == ["OCC", "CCO", "CCN"]
    with pytest.raises(ValueError, match="cover target"):
        audit_candidate_list(("CCO", ["CCN"]))


def test_candidate_audit_records_unencodable_decoys_but_never_drops_target():
    values = ["CCO", "CCCC[Sn](CCCC)(CCCC)c1ccc(CO)c(C[SiH2+](C)C(C)(C)C)n1", "OCC"]
    summary = audit_candidate_list(("CCO", values))
    assert summary["entries"] == 3
    assert summary["graph_eligible_entries"] == 2
    assert summary["invalid_graph_entries"] == 1
    assert summary["duplicate_2d_identities"] == 1
    assert summary["invalid_graph_records"] == [{"candidate_index": 1, "smiles": values[1]}]
    assert len(values) == 3
    with pytest.raises(ValueError, match="Invalid molecule"):
        audit_candidate_list((values[1], values))


def sequence(value, smiles="CCO"):
    return {"mz": np.array([value, 2], dtype=np.float32), "intensity": np.array([1, 1], dtype=np.float32),
            "mask": np.array([False, False]), "smiles": smiles}


def dataset():
    return AlignGraphDataset(data={"a": [sequence(1), sequence(2)], "b": [sequence(3, "CCN")]},
                             keys=["a", "b"], n_views=1, is_augment=False, full_spectra=True)


def test_alignment_all_spectra_once_including_repeat_molecule():
    data = dataset()
    assert len(data) == 3
    seen = [data[i][0][0, 0].item() for i in range(len(data))]
    assert seen == [1, 2, 3]
    assert [data[i][4][0] for i in range(len(data))] == ["a", "a", "b"]
    with pytest.raises(ValueError, match="every molecule"):
        AlignGraphDataset(data=data._data, keys=["a"], n_views=1, full_spectra=True)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 3)

    def forward(self, mzs, ints, masks, mols):
        value = self.projection(mzs)
        return value, value, torch.tensor(1.)


def test_formal_alignment_detects_dropped_last_batch(tmp_path):
    data = dataset()
    loader = DataLoader(data, batch_size=2, drop_last=True, collate_fn=align_collate_fn)
    trainer = TrainerAlign(TinyModel(), loader, loader, "cpu", str(tmp_path))
    trainer.expected_epoch_counts = {"train": 3, "val": 3}
    with pytest.raises(RuntimeError, match="Incomplete alignment epoch"):
        trainer.train_epoch(torch.optim.SGD(trainer.model.parameters(), lr=0.01), 1, "stage2")


def test_formal_alignment_records_complete_epochs_and_rejects_nonfinite_loss(tmp_path):
    data = dataset()
    loader = DataLoader(data, batch_size=2, collate_fn=align_collate_fn)
    trainer = TrainerAlign(TinyModel(), loader, loader, "cpu", str(tmp_path))
    trainer.expected_epoch_counts = {"train": 3, "val": 3}
    trainer.fit(1, torch.optim.SGD(trainer.model.parameters(), lr=0.01), stage_name="stage2")
    assert trainer.epoch_counts == [{"stage": "stage2", "epoch": 1, "train": 3, "val": 3}]
    with patch.object(trainer.criterion, "forward", return_value=torch.tensor(float("nan"))):
        with pytest.raises(RuntimeError, match="Non-finite"):
            trainer.validate(2, "stage2")


def test_v15_loader_refuses_unproven_or_legacy_alignment(tmp_path):
    checkpoint = tmp_path / "best.pth"
    with patch.object(config.model.mol_encoder, "graph_policy", "rdkit_sanitized"):
        with pytest.raises(ValueError, match="provenance"):
            load_align_model(str(checkpoint), torch.device("cpu"), "layernorm", 1e-5)
        (tmp_path / "alignment_selection.json").write_text(json.dumps({"seed": 42}))
        with pytest.raises(ValueError, match="graph policy"):
            load_align_model(str(checkpoint), torch.device("cpu"), "layernorm", 1e-5)


def test_synthetic_formal_alignment_saves_counts_and_loadable_provenance(tmp_path):
    model_config = config.model.to_dict()
    model_config["spec_encoder"].update(embedding_dim=8, n_head=2, n_layer=1, dim_feedward=8, dim_target=8)
    model_config["mol_encoder"].update(emb_dim=8, n_layers=1, dropout_rate=0, size_feature_dim=4, graph_policy="rdkit_sanitized")
    model_config["align"].update(final_dim=8, dropout_rate=0)
    data = dataset()._data
    with patch.object(config, "model", ConfigObject(model_config)), \
         patch.object(config.train.align, "epochs_stage2", 1), \
         patch.object(config.train.align, "num_workers", 0), \
         patch.object(config.augmentation, "prob", 0):
        train_align.train_align(data, list(data), data, list(data), None, batch_size=2, device="cpu",
                                save_dir=str(tmp_path), formal_fulltrain=True,
                                selection_metadata={"seed": 42, "fulltrain_audit": {"expected_epoch_counts": {"train": 3, "val": 3}}})
        selection = json.loads((tmp_path / "alignment_selection.json").read_text())
        assert selection["fulltrain_audit"]["epochs"] == [{"stage": "stage2", "epoch": 1, "train": 3, "val": 3}]
        assert selection["graph_policy"] == "rdkit_sanitized"
        loaded = load_align_model(str(tmp_path / "best_model_stage2.pth"), torch.device("cpu"), "layernorm", 1e-5)
        assert loaded.training is False


def test_real_synthetic_preparation_spawn_pool_and_fresh_full_tokenization(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for i, (fold, smiles) in enumerate([("train", "CCO"), ("train", "CCO"), ("train", "CCO"),
                                       ("val", "CCN"), ("val", "CCC"), ("test", "CCC")]):
        rows.append({"identifier": f"ID{i}", "fold": fold, "smiles": smiles, "mzs": "10,20", "intensities": "0.5,1",
                     "precursor_mz": 50., "parent_mass": 49., "collision_energy": 10.})
    frame = pd.DataFrame(rows)
    source_tsv = source / "MassSpecGym1.5.tsv"
    frame.to_csv(source_tsv, sep="\t", index=False)
    legacy = tmp_path / "legacy.tsv"
    frame.to_csv(legacy, sep="\t", index=False)
    values = {s: [s, "CO", "C(C)(C)(C)(C)C", FUSED_RING_CANDIDATE] for s in frame.smiles.unique()}
    for kind in ("mass", "formula"):
        (source / f"MassSpecGym1.5_retrieval_candidates_{kind}.json").write_text(json.dumps(values))
    settings = ConfigObject({"sources": {p.name: sha256_file(p) for p in source.iterdir()},
                             "graph_policy": "rdkit_sanitized", "audit_workers": 2})
    expected = {"train": 3, "val": 2, "test": 1}
    output = tmp_path / "output"
    prepared = prepare_dataset(source, legacy, output, settings, expected, [1])
    assert prepared["state"] == "complete"
    assert prepared["target_audit"]["excluded_val_identifiers"] == ["ID4"]
    for kind in ("mass", "formula"):
        audit = prepared["candidate_audits"][kind]
        assert audit["invalid_graph_entries"] == 3
        rejected = [json.loads(line) for line in (output / audit["graph_rejections"]["file"]).read_text().splitlines()]
        assert {row["target"] for row in rejected} == set(values)
        retries_path = output / audit["identity_retries"]["file"]
        retries = [json.loads(line) for line in retries_path.read_text().splitlines()]
        assert sum(len(row["retried"]) for row in retries) == audit["identity_retry_entries"]
    for kind in ("mass", "formula"):
        with (output / f"candidates_{kind}.pkl").open("rb") as handle:
            assert pickle.load(handle) == values
    classified, audit = classified_full_spectra(output, expected, [1], Tokenizer(4, show_progress_bar=False))
    assert sum(map(len, classified["train_data"].values())) == 3
    assert sum(map(len, classified["val_data"].values())) == 1
    assert audit["expected_epoch_counts"] == {"train": 3, "val": 1}
    retries_path = output / "identity_retry_mass.jsonl"
    original_retries = retries_path.read_bytes()
    retries_path.write_text("tampered")
    with pytest.raises(ValueError, match="identity retry record changed"):
        verify_dataset(output, expected, [1])
    retries_path.write_bytes(original_retries)
    with pytest.raises(FileExistsError):
        prepare_dataset(source, legacy, output, settings, expected, [1])
    (output / "train.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed v1.5"):
        verify_dataset(output, expected, [1])


def test_target_audit_refuses_new_train_test_overlap():
    frame = pd.DataFrame({"identifier": ["a", "b", "c"], "fold": ["train", "val", "test"], "smiles": ["CCO", "CCN", "OCC"]})
    with pytest.raises(ValueError, match="split overlap"):
        audit_targets(frame, frame.copy(), [])


def test_target_audit_keeps_frozen_exclusions_without_inventing_overlap():
    frame = pd.DataFrame({"identifier": ["a", "b", "c"], "fold": ["train", "val", "test"], "smiles": ["CCO", "CCN", "CCC"]})
    _, audit = audit_targets(frame, frame.copy(), [0])
    assert audit["excluded_val_query_indices"] == [0]
    assert audit["observed_val_test_2d_overlap_indices"] == []


def test_v15_queue_commands_separate_cpu_audit_from_gpu_full_training(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, source_dir=tmp_path / "sources", legacy_tsv=tmp_path / "old.tsv", gpu=1, device="cuda:1")
    stages = runner.commands(args)
    assert [s["name"] for s in stages] == ["prepare_v15", "alignment42", "rerank_matrix"]
    assert stages[0]["gpu"] is False
    assert stages[1]["gpu"] is True
    assert "--formal-fulltrain" in stages[1]["command"]
    assert "--pretrained_spec" not in stages[1]["command"]
    assert "--tokenset_cache" not in stages[1]["command"]
    assert all("--limit" not in s["command"] and "--max-train-queries" not in s["command"] for s in stages)


def test_v15_queue_cpu_failure_prevents_gpu_and_records_failure(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, source_dir=tmp_path / "sources", legacy_tsv=tmp_path / "old.tsv", gpu=1, device="cuda:1")
    manifest = {"runtime_config": config.to_dict(), "stages": runner.commands(args)}
    with patch.object(runner, "gpu_inventory", return_value={0: "GPU-a", 1: "GPU-b"}), \
         patch.object(runner, "preflight", return_value=manifest), \
         patch.object(runner.subprocess, "run", side_effect=RuntimeError("synthetic CPU audit failure")) as child, \
         patch.object(runner, "wait_for_gpu") as gpu:
        with pytest.raises(RuntimeError, match="CPU audit failure"):
            runner.execute(args, manifest)
    child.assert_called_once()
    gpu.assert_not_called()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed_or_interrupted"
    with pytest.raises(FileExistsError):
        runner.execute(args, manifest)
