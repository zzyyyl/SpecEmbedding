import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import alignment_validation
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import PROTOCOL, AlignmentRetrievalValidator
from SpecEmbedding.utils.spectrum_controls import ControlledValidationSpectra


def control_index():
    sequences = [{"mz": np.array([100. + i, 40. + i, 0., 0., 0., 0.]),
                  "intensity": np.array([2., .2 * (i + 1), 0., 0., 0., 0.]),
                  "mask": np.array([False, False, True, True, True, True]), "smiles": smiles}
                 for i, smiles in enumerate(["CCO", "CCC", "CCN", "CCO"])]
    return {"sequences": sequences, "raw_query_indices": [0, 2, 3, 7], "mol_smiles": ["CCO", "CCC", "CCN"],
            "candidate_indices": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [-1, -1, -1]]),
            "positive_mask": torch.tensor([[True, False, False], [False, True, False],
                                            [False, False, False], [False, False, False]]), "protocol": PROTOCOL}


def settings():
    return SimpleNamespace(permutation_seed=42, constant_mz=[100., 50.], constant_intensity=[2., 1.])


def test_permutation_is_reproducible_full_derangement_independent_of_labels_and_rng():
    index = control_index()
    rng = torch.get_rng_state().clone()
    first = ControlledValidationSpectra(index, "permuted", settings())
    changed = copy.deepcopy(index)
    changed["positive_mask"].logical_not_()
    changed["candidate_indices"].fill_(-1)
    second = ControlledValidationSpectra(changed, "permuted", settings())
    assert torch.equal(rng, torch.get_rng_state())
    assert first.metadata == second.metadata
    assert sorted(first.donors.tolist()) == list(range(4))
    assert all(i != j for i, j in enumerate(first.donors.tolist()))
    assert first.metadata["donor_raw_query_indices"] == [index["raw_query_indices"][i] for i in first.donors]
    for i, donor in enumerate(first.donors.tolist()):
        for output, original in (("spec_mz", "mz"), ("spec_intensity", "intensity"), ("spec_mask", "mask")):
            assert torch.equal(first[i][output], torch.as_tensor(index["sequences"][donor][original], dtype=first[i][output].dtype))
        assert "smiles" not in first[i]


def test_constant_removes_precursor_length_mask_and_identity_information():
    index = control_index()
    before = copy.deepcopy(index)
    control = ControlledValidationSpectra(index, "constant", settings())
    for i in range(4):
        sample = control[i]
        assert sample["spec_mz"].tolist() == [100., 50., 0., 0., 0., 0.]
        assert sample["spec_intensity"].tolist() == [2., 1., 0., 0., 0., 0.]
        assert sample["spec_mask"].tolist() == [False, False, True, True, True, True]
        sample["spec_mz"].fill_(999.)
    for a, b in zip(index["sequences"], before["sequences"], strict=True):
        assert all(np.array_equal(a[key], b[key]) for key in ("mz", "intensity", "mask"))
    assert torch.equal(index["positive_mask"], before["positive_mask"])


def test_precursor_only_retains_mass_but_ignores_fragments_labels_and_peak_count():
    index = control_index()
    before = copy.deepcopy(index)
    changed = copy.deepcopy(index)
    changed["positive_mask"].logical_not_()
    changed["candidate_indices"].fill_(-1)
    for seq in changed["sequences"]:
        seq["smiles"] = "not a molecule; never inspected by this control"
        seq["mz"][1:] = 900.
        seq["intensity"][1:] = .9
        seq["mask"][1:] = False
    rng = torch.get_rng_state().clone()
    first = ControlledValidationSpectra(index, "precursor_only", settings())
    second = ControlledValidationSpectra(changed, "precursor_only", settings())
    assert first.metadata == second.metadata
    expected_masses = np.array([100., 101., 102., 103.], dtype="<f4")
    assert first.metadata["precursor_mz_float32_le_sha256"] == hashlib.sha256(expected_masses.tobytes()).hexdigest()
    for i in range(4):
        sample = first[i]
        assert sample["spec_mz"].tolist() == [100. + i, 0., 0., 0., 0., 0.]
        assert sample["spec_intensity"].tolist() == [2., 0., 0., 0., 0., 0.]
        assert sample["spec_mask"].tolist() == [False, True, True, True, True, True]
        assert "smiles" not in sample
        assert all(torch.equal(sample[key], value) for key, value in second[i].items())
        for value in sample.values():
            value.fill_(1)
        assert all(torch.equal(first[i][key], value) for key, value in second[i].items())
    assert torch.equal(rng, torch.get_rng_state())
    for a, b in zip(index["sequences"], before["sequences"], strict=True):
        assert all(np.array_equal(a[key], b[key]) for key in ("mz", "intensity", "mask"))
    index["sequences"][0]["mz"][0] = 999.
    assert first[0]["spec_mz"][0] == 100.  # Captured input is independent of later source mutation.


@pytest.mark.parametrize("key,value", [("mz", 0.), ("mz", float("nan")), ("mz", float("inf")),
                                      ("intensity", 1.), ("intensity", float("nan")), ("mask", True)])
def test_precursor_only_rejects_invalid_precursor_token(key, value):
    index = control_index()
    index["sequences"][2][key][0] = value
    with pytest.raises(ValueError, match="unmasked measured precursor"):
        ControlledValidationSpectra(index, "precursor_only", settings())


@pytest.mark.parametrize("shape", [(0,), (5,), (6, 1)])
def test_precursor_only_rejects_inconsistent_token_width(shape):
    index = control_index()
    index["sequences"][2]["mz"] = np.ones(shape)
    with pytest.raises(ValueError, match="consistent nonempty tokenizer widths"):
        ControlledValidationSpectra(index, "precursor_only", settings())


@pytest.mark.parametrize("mode", ["permuted", "constant", "precursor_only"])
def test_real_control_forward_keeps_all_queries_and_normal_audit_rejects_it(tmp_path, mode):
    index = control_index()
    ids, labels = index["candidate_indices"].clone(), index["positive_mask"].clone()
    model = SpecMolAlignModel(SiameseModel(embedding_dim=16, n_head=2, n_layer=1, dim_feedward=16, dim_target=16),
                              GINEEncoder(emb_dim=8, n_layers=1, size_feature_dim=4, dropout_rate=0.),
                              spec_dim=16, hidden_dim=16, final_dim=16, dropout_rate=0., tau=.07)
    runtime = SimpleNamespace(mol_batch_size=2, spec_batch_size=3, num_workers=0, top_k=[1, 5, 10, 20])
    rng = torch.get_rng_state().clone()
    validator = AlignmentRetrievalValidator(index, runtime, tmp_path, spectrum_control=mode, control_settings=settings())
    metrics = validator(model, torch.device("cpu"), 0, f"control_{mode}")
    assert torch.equal(rng, torch.get_rng_state())
    assert metrics["queries"] == 4 and metrics["positive_queries"] == 2
    assert torch.equal(ids, index["candidate_indices"]) and torch.equal(labels, index["positive_mask"])
    path = tmp_path / f"control_{mode}_epoch000.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["raw_query_indices"] == [0, 2, 3, 7]
    assert saved["ranks"][-2:].tolist() == [0, 0]
    if mode == "constant":
        torch.testing.assert_close(saved["scores"][0], saved["scores"][1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unexpected spectrum control"):
        audit_snapshot(path, index)
    verified = audit_snapshot(path, index, expected_spectrum_control=validator.control_metadata)
    assert verified["mrr"] == pytest.approx(metrics["mrr"])
    with pytest.raises(FileExistsError):
        validator(model, torch.device("cpu"), 0, f"control_{mode}")


@pytest.mark.parametrize("mz,intensity", [([], []), ([100.], []), ([float("nan")], [1.]),
                                        ([100.], [-1.]), ([100.], [0.]), (list(range(1, 8)), [1.] * 7)])
def test_invalid_constant_fails_before_forward(mz, intensity):
    value = settings()
    value.constant_mz, value.constant_intensity = mz, intensity
    with pytest.raises(ValueError, match="Invalid fixed spectrum"):
        ControlledValidationSpectra(control_index(), "constant", value)


def test_single_query_permutation_and_implicit_control_settings_are_rejected():
    index = control_index()
    index["sequences"], index["raw_query_indices"] = index["sequences"][:1], [0]
    with pytest.raises(ValueError, match="at least two"):
        ControlledValidationSpectra(index, "permuted", settings())
    with pytest.raises(ValueError, match="explicit spectrum control"):
        AlignmentRetrievalValidator(index, SimpleNamespace(), control_settings=settings())


@pytest.mark.parametrize("mode", ["constant", "precursor_only"])
def test_cli_controls_cannot_prepare_index_or_fall_back_to_cpu(tmp_path, mode):
    common = ["--data-path", str(tmp_path), "--index", str(tmp_path / "index.pt"), "--spectrum-control", mode]
    with pytest.raises(SystemExit):
        alignment_validation.main(common + ["--prepare-only"])
    with pytest.raises(SystemExit):
        alignment_validation.main(common + ["--checkpoint", str(tmp_path / "weights.pth"),
                                            "--output", str(tmp_path / "output"), "--device", "cpu"])
    assert not (tmp_path / "index.pt").exists() and not (tmp_path / "output").exists()


@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("mode", ["constant", "precursor_only"])
def test_control_cli_binds_checkpoint_and_publishes_audited_separate_receipt(tmp_path, monkeypatch, tamper, mode):
    index = control_index()
    index["dataset_outputs"] = {"synthetic": True}
    index_file = tmp_path / "index.pt"
    torch.save(index, index_file)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic checkpoint; loader mocked in this CPU test")
    selection = {"seed": 42, "checkpoint_sha256": sha256_file(checkpoint), "model_config": {"synthetic": True},
                 "exclude_val_query_indices": [], "fulltrain_audit": {"dataset_version": "1.5",
                    "input_outputs": index["dataset_outputs"], "formal_fulltrain": True}}
    (tmp_path / "alignment_selection.json").write_text(json.dumps(selection))
    if tamper:
        checkpoint.write_bytes(b"changed weights")
    runtime = SimpleNamespace(mol_batch_size=2, spec_batch_size=2, num_workers=0, top_k=[1, 5, 10, 20],
                              spectrum_controls=settings())
    config = SimpleNamespace(model=SimpleNamespace(to_dict=lambda: {"synthetic": True},
                                mol_encoder=SimpleNamespace(graph_policy="rdkit_sanitized", norm_type="layernorm", norm_eps=1e-5)),
                             fulltrain=SimpleNamespace(expected_counts=SimpleNamespace(to_dict=lambda: {}), exclude_val_query_indices=[]),
                             data=SimpleNamespace(tokenizer=SimpleNamespace(to_dict=lambda: {})), retrieval_validation=runtime)
    monkeypatch.setattr(alignment_validation, "config", config)
    monkeypatch.setattr(alignment_validation, "load_validation_index", lambda *args: index)
    monkeypatch.setattr(alignment_validation, "resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setenv("SPECEMBEDDING_REQUIRE_CUDA", "1")
    model = SpecMolAlignModel(SiameseModel(embedding_dim=16, n_head=2, n_layer=1, dim_feedward=16, dim_target=16),
                              GINEEncoder(emb_dim=8, n_layers=1, size_feature_dim=4, dropout_rate=0.),
                              spec_dim=16, hidden_dim=16, final_dim=16, dropout_rate=0., tau=.07)
    monkeypatch.setattr(alignment_validation, "load_align_model", lambda *args: model)
    output = tmp_path / "output"
    argv = ["--data-path", str(tmp_path), "--index", str(index_file), "--checkpoint", str(checkpoint),
            "--output", str(output), "--device", "cuda:0", "--spectrum-control", mode]
    if tamper:
        with pytest.raises(ValueError, match="provenance mismatch"):
            alignment_validation.main(argv)
        assert not output.exists()
        return
    alignment_validation.main(argv)
    receipt = json.loads((output / "metrics.json").read_text())
    assert receipt["spectrum_control"]["mode"] == mode
    assert receipt["saved_score_audit"] == "passed" and not receipt["used_for_checkpoint_selection"]
    assert receipt["checkpoint_sha256"] == sha256_file(checkpoint)
    assert receipt["snapshot_sha256"] == sha256_file(output / f"control_{mode}_epoch000.pt")
    assert receipt["metrics"]["queries"] == 4 and not receipt["test_evaluated"]
    assert not (output / "baseline_epoch000.pt").exists()
