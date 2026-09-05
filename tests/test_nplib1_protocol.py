import copy
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.audit_nplib1_overlap import (
    overlap_summary,
    spectrum_signature,
    split_inventory,
)
from analysis.audit_nplib1_source_bundle import (
    summarize_candidate_source,
    summarize_split,
)
from data_processing.download_nplib1 import SourceFile, git_blob_sha1, validate_source_file
from data_processing.nplib1 import (
    build_candidate_mapping,
    build_inchikey_to_smiles,
    canonicalize_2d_smiles,
    spectrum_from_entry,
    unpack_candidate_inchikeys,
    validate_split_identity_disjointness,
)
from prepare_rerank_cache import build_query_record
from src.data.NPLIB1 import NPLIB1Provider

REPO_ROOT = Path(__file__).resolve().parents[1]


class NPLIB1ProcessingTest(unittest.TestCase):
    def test_build_inchikey_to_smiles_accepts_rdkit_molecules(self):
        from rdkit import Chem

        mapping, summary = build_inchikey_to_smiles(
            {"AAAA-BBBB-CC": Chem.MolFromSmiles("C[C@H](O)F")}
        )

        self.assertEqual(mapping["AAAA-BBBB-CC"], "C[C@H](O)F")
        self.assertEqual(summary["mapped_molecules"], 1)
        self.assertEqual(summary["invalid_molecules"], 0)

    def test_candidate_mapping_preserves_order_without_inserting_positive(self):
        query = "AAAA-BBBB-CC"
        other_a = "DDDD-EEEE-FF"
        other_b = "GGGG-HHHH-II"
        mapping, summary = build_candidate_mapping(
            {query: [other_b, other_a, other_b]},
            {
                query: "CCO",
                other_a: "CCC",
                other_b: "CCN",
            },
        )

        self.assertEqual(mapping["CCO"], ["CCN", "CCC", "CCN"])
        self.assertNotIn("CCO", mapping["CCO"])
        self.assertEqual(summary["exact_inchikey_positive_sets"], 0)
        self.assertEqual(summary["two_dimensional_inchikey_positive_sets"], 0)
        self.assertEqual(summary["mapped_2d_smiles_positive_sets"], 0)
        self.assertEqual(summary["mapped_candidate_size"]["median"], 3.0)
        self.assertEqual(
            summary["candidate_policy"],
            "preserve_supplied_candidates_without_positive_insertion",
        )

    def test_candidate_mapping_reports_two_dimensional_positive(self):
        query = "AAAA-BBBB-CC"
        same_2d = "AAAA-ZZZZ-YY"
        mapping, summary = build_candidate_mapping(
            {query: [same_2d]},
            {query: "C[C@H](O)F", same_2d: "C[C@@H](O)F"},
        )

        self.assertEqual(mapping["CC(O)F"], ["CC(O)F"])
        self.assertEqual(summary["exact_inchikey_positive_sets"], 0)
        self.assertEqual(summary["two_dimensional_inchikey_positive_sets"], 1)
        self.assertEqual(summary["mapped_2d_smiles_positive_sets"], 1)
        self.assertEqual(
            summary["smiles_identity_policy"],
            "rdkit_canonical_non_isomeric_smiles",
        )

    def test_candidate_mapping_accepts_stateful_scored_train_candidates(self):
        query = "AAAA-BBBB-CC"
        candidate = "DDDD-EEEE-FF"

        mapping, summary = build_candidate_mapping(
            {query: [7, [(query, 1.0), (candidate, 0.75)]]},
            {query: "CCO", candidate: "CCC"},
        )

        self.assertEqual(mapping["CCO"], ["CCO", "CCC"])
        self.assertEqual(
            summary["source_formats"],
            {"state_and_scored_candidate_list": 1},
        )

    def test_candidate_unpacking_rejects_unknown_entry(self):
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            unpack_candidate_inchikeys([object()])

    def test_source_candidate_audit_reports_split_and_positive_coverage(self):
        split = {
            "train": ["AAAA-BBBB-CC"],
            "valid": ["VVVV-WWWW-XX"],
            "test": ["TTTT-UUUU-VV"],
        }
        summary = summarize_candidate_source(
            {"AAAA-BBBB-CC": [4, [("AAAA-BBBB-CC", 1.0), ("DDDD-EEEE-FF", 0.5)]]},
            split,
        )

        self.assertEqual(summary["candidate_sets"], 1)
        self.assertEqual(summary["candidate_entries"], 2)
        self.assertEqual(summary["exact_inchikey_positive_coverage"], 1.0)
        self.assertEqual(
            summary["query_key_coverage_by_split"]["val"]["coverage"],
            0.0,
        )

    def test_source_split_audit_reports_train_validation_leakage(self):
        summary = summarize_split(
            {
                "train": ["AAAA-BBBB-CC", "SHARED-AAAA-BB"],
                "valid": ["SHARED-AAAA-BB"],
                "test": ["TTTT-UUUU-VV"],
            }
        )

        self.assertEqual(
            summary["pairwise_overlap"]["train_vs_val"]["shared_full_inchikeys"],
            1,
        )
        self.assertEqual(
            summary["pairwise_overlap"]["train_vs_test"]["shared_2d_inchikeys"],
            0,
        )

    def test_canonicalize_2d_smiles_removes_stereochemistry(self):
        self.assertEqual(
            canonicalize_2d_smiles("C[C@H](O)F"),
            canonicalize_2d_smiles("C[C@@H](O)F"),
        )
        self.assertIsNone(canonicalize_2d_smiles("not-a-smiles"))

    def test_processing_rejects_identity_overlap_between_splits(self):
        with self.assertRaisesRegex(ValueError, "train_vs_val=1"):
            validate_split_identity_disjointness(
                {
                    "train": ["SHARED-AAAA-BB"],
                    "valid": ["SHARED-CCCC-DD"],
                    "test": ["TTTT-UUUU-VV"],
                }
            )

    def test_spectrum_conversion_does_not_mutate_source_entry(self):
        entry = {
            "inchikey": "AAAA-BBBB-CC",
            "Precursor": "[M+H]+",
            "PrecursorMZ": 101.25,
            "ms": np.array([[2.0, 80.0], [1.0, 20.0]], dtype=np.float32),
        }
        original = copy.deepcopy(entry)

        spectrum = spectrum_from_entry(entry, entry["inchikey"], "CCO")

        self.assertIsNotNone(spectrum)
        np.testing.assert_array_equal(entry["ms"], original["ms"])
        self.assertEqual(set(entry), set(original))
        self.assertEqual(spectrum.peaks.mz.tolist(), [20.0, 80.0])
        self.assertEqual(spectrum.get("smiles"), "CCO")
        self.assertEqual(spectrum.get("inchikey_2d"), "AAAA")


class NPLIB1ProviderTest(unittest.TestCase):
    def test_provider_loads_supplied_protocol_and_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "NPLIB1"
            dataset_dir.mkdir()
            with (dataset_dir / "train.pkl").open("wb") as handle:
                pickle.dump([{"smiles": "CCO"}], handle)
            with (dataset_dir / "candidates_supplied.pkl").open("wb") as handle:
                pickle.dump({"CCO": ["CCC"]}, handle)

            provider = NPLIB1Provider(data_dir=root)
            self.assertEqual(provider.load_data("train"), [{"smiles": "CCO"}])
            self.assertEqual(provider.load_candidates("supplied"), {"CCO": ["CCC"]})
            with self.assertRaises(FileNotFoundError):
                provider.load_data("test")
            with self.assertRaisesRegex(ValueError, "supplied"):
                provider.load_candidates("mass")


class NPLIB1DownloadTest(unittest.TestCase):
    def test_validate_source_file_records_sha256_and_git_blob_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.pkl"
            path.write_bytes(b"nplib1-test")
            source = SourceFile(
                filename="split.pkl",
                url="https://example.invalid/split.pkl",
                size_bytes=path.stat().st_size,
                checksum_kind="git_blob_sha1",
                checksum=git_blob_sha1(path),
                source_record="https://example.invalid/source",
            )

            record = validate_source_file(path, source)

            self.assertEqual(record["bytes"], len(b"nplib1-test"))
            self.assertEqual(
                record["sha256"],
                "f247b8d001072dc5b929386bdcacbdd5a3b66338b2cac4721cb3fbc96895594e",
            )


class NPLIB1OverlapAuditTest(unittest.TestCase):
    @staticmethod
    def _record(smiles, inchikey, peaks, precursor_mz=100.0):
        return {
            "smiles": smiles,
            "inchikey": inchikey,
            "peaks": peaks,
            "precursor_mz": precursor_mz,
        }

    def test_spectrum_signature_is_order_and_scale_invariant(self):
        first = self._record("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", [[20, 1], [30, 2]])
        second = self._record("CCO", "LFQSCWFLJHTTHZ-TESTTEST-N", [[30, 20], [20, 10]])

        self.assertEqual(spectrum_signature(first), spectrum_signature(second))

    def test_overlap_uses_two_dimensional_identity_and_spectrum_signature(self):
        left = split_inventory(
            [self._record("C[C@H](O)F", "AAAA-BBBB-CC", [[20, 1]])]
        )
        right = split_inventory(
            [self._record("C[C@@H](O)F", "AAAA-ZZZZ-YY", [[20, 5]])]
        )

        summary = overlap_summary(left, right)

        self.assertEqual(summary["shared_2d_inchikeys"], 1)
        self.assertEqual(summary["shared_canonical_2d_smiles"], 1)
        self.assertEqual(summary["shared_spectrum_signatures"], 1)
        self.assertEqual(
            summary["left_spectra_with_shared_signature_but_different_2d_identity"],
            0,
        )

    def test_overlap_flags_identical_spectrum_with_different_identity(self):
        left = split_inventory([self._record("CCO", "AAAA-BBBB-CC", [[20, 1]])])
        right = split_inventory([self._record("CCN", "DDDD-EEEE-FF", [[20, 2]])])

        summary = overlap_summary(left, right)

        self.assertEqual(summary["shared_2d_inchikeys"], 0)
        self.assertEqual(
            summary["left_spectra_with_shared_signature_but_different_2d_identity"],
            1,
        )
        self.assertEqual(
            summary["different_2d_identity_signature_examples"][0][
                "left_2d_inchikey"
            ],
            "AAAA",
        )
        self.assertEqual(
            summary["different_2d_identity_signature_examples"][0][
                "right_2d_inchikeys"
            ],
            ["DDDD"],
        )


class NPLIB1RerankCacheProtocolTest(unittest.TestCase):
    @staticmethod
    def _args(*, force_include_positive: bool) -> SimpleNamespace:
        return SimpleNamespace(
            candidate_chunk_size=2,
            pre_top_k=2,
            force_include_positive=force_include_positive,
        )

    def test_missing_positive_is_retained_as_coverage_miss(self):
        record = build_query_record(
            spec_index=0,
            spec_emb=torch.tensor([0.0, 0.0, 1.0]),
            true_smiles="CO",
            candidates=["CC", "CCC"],
            smiles_to_idx={"CC": 0, "CCC": 1, "CO": 2},
            mol_embs=torch.eye(3),
            valid_mol_mask=torch.ones(3, dtype=torch.bool),
            args=self._args(force_include_positive=False),
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(record)
        self.assertIsNone(record["label"])
        self.assertFalse(record["positive_in_source_candidates"])
        self.assertFalse(record["positive_in_candidate_pool"])
        self.assertFalse(record["positive_in_base_topk"])
        self.assertFalse(record["positive_forced_into_candidate_pool"])
        self.assertNotIn(2, record["candidate_indices"].tolist())

    def test_legacy_forcing_is_explicitly_recorded(self):
        record = build_query_record(
            spec_index=0,
            spec_emb=torch.tensor([0.0, 0.0, 1.0]),
            true_smiles="CO",
            candidates=["CC", "CCC"],
            smiles_to_idx={"CC": 0, "CCC": 1, "CO": 2},
            mol_embs=torch.eye(3),
            valid_mol_mask=torch.ones(3, dtype=torch.bool),
            args=self._args(force_include_positive=True),
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(record["label"])
        self.assertFalse(record["positive_in_source_candidates"])
        self.assertTrue(record["positive_in_candidate_pool"])
        self.assertFalse(record["positive_in_base_topk"])
        self.assertTrue(record["positive_forced_into_candidate_pool"])
        self.assertTrue(record["positive_forced_into_topk"])

    def test_natural_positive_outside_topk_is_not_marked_as_forced(self):
        record = build_query_record(
            spec_index=0,
            spec_emb=torch.tensor([1.0, 0.0]),
            true_smiles="CO",
            candidates=["CC", "CCC", "CO"],
            smiles_to_idx={"CC": 0, "CCC": 1, "CO": 2},
            mol_embs=torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]]),
            valid_mol_mask=torch.ones(3, dtype=torch.bool),
            args=self._args(force_include_positive=False),
            device=torch.device("cpu"),
        )

        self.assertIsNone(record["label"])
        self.assertTrue(record["positive_in_source_candidates"])
        self.assertTrue(record["positive_in_candidate_pool"])
        self.assertFalse(record["positive_in_base_topk"])
        self.assertFalse(record["positive_forced_into_candidate_pool"])
        self.assertFalse(record["positive_forced_into_topk"])

    def test_pipeline_defaults_nplib1_to_supplied_candidates(self):
        environment = os.environ.copy()
        environment.pop("SPECEMBEDDING_CONFIG", None)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "run_rerank_pipeline.py"),
                "nplib1",
                "--mode",
                "prepare",
                "--checkpoint",
                "checkpoints_align/nplib1.pth",
                "--data_path",
                "data/processed",
                "--pre_top_k",
                "256",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Candidate Source   : supplied", result.stdout)
        self.assertIn("--candidate_type supplied", result.stdout)
        self.assertIn("--no-force_include_positive", result.stdout)


if __name__ == "__main__":
    unittest.main()
