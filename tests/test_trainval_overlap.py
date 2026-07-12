import unittest

import numpy as np

from SpecEmbedding.data.overlap import filter_classified_validation


class SpectrumStub:
    def __init__(self, smiles: str):
        self.smiles = smiles

    def get(self, key: str):
        if key == "smiles":
            return self.smiles
        return None


class TrainValidationOverlapTest(unittest.TestCase):
    def setUp(self):
        self.classified = {
            "train_data": {1: [{"smiles": "TRAIN"}]},
            "train_keys": np.asarray([1]),
            "val_data": {
                10: [{"smiles": "A"}, {"smiles": "A"}, {"smiles": "A"}],
                20: [{"smiles": "B"}],
            },
            "val_keys": np.asarray([10, 20]),
        }
        self.val_raw = [SpectrumStub("A"), SpectrumStub("A"), SpectrumStub("A"), SpectrumStub("B")]

    def test_complete_validation_molecule_group_is_removed(self):
        filtered, report = filter_classified_validation(
            self.classified,
            self.val_raw,
            [2, 0, 1, 1],
        )

        np.testing.assert_array_equal(filtered["train_keys"], np.asarray([1]))
        np.testing.assert_array_equal(filtered["val_keys"], np.asarray([20]))
        self.assertEqual(set(filtered["val_data"]), {20})
        self.assertEqual(report["query_indices"], [0, 1, 2])
        self.assertEqual(report["smiles"], ["A"])
        self.assertEqual(report["keys"], [10])
        self.assertEqual(report["original_num_keys"], 2)
        self.assertEqual(report["filtered_num_keys"], 1)

    def test_partial_validation_molecule_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete molecule groups"):
            filter_classified_validation(self.classified, self.val_raw, [0, 1])

    def test_out_of_range_query_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            filter_classified_validation(self.classified, self.val_raw, [4])

    def test_raw_and_cached_validation_mismatch_is_rejected(self):
        mismatched = [SpectrumStub("A"), SpectrumStub("A"), SpectrumStub("B")]

        with self.assertRaisesRegex(RuntimeError, "different SMILES counts"):
            filter_classified_validation(self.classified, mismatched, [0, 1])


if __name__ == "__main__":
    unittest.main()
