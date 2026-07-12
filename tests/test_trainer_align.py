import tempfile
import unittest
from pathlib import Path

import torch

from SpecEmbedding.trainer.trainer_align import TrainerAlign


class TrainerAlignSelectionTest(unittest.TestCase):
    def test_fit_records_best_and_stop_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = TrainerAlign.__new__(TrainerAlign)
            trainer.model = torch.nn.Linear(1, 1)
            trainer.save_dir = temporary
            trainer.stage_summaries = {}
            validation_losses = iter([0.8, 0.5, 0.6, 0.7])
            trainer.train_epoch = lambda optimizer, epoch, stage_name: 1.0
            trainer.validate = lambda epoch, stage_name: next(validation_losses)
            optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

            best_loss = trainer.fit(
                epochs=10,
                optimizer=optimizer,
                stage_name="stage2",
                patience=2,
            )

            summary = trainer.stage_summaries["stage2"]
            self.assertEqual(best_loss, 0.5)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["stop_epoch"], 4)
            self.assertTrue(summary["early_stopped"])
            self.assertEqual(summary["configured_epochs"], 10)
            self.assertTrue((Path(temporary) / "best_model_stage2.pth").is_file())


if __name__ == "__main__":
    unittest.main()
