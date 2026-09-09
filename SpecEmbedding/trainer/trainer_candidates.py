"""Candidate-supervised alignment component; formal runner integration is separate."""

import hashlib
import math
from numbers import Real

import torch

from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset, CandidateAlignmentBatch
from SpecEmbedding.loss_candidates import candidate_alignment_loss
from SpecEmbedding.trainer.trainer_align import TrainerAlign


class CandidateTrainerAlign(TrainerAlign):
    """Add a per-query candidate CE term while keeping the baseline validation path.

    No parameters or retrieval scoring change. Candidate graphs are encoded with
    gradients by the current molecular tower; no stale embedding cache is used.
    Full graph batches are retained for backward, so actual CUDA peak memory must
    be measured before a formal candidate trial is accepted.
    """

    def __init__(self, *args, candidate_loss_weight, **kwargs):
        if (not isinstance(candidate_loss_weight, Real) or isinstance(candidate_loss_weight, bool)
                or not math.isfinite(candidate_loss_weight) or candidate_loss_weight <= 0):
            raise ValueError("Candidate loss weight must be explicitly finite and positive")
        super().__init__(*args, **kwargs)
        if not isinstance(self.train_loader.dataset, CandidateAlignDataset):
            raise ValueError("Candidate trainer requires the verified full-query candidate dataset")
        if self.train_loader.persistent_workers:
            raise ValueError("Recreate candidate workers each epoch so they receive the current sampling epoch")
        if self.train_loader.drop_last:
            raise ValueError("Candidate training cannot drop the final query batch")
        self.candidate_loss_weight = float(candidate_loss_weight)
        self.candidate_epoch_audits = []

    def training_batch_loss(self, batch):
        if not isinstance(batch, CandidateAlignmentBatch):
            raise ValueError("Candidate trainer received an ordinary or malformed batch")
        mzs, ints, masks, mols, labels = batch.anchor
        n = mzs.shape[0]
        raw = batch.raw_query_indices
        if (raw.dtype != torch.long or raw.ndim != 1 or len(raw) != n or raw.device.type != "cpu"
                or raw.min() < 0 or raw.max() >= len(self._candidate_seen)):
            raise ValueError("Candidate batch has invalid original query indices")
        f_spec, f_positive, scale = self.model(mzs.to(self.device), ints.to(self.device),
                                             masks.to(self.device), mols.to(self.device))
        f_negative = (f_positive[:0] if batch.negative_graphs is None
                      else self.model.encode_mol(batch.negative_graphs.to(self.device)))
        if len(f_negative) != len(batch.candidate_indices) or len(f_negative) != len(batch.source_positions):
            raise ValueError("Encoded negative graphs and source metadata differ")
        candidate_loss = candidate_alignment_loss(f_spec, f_positive, f_negative, batch.negative_ptr, scale)
        base_loss = self.criterion(f_spec, f_positive, scale, labels)
        self._candidate_seen.index_add_(0, raw, torch.ones_like(raw))
        counts = batch.negative_ptr[1:] - batch.negative_ptr[:-1]
        self._candidate_counts[raw] = counts
        for value in (raw, batch.negative_ptr, batch.candidate_indices, batch.source_positions):
            self._candidate_sample_hash.update(len(value).to_bytes(8, "little"))
            self._candidate_sample_hash.update(value.numpy().astype("<i8").tobytes())
        self._candidate_loss_sum += float(candidate_loss.detach()) * n
        return base_loss + self.candidate_loss_weight * candidate_loss, n

    def train_epoch(self, optimizer, epoch, stage_name):
        dataset = self.train_loader.dataset
        dataset.set_epoch(epoch)
        self._candidate_seen = torch.zeros(len(dataset), dtype=torch.long)
        self._candidate_counts = torch.zeros(len(dataset), dtype=torch.long)
        self._candidate_sample_hash = hashlib.sha256()
        self._candidate_loss_sum = 0.0
        loss = super().train_epoch(optimizer, epoch, stage_name)
        if not torch.all(self._candidate_seen == 1):
            raise RuntimeError("Candidate training did not visit each original training query exactly once")
        counts = self._candidate_counts
        self.candidate_epoch_audits.append({
            "stage": stage_name, "epoch": epoch, "seed": dataset.seed, "queries": len(dataset),
            "unique_queries": len(dataset), "negative_samples": int(counts.sum()),
            "minimum_negatives": int(counts.min()), "maximum_negatives": int(counts.max()),
            "queries_without_negatives": int((counts == 0).sum()),
            "candidate_loss_query_mean": self._candidate_loss_sum / len(dataset),
            "observed_query_sample_order_sha256": self._candidate_sample_hash.hexdigest(),
        })
        return loss

    def fit(self, epochs, optimizer, scheduler=None, stage_name="Stage", patience=5):
        result = super().fit(epochs, optimizer, scheduler=scheduler, stage_name=stage_name, patience=patience)
        self.stage_summaries[stage_name]["candidate_training"] = {
            "loss": "baseline_bidirectional_inbatch_plus_per_query_candidate_ce",
            "candidate_loss_weight": self.candidate_loss_weight,
            "data": self.train_loader.dataset.provenance,
            "epochs": [record for record in self.candidate_epoch_audits if record["stage"] == stage_name],
        }
        return result
