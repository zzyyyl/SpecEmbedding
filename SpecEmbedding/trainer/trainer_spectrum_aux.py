"""Reuse candidate training embeddings for the train-only conditional spectrum objective."""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from SpecEmbedding.models_spectrum_aux import MoleculeSpectrumAuxiliaryHead, sparse_spectrum_cosine_loss
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils.spectrum_auxiliary_inputs import (
    bind_spectrum_targets,
    spectrum_target_batch,
    update_target_batch_hash,
    verify_spectrum_target_files,
)
from SpecEmbedding.utils.spectrum_targets import require


class SpectrumAuxiliaryTrainer(CandidateTrainerAlign):
    def __init__(self, *args, spectrum_targets, auxiliary_loss_weight, **kwargs):
        require(type(auxiliary_loss_weight) in (int, float) and math.isfinite(auxiliary_loss_weight)
                and auxiliary_loss_weight >= 0, 'Invalid auxiliary loss weight')
        super().__init__(*args, **kwargs)
        require(isinstance(getattr(self.model, 'spectrum_auxiliary', None), MoleculeSpectrumAuxiliaryHead),
                'Auxiliary trainer requires its configured prediction head')
        require(self.model.spectrum_auxiliary.target_settings == spectrum_targets.provenance['source']['settings'],
                'Auxiliary model and target settings differ')
        bind_spectrum_targets(self.train_loader.dataset, spectrum_targets)
        self.spectrum_targets = spectrum_targets
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.auxiliary_epoch_audits = []

    def additional_training_loss(self, batch, f_positive, base_loss, candidate_loss):
        target = spectrum_target_batch(self.spectrum_targets, batch.raw_query_indices)
        update_target_batch_hash(self._target_hash, target)
        n = len(batch.raw_query_indices)
        self._target_seen.index_add_(0, batch.raw_query_indices, torch.ones(n, dtype=torch.long))
        # Zero is used only for synthetic parent-equivalence checks, never a formal enabled run.
        if self.auxiliary_loss_weight == 0:
            return None
        head = self.model.spectrum_auxiliary
        predicted = head(f_positive, target['adduct_ids'].to(self.device))
        auxiliary = sparse_spectrum_cosine_loss(predicted, target['row_ptr'].to(self.device),
            target['bin_indices'].to(self.device), target['values'].to(self.device), eps=head.settings['normalization_eps'])
        weighted = self.auxiliary_loss_weight * auxiliary
        main = base_loss + self.candidate_loss_weight * candidate_loss
        if self._gradient_probe is None:
            main_gradient = torch.autograd.grad(main, f_positive, retain_graph=True)[0].detach()
            auxiliary_gradient = torch.autograd.grad(weighted, f_positive, retain_graph=True)[0].detach()
            norms = [torch.linalg.vector_norm(value) for value in (main_gradient, auxiliary_gradient)]
            denominator = norms[0] * norms[1]
            cosine = (None if denominator.item() == 0 else
                      float((main_gradient * auxiliary_gradient).sum() / denominator))
            self._gradient_probe = {'scope': 'First batch each epoch, shared positive molecule embedding gradients',
                'main_norm': float(norms[0]), 'weighted_auxiliary_norm': float(norms[1]), 'cosine': cosine,
                'queries': n}
        for key, value in (('contrastive', base_loss), ('candidate', candidate_loss), ('auxiliary', auxiliary), ('total', main + weighted)):
            self._loss_sums[key] += float(value.detach()) * n
        return weighted

    def after_backward(self, gradient_norm):
        value = float(gradient_norm)
        require(math.isfinite(value), 'Non-finite auxiliary training parameter gradient')
        self._gradient_norms.append(value)

    def train_epoch(self, optimizer, epoch, stage_name):
        verify_spectrum_target_files(self.spectrum_targets.provenance)
        self._target_hash = hashlib.sha256()
        self._target_seen = torch.zeros(len(self.spectrum_targets), dtype=torch.long)
        self._gradient_probe = None
        self._gradient_norms = []
        self._loss_sums = dict.fromkeys(('contrastive', 'candidate', 'auxiliary', 'total'), 0.)
        loss = super().train_epoch(optimizer, epoch, stage_name)
        require(bool((self._target_seen == 1).all()), 'Spectrum auxiliary epoch lost or duplicated training queries')
        verify_spectrum_target_files(self.spectrum_targets.provenance)
        record = {'stage': stage_name, 'epoch': epoch, 'queries': len(self.spectrum_targets),
            'unique_queries': len(self.spectrum_targets), 'loss_weight': self.auxiliary_loss_weight,
            'observed_target_batch_sha256': self._target_hash.hexdigest(), 'gradient_probe': self._gradient_probe,
            'loss_query_means': {key: value / len(self.spectrum_targets) for key, value in self._loss_sums.items()},
            'optimizer_steps': len(self._gradient_norms), 'gradient_clip_max_norm': 1.,
            'preclip_gradient_norm_mean': sum(self._gradient_norms) / len(self._gradient_norms),
            'preclip_gradient_norm_max': max(self._gradient_norms),
            'clipped_steps': sum(value > 1 for value in self._gradient_norms)}
        directory = Path(self.save_dir) / 'spectrum_auxiliary'
        directory.mkdir(exist_ok=True)
        with (directory / f'{stage_name}_epoch{epoch:03d}.json').open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        self.auxiliary_epoch_audits.append(record)
        return loss

    def fit(self, epochs, optimizer, scheduler=None, stage_name='Stage', patience=5):
        result = super().fit(epochs, optimizer, scheduler=scheduler, stage_name=stage_name, patience=patience)
        self.stage_summaries[stage_name]['spectrum_auxiliary'] = {
            'loss': 'mean_query_cosine_distance_raw_observed_spectrum', 'loss_weight': self.auxiliary_loss_weight,
            'targets': copy.deepcopy(self.spectrum_targets.provenance),
            'epochs': [item for item in self.auxiliary_epoch_audits if item['stage'] == stage_name]}
        return result
