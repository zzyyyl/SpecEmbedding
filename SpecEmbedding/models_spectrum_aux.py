"""Prepared training-only spectrum predictor; no formal alignment activation yet."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.utils.spectrum_targets import require, validate_target_settings


class MoleculeSpectrumAuxiliaryHead(nn.Module):
    """Predict a conditional spectrum from the molecule's shared embedding."""

    def __init__(self, settings, target_settings):
        super().__init__()
        require(isinstance(settings, dict) and set(settings) == {
            'embedding_dim', 'hidden_dim', 'normalization_eps', 'initialization_seed'}, 'Incomplete auxiliary head settings')
        require(all(type(settings[k]) is int and settings[k] > 0 for k in ('embedding_dim', 'hidden_dim'))
                and type(settings['initialization_seed']) is int and 0 <= settings['initialization_seed'] < 2**63,
                'Invalid auxiliary head dimensions or seed')
        eps = settings['normalization_eps']
        require(type(eps) in (int, float) and math.isfinite(eps) and eps > 0, 'Invalid auxiliary normalization epsilon')
        self.settings, self.target_settings = copy.deepcopy(settings), copy.deepcopy(target_settings)
        self.output_dim = validate_target_settings(target_settings)
        # Seed only the CPU generator and restore it. Do not touch any CUDA RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(settings['initialization_seed'])
            self.input = nn.Linear(settings['embedding_dim'] + 2, settings['hidden_dim'], device='cpu')
            self.output = nn.Linear(settings['hidden_dim'], self.output_dim, device='cpu')

    def forward(self, molecule_embeddings, adduct_ids):
        require(isinstance(molecule_embeddings, torch.Tensor) and molecule_embeddings.ndim == 2
                and molecule_embeddings.shape[0] > 0
                and molecule_embeddings.shape[1] == self.settings['embedding_dim']
                and molecule_embeddings.dtype in (torch.float32, torch.float64)
                and bool(torch.isfinite(molecule_embeddings).all()), 'Invalid molecule embeddings for auxiliary prediction')
        require(isinstance(adduct_ids, torch.Tensor) and adduct_ids.dtype == torch.long
                and adduct_ids.shape == molecule_embeddings.shape[:1] and adduct_ids.device == molecule_embeddings.device
                and bool(((adduct_ids >= 0) & (adduct_ids <= 2)).all()), 'Auxiliary adduct IDs must be explicit int64 values 0, 1, or 2')
        conditions = torch.stack((adduct_ids == 0, adduct_ids == 1), dim=1).to(molecule_embeddings.dtype)
        normalized = F.normalize(molecule_embeddings, dim=-1, eps=self.settings['normalization_eps'])
        hidden = F.silu(self.input(torch.cat((normalized, conditions), dim=1)))
        predicted = F.softplus(self.output(hidden))
        require(bool(torch.isfinite(predicted).all()), 'Non-finite auxiliary spectrum prediction')
        return predicted


def sparse_spectrum_cosine_loss(predicted, row_ptr, bin_indices, values, *, eps):
    """Mean per-query cosine distance, with a sparse unit-norm target and full dense prediction norm."""
    require(isinstance(predicted, torch.Tensor) and predicted.ndim == 2 and min(predicted.shape) > 0
            and predicted.dtype in (torch.float32, torch.float64)
            and bool(torch.isfinite(predicted).all()) and bool((predicted >= 0).all()),
            'Invalid nonnegative auxiliary predictions')
    require(type(eps) in (int, float) and math.isfinite(eps) and eps > 0, 'Invalid cosine epsilon')
    require(all(isinstance(x, torch.Tensor) and x.device == predicted.device for x in (row_ptr, bin_indices, values))
            and row_ptr.dtype == bin_indices.dtype == torch.long and values.dtype == predicted.dtype,
            'Sparse target tensors require matching devices, floating values and int64 indices')
    require(row_ptr.shape == (len(predicted) + 1,) and bin_indices.ndim == values.ndim == 1
            and len(bin_indices) == len(values) > 0 and row_ptr[0].item() == 0 and row_ptr[-1].item() == len(values)
            and bool((torch.diff(row_ptr) > 0).all()) and bool(((bin_indices >= 0) & (bin_indices < predicted.shape[1])).all())
            and bool(torch.isfinite(values).all()) and bool((values > 0).all()), 'Malformed sparse spectrum targets')
    rows = torch.repeat_interleave(torch.arange(len(predicted), device=predicted.device), torch.diff(row_ptr))
    require(bool(((rows[1:] != rows[:-1]) | (bin_indices[1:] > bin_indices[:-1])).all()),
            'Sparse bins must be strictly increasing within each query')
    target_norms = torch.zeros(len(predicted), device=predicted.device, dtype=predicted.dtype).index_add(0, rows, values.square())
    require(bool(torch.allclose(target_norms, torch.ones_like(target_norms), rtol=2e-6, atol=2e-6)),
            'Sparse spectrum target must have unit norm for every query')
    normalized = F.normalize(predicted, dim=1, eps=eps)
    products = normalized[rows, bin_indices] * values
    dots = torch.zeros(len(predicted), device=predicted.device, dtype=predicted.dtype).index_add(0, rows, products)
    return (1 - dots).mean()
