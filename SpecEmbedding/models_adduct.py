"""Prepared downstream adduct conditioning; formal data/runner integration is separate."""

import copy

import torch
from torch import nn

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_attention_pool import AttentionPoolingEncoder


def validate_adduct_settings(settings):
    if (not isinstance(settings, dict)
            or set(settings) != {'adducts', 'unknown_id', 'unknown_policy'}
            or settings['adducts'] != ['[M+H]+', '[M+Na]+']
            or type(settings['unknown_id']) is not int or settings['unknown_id'] != 2
            or settings['unknown_policy'] != 'identity'):
        raise ValueError('Adduct conditioning requires the explicit ordered H/Na vocabulary and unknown identity policy')


class AdductFiLM(nn.Module):
    """One affine transform for known adducts; unknown input always preserves features."""

    def __init__(self, embedding_dim, settings):
        super().__init__()
        validate_adduct_settings(settings)
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError('Adduct conditioning width must be a positive integer')
        self.settings = copy.deepcopy(settings)
        self.embedding_dim = embedding_dim
        # Zeros consume no RNG. The two coefficient slots are delta-scale and shift.
        self.affine = nn.Parameter(torch.zeros(2, 2, embedding_dim))

    def forward(self, features, padding, adduct_ids):
        if (features.ndim != 3 or not features.is_floating_point()
                or features.shape[-1] != self.embedding_dim or min(features.shape[:2]) == 0
                or padding.dtype != torch.bool or padding.shape != features.shape[:2]
                or padding.device != features.device):
            raise ValueError('Adduct conditioning requires nonempty features and matching boolean padding')
        if (not isinstance(adduct_ids, torch.Tensor) or adduct_ids.dtype != torch.long
                or adduct_ids.shape != features.shape[:1] or adduct_ids.device != features.device
                or bool(((adduct_ids < 0) | (adduct_ids > 2)).any())):
            raise ValueError('Adduct IDs must be an explicit int64 vector with values 0, 1, or 2')
        if bool(padding.all(dim=1).any()):
            raise ValueError('Adduct conditioning requires at least one valid token per spectrum')
        safe = features.masked_fill(padding.unsqueeze(-1), 0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError('Non-finite valid features in adduct conditioning')
        coefficients = self.affine[adduct_ids.clamp_max(1)].to(dtype=features.dtype)
        coefficients = coefficients.masked_fill((adduct_ids == 2)[:, None, None], 0)
        result = safe * (1 + coefficients[:, 0, None, :]) + coefficients[:, 1, None, :]
        return result.masked_fill(padding.unsqueeze(-1), 0)


class AdductConditionedEncoder(nn.Module):
    """Condition peak embeddings once, preserving the parent's Transformer and readout."""

    def __init__(self, encoder, settings):
        super().__init__()
        validate_adduct_settings(settings)
        if type(encoder) is AttentionPoolingEncoder:
            base = encoder.base
        elif type(encoder) is SiameseModel:
            base = encoder
        else:
            raise ValueError('Unsupported parent spectrum forward for adduct conditioning')
        if type(base) is not SiameseModel:
            raise ValueError('Unsupported parent spectrum forward for adduct conditioning')
        self.encoder = encoder
        self.conditioning = AdductFiLM(base._encoder.layers[0].self_attn.embed_dim, settings)

    def forward(self, mz, intensity, mask, adduct_ids):
        if (mz.ndim != 2 or intensity.shape != mz.shape or mask.shape != mz.shape
                or not mz.is_floating_point() or not intensity.is_floating_point()
                or mask.dtype != torch.bool or min(mz.shape) == 0
                or intensity.device != mz.device or mask.device != mz.device):
            raise ValueError('Adduct spectrum encoder requires nonempty matching peaks and boolean padding')
        mz, intensity = mz.masked_fill(mask, 0), intensity.masked_fill(mask, 0)
        if not bool((torch.isfinite(mz) & torch.isfinite(intensity)).all()):
            raise ValueError('Non-finite valid peaks in adduct spectrum encoder')
        pooled_parent = type(self.encoder) is AttentionPoolingEncoder
        base = self.encoder.base if pooled_parent else self.encoder
        hidden = self.conditioning(base.embedding(mz, intensity), mask, adduct_ids)
        hidden = base._encoder(hidden, src_key_padding_mask=mask)
        if pooled_parent:
            pooled = self.encoder.pool(hidden, mask)
        else:
            pooled = hidden.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True)
        return base._activation(base._decoder(pooled))
