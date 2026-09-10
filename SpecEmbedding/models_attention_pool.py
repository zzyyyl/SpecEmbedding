"""Prepared downstream readout component; formal construction is deliberately not enabled yet."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.models import SiameseModel


def validate_attention_pool(settings):
    if not isinstance(settings, dict) or set(settings) != {'norm_eps'}:
        raise ValueError('Attention pooling requires explicit norm_eps only')
    eps = settings['norm_eps']
    if (isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps)
            or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max):
        raise ValueError('Attention pooling norm_eps must be positive and representable in float32')


class MaskedAttentionPool(nn.Module):
    """One zero-initialized query; normalized keys, original values, no learned projection."""

    def __init__(self, embedding_dim, *, norm_eps):
        super().__init__()
        validate_attention_pool({'norm_eps': norm_eps})
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError('Attention pooling dimension must be a positive integer')
        self.norm_eps = float(norm_eps)
        self.query = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, features, padding, *, return_weights=False):
        if (features.ndim != 3 or not features.is_floating_point() or features.shape[-1] != self.query.numel()
                or padding.dtype != torch.bool or padding.shape != features.shape[:2]
                or padding.device != features.device or features.shape[0] == 0 or features.shape[1] == 0):
            raise ValueError('Attention pooling requires nonempty features and a matching boolean padding mask')
        if bool(padding.all(dim=1).any()):
            raise ValueError('Attention pooling requires at least one valid token per query')
        values = features.masked_fill(padding.unsqueeze(-1), 0)
        keys = F.layer_norm(values, (self.query.numel(),), eps=self.norm_eps)
        scores = (keys * self.query).sum(-1) / math.sqrt(self.query.numel())
        weights = scores.masked_fill(padding, -torch.inf).softmax(dim=1)
        pooled = (weights.unsqueeze(-1) * values).sum(dim=1)
        return (pooled, weights) if return_weights else pooled


class AttentionPoolingEncoder(nn.Module):
    """Keep an existing downstream encoder and replace only its final mean readout.

    Q/K normalization may already be installed in the inherited Transformer layers.
    Other forward overrides are rejected rather than silently bypassed.
    """

    def __init__(self, encoder, settings):
        super().__init__()
        validate_attention_pool(settings)
        if type(encoder) is not SiameseModel:
            raise ValueError('Attention pooling requires an explicitly supported SiameseModel forward')
        self.base = encoder
        self.pool = MaskedAttentionPool(encoder._encoder.layers[0].self_attn.embed_dim,
                                       norm_eps=settings['norm_eps'])

    def forward(self, mz, intensity, mask):
        if (mz.ndim != 2 or intensity.shape != mz.shape or mask.shape != mz.shape or mask.dtype != torch.bool
                or not mz.is_floating_point() or not intensity.is_floating_point()):
            raise ValueError('Attention encoder requires matching floating peaks and boolean padding')
        mz, intensity = mz.masked_fill(mask, 0), intensity.masked_fill(mask, 0)
        hidden = self.base.embedding(mz, intensity)
        hidden = self.base._encoder(hidden, src_key_padding_mask=mask)
        return self.base._activation(self.base._decoder(self.pool(hidden, mask)))
