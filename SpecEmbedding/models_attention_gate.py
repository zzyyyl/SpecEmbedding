"""Prepared downstream SDPA head gates; the shared pretraining model stays unchanged."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_adduct import AdductConditionedEncoder
from SpecEmbedding.models_attention_pool import AttentionPoolingEncoder
from SpecEmbedding.models_qk_norm import QKNormEncoderLayer

GATE_SETTINGS = {
    'granularity': 'headwise', 'placement': 'sdpa_output', 'activation': 'sigmoid',
    'scale': 2.0, 'initialization': 'zero_weight', 'bias': False,
}


def validate_attention_gate(settings):
    if (not isinstance(settings, dict) or set(settings) != set(GATE_SETTINGS) or settings != GATE_SETTINGS
            or type(settings['scale']) not in (int, float) or type(settings['bias']) is not bool):
        raise ValueError('Attention gate requires the complete registered headwise identity-initialized configuration')


class GatedAttentionEncoderLayer(nn.Module):
    """Keep the parent layer's weights and apply query-conditioned gates before W_O.

    An explicit forward is necessary in both training and inference: the stock
    fused TransformerEncoderLayer path cannot execute the added gate.
    """

    def __init__(self, original, settings):
        super().__init__()
        validate_attention_gate(settings)
        if type(original) not in (nn.TransformerEncoderLayer, QKNormEncoderLayer):
            raise ValueError('Attention gates require a fresh stock or Q/K-normalized downstream layer')
        attention = original.self_attn
        if (not original.norm_first or not attention.batch_first or not attention._qkv_same_embed_dim
                or attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn):
            raise ValueError('Attention gates require the existing pre-norm batch-first spectral attention')
        for name in ('self_attn', 'linear1', 'linear2', 'norm1', 'norm2', 'dropout', 'dropout1', 'dropout2'):
            setattr(self, name, getattr(original, name))
        self.activation = original.activation
        self.norm_first = True
        self._qk_norm_enabled = isinstance(original, QKNormEncoderLayer)
        if self._qk_norm_enabled:
            self.q_norm, self.k_norm = original.q_norm, original.k_norm
        factory = {'device': attention.in_proj_weight.device, 'dtype': attention.in_proj_weight.dtype}
        # zeros() does not consume the parent's initialization RNG stream.
        self.gate_weight = nn.Parameter(torch.zeros(attention.num_heads, attention.embed_dim, **factory))
        self._gate_settings = copy.deepcopy(settings)
        self.train(original.training)

    def gate_values(self, normalized_tokens):
        """Return ephemeral [batch, peak, head] values; do not retain learned activations."""
        return self._gate_settings['scale'] * torch.sigmoid(F.linear(normalized_tokens, self.gate_weight))

    def attention(self, normalized, padding):
        batch, length, width = normalized.shape
        qkv = F.linear(normalized, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = (part.reshape(batch, length, self.self_attn.num_heads, self.self_attn.head_dim)
                   .transpose(1, 2) for part in qkv.chunk(3, dim=-1))
        if self._qk_norm_enabled:
            q, k = self.q_norm(q), self.k_norm(k)
        allowed = None if padding is None else (~padding)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.self_attn.dropout if self.training else 0.0, is_causal=False,
        )
        gates = self.gate_values(normalized).transpose(1, 2).unsqueeze(-1)
        attended = (attended * gates).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.dropout1(self.self_attn.out_proj(attended))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if is_causal or src_mask is not None:
            raise ValueError('Spectral gates require full bidirectional attention with padding only')
        if src.ndim != 3 or not src.shape[0] or not src.shape[1] or src.shape[-1] != self.self_attn.embed_dim:
            raise ValueError('Spectral gates require nonempty batch-first tokens of the configured width')
        padding = None
        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != src.shape[:2]:
                raise ValueError('Spectral padding shape differs from token batch')
            if src_key_padding_mask.dtype == torch.bool:
                padding = src_key_padding_mask
            elif src_key_padding_mask.is_floating_point():
                if not bool(((src_key_padding_mask == 0) | torch.isneginf(src_key_padding_mask)).all()):
                    raise ValueError('Spectral gates accept only boolean or canonical 0/-inf padding')
                padding = torch.isneginf(src_key_padding_mask)
            else:
                raise ValueError('Spectral gates accept only boolean or canonical floating padding')
            if bool(padding.all(dim=1).any()):
                raise ValueError('Every spectrum must contain a valid peak')
            src = src.masked_fill(padding.unsqueeze(-1), 0)
        x = src + self.attention(self.norm1(src), padding)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x if padding is None else x.masked_fill(padding.unsqueeze(-1), 0)


def enable_attention_gate(encoder, settings):
    """Apply only to a fresh downstream instance, preserving inherited state keys and RNG."""
    validate_attention_gate(settings)
    base = encoder.encoder if type(encoder) is AdductConditionedEncoder else encoder
    base = base.base if type(base) is AttentionPoolingEncoder else base
    if type(base) is not SiameseModel:
        raise ValueError('Gates require an explicitly supported downstream spectrum forward')
    layers = base._encoder.layers
    if (base._encoder.enable_nested_tensor or not layers
            or any(type(layer) not in (nn.TransformerEncoderLayer, QKNormEncoderLayer) for layer in layers)):
        raise ValueError('Gates require non-nested original layers and cannot be installed twice')
    base._encoder.layers = nn.ModuleList(GatedAttentionEncoderLayer(layer, settings) for layer in layers)
    return encoder


def build_gated_spectrum_encoder(parent_config, gate_config):
    """Component factory; formal configuration acceptance is intentionally a later step."""
    from SpecEmbedding.models_precursor_delta import build_spectrum_encoder

    validate_attention_gate(gate_config)
    if 'attention_output_gate' in parent_config or 'precursor_delta' in parent_config:
        raise ValueError('Gate component requires a supported parent without a gate or precursor override')
    return enable_attention_gate(build_spectrum_encoder(parent_config), gate_config)
