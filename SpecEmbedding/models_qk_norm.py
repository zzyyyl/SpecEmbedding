"""Optional downstream Q/K RMS normalization; no changes to the pretraining encoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_qk_norm(config):
    if (not isinstance(config, dict) or set(config) != {'eps'}
            or isinstance(config['eps'], bool) or not isinstance(config['eps'], (int, float))
            or not math.isfinite(config['eps']) or config['eps'] <= 0):
        raise ValueError('Q/K normalization requires one explicit finite positive eps')
    if not 0 < config['eps'] <= torch.finfo(torch.float32).max or float(torch.tensor(config['eps'])) == 0:
        raise ValueError('Q/K normalization eps must be representable in float32')


class QKNormEncoderLayer(nn.Module):
    """Reuse a newly built downstream layer, adding only per-head-dimension Q/K norms.

    This explicit forward also runs in inference_mode: a stock fused encoder-layer
    fast path must not bypass normalization. State keys and RNG of inherited modules
    remain unchanged. Affine vectors are shared across heads, separately for Q and K.
    """

    def __init__(self, original, config):
        super().__init__()
        validate_qk_norm(config)
        attention = original.self_attn
        if (not original.norm_first or not attention.batch_first or not attention._qkv_same_embed_dim
                or attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn):
            raise ValueError('Q/K variant requires the existing pre-norm, batch-first spectral attention')
        for name in ('self_attn', 'linear1', 'linear2', 'norm1', 'norm2', 'dropout', 'dropout1', 'dropout2'):
            setattr(self, name, getattr(original, name))
        self.activation = original.activation
        self.norm_first = True
        factory = {'device': attention.in_proj_weight.device, 'dtype': attention.in_proj_weight.dtype}
        self.q_norm = nn.RMSNorm(attention.head_dim, eps=config['eps'], **factory)
        self.k_norm = nn.RMSNorm(attention.head_dim, eps=config['eps'], **factory)
        self.train(original.training)

    def attention(self, x, padding):
        batch, length, width = x.shape
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = (part.reshape(batch, length, self.self_attn.num_heads, self.self_attn.head_dim)
                   .transpose(1, 2) for part in qkv.chunk(3, dim=-1))
        q, k = self.q_norm(q), self.k_norm(k)
        # SDPA's boolean mask uses True=allowed, unlike key_padding_mask.
        allowed = None if padding is None else (~padding)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.self_attn.dropout if self.training else 0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.dropout1(self.self_attn.out_proj(attended))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if is_causal or src_mask is not None:
            raise ValueError('Spectral Q/K attention only supports full bidirectional attention with padding')
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0 or src.shape[-1] != self.self_attn.embed_dim:
            raise ValueError('Spectral attention requires a batch-first tensor of the configured width')
        padding = None
        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != src.shape[:2]:
                raise ValueError('Spectral padding shape differs from the peak batch')
            if src_key_padding_mask.dtype == torch.bool:
                padding = src_key_padding_mask
            elif src_key_padding_mask.is_floating_point():
                # TransformerEncoder canonicalizes boolean padding to 0/-inf.
                if not bool(((src_key_padding_mask == 0) | torch.isneginf(src_key_padding_mask)).all()):
                    raise ValueError('Spectral padding only accepts boolean or canonical 0/-inf masks')
                padding = torch.isneginf(src_key_padding_mask)
            else:
                raise ValueError('Spectral padding requires boolean or canonical floating masks')
            if bool(padding.all(dim=1).any()):
                raise ValueError('Every spectrum must have at least one valid peak')
            src = src.masked_fill(padding.unsqueeze(-1), 0)
        x = src + self.attention(self.norm1(src), padding)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x if padding is None else x.masked_fill(padding.unsqueeze(-1), 0)


def enable_qk_norm(encoder, config):
    """Apply only to a fresh, explicitly configured downstream spectrum instance."""
    validate_qk_norm(config)
    if encoder._encoder.enable_nested_tensor or any(isinstance(layer, QKNormEncoderLayer) for layer in encoder._encoder.layers):
        raise ValueError('Q/K normalization requires non-nested layers and cannot be installed twice')
    encoder._encoder.layers = nn.ModuleList(QKNormEncoderLayer(layer, config) for layer in encoder._encoder.layers)
    return encoder
