"""Opt-in downstream spectrum features; the SpecEmbedding pretraining class is unchanged."""

import math

import torch
from torch import nn

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_attention_pool import AttentionPoolingEncoder, validate_attention_pool
from SpecEmbedding.models_qk_norm import enable_qk_norm, validate_qk_norm

SPEC_FIELDS = {'embedding_dim', 'n_head', 'n_layer', 'dim_feedward', 'dim_target', 'feedward_activation'}
DELTA_FIELDS = {'fourier_dim', 'hidden_dim', 'min_wavelength', 'max_wavelength'}


def validate_spectrum_config(config):
    if (not isinstance(config, dict) or not SPEC_FIELDS <= set(config)
            or set(config) - SPEC_FIELDS - {'precursor_delta', 'qk_norm', 'attention_pool'}):
        raise ValueError('Incomplete spectral configuration')
    if 'attention_pool' in config:
        validate_attention_pool(config['attention_pool'])
        if 'precursor_delta' in config:
            raise ValueError('Attention pooling does not support precursor delta forward overrides')
    if 'qk_norm' in config:
        validate_qk_norm(config['qk_norm'])
    if 'precursor_delta' not in config:
        return
    delta = config['precursor_delta']
    if not isinstance(delta, dict) or set(delta) != DELTA_FIELDS:
        raise ValueError('Incomplete precursor delta configuration')
    if (any(type(delta[key]) is not int or delta[key] <= 0 for key in ('fourier_dim', 'hidden_dim'))
            or delta['fourier_dim'] % 2):
        raise ValueError('Precursor delta dimensions must be positive integers with even Fourier width')
    if (any(isinstance(delta[key], bool) or not isinstance(delta[key], (int, float))
            or not math.isfinite(delta[key]) or delta[key] <= 0
            for key in ('min_wavelength', 'max_wavelength'))
            or delta['max_wavelength'] <= delta['min_wavelength']):
        raise ValueError('Precursor delta wavelengths must be finite, positive and increasing')


class PrecursorDeltaEncoder(SiameseModel):
    """Encode signed m/z differences before the inherited Transformer, without clipping peaks.

    Token zero must be the precursor. Differences are m/z differences, not charge-corrected
    neutral losses. The last residual projection starts at zero; the branch learns during
    downstream training and does not require a pretrained checkpoint.
    """

    def __init__(self, *, spec_config, delta_config):
        validate_spectrum_config({**spec_config, 'precursor_delta': delta_config})
        super().__init__(**spec_config)
        wavelengths = torch.logspace(math.log10(delta_config['min_wavelength']),
                                    math.log10(delta_config['max_wavelength']),
                                    delta_config['fourier_dim'] // 2, dtype=torch.float32)
        frequencies = 2 * math.pi / wavelengths
        if not bool((torch.isfinite(wavelengths) & (wavelengths > 0)
                     & torch.isfinite(frequencies) & (frequencies > 0)).all()):
            raise ValueError('Precursor delta wavelengths are not representable in float32')
        self.register_buffer('delta_frequencies', frequencies, persistent=False)
        self.delta_projection = nn.Sequential(
            nn.Linear(delta_config['fourier_dim'] + 1, delta_config['hidden_dim']),
            nn.GELU(), nn.Linear(delta_config['hidden_dim'], spec_config['embedding_dim']),
        )
        nn.init.zeros_(self.delta_projection[-1].weight)
        nn.init.zeros_(self.delta_projection[-1].bias)

    def delta_features(self, mz, mask):
        """Keep negative differences and distinguish the precursor from zero-difference fragments."""
        safe_mz = mz.masked_fill(mask, 0)
        delta = safe_mz[:, :1] - safe_mz
        phase = delta.unsqueeze(-1) * self.delta_frequencies
        role = torch.zeros_like(safe_mz)
        role[:, 0] = 1
        features = torch.cat((phase.sin(), phase.cos(), role.unsqueeze(-1)), dim=-1)
        return features.masked_fill(mask.unsqueeze(-1), 0)

    def forward(self, mz, intensity, mask):
        if (mz.ndim != 2 or intensity.shape != mz.shape or mask.shape != mz.shape
                or mask.dtype != torch.bool or not mz.is_floating_point() or not intensity.is_floating_point()
                or mz.shape[0] == 0 or mz.shape[1] == 0):
            raise ValueError('Precursor delta encoder requires nonempty, equally shaped peak tensors and boolean mask')
        if bool(mask[:, 0].any()):
            raise ValueError('Precursor delta encoder requires an unmasked precursor at token zero')
        # Padding must never contribute NaNs to attention or mean pooling.
        mz, intensity = mz.masked_fill(mask, 0), intensity.masked_fill(mask, 0)
        if not bool((torch.isfinite(mz) & torch.isfinite(intensity)).all()):
            raise ValueError('Precursor delta encoder received non-finite unmasked peaks')
        x = self.embedding(mz, intensity)
        residual = self.delta_projection(self.delta_features(mz, mask)).masked_fill(mask.unsqueeze(-1), 0)
        x = self._encoder(x + residual, src_key_padding_mask=mask)
        pooled = x.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        return self._activation(self._decoder(pooled))


def build_spectrum_encoder(config):
    validate_spectrum_config(config)
    base = {key: config[key] for key in SPEC_FIELDS}
    if 'precursor_delta' not in config:
        encoder = SiameseModel(**base)
    else:
        encoder = PrecursorDeltaEncoder(spec_config=base, delta_config=config['precursor_delta'])
    if 'qk_norm' in config:
        encoder = enable_qk_norm(encoder, config['qk_norm'])
    return AttentionPoolingEncoder(encoder, config['attention_pool']) if 'attention_pool' in config else encoder
