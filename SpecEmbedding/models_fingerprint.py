"""Small learned molecular tower over fixed Morgan inputs; spectral architecture is inherited."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from SpecEmbedding.models_align import SpecMolAlignModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder, validate_spectrum_config


class FingerprintEncoder(nn.Module):
    def __init__(self, *, input_bits, hidden_dim, emb_dim, dropout_rate, norm_eps):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (input_bits, hidden_dim, emb_dim)):
            raise ValueError("Fingerprint dimensions must be explicit positive integers")
        if input_bits % 8:
            raise ValueError("Fingerprint width must be a multiple of eight")
        if (isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float))
                or not math.isfinite(dropout_rate) or not 0 <= dropout_rate < 1):
            raise ValueError("Invalid fingerprint dropout")
        if (isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float))
                or not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError("Invalid fingerprint normalization epsilon")
        self.input_bits, self.emb_dim = input_bits, emb_dim
        self.layers = nn.Sequential(
            nn.Linear(input_bits, hidden_dim), nn.LayerNorm(hidden_dim, eps=norm_eps),
            nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, emb_dim), nn.ReLU(),
        )

    def forward(self, fingerprints):
        if (not isinstance(fingerprints, torch.Tensor) or fingerprints.ndim != 2
                or fingerprints.shape[1] != self.input_bits or not fingerprints.is_floating_point()):
            raise ValueError("Molecular tower requires a floating batch of the configured fingerprint width")
        return self.layers(fingerprints)


class FingerprintAlignmentModel(SpecMolAlignModel):
    """Use unchanged spectral/projection paths and cosine scoring with a trainable fingerprint MLP."""

    def __init__(self, *, spec_config, molecule_config, alignment_config):
        validate_spectrum_config(spec_config)
        if set(molecule_config) != {'input_bits', 'hidden_dim', 'emb_dim', 'dropout_rate', 'norm_eps'}:
            raise ValueError("Incomplete fingerprint molecular configuration")
        if set(alignment_config) != {'final_dim', 'dropout_rate', 'tau'}:
            raise ValueError("Incomplete fingerprint alignment configuration")
        super().__init__(
            spec_encoder=build_spectrum_encoder(spec_config), mol_encoder=FingerprintEncoder(**molecule_config),
            spec_dim=spec_config['dim_target'], hidden_dim=alignment_config['final_dim'],
            final_dim=alignment_config['final_dim'], dropout_rate=alignment_config['dropout_rate'],
            tau=alignment_config['tau'],
        )
        self._construction_config = copy.deepcopy({
            'spec_config': spec_config, 'molecule_config': molecule_config, 'alignment_config': alignment_config,
        })

    def construction_config(self):
        return copy.deepcopy(self._construction_config)

    def encode_mol(self, fingerprints, normalize=False):
        result = self.mol_proj(self.mol_encoder(fingerprints))
        return F.normalize(result, dim=-1) if normalize else result
