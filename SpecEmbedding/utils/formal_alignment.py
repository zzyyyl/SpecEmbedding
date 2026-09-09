"""Construct formal models from their own immutable checkpoint metadata."""

import copy
import json
from pathlib import Path

import torch
from rdkit import rdBase

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_fingerprint import FingerprintAlignmentModel
from SpecEmbedding.utils.fulltrain import sha256_file

SPEC_FIELDS = {'embedding_dim', 'n_head', 'n_layer', 'dim_feedward', 'dim_target', 'feedward_activation'}
ALIGN_FIELDS = {'final_dim', 'dropout_rate', 'tau'}
GINE_FIELDS = {'emb_dim', 'n_layers', 'dropout_rate', 'size_feature_dim', 'norm_type', 'norm_eps', 'graph_policy'}
FINGERPRINT_FIELDS = {'input_bits', 'hidden_dim', 'emb_dim', 'dropout_rate', 'norm_eps', 'graph_policy'}


def formal_model_type(model_config):
    """A missing tag identifies the existing GINE metadata format, never a new tower."""
    if not isinstance(model_config, dict) or set(model_config) not in (
        {'spec_encoder', 'mol_encoder', 'align'}, {'type', 'spec_encoder', 'mol_encoder', 'align'}
    ):
        raise ValueError('Incomplete formal model configuration')
    kind = model_config.get('type', 'gine')
    if kind not in ('gine', 'fingerprint'):
        raise ValueError('Unknown formal alignment model type')
    if (set(model_config['spec_encoder']) != SPEC_FIELDS or set(model_config['align']) != ALIGN_FIELDS
            or set(model_config['mol_encoder']) != (GINE_FIELDS if kind == 'gine' else FINGERPRINT_FIELDS)
            or model_config['mol_encoder']['graph_policy'] != 'rdkit_sanitized'):
        raise ValueError('Incomplete formal model construction fields or graph policy')
    return kind


def build_formal_alignment(model_config):
    kind = formal_model_type(model_config)
    molecule = {key: value for key, value in model_config['mol_encoder'].items() if key != 'graph_policy'}
    spec, align = model_config['spec_encoder'], model_config['align']
    if kind == 'fingerprint':
        return FingerprintAlignmentModel(spec_config=spec, molecule_config=molecule, alignment_config=align)
    return SpecMolAlignModel(spec_encoder=SiameseModel(**spec), mol_encoder=GINEEncoder(**molecule),
                             spec_dim=spec['dim_target'], hidden_dim=align['final_dim'], final_dim=align['final_dim'],
                             dropout_rate=align['dropout_rate'], tau=align['tau'])


def read_formal_alignment_checkpoint(checkpoint, *, dataset_outputs, dataset_manifest_sha256,
                                    tokenizer_config, expected_counts, exclusions):
    """Bind model-specific construction to a shared data/identity protocol, not to another model."""
    checkpoint = Path(checkpoint).resolve()
    selection_path = checkpoint.parent / 'alignment_selection.json'
    before = {'checkpoint': sha256_file(checkpoint), 'selection': sha256_file(selection_path)}
    selection = json.loads(selection_path.read_text())
    model = selection['model_config']
    kind = formal_model_type(model)
    audit, snapshot = selection['fulltrain_audit'], selection['config_snapshot']
    if (selection['checkpoint_sha256'] != before['checkpoint'] or selection['seed'] != 42
            or selection['graph_policy'] != 'rdkit_sanitized' or selection['rdkit_version'] != rdBase.rdkitVersion
            or not audit['formal_fulltrain'] or audit['dataset_version'] != '1.5'
            or audit['dataset_manifest_sha256'] != dataset_manifest_sha256 or audit['input_outputs'] != dataset_outputs
            or audit['expected_epoch_counts'] != expected_counts or selection['exclude_val_query_indices'] != exclusions
            or snapshot['model'] != model or snapshot['data']['tokenizer'] != tokenizer_config):
        raise ValueError('Formal checkpoint construction or shared data/tokenizer/exclusion provenance mismatch')
    if kind == 'fingerprint' and (not selection.get('training_fingerprint_cache')
                                  or not selection.get('validation_fingerprint_cache')):
        raise ValueError('Fingerprint checkpoint is missing its fixed input provenance')
    if sha256_file(checkpoint) != before['checkpoint'] or sha256_file(selection_path) != before['selection']:
        raise ValueError('Formal checkpoint or selection changed during verification')
    receipt = {'checkpoint': str(checkpoint), 'checkpoint_sha256': before['checkpoint'],
               'selection': str(selection_path), 'selection_sha256': before['selection'],
               'model_type': kind, 'model_config': copy.deepcopy(model),
               'dataset_manifest_sha256': dataset_manifest_sha256,
               'tokenizer_config': copy.deepcopy(tokenizer_config), 'expected_epoch_counts': dict(expected_counts),
               'exclude_val_query_indices': list(exclusions)}
    return selection, receipt


def load_formal_alignment(checkpoint, device, **protocol):
    selection, receipt = read_formal_alignment_checkpoint(checkpoint, **protocol)
    # No shape adaptation, missing-key fallback, or candidate-model configuration is allowed here.
    model = build_formal_alignment(receipt['model_config'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    _, after = read_formal_alignment_checkpoint(checkpoint, **protocol)
    if receipt != after:
        raise ValueError('Formal checkpoint changed during model loading')
    return model.to(device).eval(), selection, receipt
