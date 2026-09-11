"""Bind train-only peak targets to model configuration and observed query batches."""

import copy
import math
from pathlib import Path

import numpy as np
import torch

from SpecEmbedding.models_spectrum_aux import auxiliary_model_settings
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.spectrum_targets import SpectrumTargetCache, load_spectrum_target_cache, require


def auxiliary_training_weight(model_config, training_config):
    settings = auxiliary_model_settings(model_config)
    active = 'spectrum_auxiliary' in training_config
    require((settings is not None) == active, 'Auxiliary model and training settings must be supplied together')
    if not active:
        return None
    value = training_config['spectrum_auxiliary']
    require(isinstance(value, dict) and set(value) == {'loss_weight'}, 'Incomplete auxiliary training settings')
    weight = value['loss_weight']
    require(type(weight) in (int, float) and math.isfinite(weight) and weight > 0,
            'Formal auxiliary loss weight must be finite and positive')
    return float(weight)


def bind_spectrum_targets(dataset, cache):
    from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset

    require(isinstance(dataset, CandidateAlignDataset) and isinstance(cache, SpectrumTargetCache),
            'Spectrum targets require the complete candidate dataset and audited cache')
    require(cache.provenance['source']['dataset_manifest_sha256'] == dataset.provenance['dataset_manifest_sha256'],
            'Candidate and spectrum-target dataset manifests differ')
    require(np.array_equal(cache.bind_dataset(dataset.base), dataset.raw_query_indices),
            'Candidate and spectrum-target query order differs')


def spectrum_target_batch(cache, raw):
    require(isinstance(raw, torch.Tensor) and raw.device.type == 'cpu' and raw.dtype == torch.long
            and raw.ndim == 1 and raw.numel() > 0 and bool(((raw >= 0) & (raw < len(cache))).all()),
            'Invalid spectrum-target raw query batch')
    rows = [cache[int(index)] for index in raw]
    require([row['raw_query_index'] for row in rows] == raw.tolist(), 'Spectrum-target row binding changed')
    return {
        'raw_query_indices': raw.clone(),
        'adduct_ids': torch.tensor([row['adduct_id'] for row in rows], dtype=torch.long),
        'row_ptr': torch.tensor([0, *np.cumsum([len(row['values']) for row in rows]).tolist()], dtype=torch.long),
        'bin_indices': torch.from_numpy(np.concatenate([row['bin_indices'] for row in rows])),
        'values': torch.from_numpy(np.concatenate([row['values'] for row in rows])),
    }


def update_target_batch_hash(digest, batch):
    for key in ('raw_query_indices', 'adduct_ids', 'row_ptr', 'bin_indices', 'values'):
        value = batch[key]
        require(value.device.type == 'cpu' and value.ndim == 1, 'Target audit hash requires CPU vectors')
        dtype = '<f4' if key == 'values' else '<i8'
        digest.update(len(value).to_bytes(8, 'little'))
        digest.update(value.numpy().astype(dtype, copy=False).tobytes())


def spectrum_target_files(receipt, prefix='spectrum_targets'):
    require(isinstance(receipt, dict) and {'directory', 'source', 'manifest_sha256', 'audit_sha256'} <= set(receipt),
            'Missing spectrum target input provenance')
    root = Path(receipt['directory'])
    require(root.is_absolute() and root.resolve() == root, 'Spectrum target directory must be explicit and resolved')
    # The manifest binds all payload bytes. Pin its complete inventory as well for the runner.
    import json
    require(sha256_file(root / 'manifest.json') == receipt['manifest_sha256'], 'Spectrum target manifest changed')
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['source'] == receipt['source'], 'Spectrum target source differs from receipt')
    files = {name: entry['sha256'] for name, entry in manifest['files'].items()}
    files.update({'manifest.json': receipt['manifest_sha256'], 'audit.json': receipt['audit_sha256']})
    return {f'{prefix}_{name}': {'path': str(root / name), 'sha256': digest} for name, digest in files.items()}


def verify_spectrum_target_files(receipt):
    files = spectrum_target_files(receipt)
    require(all(sha256_file(item['path']) == item['sha256'] for item in files.values()), 'Spectrum target files changed')
    return files


def validate_selection_spectrum_targets(selection, *, dataset_manifest_sha256, tokenizer_config,
                                         expected_counts, exclusions):
    settings = auxiliary_model_settings(selection['model_config'])
    receipt = selection.get('spectrum_targets')
    if settings is None:
        require(receipt is None and 'spectrum_auxiliary' not in selection.get('training_config', {}),
                'Unexpected spectrum targets in an inactive checkpoint')
        return None
    weight = auxiliary_training_weight(selection['model_config'], selection['training_config'])
    require(selection['config_snapshot']['train']['align']['spectrum_auxiliary'] == {'loss_weight': weight},
            'Checkpoint auxiliary weight differs from its configuration snapshot')
    spectrum_target_files(receipt)
    source = receipt['source']
    require(source['settings'] == settings['target'] and source['dataset_manifest_sha256'] == dataset_manifest_sha256
            and source['tokenizer_config'] == tokenizer_config and source['exclude_val_query_indices'] == exclusions
            and {'train': source['expected_counts']['train'],
                 'val': source['expected_counts']['val'] - len(exclusions)} == expected_counts,
            'Checkpoint spectrum targets changed the shared data/tokenizer protocol')
    observed = load_spectrum_target_cache(selection['data_path'], receipt['directory'], source['expected_counts'],
                                          exclusions, tokenizer_config, settings['target'])
    require(observed.provenance == receipt, 'Checkpoint spectrum targets differ from their complete source binding')
    return copy.deepcopy(receipt)
