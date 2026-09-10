"""Bind observed spectrum metadata to formal construction, input manifests and audits."""

import copy
from pathlib import Path

from SpecEmbedding.models_adduct import validate_adduct_settings
from SpecEmbedding.utils.adduct_metadata import load_adduct_cache, require
from SpecEmbedding.utils.fulltrain import sha256_file


def adduct_model_settings(model_config):
    settings = model_config.get('spec_encoder', {}).get('adduct_conditioning')
    if 'adduct_conditioning' in model_config.get('spec_encoder', {}):
        validate_adduct_settings(settings)
        require(model_config.get('type', 'gine') in ('gine', 'gine_fingerprint'),
                'Adduct formal alignment supports the registered GINE parent branches')
        return copy.deepcopy(settings)
    return None


def load_spectrum_metadata(data_path, root, *, counts, exclusions, tokenizer_config, settings, index=None):
    """Read both full splits, preserving raw query IDs and independently binding validation tokens."""
    metadata = {split: load_adduct_cache(data_path, root, split, counts, exclusions, tokenizer_config, settings)
                for split in ('train', 'val')}
    if index is not None:
        metadata['val'].bind_index(index)
    receipts = {split: copy.deepcopy(item.provenance) for split, item in metadata.items()}
    require(receipts['train']['source'] == receipts['val']['source'], 'Adduct train/validation sources differ')
    return metadata, receipts


def spectrum_metadata_files(receipts, prefix='spectrum_metadata'):
    require(isinstance(receipts, dict) and set(receipts) == {'train', 'val'}, 'Missing full adduct split receipts')
    train, val = receipts['train'], receipts['val']
    require(train['split'] == 'train' and val['split'] == 'val'
            and all(train[key] == val[key] for key in ('directory', 'source', 'manifest_sha256', 'audit_sha256')),
            'Adduct split receipts describe different caches')
    root = Path(train['directory'])
    require(root.is_absolute() and root.resolve() == root, 'Adduct cache must use its explicit resolved directory')
    files = {'manifest.json': train['manifest_sha256'], 'audit.json': train['audit_sha256'],
             'train.json': train['payload_sha256'], 'val.json': val['payload_sha256']}
    return {f'{prefix}_{name}': {'path': str(root / name), 'sha256': digest} for name, digest in files.items()}


def validate_selection_spectrum_metadata(selection, *, dataset_manifest_sha256, tokenizer_config,
                                         expected_counts, exclusions):
    """A conditioned checkpoint must carry its own complete, still valid metadata provenance."""
    settings = adduct_model_settings(selection['model_config'])
    receipts = selection.get('spectrum_metadata')
    if settings is None:
        require(receipts is None, 'Unexpected adduct metadata in an unconditioned checkpoint')
        return None
    spectrum_metadata_files(receipts)
    source = receipts['train']['source']
    require(source['settings'] == settings and source['dataset_manifest_sha256'] == dataset_manifest_sha256
            and source['tokenizer_config'] == tokenizer_config and source['exclude_val_query_indices'] == exclusions
            and {'train': source['expected_counts']['train'],
                 'val': source['expected_counts']['val'] - len(exclusions)} == expected_counts,
            'Checkpoint adduct metadata changed the shared query/tokenizer protocol')
    _, observed = load_spectrum_metadata(selection['data_path'], receipts['train']['directory'],
                                         counts=source['expected_counts'], exclusions=exclusions,
                                         tokenizer_config=tokenizer_config, settings=settings)
    require(observed == receipts, 'Checkpoint adduct inputs differ from its own complete source binding')
    return copy.deepcopy(receipts)


def audit_model_spectrum_metadata(manifest, selection, data_path, index):
    """Validate actual model input provenance against the immutable preflight and full raw inputs."""
    runtime = manifest['runtime_config']
    settings = adduct_model_settings(runtime['model'])
    require(selection['model_config'] == runtime['model'], 'Adduct model configuration differs from preflight')
    if settings is None:
        require(manifest.get('spectrum_metadata') is None and selection.get('spectrum_metadata') is None,
                'Unexpected adduct inputs in an inactive run')
        return None, None, {}
    receipts = manifest.get('spectrum_metadata')
    files = spectrum_metadata_files(receipts)
    require(selection.get('spectrum_metadata') == receipts
            and all(manifest['inputs'].get(name) == item for name, item in files.items()),
            'Adduct training inputs differ from preflight/selection')
    fulltrain = runtime['fulltrain']
    metadata, observed = load_spectrum_metadata(data_path, receipts['train']['directory'],
        counts=fulltrain['expected_counts'], exclusions=fulltrain['exclude_val_query_indices'],
        tokenizer_config=runtime['data']['tokenizer'], settings=settings, index=index)
    require(observed == receipts, 'Adduct source changed after preflight')
    hashes = {item['path']: item['sha256'] for item in files.values()}
    require(all(sha256_file(path) == digest for path, digest in hashes.items()), 'Adduct source changed during audit')
    return metadata, receipts, hashes
