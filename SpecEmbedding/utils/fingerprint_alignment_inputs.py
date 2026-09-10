"""Bind existing fixed molecular inputs to a formal alignment run."""

from pathlib import Path

from SpecEmbedding.utils.fingerprint_cache import fingerprint_provenance, load_fingerprint_cache
from SpecEmbedding.utils.fingerprint_preparation import fingerprint_source
from SpecEmbedding.utils.formal_alignment import fingerprint_input_bits
from SpecEmbedding.utils.fulltrain import sha256_file


def load_alignment_fingerprints(data_path, index_path, kind, cache_root, *, counts, exclusions,
                                tokenizer_config, settings, pool_cache_size):
    smiles, source = fingerprint_source(data_path, index_path, kind, expected_counts=counts,
                                         exclusions=exclusions, tokenizer_config=tokenizer_config,
                                         pool_cache_size=pool_cache_size)
    provenance = fingerprint_provenance(smiles, index_sha256=source['sha256'],
                                        dataset_manifest_sha256=source['dataset_manifest_sha256'],
                                        radius=settings['radius'], bits=settings['bits'])
    _, receipt = load_fingerprint_cache(smiles, cache_root, provenance)
    return smiles, {'source': source, 'cache': receipt}


def fingerprint_input_files(receipt, prefix):
    source, cache = receipt['source'], receipt['cache']
    index = Path(source['index_path'])
    root = Path(cache['directory'])
    files = {f'{prefix}_index': {'path': str(index), 'sha256': source['sha256']},
             f'{prefix}_manifest': {'path': str(root / 'manifest.json'), 'sha256': cache['manifest_sha256']},
             f'{prefix}_audit': {'path': str(root / 'audit.json'), 'sha256': cache['audit_sha256']},
             f'{prefix}_bits': {'path': str(root / cache['file']['filename']), 'sha256': cache['file']['sha256']}}
    if source['kind'] == 'train':
        files.update({f'{prefix}_preparation': {'path': str(index.parent / 'receipt.json'), 'sha256': source['receipt_sha256']},
                      f'{prefix}_verification': {'path': str(index.parent / 'verification.json'),
                                                'sha256': source['verification_sha256']}})
    else:
        files[f'{prefix}_index_receipt'] = {'path': str(index.with_suffix('.json')), 'sha256': source['receipt_sha256']}
    return files


def fingerprint_training_provenance(provenance, receipt, *, graph_fingerprint=False):
    if graph_fingerprint:
        return {**provenance, 'molecule_input': 'graph_with_fixed_morgan_bits', 'fingerprint_cache': receipt,
                'fingerprint_augmentation': 'None: fixed bits describe original positive and negative molecules'}
    return {**provenance, 'graph_cache_size': 0, 'molecule_input': 'fixed_morgan_bits', 'fingerprint_cache': receipt,
            'negative_graph_augmentation': 'None: fixed fingerprints for both positives and negatives'}


def audit_pinned_fingerprint_input(pinned, prefix, manifest, data_path, runtime, *, settings):
    source, cache = pinned['source'], pinned['cache']
    _, verified = load_alignment_fingerprints(
        data_path, source['index_path'], source['kind'], cache['directory'],
        counts=runtime['fulltrain']['expected_counts'], exclusions=runtime['fulltrain']['exclude_val_query_indices'],
        tokenizer_config=runtime['data']['tokenizer'], settings=settings,
        pool_cache_size=runtime['train']['align']['candidate_supervision']['pool_cache_size'])
    if verified != pinned:
        raise ValueError('Fingerprint inputs differ from the complete preflight verification')
    files = fingerprint_input_files(pinned, prefix)
    if any(manifest['inputs'].get(key) != value for key, value in files.items()):
        raise ValueError('Fingerprint source or cache file was not pinned in preflight')
    return {value['path']: value['sha256'] for value in files.values()}


def audit_model_fingerprint_inputs(manifest, selection, data_path, validation_index):
    runtime = manifest['runtime_config']
    pinned = manifest.get('fingerprint_inputs')
    recorded = {key: selection.get(key) for key in ('training_fingerprint_cache', 'validation_fingerprint_cache')}
    kind = runtime['model'].get('type', 'gine')
    if kind not in ('fingerprint', 'gine_fingerprint'):
        if pinned is not None or any(value is not None for value in recorded.values()):
            raise ValueError('Unexpected fingerprint inputs in a GINE model run')
        return None, {}
    if (not isinstance(pinned, dict) or set(pinned) != {'train', 'validation'}
            or (kind == 'fingerprint' and any(runtime['augmentation'][key] != 0
                                             for key in ('node_drop_rate', 'edge_mask_rate')))):
        raise ValueError('Missing fingerprint inputs or active graph augmentation')
    hashes = {}
    for kind, field in (('train', 'training_fingerprint_cache'), ('validation', 'validation_fingerprint_cache')):
        item = pinned[kind]
        if (item['source']['kind'] != kind or recorded[field] != item['cache']
                or item['cache']['provenance']['options']['bits'] != fingerprint_input_bits(runtime['model'])):
            raise ValueError('Selected fingerprint model/input provenance mismatch')
        hashes.update(audit_pinned_fingerprint_input(item, f'fingerprint_{kind}', manifest, data_path, runtime,
                                                    settings=runtime['molecule_fingerprints']))
    if pinned['validation']['cache']['provenance']['index_sha256'] != sha256_file(validation_index):
        raise ValueError('Fingerprint cache used another validation index')
    return {'state': 'verified_complete_fingerprint_model_inputs', **recorded}, hashes
