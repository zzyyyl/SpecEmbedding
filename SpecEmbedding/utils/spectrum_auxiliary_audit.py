"""CPU verification of target provenance and each completed auxiliary training epoch."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from SpecEmbedding.models_spectrum_aux import auxiliary_model_settings
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.spectrum_auxiliary_inputs import auxiliary_training_weight, verify_spectrum_target_files
from SpecEmbedding.utils.spectrum_targets import load_spectrum_target_cache, require


def audit_model_spectrum_targets(manifest, selection, data_path):
    runtime = manifest['runtime_config']
    settings = auxiliary_model_settings(runtime['model'])
    weight = auxiliary_training_weight(runtime['model'], runtime['train']['align'])
    receipt = manifest.get('spectrum_targets')
    require(selection['model_config'] == runtime['model'], 'Spectrum auxiliary construction differs from preflight')
    if settings is None:
        require(receipt is None and selection.get('spectrum_targets') is None
                and 'spectrum_auxiliary' not in selection['stages']['stage2'], 'Unexpected inactive auxiliary artifacts')
        return None, None, {}
    require(selection.get('spectrum_targets') == receipt
            and auxiliary_training_weight(selection['model_config'], selection['training_config']) == weight,
            'Spectrum auxiliary training differs from preflight')
    files = verify_spectrum_target_files(receipt)
    require(all(manifest['inputs'].get(name) == item for name, item in files.items()),
            'Spectrum auxiliary inputs were not pinned in preflight')
    fulltrain = runtime['fulltrain']
    cache = load_spectrum_target_cache(data_path, receipt['directory'], fulltrain['expected_counts'],
        fulltrain['exclude_val_query_indices'], runtime['data']['tokenizer'], settings['target'])
    require(cache.provenance == receipt, 'Spectrum auxiliary inputs changed after preflight')
    return cache, weight, {item['path']: item['sha256'] for item in files.values()}


def replay_target_digest(cache, order, batch_sizes):
    """Reconstruct consumed tensor bytes directly from mmap rows, independently of batch construction."""
    require(sum(batch_sizes) == len(order) and all(type(n) is int and n > 0 for n in batch_sizes),
            'Auxiliary replay batch sizes do not cover the full query order')
    arrays, digest, start = cache._open(), hashlib.sha256(), 0
    for count in batch_sizes:
        raw = order[start:start + count]
        slices = [slice(int(arrays['ptr'][i]), int(arrays['ptr'][i + 1])) for i in raw]
        lengths = [part.stop - part.start for part in slices]
        values = [raw, cache.metadata.adduct_ids[raw], np.r_[0, np.cumsum(lengths)],
                  np.concatenate([arrays['bins'][part] for part in slices]),
                  np.concatenate([arrays['values'][part] for part in slices])]
        for position, value in enumerate(values):
            digest.update(len(value).to_bytes(8, 'little'))
            digest.update(value.astype('<f4' if position == 4 else '<i8', copy=False).tobytes())
        start += count
    return digest.hexdigest()


def audit_spectrum_auxiliary_training(directory, stage, cache, weight):
    directory = Path(directory)
    if cache is None:
        require('spectrum_auxiliary' not in stage and not (directory / 'spectrum_auxiliary').exists(),
                'Unexpected inactive spectrum auxiliary training records')
        return None, {}
    report = stage.get('spectrum_auxiliary')
    require(isinstance(report, dict) and report['targets'] == cache.provenance and report['loss_weight'] == weight
            and report['loss'] == 'mean_query_cosine_distance_raw_observed_spectrum',
            'Missing or changed spectrum auxiliary stage configuration')
    candidates = stage['candidate_training']
    require(len(report['epochs']) == len(candidates['epochs']) == stage['stop_epoch'], 'Missing auxiliary training epochs')
    hashes = {}
    for number, (record, candidate) in enumerate(zip(report['epochs'], candidates['epochs'], strict=True), 1):
        path = directory / 'spectrum_auxiliary' / f'stage2_epoch{number:03d}.json'
        hashes[str(path)] = sha256_file(path)
        require(json.loads(path.read_text()) == record, 'Auxiliary epoch summary differs from its original record')
        require(record['stage'] == 'stage2' and record['epoch'] == number and record['loss_weight'] == weight
                and record['queries'] == record['unique_queries'] == len(cache), 'Incomplete auxiliary query coverage')
        order_path = directory / 'candidate_training' / candidate['query_order_file']
        hashes[str(order_path)] = sha256_file(order_path)
        require(hashes[str(order_path)] == candidate['query_order_sha256'], 'Auxiliary candidate query order changed')
        order = np.load(order_path, allow_pickle=False)
        require(order.dtype == np.int64 and np.array_equal(np.sort(order), np.arange(len(cache))),
                'Auxiliary query order is not a complete permutation')
        require(record['observed_target_batch_sha256'] == replay_target_digest(cache, order, candidate['batch_sizes']),
                'Auxiliary target bytes differ from independently replayed source queries')
        means = record['loss_query_means']
        require(set(means) == {'contrastive', 'candidate', 'auxiliary', 'total'}
                and all(math.isfinite(value) and value >= -1e-6 for value in means.values())
                and means['auxiliary'] <= 1 + 1e-6
                and math.isclose(means['candidate'], candidate['candidate_loss_query_mean'], rel_tol=1e-6, abs_tol=1e-7)
                and math.isclose(means['total'], means['contrastive'] + candidates['candidate_loss_weight'] * means['candidate']
                                 + weight * means['auxiliary'], rel_tol=1e-6, abs_tol=1e-7),
                'Auxiliary loss composition differs from the declared training objective')
        probe = record['gradient_probe']
        require(isinstance(probe, dict) and probe['queries'] == candidate['batch_sizes'][0]
                and all(math.isfinite(probe[key]) and probe[key] >= 0 for key in ('main_norm', 'weighted_auxiliary_norm'))
                and (probe['cosine'] is None or (math.isfinite(probe['cosine']) and abs(probe['cosine']) <= 1 + 1e-6)),
                'Missing or invalid shared embedding gradient measurement')
        require(record['optimizer_steps'] == len(candidate['batch_sizes']) and record['gradient_clip_max_norm'] == 1.
                and type(record['clipped_steps']) is int and 0 <= record['clipped_steps'] <= record['optimizer_steps']
                and all(math.isfinite(record[key]) and record[key] >= 0
                        for key in ('preclip_gradient_norm_mean', 'preclip_gradient_norm_max')),
                'Auxiliary optimizer/gradient clipping measurements are incomplete')
    require(all(sha256_file(path) == digest for path, digest in hashes.items()), 'Auxiliary records changed during audit')
    verify_spectrum_target_files(cache.provenance)
    return {'state': 'verified_full_auxiliary_target_replay', 'queries': len(cache), 'epochs': stage['stop_epoch'],
            'loss_weight': weight, 'targets': cache.provenance}, hashes
