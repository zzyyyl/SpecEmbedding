"""Pure checks for preparing one successor after completed validation evidence."""

import copy
import math

from SpecEmbedding.utils.optimization_audit import METRICS, TOLERANCE, compare_metrics


def completed_dependencies(run_status, audit_status, pinned, observed):
    """A live original process is a wait, even when its status already says complete."""
    if run_status['state'] not in {'running', 'complete'}:
        raise ValueError('Predecessor failed or was interrupted')
    if audit_status['state'] not in {'waiting_parent', 'auditing', 'complete'}:
        raise ValueError('Predecessor completion audit failed or has an unknown state')
    if set(pinned) != set(observed) or not pinned:
        raise ValueError('Incomplete predecessor process observations')
    live = False
    for name, identity in pinned.items():
        item = observed[name]
        if item is None or item['starttime'] != identity['starttime']:
            continue
        if item['state'] in {'Z', 'X'}:
            if item['exit_code'] != 0:
                raise ValueError(f'Predecessor process {name} exited unsuccessfully')
        else:
            live = True
    complete = run_status['state'] == audit_status['state'] == 'complete'
    if not complete and not live:
        raise ValueError('Original processes ended with incomplete dependencies')
    if not complete or live:
        return False
    if not run_status['stages'] or any(s['state'] != 'complete' for s in run_status['stages']):
        raise ValueError('Predecessor has incomplete stages')
    return True


def _metrics(values):
    if any(isinstance(values[k], bool) or not isinstance(values[k], (int, float))
           or not math.isfinite(values[k]) or not 0 <= values[k] <= 1 for k in METRICS):
        raise ValueError('Invalid validation metrics')
    return {key: values[key] for key in METRICS}


def choose_completed_parent(current, incumbent):
    """Apply existing tolerances to all audited candidates; ambiguous promotion needs review.

    Callers must first bind complete run/audit states, hashes, and finished processes.
    This helper neither reads checkpoints nor turns a partial report into accepted evidence.
    """
    for report in (current, incumbent):
        if (report['state'] != 'complete_validation_audit' or report['test_evaluated_by_this_audit'] is not False
                or report['full_epoch_counts'] != {'train': 194119, 'val': 19423}
                or report['tolerance_raw'] != TOLERANCE
                or not report['candidate_training']['negative_sampling_replayed']):
            raise ValueError('Parent requires complete validation-only evidence with the fixed protocol')
    if current['protocol'] != incumbent['protocol']:
        raise ValueError('Parent protocols differ')
    baseline = _metrics(current['baseline_metrics'])
    retained = _metrics(incumbent['selected_metrics'])
    if any(abs(baseline[k] - retained[k]) > TOLERANCE[k] for k in METRICS):
        raise ValueError('Fresh predecessor baseline differs materially from the incumbent')
    selected = _metrics(current['selected_metrics'])
    selected_comparison = compare_metrics(selected, baseline)
    if selected_comparison != current['selected_vs_baseline']:
        raise ValueError('Selected comparison disagrees with recomputed metrics')
    rows, epochs = [], set()
    for candidate in current['pareto_candidates']:
        epoch = candidate['epoch']
        if type(epoch) is not int or epoch < 1 or epoch in epochs:
            raise ValueError('Invalid or duplicate Pareto epoch')
        epochs.add(epoch)
        metrics = _metrics(candidate['metrics'])
        comparison = compare_metrics(metrics, baseline)
        if comparison != candidate['vs_baseline']:
            raise ValueError('Pareto comparison disagrees with recomputed metrics')
        rows.append({'epoch': epoch, 'metrics': metrics, 'comparison': comparison})
    chosen = [row for row in rows if row['epoch'] == current['selected_epoch']]
    if len(chosen) != 1 or chosen[0]['metrics'] != selected:
        raise ValueError('Selected epoch is absent from the audited Pareto candidates')
    if any((row['metrics']['top1'], row['metrics']['mrr']) > (selected['top1'], selected['mrr']) for row in rows):
        raise ValueError('Selected epoch violates the fixed selection ordering')
    eligible = [row['epoch'] for row in rows if row['comparison']['outcome'] == 'eligible_for_incumbent_review']
    if current['selected_epoch'] in eligible:
        parent, reason = 'current', 'Selected checkpoint improves Top-k within every existing regression tolerance'
    elif eligible:
        raise ValueError('A non-selected Pareto checkpoint is eligible; resolve candidate provenance before dispatch')
    else:
        parent, reason = 'incumbent', 'No audited Pareto checkpoint passes the existing improvement rule'
    return {'parent': parent, 'reason': reason, 'reviewed_pareto_epochs': sorted(epochs),
            'eligible_pareto_epochs': eligible, 'selected_comparison': selected_comparison,
            'test_used_for_decision': False}


def delta_successor_configuration(parent_runtime, selection, delta_settings, gpu_settings, *, storage_template):
    return _spectral_successor_configuration(parent_runtime, selection, 'precursor_delta', delta_settings,
                                            gpu_settings, storage_template=storage_template)


def qk_successor_configuration(parent_runtime, selection, qk_settings, gpu_settings, *, storage_template):
    return _spectral_successor_configuration(parent_runtime, selection, 'qk_norm', qk_settings,
                                            gpu_settings, storage_template=storage_template)


def attention_pool_successor_configuration(parent_runtime, selection, pool_settings, gpu_settings, *, storage_template):
    from SpecEmbedding.utils.formal_alignment import formal_model_type

    if formal_model_type(parent_runtime['model']) not in ('gine', 'gine_fingerprint'):
        raise ValueError('Attention pooling successor requires a retained GINE or GINE+fingerprint parent')
    return _spectral_successor_configuration(parent_runtime, selection, 'attention_pool', pool_settings,
                                            gpu_settings, storage_template=storage_template)


def adduct_successor_configuration(parent_runtime, selection, adduct_settings, gpu_settings, *, storage_template):
    """Keep the audited parent's towers and candidate objective, adding one observed-input conditioner."""
    from SpecEmbedding.utils.candidate_training import validate_candidate_settings
    from SpecEmbedding.utils.formal_alignment import formal_model_type

    if formal_model_type(parent_runtime['model']) not in ('gine', 'gine_fingerprint'):
        raise ValueError('Adduct successor requires a retained GINE parent branch')
    candidates = parent_runtime['train']['align']['candidate_supervision']
    validate_candidate_settings(candidates)
    if not candidates['enabled']:
        raise ValueError('Adduct successor requires the registered candidate supervision')
    return _spectral_successor_configuration(parent_runtime, selection, 'adduct_conditioning', adduct_settings,
                                            gpu_settings, storage_template=storage_template)


def candidate_weight_successor_configuration(parent_runtime, selection, weight_settings, gpu_settings, *, storage_template):
    """Change only the candidate loss coefficient after checking the declared parent coefficient."""
    from SpecEmbedding.utils.candidate_training import validate_candidate_settings
    from SpecEmbedding.utils.formal_alignment import formal_model_type

    if formal_model_type(parent_runtime['model']) not in ('gine', 'gine_fingerprint'):
        raise ValueError('Candidate weight successor requires a retained GINE or GINE+fingerprint parent')
    if not isinstance(weight_settings, dict) or set(weight_settings) != {'expected_parent_weight', 'loss_weight'}:
        raise ValueError('Incomplete explicit candidate weight change')
    settings = parent_runtime['train']['align']['candidate_supervision']
    validate_candidate_settings(settings)
    for weight in weight_settings.values():
        validate_candidate_settings({**settings, 'loss_weight': weight})
    if not settings['enabled'] or settings['loss_weight'] != weight_settings['expected_parent_weight']:
        raise ValueError('Parent candidate supervision or loss weight differs from the declared baseline')
    if settings['loss_weight'] == weight_settings['loss_weight']:
        raise ValueError('Candidate weight change must differ from the parent')
    result = _inherited_successor_configuration(parent_runtime, selection, gpu_settings,
                                                storage_template=storage_template)
    result['train']['align']['candidate_supervision']['loss_weight'] = float(weight_settings['loss_weight'])
    return result


def _spectral_successor_configuration(parent_runtime, selection, feature, settings, gpu_settings, *, storage_template):
    """Inherit scientific settings, adding the feature and rebinding explicit external storage."""
    from SpecEmbedding.models_precursor_delta import validate_spectrum_config

    if feature in parent_runtime['model']['spec_encoder']:
        raise ValueError(f'Parent already has the proposed {feature} feature')
    result = _inherited_successor_configuration(parent_runtime, selection, gpu_settings,
                                                storage_template=storage_template)
    result['model']['spec_encoder'][feature] = copy.deepcopy(settings)
    validate_spectrum_config(result['model']['spec_encoder'])
    return result


def _inherited_successor_configuration(parent_runtime, selection, gpu_settings, *, storage_template):
    if (selection['model_config'] != parent_runtime['model']
            or selection['training_config'] != parent_runtime['train']['align']
            or selection['config_snapshot']['augmentation'] != parent_runtime['augmentation']):
        raise ValueError('Parent model/training/augmentation provenance differs')
    if set(gpu_settings) != {'min_free_mib', 'max_utilization', 'poll_seconds', 'hold_seconds'}:
        raise ValueError('Incomplete explicitly authorized GPU policy')
    result = copy.deepcopy(parent_runtime)
    result['fulltrain'].update(gpu_settings)
    from SpecEmbedding.config import PATH_FIELDS
    from SpecEmbedding.utils.storage import external_storage_root, storage_path

    root = external_storage_root(storage_template['storage']['root'])
    result['storage'] = copy.deepcopy(storage_template['storage'])
    for keys in PATH_FIELDS:
        target, template = result, storage_template
        for key in keys[:-1]:
            target = target.get(key, {})
            template = template.get(key, {})
        if keys[-1] in target:
            value = template[keys[-1]]
            storage_path(value, root)
            target[keys[-1]] = value
    return result
