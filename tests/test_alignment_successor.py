import copy

import pytest

from SpecEmbedding.utils.alignment_successor import (
    choose_completed_parent,
    completed_dependencies,
    delta_successor_configuration,
)
from SpecEmbedding.utils.optimization_audit import METRICS, TOLERANCE, compare_metrics
from tests.test_formal_alignment import model_config


def reports(improved=True):
    base = dict(zip(METRICS, [.2, .4, .55, .7, .3], strict=True))
    selected = {k: v + (.03 if improved else -.03) for k, v in base.items()}
    common = {'state': 'complete_validation_audit', 'test_evaluated_by_this_audit': False,
              'full_epoch_counts': {'train': 194119, 'val': 19423}, 'tolerance_raw': TOLERANCE,
              'candidate_training': {'negative_sampling_replayed': True}, 'protocol': 'fixed_protocol'}
    current = {**copy.deepcopy(common), 'baseline_metrics': base, 'selected_metrics': selected,
               'selected_epoch': 7, 'selected_vs_baseline': compare_metrics(selected, base),
               'pareto_candidates': [{'epoch': 7, 'metrics': selected, 'vs_baseline': compare_metrics(selected, base)}]}
    return current, {**copy.deepcopy(common), 'selected_metrics': copy.deepcopy(base)}


@pytest.mark.parametrize('improved,parent', [(True, 'current'), (False, 'incumbent')])
def test_parent_choice_uses_audited_metrics(improved, parent):
    current, incumbent = reports(improved)
    decision = choose_completed_parent(current, incumbent)
    assert decision['parent'] == parent and decision['reviewed_pareto_epochs'] == [7]
    assert not decision['test_used_for_decision']


@pytest.mark.parametrize('mutation', ['partial', 'test', 'counts', 'tolerance', 'replay', 'protocol', 'nan', 'forged_comparison', 'missing_selected', 'duplicate'])
def test_invalid_evidence_never_promotes(mutation):
    current, incumbent = reports()
    if mutation == 'partial':
        current['state'] = 'running'
    elif mutation == 'test':
        current['test_evaluated_by_this_audit'] = True
    elif mutation == 'counts':
        current['full_epoch_counts']['train'] = 20000
    elif mutation == 'tolerance':
        current['tolerance_raw'] = {k: .1 for k in METRICS}
    elif mutation == 'replay':
        current['candidate_training']['negative_sampling_replayed'] = False
    elif mutation == 'protocol':
        current['protocol'] = 'other'
    elif mutation == 'nan':
        current['selected_metrics']['top1'] = float('nan')
    elif mutation == 'forged_comparison':
        current['selected_vs_baseline']['outcome'] = 'fake'
    elif mutation == 'missing_selected':
        current['pareto_candidates'] = []
    elif mutation == 'duplicate':
        current['pareto_candidates'] *= 2
    with pytest.raises(ValueError):
        choose_completed_parent(current, incumbent)


def test_eligible_nonselected_candidate_requires_review():
    current, incumbent = reports()
    base = current['baseline_metrics']
    selected = {**base, 'top1': .3, 'top20': .6}
    other = {k: v + .02 for k, v in base.items()}
    current.update(selected_metrics=selected, selected_vs_baseline=compare_metrics(selected, base))
    current['pareto_candidates'] = [{'epoch': e, 'metrics': m, 'vs_baseline': compare_metrics(m, base)} for e, m in [(6, other), (7, selected)]]
    with pytest.raises(ValueError, match='non-selected'):
        choose_completed_parent(current, incumbent)


def test_materially_different_baseline_is_not_silently_accepted():
    current, incumbent = reports()
    incumbent['selected_metrics']['top1'] += .01
    with pytest.raises(ValueError, match='differs materially'):
        choose_completed_parent(current, incumbent)


def test_original_process_liveness_dominates_complete_status():
    run = {'state': 'complete', 'stages': [{'state': 'complete'}]}
    audit = {'state': 'complete'}
    pinned = {'trainer': {'starttime': 10}}
    assert not completed_dependencies(run, audit, pinned, {'trainer': {'starttime': 10, 'state': 'S'}})
    assert completed_dependencies(run, audit, pinned, {'trainer': None})
    assert completed_dependencies(run, audit, pinned, {'trainer': {'starttime': 11, 'state': 'S'}})
    assert completed_dependencies(run, audit, pinned, {'trainer': {'starttime': 10, 'state': 'Z', 'exit_code': 0}})
    with pytest.raises(ValueError, match='unsuccessfully'):
        completed_dependencies(run, audit, pinned, {'trainer': {'starttime': 10, 'state': 'Z', 'exit_code': 256}})


def test_incomplete_or_failed_dependencies_never_dispatch():
    pinned = {'trainer': {'starttime': 10}}
    live = {'trainer': {'starttime': 10, 'state': 'S'}}
    run = {'state': 'running', 'stages': [{'state': 'running'}]}
    assert not completed_dependencies(run, {'state': 'waiting_parent'}, pinned, live)
    with pytest.raises(ValueError, match='incomplete'):
        completed_dependencies(run, {'state': 'complete'}, pinned, {'trainer': None})
    for state in ['failed', 'failed_or_interrupted']:
        with pytest.raises(ValueError):
            completed_dependencies({**run, 'state': state}, {'state': 'complete'}, pinned, live)
        with pytest.raises(ValueError):
            completed_dependencies(run, {'state': state}, pinned, live)


@pytest.mark.parametrize('kind', ['gine', 'fingerprint'])
def test_successor_inherits_model_training_and_inputs_except_declared_change(kind):
    runtime = {'model': model_config(kind), 'train': {'align': {'batch_size': 128, 'lr': .0001}},
               'augmentation': {'prob': .5, 'node_drop_rate': 0.}, 'fulltrain': {'min_free_mib': 20000},
               'data': {'tokenizer': {'max_len': 100}, 'cache_path': '/old/source/data/train_cache'},
               'unchanged_extra': [1, 2]}
    original = copy.deepcopy(runtime)
    selection = {'model_config': runtime['model'], 'training_config': runtime['train']['align'],
                 'config_snapshot': {'augmentation': runtime['augmentation']}}
    delta = {'fourier_dim': 64, 'hidden_dim': 128, 'min_wavelength': .01, 'max_wavelength': 10000.}
    gpu = {'min_free_mib': 10000, 'max_utilization': 100, 'poll_seconds': 30, 'hold_seconds': 120}
    storage = {'storage': {'root': '/tmp/approved_external'},
               'data': {'cache_path': '/tmp/approved_external/train_cache'}}
    result = delta_successor_configuration(runtime, selection, delta, gpu, storage_template=storage)
    assert runtime == original
    expected = copy.deepcopy(runtime)
    expected['model']['spec_encoder']['precursor_delta'] = delta
    expected['fulltrain'].update(gpu)
    expected['storage'] = storage['storage']
    expected['data']['cache_path'] = storage['data']['cache_path']
    assert result == expected
    bad_storage = copy.deepcopy(storage)
    bad_storage['data']['cache_path'] = '/old/source/data/train_cache'
    with pytest.raises(ValueError):
        delta_successor_configuration(runtime, selection, delta, gpu, storage_template=bad_storage)
    selection['training_config'] = {'batch_size': 256}
    with pytest.raises(ValueError, match='provenance'):
        delta_successor_configuration(runtime, selection, delta, gpu, storage_template=storage)
