"""Scientific parameter inheritance and rejection of incompatible candidate-budget trials."""
import copy

import pytest

from SpecEmbedding.config import config
from SpecEmbedding.utils.alignment_successor import candidate_budget_successor_configuration
from tests.test_candidate_weight_successor import parent_inputs


@pytest.mark.parametrize('kind', ['gine', 'gine_fingerprint'])
def test_budget_successor_preserves_query_batch_objective_and_independent_parent(tmp_path, kind):
    runtime, selection, _, gpu, storage = parent_inputs(tmp_path, kind)
    settings = config.training_candidate_budget.to_dict()
    originals = copy.deepcopy((runtime, selection, settings, gpu, storage))
    result = candidate_budget_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)
    train = copy.deepcopy(result['train'])
    assert train['align']['candidate_supervision'].pop('negative_count') == 32
    expected_train = copy.deepcopy(runtime['train'])
    assert expected_train['align']['candidate_supervision'].pop('negative_count') == 16
    assert train == expected_train and result['model'] == runtime['model']
    assert result['augmentation'] == runtime['augmentation'] and result['data']['tokenizer'] == runtime['data']['tokenizer']
    assert result['data']['cache_path'] == storage['data']['cache_path']
    result['train']['align']['candidate_supervision']['loss_weight'] = 999
    assert (runtime, selection, settings, gpu, storage) == originals


@pytest.mark.parametrize('damage', ['changed_parent', 'structural', 'disabled', 'same', 'smaller', 'float', 'bool',
                                   'too_many', 'missing_expected', 'unexpected_weight', 'selection', 'storage'])
def test_budget_successor_refuses_protocol_drift(tmp_path, damage):
    runtime, selection, _, gpu, storage = parent_inputs(tmp_path, structural=damage == 'structural')
    settings = config.training_candidate_budget.to_dict()
    if damage == 'changed_parent':
        runtime['train']['align']['candidate_supervision']['negative_count'] = 64
    elif damage == 'disabled':
        runtime['train']['align']['candidate_supervision']['enabled'] = False
    elif damage in ('same', 'smaller', 'float', 'bool', 'too_many'):
        settings['negative_count'] = {'same': 16, 'smaller': 8, 'float': 32., 'bool': True, 'too_many': 256}[damage]
    elif damage == 'missing_expected':
        del settings['expected_parent_negative_count']
    elif damage == 'unexpected_weight':
        settings['loss_weight'] = 2.
    elif damage == 'selection':
        selection['training_config']['batch_size'] = 256
    elif damage == 'storage':
        storage['data']['cache_path'] = '/old/source/train_cache'
    with pytest.raises(ValueError):
        candidate_budget_successor_configuration(runtime, selection, settings, gpu, storage_template=storage)
