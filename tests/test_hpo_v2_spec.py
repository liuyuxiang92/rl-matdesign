from pathlib import Path

import pytest
import yaml

from rl_matdesign.config_schema import ConfigError
from rl_matdesign.hpo.spec import load_plan


def write_plan(tmp_path, extra=None, scene=None):
    base = scene if scene is not None else {
        'properties': [{'name': 'working_temperature', 'output': 'T3', 'direction': 'min', 'unit': 'K'}],
        'gen_temperature': 1.0,
    }
    (tmp_path / 'scene.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    data = {'base_config': 'scene.yaml', 'method': 'a2c', 'evaluation': {'threshold': 0}}
    data.update(extra or {})
    path = tmp_path / 'hpo.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.mark.parametrize('method', ['a2c', 'dqn-mc', 'dqn-bootstrap'])
def test_simplified_defaults_inherit_named_property_and_fixed_rules(tmp_path, method):
    plan = load_plan(write_plan(tmp_path, {'method': method}))
    assert plan['protocol'] == 'hpo_v2'
    assert plan['parallel_jobs'] == 1
    assert plan['sources']['parallel_jobs'] == 'default'
    assert plan['objective'] == {
        'field': 'working_temperature', 'direction': 'minimize', 'threshold': 0,
        'threshold_inclusive': True, 'unit': 'K',
        'normalization': {'kind': 'sigmoid', 'scale': None, 'calibration': 'reference'},
    }
    assert plan['scoring_scope'] == 'score_only'
    assert plan['budget']['tolerances'] == {'score': .01}
    assert plan['range_search']['score_improvement'] == .01
    assert plan['range_search']['max_extra_values_per_direction'] == 2
    assert plan['generation_search']['max_extra_values_per_direction'] == 2
    assert plan['training_search']['finalists'] == 3
    assert plan['budget']['train']['factor'] == plan['budget']['generation']['factor'] == 2
    assert plan['budget']['train']['min_unique'] == 1
    assert plan['efficiency']['comparison'] == 'lt'
    assert plan['efficiency']['max_attempts_per_unique'] == 1.5
    assert plan['generation_search']['parameter'] == 'gen_temperature'
    assert plan['generation_search']['repeats_per_model'] == 3


@pytest.mark.parametrize('parallel_jobs', [1, 3])
def test_parallel_jobs_changes_execution_capacity_only(tmp_path, parallel_jobs):
    serial = load_plan(write_plan(tmp_path))
    plan = load_plan(write_plan(tmp_path, {'parallel_jobs': parallel_jobs}))
    assert plan['parallel_jobs'] == parallel_jobs
    assert plan['sources']['parallel_jobs'] == 'yaml'
    assert plan['authored_config']['parallel_jobs'] == parallel_jobs
    # Protocol, budgets, scoring, parameter ranges and every seed stay identical.
    execution_fields = {'parallel_jobs', 'sources', 'authored_config'}
    assert {key: value for key, value in plan.items() if key not in execution_fields} == {
        key: value for key, value in serial.items() if key not in execution_fields}


@pytest.mark.parametrize('parallel_jobs', [0, -1, True, False, 1.0, '3', None])
def test_parallel_jobs_requires_positive_integer(tmp_path, parallel_jobs):
    with pytest.raises(ConfigError, match=r'parallel_jobs must be an integer >= 1'):
        load_plan(write_plan(tmp_path, {'parallel_jobs': parallel_jobs}))


@pytest.mark.parametrize('direction,result', [('min', 'minimize'), ('minimize', 'minimize'),
                                              ('max', 'maximize'), ('maximize', 'maximize')])
def test_property_direction_aliases_and_negative_threshold(tmp_path, direction, result):
    plan = load_plan(write_plan(tmp_path, {'evaluation': {'threshold': -3.2}},
                                {'properties': [{'name': 'energy', 'direction': direction}]}))
    assert plan['objective']['direction'] == result
    assert plan['objective']['threshold'] == -3.2


def test_reward_objective_selects_named_property_and_ignores_gate_properties(tmp_path):
    scene = {
        'properties': [{'name': 'working_temperature', 'direction': 'min', 'output': 'T3', 'weight': 0},
                       {'name': 'transition_temperature', 'direction': 'max', 'output': 'Tg'},
                       {'name': 'expansion', 'direction': 'min', 'output': 'CTE'}],
        'reward': {'type': 'gated', 'objective': {'terms': {'working_temperature': 2, 'expansion': 0}},
                   'gates': [{'name': 'tg', 'score': {'terms': {'transition_temperature': 1}}, 'min': 500},
                             {'name': 'cte', 'score': {'terms': {'expansion': 1}}, 'max': 1e-5}],
                   'failure': {'reward': -1000}},
    }
    plan = load_plan(write_plan(tmp_path, scene=scene))
    assert plan['objective']['field'] == 'working_temperature'
    assert plan['base_config_data']['reward']['gates'] == scene['reward']['gates']


def test_legacy_property_weights_can_select_single_target(tmp_path):
    plan = load_plan(write_plan(tmp_path, scene={'properties': [
        {'name': 'energy', 'direction': 'min', 'weight': 1},
        {'name': 'density', 'direction': 'max', 'weight': 0},
    ]}))
    assert plan['objective']['field'] == 'energy'


@pytest.mark.parametrize('scene,match', [
    ({'predictor': 'dummy'}, 'named properties'),
    ({'properties': [{'name': 'a', 'direction': 'min'}, {'name': 'b', 'direction': 'max'}]}, 'exactly one'),
    ({'properties': [{'name': 'a', 'direction': 'min', 'weight': 0}]}, 'exactly one'),
    ({'properties': [{'name': 'a', 'direction': 'target'}]}, 'direction target'),
    ({'properties': [{'name': 'a'}]}, 'direction min or max'),
    ({'properties': [{'name': 'a', 'direction': 'min'}, {'name': 'a', 'direction': 'max'}]}, 'distinct'),
    ({'properties': [{'name': 'a', 'direction': 'min', 'scale': -1}]}, 'contradicts'),
    ({'properties': [{'name': 'a', 'direction': 'min', 'weight': -1}]}, 'contradicts'),
    ({'properties': [{'name': 'a', 'direction': 'min', 'scale': 0}]}, 'contradicts'),
    ({'properties': [{'name': 'a', 'direction': 'min'}],
      'reward': {'objective': {'terms': {'T3': 1}}}}, 'unknown property'),
])
def test_ambiguous_or_contradictory_scenes_rejected(tmp_path, scene, match):
    with pytest.raises(ConfigError, match=match):
        load_plan(write_plan(tmp_path, scene=scene))


def test_one_training_repeat_count_controls_all_stages_with_fresh_validation(tmp_path):
    path = write_plan(tmp_path, {'random_seed': 123, 'training': {'repeats': 4},
                                'generation': {'repeats': 2}})
    plan = load_plan(path)
    assert plan['range_search']['seeds'] == plan['training_search']['seeds']
    assert len(plan['range_search']['seeds']) == len(plan['training_search']['validation_seeds']) == 4
    assert not set(plan['range_search']['seeds']) & set(plan['training_search']['validation_seeds'])
    assert 'confirmation_seed' not in plan['range_search']
    assert set(plan['seed_schedule']) == {'exploration', 'validation'}
    assert plan['range_search']['random_seed'] == 123
    assert plan['seed_schedule'] == load_plan(path)['seed_schedule']
    other = load_plan(write_plan(tmp_path, {'random_seed': 124, 'training': {'repeats': 4}}))
    assert other['seed_schedule'] != plan['seed_schedule']
    assert plan['generation_search']['repeats_per_model'] == 2
    assert 'paired_seeds' not in plan['generation_search']


@pytest.mark.parametrize('method', ['a2c', 'dqn-mc', 'dqn-bootstrap'])
def test_rounds_reuse_one_search_trial_policy_and_tolerance(tmp_path, method):
    plan = load_plan(write_plan(tmp_path, {
        'method': method,
        'evaluation': {'threshold': 2, 'score_tolerance': .03},
        'training': {'search': {'max_rounds': 2, 'trials_per_round': 9, 'startup_random_trials': 3}},
    }))
    cfg = plan['training_search']
    assert cfg['max_budget_revisions'] == 1
    assert cfg['trials'] == cfg['additional_trials_per_revision'] == 9
    assert cfg['startup_random_trials'] == cfg['additional_random_trials'] == 3
    assert cfg['finalists'] == 3
    assert plan['range_search']['score_improvement'] == plan['budget']['tolerances']['score'] == .03


def test_independent_budget_modes_and_no_hidden_fraction_floor(tmp_path):
    plan = load_plan(write_plan(tmp_path, {
        'training': {'budget': {'mode': 'fraction_of_space', 'min_fraction': 1e-8, 'initial_fraction': 1e-7, 'max_fraction': .01}},
        'generation': {'budget': {'mode': 'absolute_unique', 'min_unique': 3, 'initial_unique': 7, 'max_unique': 19}},
    }))
    assert plan['budget']['mode'] == 'per_phase'
    assert plan['budget']['train']['mode'] == 'fraction_of_space'
    assert plan['budget']['train']['min_unique'] == 1
    assert plan['budget']['generation']['mode'] == 'absolute_unique'
    assert plan['budget']['generation']['min_unique'] == 3
    assert plan['budget']['train']['factor'] == plan['budget']['generation']['factor'] == 2


@pytest.mark.parametrize('phase', ['training', 'generation'])
@pytest.mark.parametrize('budget', [
    {'mode': 'fraction_of_space'},
    {'mode': 'absolute_unique', 'min_unique': 8, 'initial_unique': 16, 'max_unique': 32},
])
@pytest.mark.parametrize('factor', [2, 3])
def test_budget_factor_is_internal_even_when_explicit_value_matches_default(tmp_path, phase, budget, factor):
    with pytest.raises(ConfigError, match='Unknown fields.*factor'):
        load_plan(write_plan(tmp_path, {phase: {'budget': {**budget, 'factor': factor}}}))


@pytest.mark.parametrize('extra_values', [0, 2, 5])
def test_training_range_probe_limit_is_internal(tmp_path, extra_values):
    with pytest.raises(ConfigError, match='Unknown fields.*range_search'):
        load_plan(write_plan(tmp_path, {'training': {
            'range_search': {'max_extra_values_per_direction': extra_values}}}))


def test_default_seed_42_and_three_repeats_have_reproducible_distinct_cohorts(tmp_path):
    plan = load_plan(write_plan(tmp_path))
    assert plan['seed_schedule'] == {
        'exploration': [2746317213, 1181241943, 958682846],
        'validation': [3163119785, 1812140441, 127978094],
    }


@pytest.mark.parametrize('extra,match', [
    ({'objective': {'field': 'working_temperature'}}, 'Unknown fields'),
    ({'evaluation': {'threshold': None}}, 'finite number'),
    ({'evaluation': {'threshold': 1, 'normalization': {'kind': 'sigmoid'}}}, 'Unknown fields'),
    ({'evaluation': {'threshold': 1, 'threshold_inclusive': True}}, 'Unknown fields'),
    ({'evaluation': {'threshold': 1, 'max_attempts_per_unique': 1}}, '> 1'),
    ({'training': {'repeats': 0}}, 'integer >= 1'),
    ({'training': {'seeds': [1, 2]}}, 'Unknown fields'),
    ({'training': {'validation': {'top_configs': 3}}}, 'Unknown fields.*validation'),
    ({'training': {'validation': {'top_configs': 1}}}, 'Unknown fields.*validation'),
    ({'training': {'range_search': {'score_improvement': .1}}}, 'Unknown fields'),
    ({'training': {'search': {'trials_per_round': 2}}}, 'must not exceed'),
    ({'training': {'budget': {'min_unique': 100}}}, 'Unknown fields'),
    ({'training': {'budget': {'strategy': 'bidirectional_grid'}}}, 'Unknown fields'),
    ({'training': {'budget': {'mode': 'unknown'}}}, 'must be fraction'),
    ({'training': {'budget': {'min_fraction': .5}}}, 'min <= initial'),
    ({'generation': {'paired_seeds': [1]}}, 'Unknown fields'),
    ({'generation': {'fixed': {'gen_epsilon': 0}}}, 'Unknown fields'),
    ({'generation': {'max_extra_values_per_direction': 0}}, 'Unknown fields.*max_extra_values_per_direction'),
    ({'generation': {'max_extra_values_per_direction': 2}}, 'Unknown fields.*max_extra_values_per_direction'),
    ({'generation': {'repeats': 0}}, r'generation\.repeats must be an integer >= 1'),
    ({'generation': {'repeats_per_model': 2}}, 'Unknown fields.*repeats_per_model'),
    ({'random_seed': 2**32}, '< 2'),
])
def test_invalid_or_redundant_public_settings_rejected(tmp_path, extra, match):
    with pytest.raises(ConfigError, match=match):
        load_plan(write_plan(tmp_path, extra))


@pytest.mark.parametrize('parameter,values', [
    ('gen_temperature', [0, .1, 100]), ('gen_epsilon', [0, .1, 1]), ('gen_top_frac', [0, .1, 1]),
])
def test_scene_last_explicit_generation_method_selects_search_dimension(tmp_path, parameter, values):
    scene = {'properties': [{'name': 'quality', 'direction': 'max'}],
             'gen_temperature': 9, 'gen_epsilon': .2, 'gen_top_frac': .4}
    scene.pop(parameter)
    scene[parameter] = 0  # Explicit zero still selects this strategy.
    plan = load_plan(write_plan(tmp_path, {'generation': {'initial_values': values}}, scene))
    assert plan['generation_search']['parameter'] == parameter
    assert plan['generation_search']['initial_values'] == values
    assert parameter not in plan['generation_search']['fixed']


@pytest.mark.parametrize('parameter,values', [
    ('gen_temperature', [-1, 1]), ('gen_epsilon', [0, 1.01]), ('gen_top_frac', [0, 2]),
    ('gen_temperature', [0, float('inf')]), ('gen_epsilon', [0, 0]),
])
def test_generation_parameter_domains(tmp_path, parameter, values):
    with pytest.raises(ConfigError):
        load_plan(write_plan(tmp_path, {'generation': {'initial_values': values}},
                             {'properties': [{'name': 'quality', 'direction': 'max'}], parameter: 0}))


def test_explicit_ranges_override_profiles_and_fixed_dqn_steps_remain(tmp_path):
    plan = load_plan(write_plan(tmp_path, {'method': 'dqn-bootstrap', 'training': {'parameter_ranges': {
        'dqn_lr': {'type': 'categorical', 'choices': [.0003]},
    }}}, {'properties': [{'name': 'quality', 'direction': 'max'}], 'dqn_grad_steps_per_ep': 7}))
    assert plan['search_space']['dqn_lr']['choices'] == [.0003]
    assert 'dqn_target_update_freq' in plan['search_space']
    assert 'dqn_grad_steps_per_ep' not in plan['search_space']
    assert plan['base_config_data']['dqn_grad_steps_per_ep'] == 7


@pytest.mark.parametrize('name', [
    'dummy_a2c.yaml', 'dummy_dqn_bootstrap.yaml', 'dummy_dqn_mc.yaml',
    'glassnet_a2c.example.yaml', 'glassnet_dqn_bootstrap.example.yaml', 'glassnet_dqn_mc.example.yaml',
])
def test_templates_resolve_shared_rules_using_dummy_scene(tmp_path, name):
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / 'configs/hpo/v1' / name).read_text())
    assert not {'validation', 'range_search'} & data['training'].keys()
    assert 'repeats_per_model' not in data['generation']
    assert 'max_extra_values_per_direction' not in data['generation']
    assert all('factor' not in data[phase]['budget'] for phase in ('training', 'generation'))
    # GlassNet assets/threshold belong to the user's host; exercise all actual
    # template controls against a local named scene without pretending to run it.
    data['base_config'] = str(root / 'configs/hpo/scenarios/dummy.yaml')
    data['evaluation']['threshold'] = .5
    source = tmp_path / 'hpo.yaml'
    source.write_text(yaml.safe_dump(data, sort_keys=False))
    plan = load_plan(source)
    assert plan['protocol'] == 'hpo_v2'
    assert plan['parallel_jobs'] == data['parallel_jobs'] == 1
    assert plan['sources']['parallel_jobs'] == 'yaml'
    assert plan['method'] == data['method']
    assert plan['objective']['field'] == 'quality'
    assert plan['generation_search']['parameter'] == 'gen_temperature'
    assert plan['generation_search']['repeats_per_model'] == data['generation']['repeats']
    assert plan['generation_search']['max_extra_values_per_direction'] == 2
    assert plan['training_search']['finalists'] == 3
    assert plan['range_search']['max_extra_values_per_direction'] == 2
    assert plan['budget']['train']['factor'] == plan['budget']['generation']['factor'] == 2
    assert len(plan['training_search']['validation_seeds']) == data['training']['repeats']
    if plan['method'].startswith('dqn-'):
        assert 'dqn_grad_steps_per_ep' not in plan['search_space']
        assert ('dqn_target_update_freq' in plan['search_space']) == (plan['method'] == 'dqn-bootstrap')


@pytest.mark.parametrize('parameter', ['gen_epsilon', 'gen_top_frac'])
def test_strategy_defaults_do_not_inherit_temperature_domain(tmp_path, parameter):
    scene = {'properties': [{'name': 'quality', 'direction': 'max'}], parameter: .1}
    plan = load_plan(write_plan(tmp_path, scene=scene))
    assert plan['generation_search']['parameter'] == parameter
    assert len(plan['generation_search']['initial_values']) > 1
    assert all(0 <= value <= 1 for value in plan['generation_search']['initial_values'])


def test_legacy_compatibility_warns_about_preserved_semantics(tmp_path):
    path = write_plan(tmp_path)
    path.write_text(yaml.safe_dump({
        'protocol': 'hpo_v1', 'base_config': 'scene.yaml', 'method': 'a2c',
        'objective': {'field': 'working_temperature', 'direction': 'minimize',
                      'unit': 'K', 'threshold': 1000},
    }))
    with pytest.warns(FutureWarning, match='retains its original scoring and seed settings'):
        plan = load_plan(path)
    assert plan['protocol'] == 'hpo_v1'
    assert plan['objective']['threshold_inclusive'] is False
