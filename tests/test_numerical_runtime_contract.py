"""Resume compares numerical runtime, without binding unrelated process state."""
import copy
import json
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.execution_contract import (
    execution_contract, numerical_runtime_contract, require_execution_contract,
)
from rl_matdesign.utils import provenance
from rl_matdesign.utils.timing import PredictorTimer


def runtime():
    return {
        'python': '3.11.10', 'platform': 'linux', 'machine': 'x86_64',
        'executable': '/first/env/bin/python',
        'packages': {'matminer': '0.9', 'pymatgen': '2025.1', 'scikit-learn': '1.5'},
        'determinism_environment': {'MKL_NUM_THREADS': '1', 'OMP_SCHEDULE': 'static'},
        'torch': {'version': '2.10', 'cuda_initialized': False,
                  'num_threads': 1, 'num_interop_threads': 1,
                  'matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                  'float32_matmul_precision': 'highest', 'cudnn_version': 90100},
        'nvidia_smi': {'status': 'available',
                       'columns': 'index,name,uuid,driver_version',
                       'output': '0, A100, GPU-first, 570.0\n', 'truncated': False},
    }


@pytest.mark.parametrize('section,key,value', [
    ('packages', 'matminer', '1.0'),
    ('packages', 'pymatgen', '2026.1'),
    ('packages', 'scikit-learn', '1.6'),
    ('determinism_environment', 'MKL_NUM_THREADS', '4'),
    ('determinism_environment', 'OMP_SCHEDULE', 'dynamic'),
    ('torch', 'num_threads', 2),
    ('torch', 'num_interop_threads', 2),
    ('torch', 'matmul_allow_tf32', True),
    ('torch', 'cudnn_allow_tf32', True),
    ('torch', 'cudnn_version', 90101),
    ('torch', 'float32_matmul_precision', 'high'),
    ('nvidia_smi', 'output', '0, A100, GPU-first, 580.0\n'),
])
def test_changed_numerical_runtime_rejects_exact_resume(section, key, value):
    original = runtime()
    changed = copy.deepcopy(original)
    changed[section][key] = value
    expected = {'evaluation': {'numerical_runtime': numerical_runtime_contract(original)}}
    actual = {'evaluation': {'numerical_runtime': numerical_runtime_contract(changed)}}
    with pytest.raises(ValueError, match='runtime'):
        require_execution_contract(actual, expected)


def test_process_location_cuda_initialization_and_gpu_uuid_do_not_change_contract():
    original = runtime()
    changed = copy.deepcopy(original)
    changed['executable'] = '/second/env/bin/python'
    changed['torch']['cuda_initialized'] = True
    changed['nvidia_smi']['output'] = '0, A100, GPU-other-uuid, 570.0\n'
    changed['packages']['scikit_learn'] = changed['packages'].pop('scikit-learn')
    assert numerical_runtime_contract(original) == numerical_runtime_contract(changed)


def test_checkpoint_frequency_changes_learning_contract_only(tmp_path):
    path = tmp_path / 'manifest.json'
    manifest = {'input_files': [], 'runtime': runtime(),
                'runtime_args': {'save_checkpoint_freq': 1}}
    path.write_text(json.dumps(manifest))
    first = execution_contract({'method': 'dqn'}, SimpleNamespace(), path, 'cpu')
    manifest['runtime_args']['save_checkpoint_freq'] = 100
    path.write_text(json.dumps(manifest))
    second = execution_contract({'method': 'dqn'}, SimpleNamespace(), path, 'cpu')
    assert first['evaluation'] == second['evaluation']
    assert first['learning']['checkpoint_frequency'] == 1
    with pytest.raises(ValueError, match='training protocol'):
        require_execution_contract(second, first)


def test_manifest_runtime_propagates_to_execution_contract(tmp_path):
    path = tmp_path / 'manifest.json'
    manifest = {'input_files': [], 'runtime': runtime()}
    path.write_text(json.dumps(manifest))
    first = execution_contract({'method': 'dqn'}, SimpleNamespace(), path, 'cpu')
    manifest['runtime']['packages']['ase'] = '3.27'
    path.write_text(json.dumps(manifest))
    changed = execution_contract({'method': 'dqn'}, SimpleNamespace(), path, 'cpu')
    with pytest.raises(ValueError, match='runtime'):
        require_execution_contract(changed, first)


@pytest.mark.parametrize('first,second,sampling_matches', [
    ({}, {'gen_temperature': 1}, True),
    ({'gen_temperature': 1}, {'gen_epsilon': .2}, False),
    ({'gen_temperature': 0}, {'gen_epsilon': 0}, False),
    ({'gen_top_frac': .2}, {'gen_top_frac': .8}, False),
])
def test_generation_selection_is_not_part_of_the_oracle_or_learning_contract(
        tmp_path, first, second, sampling_matches):
    from rl_matdesign.config_schema import resolve_config
    from rl_matdesign.utils.execution_contract import generation_sampling_contract
    from rl_matdesign.utils.generation_resume import restore_generation
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'input_files': [], 'runtime': runtime()}))
    before = resolve_config(first).config
    after = resolve_config(second).config
    assert execution_contract(before, SimpleNamespace(), path, 'cpu') == execution_contract(
        after, SimpleNamespace(), path, 'cpu')
    sampling_before = generation_sampling_contract(before)
    sampling_after = generation_sampling_contract(after)
    assert (sampling_before == sampling_after) == sampling_matches
    if not sampling_matches:
        with pytest.raises(ValueError, match='sampling settings'):
            restore_generation(None, {'generation_contract': {'sampling': sampling_before}},
                               {'sampling': sampling_after})


def test_generation_sampling_contract_ignores_source_provenance():
    from rl_matdesign.config_schema import resolve_config
    from rl_matdesign.utils.execution_contract import generation_sampling_contract
    authored = resolve_config({'gen_epsilon': .2}).config
    overridden = resolve_config({}, {'gen_epsilon': .2}).config
    assert authored['generation_selection']['source'] != overridden['generation_selection']['source']
    assert generation_sampling_contract(authored) == generation_sampling_contract(overridden)


@pytest.mark.parametrize('name', ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'MKL_CBWR', 'OMP_SCHEDULE', 'OMP_PROC_BIND',
    'OMP_DYNAMIC', 'MKL_DYNAMIC', 'KMP_AFFINITY', 'NVIDIA_TF32_OVERRIDE'])
def test_provenance_captures_numerical_environment_whitelist(monkeypatch, name):
    monkeypatch.setenv(name, 'test-setting')
    monkeypatch.setenv('UNRELATED_SECRET', 'not-for-the-manifest')
    monkeypatch.setattr(provenance.shutil, 'which', lambda _: None)
    evidence = provenance._runtime()
    assert evidence['determinism_environment'][name] == 'test-setting'
    assert 'UNRELATED_SECRET' not in evidence['determinism_environment']
    assert evidence['torch']['num_threads'] >= 1
    assert evidence['torch']['num_interop_threads'] >= 1
    assert evidence['torch']['float32_matmul_precision'] in ('highest', 'high', 'medium')


def test_failed_scalar_attempt_retains_backend_task_costs(tmp_path):
    class Failed:
        def predict(self, _candidate):
            self.last_evaluation_timings = [{'config_attempted': 3,
                'config_succeeded': 0, 'md_task_s': 4.5, 'property_s': 0.0,
                'cache_hit': False}]
            raise ValueError('all structures failed after physical work')
    timer = PredictorTimer(Failed(), episode_log_path=str(tmp_path / 'episode_timing.csv'))
    with pytest.raises(ValueError, match='physical work'):
        timer.predict({'Ni': 1.0})
    row = json.loads((tmp_path / 'evaluations.jsonl').read_text())
    assert row['status'] == 'failed'
    assert row['wall_elapsed_s'] > 0
    assert row['prediction_elapsed_s'] == timer.t_predict_s
    assert row['timing'] == timer.last_evaluation_timings[0]
    assert row['timing']['config_attempted'] == 3
    assert row['timing']['md_task_s'] == 4.5
    assert timer.summary()['phases']['train']['requests'] == 1
    assert timer.summary()['phases']['train']['failures'] == 1
    assert timer.n_unique == 0
    assert timer.t_predict_s > 0


def test_contract_error_names_changed_fields_without_values():
    original = {'learning': {'train_count_duplicates': False},
                'evaluation': {'numerical_runtime': {'packages': {'ase': 'secret-original-version'}}}}
    changed = copy.deepcopy(original)
    changed['learning']['train_count_duplicates'] = True
    changed['evaluation']['numerical_runtime']['packages']['ase'] = 'secret-new-version'
    with pytest.raises(ValueError) as error:
        require_execution_contract(changed, original)
    message = str(error.value)
    assert 'learning.train_count_duplicates' in message
    assert 'evaluation.numerical_runtime.packages.ase' in message
    assert 'secret-original-version' not in message
    assert 'secret-new-version' not in message


def test_contract_difference_paths_are_bounded_and_handle_lists_missing_fields():
    from rl_matdesign.utils.execution_contract import _contract_difference_paths
    assert _contract_difference_paths({'a': [{'b': 1}]}, {'a': [{'b': 2}, 3]}) == [
        'a[0].b', 'a[1]']
    assert _contract_difference_paths({'a': {}}, {'a': {'missing': 'hidden-value'}}) == ['a.missing']
    assert _contract_difference_paths(None, {'a': 'hidden-value'}) == ['<root>']
    original = {f'field_{index:02d}': 0 for index in range(12)}
    changed = {key: 1 for key in original}
    assert _contract_difference_paths(original, changed) == [f'field_{index:02d}' for index in range(8)]
