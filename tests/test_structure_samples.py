"""Sample identity is physical; worker count, call order and batching are not."""
from copy import deepcopy

import pytest

from rl_matdesign.structure_sources import StructureSourceSpec
from rl_matdesign.utils.structure_samples import (
    SAMPLE_INFO_KEY, attach_sample_identity, canonical_sample_candidate, sample_plan,
)


def test_sample_plan_retains_template_counts_and_uses_source_seed_protocol():
    source = StructureSourceSpec('bulk', {'builder': 'substitute', 'base_poscar': ['a', 'b']})
    templates = [{'path': 'a', 'site_symbol': 'X', 'n_configs': 2},
                 {'path': 'b', 'site_symbol': 'Fe'}]
    candidate = {'Ni': .6, 'Fe': .4}
    plan = sample_plan(candidate, n_configs=3, templates=templates, source_spec=source, eval_seed=7)
    assert [(s.template_index, s.sample_index) for s in plan] == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert [s.site_symbol for s in plan] == ['X', 'X', 'Fe', 'Fe', 'Fe']
    for sample in plan:
        assert sample.seed == source.seed_for(candidate, eval_seed=7, purpose='build',
                                             template_id=sample.template_id, sample_index=sample.sample_index)
    reordered = sample_plan(dict(reversed(list(candidate.items()))), n_configs=3, templates=templates,
                            source_spec=source, eval_seed=7)
    assert plan == reordered
    assert len({sample.seed for sample in plan}) == len(plan)


def test_sample_metadata_is_defensive_and_does_not_overwrite_builder_information():
    from ase import Atoms
    source = StructureSourceSpec('bulk', {'builder': 'native'})
    sample = sample_plan({'Fe': 1}, n_configs=1, source_spec=source, eval_seed=1)[0]
    atoms = Atoms('H')
    atoms.info['builder'] = 'kept'
    attach_sample_identity(atoms, sample)
    attach_sample_identity(atoms, sample)
    assert atoms.info['builder'] == 'kept'
    snapshot = deepcopy(atoms.info)
    other = sample_plan({'Fe': 1}, n_configs=1, source_spec=source, eval_seed=2)[0]
    with pytest.raises(ValueError, match='conflicting'):
        attach_sample_identity(atoms, other)
    assert atoms.info == snapshot
    atoms.info[SAMPLE_INFO_KEY]['source_id'] = 'changed'
    assert sample.record()['source_id'] == 'bulk'


def test_canonical_candidate_sorts_mappings_without_erasing_typed_fields_or_sequence_order():
    candidate = {'z': {'b': 1, 'a': True}, 'sequence': ['01', 1, '1']}
    result = canonical_sample_candidate(candidate)
    assert list(result) == ['sequence', 'z'] and list(result['z']) == ['a', 'b']
    assert result['sequence'] == ['01', 1, '1']
    assert result['z']['a'] is True
    assert list(candidate) == ['z', 'sequence']


@pytest.mark.parametrize('count', [0, -1, True, 1.5, '2'])
def test_sample_counts_are_not_silently_coerced(count):
    source = StructureSourceSpec('bulk', {'builder': 'native'})
    with pytest.raises(ValueError, match='positive integer'):
        sample_plan({}, n_configs=count, source_spec=source, eval_seed=1)
    with pytest.raises(ValueError, match='positive integer'):
        sample_plan({}, n_configs=1, source_spec=source, eval_seed=1,
                    templates=[{'path': 'a', 'site_symbol': 'X', 'n_configs': count}])


class _ExplicitSampleBuilder:
    def build_sample(self, candidate, *, sample):
        from ase import Atoms
        import numpy as np
        return Atoms('H2', positions=np.random.default_rng(sample.seed).random((2, 3)))


def test_generic_explicit_protocol_serial_batches_share_candidate_stable_plan():
    from rl_matdesign.utils.structure_samples import execute_many_seeded_samples, execute_seeded_samples
    source = StructureSourceSpec('bulk', {'builder': 'external:Explicit'})
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=13)
    builder = _ExplicitSampleBuilder()
    a, b = {'Ni': .5, 'Co': .5}, {'Ni': .25, 'Co': .75}
    first = execute_many_seeded_samples(builder, [a, b], **kwargs)
    reverse = execute_many_seeded_samples(builder, [b, dict(reversed(list(a.items())))], **kwargs)
    scalar = execute_seeded_samples(builder, a, **kwargs)
    def signature(cells):
        return [(cell.positions.tolist(), cell.info[SAMPLE_INFO_KEY]) for cell in cells]
    assert signature(first[0]) == signature(reverse[1]) == signature(scalar)
    assert signature(first[1]) == signature(reverse[0])


def test_missing_sample_capability_fails_before_native_build():
    from rl_matdesign.utils.structure_samples import SamplingProtocolError, execute_many_seeded_samples
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    class Native:
        def build(self, candidate, **kwargs):
            pytest.fail('Opaque native ensemble must not be called by explicit protocol')
    source = StructureSourceSpec('bulk', {'builder': 'external:Native'})
    with pytest.raises(EvaluationProtocolError, match='explicit build_sample') as exc:
        execute_many_seeded_samples(Native(), [{}, {}], n_configs=2, source_spec=source, eval_seed=1)
    assert isinstance(exc.value, SamplingProtocolError)


def test_candidate_plan_failure_is_aligned_and_does_not_discard_neighbors():
    from rl_matdesign.utils.structure_samples import execute_many_seeded_samples
    class Planner(_ExplicitSampleBuilder):
        def sample_plan(self, candidate, **kwargs):
            if candidate.get('fail'):
                raise ValueError('invalid candidate plan')
            return sample_plan(candidate, **kwargs)
    source = StructureSourceSpec('bulk', {'builder': 'external:Planner'})
    builder = Planner()
    result = execute_many_seeded_samples(builder, [{'id': 'a'}, {'fail': True}, {'id': 'c'}],
                                         n_configs=2, source_spec=source, eval_seed=1)
    assert len(result[0]) == len(result[2]) == 2
    assert isinstance(result[1], ValueError)
    assert [record[0]['status'] for record in builder.last_sample_records] == ['succeeded', 'plan_failed', 'succeeded']
    assert 'sample_index' not in builder.last_sample_records[1][0]


@pytest.mark.parametrize('corruption', [
    'empty', 'not_descriptor', 'wrong_source', 'wrong_seed', 'duplicate',
    'bool_index', 'partial_template', 'invalid_template',
])
def test_custom_plan_must_honor_explicit_source_and_sample_identity(corruption):
    from dataclasses import replace
    from rl_matdesign.utils.structure_samples import resolve_sample_plan
    source = StructureSourceSpec('bulk', {'builder': 'external:Planner'})
    class Planner(_ExplicitSampleBuilder):
        def sample_plan(self, candidate, **kwargs):
            plan = sample_plan(candidate, **kwargs)
            first = plan[0]
            return {
                'empty': (), 'not_descriptor': ({},),
                'wrong_source': (replace(first, source_id='other'),),
                'wrong_seed': (replace(first, seed=first.seed + 1),),
                'duplicate': (first, first),
                'bool_index': (replace(first, sample_index=True),),
                'partial_template': (replace(first, site_symbol='Fe'),),
                'invalid_template': (replace(first, template_index=-1, template_path='a', site_symbol='Fe'),),
            }[corruption]
    with pytest.raises(ValueError):
        resolve_sample_plan(Planner(), {'Fe': 1}, n_configs=2, source_spec=source, eval_seed=1)


def test_custom_planner_cannot_mutate_callers_candidate_or_reseed_modified_candidate():
    from rl_matdesign.utils.structure_samples import resolve_sample_plan
    source = StructureSourceSpec('bulk', {'builder': 'external:Planner'})
    class Planner(_ExplicitSampleBuilder):
        def sample_plan(self, candidate, **kwargs):
            candidate['nested']['Fe'] = 2
            return sample_plan(candidate, **kwargs)
    candidate = {'nested': {'Fe': 1}}
    with pytest.raises(ValueError, match='seed does not match'):
        resolve_sample_plan(Planner(), candidate, n_configs=1, source_spec=source, eval_seed=1)
    assert candidate == {'nested': {'Fe': 1}}


def test_explicit_sample_requires_single_atoms_and_retains_successful_samples():
    from rl_matdesign.utils.structure_samples import execute_seeded_samples
    class BrokenSample(_ExplicitSampleBuilder):
        def build_sample(self, candidate, *, sample):
            result = super().build_sample(candidate, sample=sample)
            if sample.sample_index == 1:
                raise RuntimeError('single sample construction failed')
            return result
    source = StructureSourceSpec('bulk', {'builder': 'external:BrokenSample'})
    builder = BrokenSample()
    cells = execute_seeded_samples(builder, {'Fe': 1}, n_configs=3, source_spec=source, eval_seed=1)
    assert [cell.info[SAMPLE_INFO_KEY]['sample_index'] for cell in cells] == [0, 2]
    assert [record['status'] for record in builder.last_sample_records[0]] == ['succeeded', 'failed', 'succeeded']
    assert 'single sample construction failed' in builder.last_sample_records[0][1]['error']



def test_sample_protocol_errors_propagate_without_trying_more_samples():
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    from rl_matdesign.utils.structure_samples import execute_many_seeded_samples
    class InvalidProtocol(_ExplicitSampleBuilder):
        calls = 0
        def build_sample(self, candidate, *, sample):
            self.calls += 1
            raise EvaluationProtocolError('invalid configured oracle')
    builder = InvalidProtocol()
    source = StructureSourceSpec('bulk', {'builder': 'external:InvalidProtocol'})
    with pytest.raises(EvaluationProtocolError, match='invalid configured oracle'):
        execute_many_seeded_samples(builder, [{}, {}], n_configs=2, source_spec=source, eval_seed=1)
    assert builder.calls == 1
