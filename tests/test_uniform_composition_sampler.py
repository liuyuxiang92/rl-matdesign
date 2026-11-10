"""Exact rank bijection, rather than a stochastic uniformity test."""
from itertools import product
import random

import pytest

from rl_matdesign.utils.design_count import UniformCompositionSampler, count_compositions


def reference(menus, total, required, slots):
    labels = sorted(menus)
    choices = [menus[label] + ([] if label in required else [None]) for label in labels]
    return {
        tuple((label, value) for label, value in zip(labels, values) if value)
        for values in product(*choices)
        if sum(v for v in values if v is not None) == total
        and (slots is None or sum(v is not None for v in values) == slots)
    }


def test_every_rank_maps_bijectively_to_an_independently_enumerated_terminal():
    rng = random.Random(19425)
    for _ in range(80):
        menus = {label: rng.sample(range(5), rng.randrange(5)) for label in 'ABCD'}
        required = rng.sample(list(menus), rng.randrange(5))
        total = rng.randrange(9)
        for slots in (None, 0, 1, 2, 3, 4, 5):
            sampler = UniformCompositionSampler(menus, total, required=required, slots=slots)
            expected = reference(menus, total, required, slots)
            actual = [tuple(sorted(sampler.unrank(i).items())) for i in range(sampler.size)]
            assert len(actual) == len(set(actual)) == len(expected)
            assert set(actual) == expected
            assert sampler.size == count_compositions(menus, total, required=required, slots=slots)


def test_sampling_uses_a_uniform_integer_rank_and_preserves_caller_rng():
    sampler = UniformCompositionSampler({'A': [0, 1, 2], 'B': [0, 1, 2], 'C': [0, 1]},
                                        2, required=[], slots=2)
    class RankProbe:
        def __init__(self, rank):
            self.rank = rank
        def randrange(self, bound):
            assert bound == sampler.size
            return self.rank
    assert [sampler.sample(RankProbe(i)) for i in range(sampler.size)] == [
        sampler.unrank(i) for i in range(sampler.size)]
    global_state = random.getstate()
    a, b = random.Random(42), random.Random(42)
    assert [sampler.sample(a) for _ in range(20)] == [sampler.sample(b) for _ in range(20)]
    assert random.getstate() == global_state


def test_zero_choices_have_no_extra_weight_or_rank():
    sampler = UniformCompositionSampler({'A': [0], 'B': [0], 'C': [0, 1]},
                                        1, required=[], slots=2)
    assert sampler.size == 1 and sampler.sample(random.Random(2)) == {'C': 1}


def test_large_domain_is_sampled_without_terminal_materialization():
    sampler = UniformCompositionSampler({str(i): [0, 1] for i in range(100)}, 50)
    assert sampler.size > 2**64
    row = sampler.sample(random.Random(1))
    assert len(row) == sum(row.values()) == 50
    assert sampler._completions.cache_info().currsize < 500_000


@pytest.mark.parametrize('rank', [-1, True, 0.5, 1])
def test_invalid_rank_rejected(rank):
    sampler = UniformCompositionSampler({'A': [1]}, 1)
    with pytest.raises(ValueError):
        sampler.unrank(rank)


def test_empty_domain_cannot_be_sampled():
    sampler = UniformCompositionSampler({'A': [2]}, 1)
    with pytest.raises(ValueError, match='empty'):
        sampler.sample(random.Random(1))
