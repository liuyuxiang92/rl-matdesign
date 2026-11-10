"""Integer-token domains: scientific identity, complete masks and migration."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from math import gcd
from functools import reduce
from pathlib import Path

import numpy as np
import pytest

from rl_matdesign.configuration import _TerminalReward, build_env, load_config
from rl_matdesign.env import CompositionEnv
from rl_matdesign.env_integer import IntegerRatioEnv, normalize_integer_group_spec
from rl_matdesign.env_multigroup import CategoricalGroup, MultiGroupEnv
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator
from rl_matdesign.utils.rollout_errors import InfeasibleEpisode


def descriptor(_formula):
    # Constant normalized descriptor makes missing absolute state observable.
    return np.array([1.0])


def make_env(**overrides):
    return IntegerRatioEnv(**{
        "species_set": ["Fe", "Co", "O"], "ratio_set": ["0", "1", "2"],
        "n_components": 3, "state_featurizer": descriptor,
        "token_requirements": {"O": {"count": 1, "positive": True}}, **overrides,
    })


def action(env, symbol, amount):
    return (tuple(np.eye(len(env.species_set))[env.species_set.index(symbol)]),
            tuple(np.eye(len(env.fraction_set))[env.fraction_set.index(str(amount))]))


def decoded(env):
    return {(env.species_set[int(np.argmax(first))], int(env.fraction_set[int(np.argmax(second))]))
            for first, second in env.allowed_actions()}


def key_for(tokens):
    counts = Counter()
    for label, amount in tokens:
        counts[label] += amount
    factor = reduce(gcd, counts.values())
    return tuple((label, count // factor) for label, count in sorted(counts.items()) if count)


@pytest.mark.parametrize("requirements", [
    {"O": {"count": 1, "positive": True}},
    {"O": {"count": 2, "positive": False}},
    {"O": {"count": 0}},
    {"O": {"count": 1}, "Fe": {"count": 1}, "Co": {"count": 1}},
    {},
])
def test_every_reachable_mask_matches_independently_enumerated_valid_completions(requirements):
    vocabulary = list(product(["Fe", "Co", "O"], [0, 1, 2]))
    valid = []
    for tokens in product(vocabulary, repeat=3):
        if not sum(amount for _, amount in tokens):
            continue
        if any(sum(label == required for label, _ in tokens) != rule["count"]
               or rule.get("positive", False) and any(label == required and amount == 0
                                                     for label, amount in tokens)
               for required, rule in requirements.items()):
            continue
        valid.append(tokens)
    continuations = defaultdict(set)
    for tokens in valid:
        for depth in range(3):
            continuations[tokens[:depth]].add(tokens[depth])
    for prefix, expected in continuations.items():
        env = make_env(token_requirements=requirements)
        for label, amount in prefix:
            env.step(action(env, label, amount))
        assert decoded(env) == expected, prefix


def test_free_oxygen_order_retains_exact_reserved_last_terminal_domain():
    vocabulary = list(product(["Fe", "Co", "O"], [0, 1, 2]))
    old = {key_for(prefix + (("O", oxygen),))
           for prefix in product(list(product(["Fe", "Co"], [0, 1, 2])), repeat=2)
           for oxygen in [1, 2]}
    observed, oxygen_positions = set(), set()
    for tokens in product(vocabulary, repeat=3):
        if sum(label == "O" for label, _ in tokens) != 1 or ("O", 0) in tokens:
            continue
        env = make_env()
        for label, amount in tokens:
            env.step(action(env, label, amount))
        observed.add(env.terminal_comp_key())
        oxygen_positions.add(next(index for index, pair in enumerate(tokens) if pair[0] == "O"))
    assert observed == old
    assert oxygen_positions == {0, 1, 2}


def test_counts_and_zero_selection_state_are_absolute_and_order_independent():
    first, second = make_env(n_components=4), make_env(n_components=4)
    for env, tokens in [(first, [("Fe", 1), ("Fe", 2), ("Co", 0)]),
                        (second, [("Co", 0), ("Fe", 2), ("Fe", 1)])]:
        for label, amount in tokens:
            env.step(action(env, label, amount))
    assert first.cation_digits() == {"Co": 0, "Fe": 3}
    assert first.counter == 3
    np.testing.assert_array_equal(first.current_state_features(), second.current_state_features())
    # descriptor, counts(Co/Fe/O), pick counts(Co/Fe/O), remaining budget.
    np.testing.assert_array_equal(first.current_state_features(), [1, 0, 3, 0, 1, 2, 0, 1])
    assert decoded(first) == decoded(second) == {("O", 1), ("O", 2)}
    one, two = make_env(), make_env()
    one.step(action(one, "Fe", 1))
    two.step(action(two, "Fe", 2))
    assert not np.array_equal(one.current_state_features(), two.current_state_features())


def test_invalid_integer_actions_are_atomic_and_do_not_reach_oracle():
    calls = []
    env = make_env(reward_fn=lambda formula: calls.append(formula) or 0)
    initial = env.current_state_features().copy()
    for bad in [action(env, "O", 0), ((0, 0, 0), (1, 0, 0)), ((1, 1, 0), (0, 1, 0))]:
        with pytest.raises(ValueError):
            env.step(bad)
        np.testing.assert_array_equal(env.current_state_features(), initial)
        assert not env.path and not calls
    env.step(action(env, "O", 1))
    with pytest.raises(ValueError):
        env.step(action(env, "O", 2))
    env.step(action(env, "Fe", 0))
    env.step(action(env, "Fe", 0))
    assert calls == ["O1"] and env.terminal_cation_fractions() == {"O": 1.0}
    with pytest.raises(RuntimeError):
        env.step(action(env, "Fe", 1))
    with pytest.raises(RuntimeError):
        env.sample_random_action()


def test_zero_only_integer_candidates_cannot_be_completed_or_scored():
    with pytest.raises(ValueError, match="no positive complete material"):
        make_env(ratio_set=["0"], token_requirements={})
    env = make_env(token_requirements={})
    env.step(action(env, "Fe", 0))
    env.step(action(env, "Co", 0))
    assert all(amount > 0 for _, amount in decoded(env))
    with pytest.raises(ValueError):
        env.step(action(env, "Fe", 0))
    assert not env.is_terminal


@pytest.mark.parametrize("requirements", [
    [], {"O": {"count": True}}, {"O": {"count": 1.5}}, {"O": {"count": -1}},
    {"O": {"count": 1, "positive": 1}}, {"O": {"count": 1, "extra": 2}},
    {"O": {"positive": True}}, {"Ni": {"count": 1}}, {"O": {"count": 4}},
])
def test_invalid_or_impossible_requirements_fail_at_construction(requirements):
    with pytest.raises(ValueError):
        make_env(token_requirements=requirements)


def test_domain_input_is_copied_and_exact_fully_reserved_budgets_are_checked():
    requirements = {"O": {"count": 1, "positive": True}}
    env = make_env(token_requirements=requirements)
    requirements["O"]["count"] = 2
    assert env.token_requirements == {"O": {"count": 1, "positive": True}}
    assert env.design_schema()["token_requirements"] == env.token_requirements
    with pytest.raises(ValueError, match="no positive complete material"):
        make_env(species_set=["O"], token_requirements={"O": {"count": 1}})
    with pytest.raises(ValueError, match="no positive value"):
        make_env(ratio_set=["0"])


class RejectAll:
    def filter_actions(self, **kwargs):
        return []


def test_optional_filter_switch_preserves_hard_domain_and_empty_is_infeasible():
    env = make_env(phase_filter=RejectAll())
    assert env.allowed_actions() == []
    with pytest.raises(InfeasibleEpisode):
        env.sample_random_action()
    env.constraints_enabled = False
    assert ("O", 0) not in decoded(env)
    env.step(action(env, "O", 1))
    assert all(label != "O" for label, _ in decoded(env))


@pytest.mark.parametrize("kind", ["integer", "categorical"])
def test_filter_cannot_mutate_environment_or_prior_group_context(kind):
    class Mutate:
        def filter_actions(self, *, actions, species_set, fraction_set, prior_groups, **kwargs):
            species_set.clear()
            fraction_set.clear()
            prior_groups[0].clear()
            if "code_to_value" in kwargs:
                kwargs["code_to_value"].clear()
            return actions
    env = (make_env(phase_filter=Mutate()) if kind == "integer" else
           CategoricalGroup([{"name": "temperature", "values": [0, 100]}], phase_filter=Mutate()))
    prior = [{"Fe": 1.0}]
    labels, values = list(env.species_set), list(env.fraction_set)
    expected = env.allowed_actions(prior_groups=prior)
    assert expected == env.allowed_actions(prior_groups=prior)
    assert env.species_set == labels and env.fraction_set == values
    assert prior == [{"Fe": 1.0}]


@pytest.mark.parametrize("kind", ["integer", "categorical"])
@pytest.mark.parametrize("result", [None, "string", "duplicates", "new_action", "unhashable"])
def test_invalid_plugin_results_are_errors_not_infeasible_rollouts(kind, result):
    class Invalid:
        def filter_actions(self, *, actions, **kwargs):
            if result == "duplicates":
                return [actions[0], actions[0]]
            if result == "new_action":
                actions.append(((0,), (0,)))
                return actions
            if result == "unhashable":
                return [list(actions[0])]
            return result
    env = (make_env(phase_filter=Invalid()) if kind == "integer" else
           CategoricalGroup([{"name": "temperature", "values": [0, 100]}], phase_filter=Invalid()))
    with pytest.raises(ValueError):
        env.allowed_actions()


def test_multigroup_prior_payload_is_gcd_reduced_and_field_zero_survives():
    class PriorRequired:
        def __init__(self):
            self.seen = []
        def filter_actions(self, *, actions, prior_groups, **kwargs):
            self.seen.append(deepcopy(prior_groups))
            return actions if prior_groups == [{"Fe": 1, "O": 2}] else []
    all_payloads = []
    for multiplier in [1, 2]:
        guard = PriorRequired()
        env = MultiGroupEnv(groups=[
            {"name": "oxide", "kind": "integer", "species_set": ["Fe", "O"],
             "ratio_set": ["1", "2", "4"], "n_components": 2},
            {"name": "conditions", "kind": "categorical", "choices": [
                {"name": "temperature", "values": [0]}], "constraint_filter": guard},
        ], state_featurizer=descriptor, reward_fn=lambda payload: all_payloads.append(payload) or 1)
        inner = env._active()
        for label, amount in [("Fe", multiplier), ("O", 2 * multiplier)]:
            env.step(env._to_union(action(inner, label, amount), inner))
        env.step(env.allowed_actions()[0])
        assert guard.seen and all(prior == [{"Fe": 1, "O": 2}] for prior in guard.seen)
        assert env.terminal_cation_fractions() == {"oxide": {"Fe": 1, "O": 2},
                                                  "conditions": {"temperature": 0}}
    assert all_payloads[0] == all_payloads[1]


def test_integer_step_passes_prior_group_context_to_filter_revalidation():
    class PriorRequired:
        def filter_actions(self, *, actions, prior_groups, **kwargs):
            return actions if prior_groups == [{"Fe": 1.0}] else []
    env = MultiGroupEnv(groups=[
        {"name": "base", "kind": "composition", "species_set": ["Fe"],
         "fraction_set": ["1.0"], "n_components": 1},
        {"name": "integer", "kind": "integer", "species_set": ["O"],
         "ratio_set": ["1"], "n_components": 1, "constraint_filter": PriorRequired()},
    ], state_featurizer=descriptor)
    env.step(env.allowed_actions()[0])
    env.step(env.allowed_actions()[0])
    assert env.is_terminal


@pytest.mark.parametrize("batch_size", [1, 2])
def test_integer_oracle_uses_canonical_mapping_for_scalar_and_deferred(batch_size):
    class Oracle:
        seen = []
        def evaluation_batch_size(self):
            return batch_size
        def predict(self, candidate):
            self.seen.append(candidate)
            return candidate["O"], 0.0
    oracle = Oracle()
    env = make_env(n_components=2, reward_fn=_TerminalReward(oracle, "integer_ratio"))
    paths = iter([[('Fe', 1), ('O', 2)], [('O', 2), ('Fe', 1)]])
    def rollout():
        env.initialize()
        for label, amount in next(paths):
            env.step(action(env, label, amount))
    evaluator = EpisodeBatchEvaluator(env)
    first, second = evaluator.next(rollout, limit=2), evaluator.next(rollout, limit=1)
    assert oracle.seen == [{"Fe": 1 / 3, "O": 2 / 3}] * 2
    assert first.path[-1].reward == second.path[-1].reward == 2 / 3


LEGACY = {"type": "last_step_element", "required_elements": ["O"],
          "reserve_for_last": True, "nonzero_ratio_at_last": True}


def test_only_known_singleton_reserved_positive_oxygen_is_migrated():
    group = {"kind": "integer", "filters": [LEGACY, {"type": "smact_charge"}]}
    before = deepcopy(group)
    with pytest.warns(DeprecationWarning, match="O-last"):
        result = normalize_integer_group_spec(group)
    assert group == before
    assert result["filters"] == [{"type": "smact_charge"}]
    assert result["token_requirements"] == {"O": {"count": 1, "positive": True}}
    for change in [{"required_elements": ["O", "N"]}, {"reserve_for_last": False},
                   {"nonzero_ratio_at_last": False}, {"type": "external:Filter"}]:
        unmodified = {"kind": "integer", "filters": [{**LEGACY, **change}]}
        assert normalize_integer_group_spec(unmodified) == unmodified
    with pytest.raises(ValueError, match="conflicts"):
        normalize_integer_group_spec({**group, "token_requirements": {"O": {"count": 2}}})


def test_all_ten_tracked_oxide_domains_have_free_positive_oxygen():
    configs = sorted((Path(__file__).parents[1] / "configs").glob("oxides*.yaml"))
    assert len(configs) == 10
    for path in configs:
        cfg = load_config(path)
        group = cfg["groups"][0]
        assert group["token_requirements"] == {"O": {"count": 1, "positive": True}}
        assert all(item["type"] != "last_step_element" for item in group.get("filters", []))
        env = make_env(species_set=group["species_set"], ratio_set=group["ratio_set"],
                       n_components=group["n_components"], token_requirements=group["token_requirements"])
        env.constraints_enabled = cfg.get("constrain_training", True)
        assert ("O", 1) in decoded(env) and ("O", 0) not in decoded(env)
        env.step(action(env, "O", 1))
        assert all(label != "O" for label, _ in decoded(env))


@pytest.mark.parametrize("count", [True, 1.5])
@pytest.mark.parametrize("kind", ["integer", "composition"])
def test_configuration_does_not_coerce_invalid_slot_counts(count, kind):
    group = {"name": "material", "kind": kind, "species_set": ["Fe", "O"],
             "n_components": count, "ratio_set": ["1"], "fraction_set": ["1.0"]}
    with pytest.raises(ValueError, match="n_components"):
        build_env({"groups": [group]}, predictor=object())
    with pytest.raises(ValueError, match="n_components"):
        MultiGroupEnv(groups=[group], state_featurizer=descriptor)


def test_positive_composition_bound_requires_presence_without_fixing_action_order():
    # Independent two-slot toy domain: a positive bound requires Fe, while
    # either optional species may occupy the other slot in either order.
    assignments = [("Fe", .25), ("Cu", .75)]
    keys = []
    for ordered in [assignments, list(reversed(assignments))]:
        env = CompositionEnv(species_set=["Fe", "Co", "Cu"],
                             fraction_set=["0", ".25", ".5", ".75", "1"],
                             n_components=2, element_bounds={"Fe": [.25, .75]},
                             state_featurizer=descriptor)
        for label, amount in ordered:
            chosen = next(pair for pair in env.allowed_actions()
                          if env.species_set[int(np.argmax(pair[0]))] == label
                          and float(env.fraction_set[int(np.argmax(pair[1]))]) == amount)
            env.step(chosen)
        keys.append(env.terminal_comp_key())
        assert env.terminal_cation_fractions() == {"Fe": .25, "Cu": .75}
    assert keys[0] == keys[1]
