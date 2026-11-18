"""Independent small-domain oracle checks and representative amount scenarios."""
from collections import deque
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup


def make(spec, **kwargs):
    return ChemicalAmountsGroup(spec, state_featurizer=lambda _: np.zeros(2), **kwargs)


def action(env, label, amount):
    value = next(i for i, code in enumerate(env.action_values) if Fraction(code) == Fraction(str(amount)))
    return (tuple(float(i == env.species_set.index(label)) for i in range(len(env.species_set))),
            tuple(float(i == value) for i in range(len(env.action_values))))


def decoded(env):
    return {(env.species_set[int(np.argmax(a))], Fraction(env.action_values[int(np.argmax(b))]))
            for a, b in env.allowed_actions()}


@pytest.mark.parametrize("spec", [
    {"species_set": ["Fe", "Ni", "O"], "amount_set": [0, .25, .5, .75, 1],
     "n_steps": 3, "total": 1, "element_bounds": {"Fe": [.5, .75], "O": [0, .5]}},
    {"species_set": ["Fe", "Ni", "O"], "amount_set": [0, 1, 2], "n_steps": 3,
     "element_bounds": {"O": [1, 2], "Ni": [0, 0]}},
    {"species_set": ["Fe", "Ni"], "amount_set": [-1, 0, 2], "n_steps": 3,
     "total": 1, "element_bounds": {"Fe": [-1, 2], "Ni": [-2, 2]}},
    {"species_set": ["Fe", "Ni", "O"], "amount_set": [0, .5, 1], "n_steps": 3,
     "total": 1, "unique": True, "element_bounds": {"O": [.5, 1]}},
    {"species_set": ["Fe", "Ni", "O"],
     "amount_set": {"Fe": [.1, .4], "Ni": [.2, .5], "O": [0, .4]},
     "n_steps": 3, "total": .8, "element_bounds": {"Fe": [.1, .6]}},
])
def test_every_reachable_prefix_matches_independent_exhaustive_oracle(spec):
    """Compare complete legal-action sets, including dead-end pruning, not just rollouts."""
    labels = spec["species_set"]
    menu_spec = spec["amount_set"]
    menus = [tuple(Fraction(str(v)) for v in (menu_spec[label] if isinstance(menu_spec, dict) else menu_spec))
             for label in labels]
    bounds = {label: tuple(Fraction(str(v)) for v in pair)
              for label, pair in spec.get("element_bounds", {}).items()}
    target = None if "total" not in spec else Fraction(str(spec["total"]))
    unique = spec.get("unique", False)
    horizon = spec["n_steps"]

    @lru_cache(None)
    def completes(amounts, picks, remaining):
        if remaining == 0:
            return (target is None or sum(amounts) == target) and all(
                lo <= amounts[labels.index(label)] <= hi for label, (lo, hi) in bounds.items())
        return bool(expected(amounts, picks, remaining))

    def expected(amounts, picks, remaining):
        result = set()
        if not remaining:
            return result
        for index, menu in enumerate(menus):
            if unique and picks[index]:
                continue
            for value in menu:
                next_amounts = list(amounts)
                next_picks = list(picks)
                next_amounts[index] += value
                next_picks[index] += 1
                if completes(tuple(next_amounts), tuple(next_picks), remaining - 1):
                    result.add((labels[index], value))
        return result

    env = make(spec)
    initial = ((Fraction(0),) * len(labels), (0,) * len(labels), horizon)
    queue = deque([(initial, ())])
    seen = set()
    while queue:
        state, prefix = queue.popleft()
        if state in seen:
            continue
        seen.add(state)
        amounts, picks, remaining = state
        env.initialize()
        for label, value in prefix:
            env.step(action(env, label, value))
        assert decoded(env) == expected(*state)
        if remaining == 0:
            assert env.is_terminal
            assert completes(*state)
            continue
        for label, value in expected(*state):
            index = labels.index(label)
            next_amounts, next_picks = list(amounts), list(picks)
            next_amounts[index] += value
            next_picks[index] += 1
            queue.append(((tuple(next_amounts), tuple(next_picks), remaining - 1), prefix + ((label, value),)))


def test_repeated_additions_have_order_invariant_raw_identity_and_distinct_budget_state():
    spec = dict(species_set=["Ni", "Fe"], amount_set=[.1, .2, .3], n_steps=3, total=.6)
    env = make(spec)
    env.step(action(env, "Ni", .1))
    env.step(action(env, "Ni", .2))
    partial = env.current_state_features().copy()
    env.step(action(env, "Fe", .3))
    key, features = env.terminal_comp_key(), env.current_state_features()
    env.initialize()
    env.step(action(env, "Fe", .3))
    env.step(action(env, "Ni", .2))
    env.step(action(env, "Ni", .1))
    assert env.terminal_comp_key() == key == (("Fe", .3), ("Ni", .3))
    np.testing.assert_array_equal(features, env.current_state_features())
    assert not np.array_equal(partial, features)
    assert env.terminal_cation_fractions() == {"Ni": .3, "Fe": .3}


def test_unique_counts_selected_labels_even_when_amount_is_zero():
    env = make(dict(species_set=["Fe", "Ni", "Co"], amount_set=[0, .5, 1],
                    n_steps=3, total=1, unique=True))
    env.step(action(env, "Ni", 0))
    assert all(label != "Ni" for label, _ in decoded(env))
    env.step(action(env, "Fe", 1))
    env.step(action(env, "Co", 0))
    assert env.terminal_comp_key() == (("Fe", 1.0),)


def test_oxygen_minimum_is_final_amount_not_last_action_or_exact_pick_count():
    spec = dict(species_set=["Fe", "O"], amount_set=[0, 1, 2], n_steps=3, element_bounds={"O": [1, 3]})
    env = make(spec)
    env.step(action(env, "O", 1))
    env.step(action(env, "O", 1))
    env.step(action(env, "Fe", 2))
    assert env.terminal_comp_key() == (("Fe", 2.0), ("O", 2.0))
    env.initialize()
    env.step(action(env, "Fe", 2))
    env.step(action(env, "Fe", 2))
    assert decoded(env) == {("O", Fraction(1)), ("O", Fraction(2))}
    with pytest.raises(ValueError, match="infeasible"):
        env.step(action(env, "Fe", 0))


def test_fraction_range_and_bounds_intersect_without_outward_rounding():
    env = make(dict(species_set=["SiO2", "B2O3"], amount_set={"min": 0, "max": .15, "step": .02},
                    n_steps=2, element_bounds={"B2O3": [.05, .15]}))
    assert "0.16" not in env.action_values
    assert ("B2O3", Fraction("0.04")) in decoded(env)  # A second addition can meet the final lower bound.
    env.step(action(env, "SiO2", .02))
    assert decoded(env) == {("B2O3", Fraction(str(value))) for value in [.06, .08, .10, .12, .14]}


def test_zero_bound_bans_compound_choice_not_elements_in_other_choices():
    env = make(dict(species_set=["NiO", "Ni2O3", "Fe"], amount_set=[0, 1], n_steps=1,
                    element_bounds={"NiO": [0, 0]}))
    assert ("NiO", Fraction(1)) not in decoded(env)
    assert ("Ni2O3", Fraction(1)) in decoded(env)


def test_signed_edit_can_temporarily_exceed_final_bounds_and_retains_sign_in_state():
    formulas = []
    env = ChemicalAmountsGroup(dict(species_set=["Sr"], amount_set=[-1, 2], n_steps=2,
                                    element_bounds={"Sr": [1, 1]}),
                               state_featurizer=lambda f: formulas.append(f) or np.zeros(2))
    assert decoded(env) == {("Sr", Fraction(-1)), ("Sr", Fraction(2))}
    env.step(action(env, "Sr", -1))
    assert env.cation_fractions() == {"Sr": -1.0}
    assert env.current_state_features()[2] == -1
    assert formulas[-1] == "Sr1"
    assert env.state_feature_schema()["descriptor_mode"] == "absolute_edit_amounts"
    env.step(action(env, "Sr", 2))
    assert env.terminal_comp_key() == (("Sr", 1.0),)


def test_composition_resolver_supplies_cation_normalized_compound_features():
    formulas = []
    env = ChemicalAmountsGroup(dict(species_set=["Ni2O3"], amount_set=[.04], n_steps=1),
                               state_featurizer=lambda f: formulas.append(f) or np.zeros(2),
                               composition_resolver=lambda a: {"Ni": a.get("Ni2O3", 0), "O": 1.5 * a.get("Ni2O3", 0)})
    env.step(action(env, "Ni2O3", .04))
    env.current_state_features()
    assert formulas[-1] == "Ni0.04O0.06"
    assert env.cation_fractions() == {"Ni2O3": .04}
    assert env.state_feature_schema()["descriptor_mode"] == "resolved_composition"


def test_default_compound_features_preserve_formula_unit_amounts():
    formulas = []
    env = ChemicalAmountsGroup(dict(species_set=["SiO2", "Na2O"], amount_set=[.4, .6], n_steps=2, total=1),
                               state_featurizer=lambda f: formulas.append(f) or np.zeros(2))
    env.step(action(env, "SiO2", .6))
    env.step(action(env, "Na2O", .4))
    env.current_state_features()
    assert formulas[-1] == "(Na2O)0.4(SiO2)0.6"


def test_partial_recipe_validator_can_be_bound_after_initialization():
    env = make(dict(species_set=["P", "Ni", "Ni2O3"], amount_set=[0, .04, .08, .92, .96],
                    n_steps=3, total=1, element_bounds={"P": [.92, .96]}))
    env.candidate_validator = lambda a: not (a.get("Ni", 0) > 0 and a.get("Ni2O3", 0) > 0)
    env.step(action(env, "P", .92))
    env.step(action(env, "Ni", .04))
    assert decoded(env) == {("Ni", Fraction(".04")), ("P", Fraction(".04"))}


def test_constraint_protocol_preserves_value_order_and_physical_amounts():
    class Filter:
        def filter_actions(self, **kwargs):
            assert kwargs["allowed_units"] == [2, 0, 1]
            assert kwargs["action_amounts"] == [2.0, 0.0, 1.0]
            assert kwargs["chemical_amounts"] == {}
            assert kwargs["amount_quantum"] == 1
            assert kwargs["prior_groups"] == [{"Sr": 1}]
            kwargs["units_map"]["corrupt"] = 100
            return kwargs["actions"]
    env = make(dict(species_set=["Fe", "O"], amount_set=[2, 0, 1], n_steps=1), filters=[Filter()])
    assert len(env.allowed_actions(prior_groups=[{"Sr": 1}])) == 6
    assert env.cation_fractions() == {}


def test_optional_filters_cannot_add_actions_and_empty_list_preserves_oxygen_requirement():
    class Filter:
        def filter_actions(self, **kwargs):
            return [((1.0, 0.0), (1.0,))]
    env = make(dict(species_set=["Fe", "O"], amount_set=[1], n_steps=1,
                    element_bounds={"O": [1, 1]}), filters=[Filter()])
    with pytest.raises(ValueError, match="subset"):
        env.allowed_actions()
    env = make(dict(species_set=["Fe", "O"], amount_set=[1], n_steps=1,
                    element_bounds={"O": [1, 1]}), filters=[])
    assert decoded(env) == {("O", Fraction(1))}


def test_tiny_decimal_lattice_is_exact_and_checkpoint_schema_has_domain_identity():
    env = make(dict(species_set=["O"], amount_set=[0.0000001], n_steps=3, total=.0000003))
    for _ in range(3):
        env.step(env.allowed_actions()[0])
    assert env.terminal_cation_fractions() == {"O": .0000003}
    assert env.remaining_units == 0
    assert env.design_schema()["state"]["quantum"] == "0.0000001"
    assert env.design_schema()["state"]["feature_dim"] == len(env.current_state_features())


@pytest.mark.parametrize("override", [
    {"n_steps": True}, {"unique": "true"}, {"amount_set": []}, {"amount_set": [1, 1.0]},
    {"amount_set": [float("nan")]}, {"element_bounds": {"X": [0, 1]}},
    {"element_bounds": {"Fe": [2, 1]}}, {"amount_set": [.2], "total": .3},
    {"n_steps": 3, "unique": True}, {"amount_set": [.2], "total": 1},
])
def test_invalid_or_initially_infeasible_domains_reject_early(override):
    with pytest.raises(ValueError):
        make({**dict(species_set=["Fe", "O"], amount_set=[0, 1], n_steps=2), **override})


@pytest.mark.parametrize("removed", ["fraction_set", "ratio_set", "n_components", "total_units",
                                   "episode_style", "token_requirements", "sites", "constraint_filter"])
def test_removed_amount_parameters_do_not_enter_the_new_domain(removed):
    with pytest.raises(ValueError, match="Unsupported composition group fields"):
        make(dict(species_set=["Fe"], amount_set=[1], n_steps=1, **{removed: 1}))


def test_scenario_scale_grids_and_73_species_catalog_have_valid_complete_rollouts():
    scenarios = [
        dict(species_set=["O"] + [f"M{i}" for i in range(72)], amount_set=list(range(10)),
             n_steps=5, element_bounds={"O": [1, 9]}),
        dict(species_set=["SiO2", "B2O3", "Al2O3", "Na2O", "K2O", "CaO", "MgO"],
             amount_set={"min": 0, "max": 1, "step": .005}, n_steps=7, total=1,
             element_bounds={"SiO2": [.5, .85], "B2O3": [.05, .25], "Al2O3": [0, .15],
                             "Na2O": [0, .2], "K2O": [0, .1], "CaO": [0, .15], "MgO": [0, .1]}),
        dict(species_set=["Mg", "Al", "Ca", "Sc", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Sr", "Y", "In", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
             amount_set={"min": .01, "max": .96, "step": .01}, n_steps=5, unique=True, total=1),
    ]
    for spec in scenarios:
        env = make(spec)
        while not env.is_terminal:
            choices = env.allowed_actions()
            assert choices
            env.step(choices[0])
        final = env.terminal_cation_fractions()
        if "total" in spec:
            assert sum(Fraction(str(v)) for v in final.values()) == 1
        else:
            assert 1 <= final.get("O", 0) <= 9
        if spec.get("unique"):
            assert len([v for v in final.values() if v > 0]) == 5
