"""Integer-valued chemical amounts use the same additive domain as fractions."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.categorical import CategoricalGroup
from rl_matdesign.configuration import load_config
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.utils.rollout_errors import InfeasibleEpisode


def descriptor(_):
    return np.zeros(2)


def make_env(**kwargs):
    spec = {"species_set": ["Fe", "O"], "amount_set": [0, 1, 2], "n_steps": 3, "element_bounds": {"O": [1, 4]}}
    filters = kwargs.pop("filters", [])
    spec.update(kwargs)
    return ChemicalAmountsGroup(spec, state_featurizer=descriptor, filters=filters)


def action(env, label, amount):
    i = env.species_set.index(label)
    j = next(j for j, code in enumerate(env.action_values) if Fraction(code) == Fraction(amount))
    return tuple(np.eye(len(env.species_set))[i]), tuple(np.eye(len(env.action_values))[j])


def decoded(env):
    return {(env.species_set[int(np.argmax(a))], int(Fraction(env.action_values[int(np.argmax(b))]))) for a,b in env.allowed_actions()}


def test_every_integer_prefix_mask_matches_independent_complete_history_enumeration():
    env = make_env()
    menu = list(product(["Fe", "O"], [0, 1, 2]))
    valid = [history for history in product(menu, repeat=3) if 1 <= sum(v for label,v in history if label=="O") <= 4]
    prefixes = {history[:length] for history in valid for length in range(4)}
    for prefix in prefixes:
        env.initialize()
        for label, amount in prefix:
            env.step(action(env, label, amount))
        expected = {history[len(prefix)] for history in valid if history[:len(prefix)]==prefix} if len(prefix)<3 else set()
        assert decoded(env) == expected


def test_oxygen_can_repeat_or_occur_first_and_final_bounds_still_hold():
    env = make_env()
    for label, amount in [("O",1),("O",1),("Fe",2)]:
        env.step(action(env,label,amount))
    assert env.terminal_cation_fractions() == {"Fe":2,"O":2}
    env.initialize()
    for _ in range(2):
        env.step(action(env,"Fe",2))
    assert decoded(env) == {("O",1),("O",2)}


def test_absolute_counts_and_zero_pick_state_are_visible_without_descriptor_information():
    first, second = make_env(), make_env()
    first.step(action(first,"Fe",0))
    second.step(action(second,"O",0))
    assert first.cation_fractions() == {"Fe":0} and second.cation_fractions() == {"O":0}
    assert not np.array_equal(first.current_state_features(),second.current_state_features())
    first.initialize(); second.initialize()
    for label,amount in [("Fe",2),("O",1)]:
        first.step(action(first,label,amount))
    for label,amount in [("O",1),("Fe",2)]:
        second.step(action(second,label,amount))
    np.testing.assert_array_equal(first.current_state_features(),second.current_state_features())


@pytest.mark.parametrize("bad", [((0,0),(1,0,0)), ((1,1),(1,0,0)), ((1,0),(0,0,0)), ((1,0),(float("nan"),0,1))])
def test_invalid_actions_are_atomic(bad):
    env=make_env(); before=env.current_state_features().copy()
    with pytest.raises(ValueError):
        env.step(bad)
    np.testing.assert_array_equal(env.current_state_features(),before)
    assert env.counter==0 and env.path==[]


def test_specification_is_copied_and_noninteger_step_counts_reject():
    spec={"species_set":["Fe","O"],"amount_set":[0,1],"n_steps":2,"element_bounds":{"O":[1,1]}}
    env=ChemicalAmountsGroup(spec,state_featurizer=descriptor)
    spec["species_set"].clear(); spec["amount_set"].clear(); spec["element_bounds"]["O"][0]=0
    assert env.species_set==["Fe","O"] and env.element_bounds=={"O":[1,1]}
    for count in [0,True,1.5,"2"]:
        with pytest.raises(ValueError):
            make_env(n_steps=count)


class RejectAll:
    def filter_actions(self, **kwargs):
        return []


def test_empty_filter_list_preserves_final_oxygen_requirement():
    env=make_env(filters=[RejectAll()])
    assert env.allowed_actions()==[]
    with pytest.raises(InfeasibleEpisode):
        env.sample_random_action()
    env=make_env(filters=[])
    for _ in range(2):
        env.step(action(env,"Fe",2))
    assert decoded(env)=={("O",1),("O",2)}


@pytest.mark.parametrize("kind",["composition","categorical"])
def test_filter_cannot_mutate_domain_or_prior_context(kind):
    class Mutate:
        def filter_actions(self,*,actions,species_set,action_values,prior_groups,**kwargs):
            species_set.clear(); action_values.clear(); prior_groups[0].clear()
            if "code_to_value" in kwargs:
                kwargs["code_to_value"].clear()
            return actions
    env=(make_env(filters=[Mutate()]) if kind=="composition" else CategoricalGroup({"choices":[{"name":"temperature","values":[0,100]}]},filters=[Mutate()]))
    labels,values=list(env.species_set),list(env.action_values)
    prior=[{"Fe":1}]
    assert env.allowed_actions(prior_groups=prior)==env.allowed_actions(prior_groups=prior)
    assert env.species_set==labels and env.action_values==values and prior==[{"Fe":1}]


@pytest.mark.parametrize("kind",["composition","categorical"])
@pytest.mark.parametrize("result",[None,"string","duplicates","new_action","unhashable"])
def test_invalid_filter_results_are_programming_errors(kind,result):
    class Invalid:
        def filter_actions(self,*,actions,**kwargs):
            if result=="duplicates": return [actions[0],actions[0]]
            if result=="new_action": return actions+[((0,),(0,))]
            if result=="unhashable": return [list(actions[0])]
            return result
    env=(make_env(filters=[Invalid()]) if kind=="composition" else CategoricalGroup({"choices":[{"name":"temperature","values":[0,100]}]},filters=[Invalid()]))
    with pytest.raises(ValueError): env.allowed_actions()


def test_multigroup_prior_amounts_remain_raw_and_field_zero_survives():
    seen=[]
    class Prior:
        def filter_actions(self,*,actions,prior_groups,**kwargs):
            seen.append(deepcopy(prior_groups)); return actions if prior_groups==[{"Fe":2,"O":4}] else []
    env=MultiGroupEnv(groups=[
        {"name":"oxide","kind":"composition","species_set":["Fe","O"],"amount_set":[2,4],"n_steps":2},
        {"name":"conditions","kind":"categorical","choices":[{"name":"temperature","values":[0]}]}],state_featurizer=descriptor)
    env._inners[1].filters=[Prior()]
    inner=env._inners[0]
    for label,amount in [("Fe",2),("O",4)]: env.step(env._to_union(action(inner,label,amount),inner))
    env.step(env.allowed_actions()[0])
    assert seen and all(prior==[{"Fe":2,"O":4}] for prior in seen)
    assert env.terminal_cation_fractions()=={"oxide":{"Fe":2,"O":4},"conditions":{"temperature":0}}


def test_three_production_oxide_configs_require_final_raw_positive_oxygen():
    root=Path(__file__).resolve().parents[1]
    paths=sorted(path for path in (root/"configs").glob("oxides_*.yaml") if path.name!="oxides_smoke.yaml")
    assert len(paths)==3
    for path in paths:
        cfg=load_config(str(path)); spec=cfg["groups"][0]
        assert spec["kind"]=="composition" and spec["element_bounds"]["O"]==[1,9]
        env=ChemicalAmountsGroup(spec,state_featurizer=descriptor)
        for _ in range(4): env.step(action(env,"Fe",1))
        assert decoded(env)=={("O",value) for value in range(1,10)}
