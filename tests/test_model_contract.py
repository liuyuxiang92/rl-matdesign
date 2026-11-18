"""Reject semantically incompatible RL artifacts even when shapes agree."""
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.utils.model_contract import (
    bind_scaler, environment_contract, load_artifact, save_weights, validate_scaler,
)


def descriptor(_formula):
    return np.zeros(2)


def make_env(species=("Ni", "Fe", "Co"), **kwargs):
    return ChemicalAmountsGroup(species_set=species, amount_set=[.25, .50, .75],
                          total=1, n_steps=2,
                          state_featurizer=descriptor, **kwargs)


def test_equal_dimensions_do_not_allow_different_species_or_action_order(tmp_path):
    original = make_env()
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "qnet.pt"
    save_weights(path, model, original, "qnet")
    assert load_artifact(path, make_env())["environment_contract"] == environment_contract(original)
    for different in (make_env(("Ni", "Fe", "Cu")), make_env(("Fe", "Ni", "Co"))):
        with pytest.raises(ValueError, match="different state/action contract"):
            load_artifact(path, different)


def test_bounds_and_uniqueness_are_part_of_artifact_identity(tmp_path):
    path = tmp_path / "policy.pt"
    save_weights(path, torch.nn.Linear(2, 1), make_env(), "policy")
    for changed in (make_env(unique=True),
                    make_env(element_bounds={"Ni": (0.25, 0.50)})):
        with pytest.raises(ValueError, match="incompatible"):
            load_artifact(path, changed)


def test_unversioned_weights_and_scaler_fail_explicitly(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(torch.nn.Linear(2, 1).state_dict(), path)
    with pytest.raises(ValueError, match="missing state/action contract"):
        load_artifact(path, make_env())
    scaler = StandardScaler().fit(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="missing state/action contract"):
        validate_scaler(scaler, make_env())
    bind_scaler(scaler, make_env())
    validate_scaler(scaler, make_env())
    with pytest.raises(ValueError, match="different state/action contract"):
        validate_scaler(scaler, make_env(("Cu", "Fe", "Co")))


def test_contract_does_not_depend_on_episode_progress():
    env = make_env()
    before = environment_contract(env)
    env.step(env.allowed_actions()[0])
    assert environment_contract(env) == before


@pytest.mark.parametrize("first,second", [(True, 1), (1, 1.0), (1, "1")])
def test_contract_does_not_use_python_coercive_equality(first, second):
    from rl_matdesign.utils.model_contract import require_contract
    left = {"complete": True, "design_spec": {"value": first}}
    right = {"complete": True, "design_spec": {"value": second}}
    with pytest.raises(ValueError, match="different state/action contract"):
        require_contract(left, right)


@pytest.mark.parametrize("change", [
    {"parent": ["B", "A", "C"]},
    {"require_change": ["peptide"]},
    {"budget": 1},
])
def test_peptide_contract_includes_initial_positions_and_edit_rules(change, tmp_path):
    from rl_matdesign.peptide_design import compile_groups
    from rl_matdesign.env_multigroup import MultiGroupEnv
    def create(**updates):
        structure = {"builder": "peptide", "parent": updates.get("parent", ["A", "B", "C"])}
        if "require_change" in updates:
            structure["require_change"] = updates["require_change"]
        cfg = {"structure": structure, "groups": [{"name": "n_terminal", "kind": "categorical",
            "choices": [{"name": "treatment", "values": ["H", "Ac"]}]},
            {"name": "peptide", "kind": "categorical",
            "labels": [1, 2, 3], "values": ["A", "B", "C"], "n_steps": updates.get("budget", 2)}]}
        groups, adapter = compile_groups(cfg)
        return MultiGroupEnv(groups=groups, candidate_adapter=adapter)
    initial, changed = create(), create(**change)
    path = tmp_path / "model.pt"
    save_weights(path, torch.nn.Linear(2, 1), initial, "policy")
    with pytest.raises(ValueError, match="different state/action contract"):
        load_artifact(path, changed)
