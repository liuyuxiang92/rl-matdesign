"""Runner boundaries preserve exploration evidence and reject incompatible resumes."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.config_schema import resolve_config
from rl_matdesign.experiment import _validate_exploration_resume
from rl_matdesign.exploration import ExplorationController, exploration_profile
from rl_matdesign.utils.execution_contract import (
    execution_contract, generation_sampling_contract, require_execution_contract,
)
from rl_matdesign.utils.model_contract import load_artifact, save_weights


def _saved():
    controller = ExplorationController(.7, True)
    controller.seed_seen(["pilot-candidate"])
    controller.observe("training-candidate", 1.5)
    controller.update_entropy(.4)
    return {"exploration_state": controller.state_dict()}


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_unchanged(before):
    after = _rng_snapshot()
    assert after[0] == before[0]
    assert after[1][0] == before[1][0]
    np.testing.assert_array_equal(after[1][1], before[1][1])
    assert after[1][2:] == before[1][2:]
    assert torch.equal(after[2], before[2])


@pytest.mark.parametrize("corruption", ["missing", "null", "malformed", "target", "profile", "settings", "ledger"])
def test_resume_validation_rejects_before_mutating_saved_state_or_rng(corruption):
    saved = _saved()
    if corruption == "missing":
        saved.pop("exploration_state")
    elif corruption == "null":
        saved["exploration_state"] = None
    elif corruption == "malformed":
        saved["exploration_state"] = []
    elif corruption == "target":
        saved["exploration_state"]["target"] = float("nan")
    elif corruption == "profile":
        saved["exploration_state"]["profile"]["version"] += 1
    elif corruption == "settings":
        saved["exploration_state"]["config"]["adaptive"] = False
    else:
        saved["exploration_state"]["seen_keys"] = ["duplicate", "duplicate"]
    before = json.dumps(saved, sort_keys=True)
    rng = _rng_snapshot()
    with pytest.raises(ValueError):
        _validate_exploration_resume(saved, {"exploration_target": .7, "exploration_adaptive": True})
    assert json.dumps(saved, sort_keys=True) == before
    _assert_rng_unchanged(rng)


@pytest.mark.parametrize("changes", [{"exploration_target": .4}, {"exploration_adaptive": False}])
def test_resume_validation_binds_current_requested_settings(changes):
    saved = _saved()
    before = deepcopy(saved)
    rng = _rng_snapshot()
    config = {"exploration_target": .7, "exploration_adaptive": True, **changes}
    with pytest.raises(ValueError, match="configuration changed"):
        _validate_exploration_resume(saved, config)
    assert saved == before
    _assert_rng_unchanged(rng)


def test_resume_validation_accepts_full_continuation_without_consuming_state():
    saved = _saved()
    before, rng = deepcopy(saved), _rng_snapshot()
    _validate_exploration_resume(saved, {"exploration_target": .7, "exploration_adaptive": True})
    assert saved == before
    _assert_rng_unchanged(rng)


@pytest.mark.parametrize("role", ["qnet", "policy"])
def test_frozen_weights_retain_primitive_novelty_archive_for_safe_loading(tmp_path, role):
    env = ChemicalAmountsGroup(species_set=["Ni", "Fe", "Co"], amount_set=[.25, .5, .75],
                               total=1, n_steps=2, state_featurizer=lambda _: np.zeros(2))
    env.exploration_seen_keys = ("pilot-candidate", "training-candidate")
    model = torch.nn.Linear(2, 1)
    path = tmp_path / f"{role}.pt"
    save_weights(path, model, env, role)
    # This is the safe loader used for generation, not checkpoint pickle loading.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["exploration_seen_keys"] == list(env.exploration_seen_keys)
    assert all(type(key) is str for key in raw["exploration_seen_keys"])
    assert "exploration_state" not in raw
    loaded = load_artifact(path, env, checkpoint=False)
    assert loaded["exploration_seen_keys"] == raw["exploration_seen_keys"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded[f"{role}_state"][name], tensor)
    generation = ExplorationController(.3)
    generation.seed_seen(loaded["exploration_seen_keys"])
    generation.observe("training-candidate", 3.0)
    assert generation.state_dict()["window"]["unique"] == 0
    assert generation.attempts == 1


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"input_files": [], "runtime": {}}))
    return path


@pytest.mark.parametrize("method", ["dqn", "a2c"])
@pytest.mark.parametrize("changes,field", [({"exploration_target": .2}, "exploration_target"),
                                            ({"exploration_adaptive": False}, "exploration_adaptive")])
def test_shared_exploration_settings_bind_learning_without_changing_oracle(manifest, method, changes, field):
    config = resolve_config({"method": method}).config
    original = execution_contract(config, object(), manifest, "cpu")
    changed = execution_contract({**config, **changes}, object(), manifest, "cpu")
    assert original["evaluation"] == changed["evaluation"]
    assert original["learning"]["exploration_profile"] == exploration_profile()
    with pytest.raises(ValueError, match=f"learning.{field}"):
        require_execution_contract(original, changed)


@pytest.mark.parametrize("method", ["dqn", "a2c"])
def test_generation_target_changes_sampling_but_preserves_training_identity(manifest, method):
    config = resolve_config({"method": method}).config
    changed = {**config, "gen_exploration_target": .1}
    assert execution_contract(config, object(), manifest, "cpu") == execution_contract(
        changed, object(), manifest, "cpu")
    assert generation_sampling_contract(config) != generation_sampling_contract(changed)


def test_internal_profile_change_invalidates_learning_and_generation_identity(manifest, monkeypatch):
    import rl_matdesign.exploration as exploration
    config = resolve_config({}).config
    original = execution_contract(config, object(), manifest, "cpu")
    sampling = generation_sampling_contract(config)
    profile = exploration_profile()
    profile["window_attempts"] += 1
    monkeypatch.setattr(exploration, "exploration_profile", lambda: deepcopy(profile))
    changed = execution_contract(config, object(), manifest, "cpu")
    assert original["evaluation"] == changed["evaluation"]
    with pytest.raises(ValueError, match="learning.exploration_profile"):
        require_execution_contract(original, changed)
    assert generation_sampling_contract(config) != sampling
