"""Configured scientific filters apply throughout preparation, learning and generation."""
from __future__ import annotations

from collections import defaultdict
import sys

import numpy as np
import pytest
import yaml


class OxygenOnly:
    def __init__(self, cfg, *, env=None):
        self.calls = 0

    def filter_actions(self, *, actions, species_set, **context):
        self.calls += 1
        return [action for action in actions if species_set[int(np.argmax(action[0]))] == "O"]


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
@pytest.mark.parametrize("filter_declaration", ["configured", "absent", "empty"])
def test_filter_list_applies_consistently_in_warmup_learning_and_generation(
    tmp_path, monkeypatch, method, filter_declaration,
):
    from rl_matdesign import experiment, training
    from rl_matdesign.amounts import ChemicalAmountsGroup
    import rl_matdesign.env_multigroup as env_module

    class CheapDescriptorEnv(env_module.MultiGroupEnv):
        def __init__(self, **kwargs):
            super().__init__(state_featurizer=lambda formula: np.asarray([len(formula)], dtype=float), **kwargs)

    monkeypatch.setattr(env_module, "MultiGroupEnv", CheapDescriptorEnv)
    config = {
        "groups": [{"name": "oxide", "kind": "composition", "species_set": ["Fe", "O"],
                    "amount_set": [1, 2], "n_steps": 2, "element_bounds": {"O": [1, 4]}}],
        "properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}],
        "method": method,
        "dqn_warmup_eps": 2, "dqn_num_train_eps": 1, "dqn_batch_size": 1,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 4, "dqn_buffer_size": 8,
        "pg_warmup_eps": 2, "pg_num_iters": 1, "pg_batch_eps": 1,
        "num_gen_eps": 1, "exploration_gen_eps": 0, "max_gen_attempts": 2,
    }
    if filter_declaration != "absent":
        config["groups"][0]["filters"] = ([{"type": f"{__name__}:OxygenOnly"}]
                                          if filter_declaration == "configured" else [])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_experiment", "--config", str(path),
                                      "--out", str(tmp_path / "run"), "--device", "cpu"])
    filter_counts, calls = {}, {}
    random_rollout_counts = defaultdict(list)
    initial_action_sets = defaultdict(list)
    active_phase = [None]
    real_rollout = training._rollout_random_episode
    real_allowed_actions = ChemicalAmountsGroup.allowed_actions

    def observe_random_rollout(env):
        random_rollout_counts[active_phase[0]].append(len(env._inners[0].filters))
        return real_rollout(env)

    def observe_allowed_actions(group, *args, **kwargs):
        actions = real_allowed_actions(group, *args, **kwargs)
        if active_phase[0] is not None and group.counter == 0:
            initial_action_sets[active_phase[0]].append({
                (group.species_set[int(np.argmax(label))], float(group.action_values[int(np.argmax(value))]))
                for label, value in actions})
        return actions

    monkeypatch.setattr(training, "_rollout_random_episode", observe_random_rollout)
    monkeypatch.setattr(ChemicalAmountsGroup, "allowed_actions", observe_allowed_actions)

    def watch(name, function):
        def invoke(*args, **kwargs):
            env = kwargs.get("env", args[0] if args else None)
            phase = "pg_scaler" if name == "_fit_scaler_from_warmup" else name
            filters = env._inners[0].filters
            filter_counts[phase] = len(filters)
            before = sum(item.calls for item in filters)
            active_phase[0] = phase
            result = function(*args, **kwargs)
            calls[phase] = sum(item.calls for item in filters) - before
            # An absent optional screen never removes intrinsic oxygen bounds.
            assert env.terminal_cation_fractions().get("O", 0) >= 1
            active_phase[0] = None
            return result
        return invoke

    for name in ("train_dqn_online", "_fit_scaler_from_warmup", "train_pg", "generate_candidates"):
        monkeypatch.setattr(experiment, name, watch(name, getattr(experiment, name)))
    experiment.main()

    phases = (["train_dqn_online"] if method == "dqn" else ["pg_scaler", "train_pg"]) + ["generate_candidates"]
    configured = filter_declaration == "configured"
    assert filter_counts == {phase: int(configured) for phase in phases}
    expected_actions = {("O", 1.), ("O", 2.)}
    if not configured:
        expected_actions |= {("Fe", 1.), ("Fe", 2.)}
    for phase in phases:
        assert (calls[phase] > 0) is configured
        assert initial_action_sets[phase]
        assert all(actions == expected_actions for actions in initial_action_sets[phase])
    warmup_phase = "train_dqn_online" if method == "dqn" else "pg_scaler"
    assert random_rollout_counts[warmup_phase] == [int(configured)] * 2
