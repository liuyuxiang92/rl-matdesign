"""Cross-formula waves retain sequential learning and budget semantics."""
from __future__ import annotations

import collections
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign import training
from rl_matdesign.configuration import _TerminalReward, _FAILED_CANDIDATE_REWARD
from rl_matdesign.env import EpisodeStep
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator


class Predictor:
    def __init__(self, size=4):
        self.size = size
        self.waves = []
        self.scalar = []
        self.raw = []
        self.observe = lambda: None

    def evaluation_batch_size(self):
        return self.size

    def score(self, candidate):
        name = next(iter(candidate))
        if name == "X":
            raise RuntimeError("failed structure realization")
        if name == "nan":
            return float("nan"), 0.0
        return float(ord(name) - ord("A") + 1), 0.0

    def predict(self, candidate):
        self.scalar.append(next(iter(candidate)))
        return self.score(candidate)

    def batch_predict(self, candidates, *, return_exceptions=False):
        assert return_exceptions
        self.waves.append([next(iter(c)) for c in candidates])
        self.observe()
        results = []
        for candidate in candidates:
            try:
                results.append(self.score(candidate))
            except Exception as exc:
                results.append(exc)
        return results

    def predict_raw(self, candidate):
        self.raw.append(next(iter(candidate)))
        mean, std = self.score(candidate)
        return 10 * mean, std


class ScriptedEnv:
    """Two-step paths with a prescribed terminal identity and real reward hook."""
    n_components = 2
    phase_filter = None

    def __init__(self, candidates, predictor):
        self.candidates = iter(candidates)
        self.starts = []
        self.path = []
        self.counter = 0
        self.reward_fn = _TerminalReward(predictor, "multi_group")

    def initialize(self):
        self.candidate = next(self.candidates)
        self.starts.append(self.candidate)
        self.counter = 0
        self.path = []

    def current_state_features(self):
        return np.array([float(self.counter), 1.0])

    def allowed_actions(self):
        return [((1.0,), (1.0,))]

    def sample_random_action(self):
        return self.allowed_actions()[0]

    def step(self, action):
        reward = self.reward_fn(self.terminal_cation_fractions()) if self.counter == 1 else 0.0
        self.path.append(EpisodeStep(
            state_material_features=self.current_state_features(),
            state_step_onehot=tuple(float(i == self.counter) for i in range(2)),
            action_elem_onehot=action[0], action_comp_onehot=action[1],
            reward=reward, allowed_actions=self.allowed_actions(),
        ))
        self.counter += 1

    def terminal_cation_fractions(self):
        return {self.candidate: 1.0}

    def terminal_comp_key(self):
        return ((self.candidate, 1),)

    @property
    def terminal_formula(self):
        return self.candidate + "1"


def run_dqn(env, tmp_path, **overrides):
    kwargs = dict(
        env=env, elem_feats_scaled=np.array([[0.1, 0.2]]), fraction_set=["0.5"],
        device=torch.device("cpu"), hidden_dim=4, n_warmup_eps=2,
        dqn_num_train_eps=3, dqn_buffer_size=100, batch_size=1,
        dqn_grad_steps_per_ep=1, dqn_target_update_freq=2,
        dqn_eps_anneal_eps=10,
        checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 2},
    )
    kwargs.update(overrides)
    return training.train_dqn_online(**kwargs)


def saved(tmp_path):
    return torch.load(tmp_path / "checkpoint.pt", weights_only=False)


def optimizer_steps(state):
    return {int(item["step"].item()) for item in state["opt_state"]["state"].values()}


@pytest.mark.parametrize("mode", ["mc", "bootstrap"])
def test_dqn_waves_keep_duplicates_failures_order_and_checkpoints(tmp_path, mode):
    predictor = Predictor(8)
    env = ScriptedEnv(["A", "A", "B", "X", "A", "C", "C", "D", "E"], predictor)
    _, _, rows = run_dqn(env, tmp_path, train_count_duplicates=False, dqn_target_mode=mode)
    assert env.starts == ["A", "A", "B", "X", "A", "C", "C", "D", "E"]
    assert predictor.waves == [["A", "A"], ["B"], ["X", "A"], ["C"], ["C", "D"], ["E"]]
    assert not predictor.scalar
    assert [r["terminal_reward"] for r in rows] == [1, 1, 2, 1, 3, 3, 4, 5]
    online = [r for r in rows if r["phase"] == "dqn_train"]
    assert [r["budget_counted"] for r in online] == [False, True, False, True, True]
    assert [r["epsilon_used"] for r in online] == pytest.approx([1, 1, .9, .9, .7])
    state = saved(tmp_path)
    assert state["episodes_completed"] == 3
    assert state["training_episodes_completed"] == 5
    assert state["training_budget_state"]["attempted"] == 8
    assert state["training_budget_state"]["counted"] == 5
    assert optimizer_steps(state) == {5}
    assert [r["reward"] for r in state["buffer"][1::2]] == [1, 1, 2, 1, 3, 3, 4, 5]
    # Checkpoints at episodes 2 and 4 cannot contain prefetched unseen candidates.
    two = torch.load(tmp_path / "checkpoint-ep2.pt", weights_only=False)
    assert two["training_episodes_completed"] == 2
    assert len(two["buffer"]) == 10
    assert optimizer_steps(two) == {2}


def test_dqn_duplicate_guard_never_prefetches_past_last_allowed_update(tmp_path):
    predictor = Predictor(20)
    env = ScriptedEnv(["A", "B", "C", "A", "A", "A", "D"], predictor)
    with pytest.raises(RuntimeError, match="3 consecutive duplicate"):
        run_dqn(env, tmp_path, n_warmup_eps=3, dqn_num_train_eps=5,
                train_count_duplicates=False, train_max_duplicate_attempts=3,
                checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 10})
    assert env.starts == ["A", "B", "C", "A", "A", "A"]
    assert predictor.waves == [["A", "B", "C"], ["A", "A", "A"]]
    assert optimizer_steps(saved(tmp_path)) == {3}
    assert saved(tmp_path)["training_budget_state"]["consecutive_duplicates"] == 3


def test_dqn_resume_preserves_actual_updates_and_seen_budget(tmp_path):
    predictor = Predictor(4)
    qnet, scaler, _ = run_dqn(ScriptedEnv(["A", "B", "A", "C"], predictor), tmp_path,
                             dqn_num_train_eps=1, train_count_duplicates=False)
    state = saved(tmp_path)
    target = copy.deepcopy(qnet)
    target.load_state_dict(state["target_net_state"])
    optimizer = torch.optim.Adam(qnet.parameters())
    optimizer.load_state_dict(state["opt_state"])
    resume = dict(
        scaler=scaler, qnet=qnet, target_net=target, optimizer=optimizer,
        buffer=collections.deque(state["buffer"], maxlen=100),
        start_ep=state["episodes_completed"],
        training_episodes_completed=state["training_episodes_completed"],
        training_budget_state=state["training_budget_state"], eps=state["eps"],
    )
    _, _, rows = run_dqn(ScriptedEnv(["C", "D"], predictor), tmp_path,
                         dqn_num_train_eps=2, train_count_duplicates=False, resume_state=resume)
    assert [r["budget_counted"] for r in rows] == [False, True]
    assert [r["episode"] for r in rows] == [3, 4]
    assert optimizer_steps(saved(tmp_path)) == {4}
    assert saved(tmp_path)["training_budget_state"]["counted"] == 4


def test_scalar_default_never_defers_or_calls_batch(tmp_path):
    predictor = Predictor(1)
    env = ScriptedEnv(["A", "X", "B", "C"], predictor)
    _, _, rows = run_dqn(env, tmp_path, n_warmup_eps=2, dqn_num_train_eps=1)
    assert predictor.scalar == ["A", "X", "B", "C"]
    assert predictor.waves == []
    assert [r["terminal_reward"] for r in rows] == [1, 2, 3]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4))

    def forward(self, _state, _step, elements, _fractions):
        return elements @ self.logits


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state, _step):
        return self.value.expand(state.shape[0], 1)


def pg_setup(monkeypatch, sequence, size=4, method="reinforce"):
    from rl_matdesign.env import CompositionEnv
    predictor = Predictor(size)
    # Real env paths/log probabilities; predictor sees slot names A-D through
    # conversion so reward identity is easy to inspect in expected sequences.
    species = ["Na", "K", "Ca", "Sr"]
    def candidate(payload):
        for i, name in enumerate(species):
            if ((isinstance(payload, dict) and payload.get(name) == 1)
                    or (isinstance(payload, str) and payload.startswith(name + "1"))):
                return {chr(ord("A") + i): 1.0}
        raise AssertionError(payload)
    adapter = _TerminalReward(predictor, "fraction")
    adapter._candidate = candidate
    env = CompositionEnv(species_set=species, fraction_set=["1.00"], n_components=1,
                         total_units=1, state_featurizer=lambda _: np.array([0.0]),
                         reward_fn=adapter)
    choices = iter(sequence)
    sampled_params = []
    policy = Policy()

    def rollout(**kwargs):
        index = next(choices)
        sampled_params.append(policy.logits.detach().clone())
        env.initialize()
        action = next(a for a in env.allowed_actions() if int(np.argmax(a[0])) == index)
        env.step(action)

    monkeypatch.setattr(training, "_rollout_pg_episode", rollout)
    return predictor, sampled_params, dict(
        policy=policy, value_net=Value() if method == "a2c" else None, env=env,
        scaler=StandardScaler().fit([env.current_state_features()]), elem_feats_scaled=np.eye(4),
        fraction_set=["1.00"], device=torch.device("cpu"), num_iters=1, batch_eps=4,
        pg_repeat_penalty_coef=.2, rl_method=method,
    )


@pytest.mark.parametrize("method", ["reinforce", "a2c"])
def test_pg_preserves_onpolicy_batches_and_duplicate_learning(monkeypatch, method):
    predictor, sampled, kwargs = pg_setup(monkeypatch, [0, 0, 1, 1, 2, 2, 3], method=method)
    rows = training.train_pg(**kwargs, train_count_duplicates=False)
    updates = [r for r in rows if r["phase"] == "pg_train"]
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    assert [r["batch_eps_actual"] for r in updates] == [4, 3]
    assert predictor.waves == [["A", "A", "B", "B"], ["C", "C"], ["D"]]
    assert [r["terminal_reward"] for r in episodes] == [1, 1, 2, 2, 3, 3, 4]
    assert episodes[1]["visit_count_before"] == 1
    assert all(torch.equal(sampled[0], p) for p in sampled[:4])
    assert all(torch.equal(sampled[4], p) for p in sampled[4:])
    assert not torch.equal(sampled[0], sampled[4])
    assert updates[-1]["training_budget_used"] == 4


def test_pg_attempt_cap_trains_partial_batch_and_never_prefetches(monkeypatch, tmp_path):
    predictor, sampled, kwargs = pg_setup(monkeypatch, [0, 1, 2, 3])
    with pytest.raises(RuntimeError, match="budget is incomplete"):
        training.train_pg(**kwargs, train_count_duplicates=False, max_train_attempts=2,
                          checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 0})
    assert predictor.waves == [["A", "B"]]
    assert len(sampled) == 2
    assert saved(tmp_path)["training_budget_state"]["attempted"] == 2
    assert saved(tmp_path)["iteration"] == 1


def test_training_stops_after_exactly_eleven_consecutive_evaluation_failures(tmp_path):
    predictor = Predictor(50)
    env = ScriptedEnv(["X"] * 20, predictor)
    original = env.reward_fn
    with pytest.raises(RuntimeError, match="11 consecutive candidates"):
        run_dqn(env, tmp_path, n_warmup_eps=20)
    assert len(env.starts) == 11
    assert predictor.waves == [["X"] * 11]
    assert env.reward_fn is original


def test_generation_uses_snapshot_metadata_and_counts_failure_attempts(monkeypatch):
    predictor = Predictor(10)
    env = ScriptedEnv(["X", "A", "B", "A", "C", "D"], predictor)
    monkeypatch.setattr(training, "_rollout_policy_episode",
                        lambda **kwargs: training._rollout_random_episode(kwargs["env"]))
    checked = []
    predictor.composition_formula = lambda comp: next(iter(comp)) + "2O3"
    predictor.check_phase = lambda comp: (checked.append(next(iter(comp))) or (True, "test"))
    predictor.per_objective_stats = lambda comp: {"property": predictor.score(comp)}
    predictor.reward_details = lambda comp: {"type": "weighted_sum"}
    rows = training.generate_candidates(
        env=env, predictor=predictor, qnet=torch.nn.Linear(1, 1),
        scaler=StandardScaler().fit([[0.0, 1.0], [1.0, 1.0]]), device=torch.device("cpu"),
        elem_feats_scaled=np.array([[0.1, 0.2]]), fraction_set=["0.5"],
        n_exploit=2, n_explore=1, max_attempts=3,
    )
    assert env.starts == ["X", "A", "B", "A", "C"]
    assert predictor.waves == [["X", "A"], ["B"], ["A"], ["C"]]
    assert [r["formula"] for r in rows] == ["A2O3", "B2O3", "C2O3"]
    assert [r["reward"] for r in rows] == [1, 2, 3]
    assert [r["dp_mean"] for r in rows] == [10, 20, 30]
    assert [r["obj_property_mean"] for r in rows] == [1, 2, 3]
    assert checked == ["A", "B", "C"]


@pytest.mark.parametrize("env_type,payload,expected", [
    ("fraction", "Ni0.50Fe0.50O2H1", {"Ni": .5, "Fe": .5}),
    ("integer_ratio", "Fe3O2", {"Fe": .6, "O": .4}),
    ("multi_group", {"cation": {"Ni": .5, "Fe": .5}}, {"cation": {"Ni": .5, "Fe": .5}}),
])
def test_adapter_scalar_and_batch_candidate_conversion_matches(env_type, payload, expected):
    seen = []
    predictor = SimpleNamespace(predict=lambda c: (seen.append(c) or (2.0, 0.0)))
    adapter = _TerminalReward(predictor, env_type, "O2H1")
    assert adapter(payload) == 2.0
    assert adapter.evaluate_many([payload]) == [2.0]
    assert seen == [expected, expected]


def test_adapter_keeps_invalid_integer_penalty_and_legacy_batch_fallback():
    predictor = SimpleNamespace(predict=lambda _: pytest.fail("invalid formula evaluated"))
    adapter = _TerminalReward(predictor, "integer_ratio")
    assert adapter("not_a_formula") == _FAILED_CANDIDATE_REWARD
    assert adapter.evaluate_many(["not_a_formula"]) == [_FAILED_CANDIDATE_REWARD]
    predictor = Predictor(4)
    predictor.batch_predict = lambda _: pytest.fail("legacy batch API invoked")
    adapter = _TerminalReward(predictor, "multi_group")
    results = adapter.evaluate_many([{"A": 1}, {"X": 1}, {"B": 1}])
    assert results[0] == 1 and results[2] == 2
    assert isinstance(results[1], training.CandidateEvaluationFailed)
    assert predictor.scalar == ["A", "X", "B"]


def test_adapter_rejects_nonfinite_scalar_and_batch_rewards():
    adapter = _TerminalReward(Predictor(), "multi_group")
    with pytest.raises(training.CandidateEvaluationFailed, match="nonfinite"):
        adapter({"nan": 1})
    result = adapter.evaluate_many([{"A": 1}, {"nan": 1}])
    assert result[0] == 1
    assert isinstance(result[1], training.CandidateEvaluationFailed)


def test_capture_restores_reward_callback_on_rollout_error():
    predictor = Predictor(4)
    env = ScriptedEnv(["A"], predictor)
    evaluator = EpisodeBatchEvaluator(env)
    original = env.reward_fn
    def fail():
        raise ValueError("invalid action")
    with pytest.raises(ValueError, match="invalid action"):
        evaluator.next(fail, limit=1)
    assert env.reward_fn is original


@pytest.mark.parametrize("later_size", [1, 2])
def test_auto_batch_size_refreshes_only_after_draining_wave(tmp_path, later_size):
    predictor = Predictor(4)
    predictor.observe = lambda: setattr(predictor, "size", later_size)
    env = ScriptedEnv(["A", "B", "C", "D", "E", "F"], predictor)
    _, _, rows = run_dqn(env, tmp_path, n_warmup_eps=6, dqn_num_train_eps=0)
    assert [row["terminal_reward"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert predictor.waves == ([["A", "B", "C", "D"], ["E", "F"]]
                               if later_size == 2 else [["A", "B", "C", "D"]])
    assert predictor.scalar == ([] if later_size == 2 else ["E", "F"])


def test_pg_failed_candidates_refill_under_the_same_policy(monkeypatch):
    predictor, sampled, kwargs = pg_setup(monkeypatch, [0, 1, 2, 3, 1])
    original_score = predictor.score
    failed = []
    def score(candidate):
        if "B" in candidate and not failed:
            failed.append(True)
            raise RuntimeError("one failed realization")
        return original_score(candidate)
    predictor.score = score
    rows = training.train_pg(**kwargs)
    assert predictor.waves == [["A", "B", "C", "D"], ["B"]]
    assert all(torch.equal(sampled[0], p) for p in sampled)
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    assert [r["terminal_reward"] for r in episodes] == [1, 3, 4, 2]
    assert episodes[-1]["training_attempts"] == 4
    assert episodes[-1]["training_budget_used"] == 4


def test_pg_duplicate_guard_learns_last_duplicate_before_stopping(monkeypatch, tmp_path):
    predictor, sampled, kwargs = pg_setup(monkeypatch, [0, 0, 0, 1])
    with pytest.raises(RuntimeError, match="2 consecutive duplicate"):
        training.train_pg(**kwargs, train_count_duplicates=False,
                          train_max_duplicate_attempts=2,
                          checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 0})
    assert predictor.waves == [["A", "A"], ["A"]]
    assert len(sampled) == 3
    state = saved(tmp_path)
    assert state["training_budget_state"]["attempted"] == 3
    assert state["training_budget_state"]["consecutive_duplicates"] == 2
    assert state["iteration"] == 1


def test_training_success_resets_consecutive_failure_guard(tmp_path):
    predictor = Predictor(30)
    env = ScriptedEnv(["X"] * 8 + ["A"] + ["X"] * 8 + ["B", "C"], predictor)
    _, _, rows = run_dqn(env, tmp_path, n_warmup_eps=2, dqn_num_train_eps=0)
    assert [r["terminal_reward"] for r in rows] == [1, 2]
    assert env.starts == ["X"] * 8 + ["A"] + ["X"] * 8 + ["B"]
    assert rows[-1]["training_attempts"] == 2


def test_generation_charge_gate_uses_each_snapshot_formula(monkeypatch):
    import rl_matdesign.constraints.charge as charge
    predictor = Predictor(5)
    env = ScriptedEnv(["A", "B", "C", "D"], predictor)
    monkeypatch.setattr(training, "_rollout_policy_episode",
                        lambda **kwargs: training._rollout_random_episode(kwargs["env"]))
    checked = []
    monkeypatch.setattr(charge, "charge_neutral",
                        lambda formula, **kwargs: checked.append(formula) or formula != "A1")
    rows = training.generate_candidates(
        env=env, predictor=predictor, qnet=torch.nn.Linear(1, 1),
        scaler=StandardScaler().fit([[0.0, 1.0], [1.0, 1.0]]), device=torch.device("cpu"),
        elem_feats_scaled=np.array([[0.1, 0.2]]), fraction_set=["0.5"],
        n_exploit=2, max_attempts=3, charge_filter=True,
    )
    assert checked == ["A1", "B1", "C1"]
    assert [r["formula"] for r in rows] == ["B1", "C1"]
    assert predictor.waves == [["A", "B"], ["C"]]


def test_auto_shrink_to_scalar_preserves_existing_failure_streak(tmp_path):
    predictor = Predictor(4)
    predictor.observe = lambda: setattr(predictor, "size", 1)
    env = ScriptedEnv(["X"] * 20, predictor)
    with pytest.raises(RuntimeError, match="11 consecutive candidates"):
        run_dqn(env, tmp_path, n_warmup_eps=20)
    assert len(env.starts) == 11
    assert predictor.waves == [["X"] * 4] + [["X"]] * 7
    assert predictor.scalar == []
