"""Pilot gradient probes cannot train or perturb the live learner."""
import copy
from dataclasses import is_dataclass
import random

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.domain_runtime import EpisodeStep
from rl_matdesign.gradient_calibration import (
    calibrate_dqn_gradients,
    calibrate_pg_gradients,
)
from rl_matdesign.model import PolicyNet, QRegressor, ValueNet
from rl_matdesign.utils.candidate_budget import NumericalTrainingError
from rl_matdesign.utils.episode_batch import EpisodeSnapshot
from rl_matdesign.utils.seeding import capture_rng_state


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_equal(first, second)
    elif is_dataclass(left):
        assert type(left) is type(right)
        _assert_equal(vars(left), vars(right))
    else:
        assert left == right


def _episodes(*, reward=None, steps=2):
    actions = [(np.eye(2)[0], np.eye(2)[0]), (np.eye(2)[1], np.eye(2)[1])]
    episodes = []
    for index in range(100):
        path = []
        for step in range(steps):
            action = actions[(index + step) % 2]
            path.append(EpisodeStep(
                state_material_features=np.array([index / 100., step / 2.]),
                state_step_onehot=np.eye(2)[step],
                action_elem_onehot=action[0].copy(),
                action_comp_onehot=action[1].copy(),
                reward=(float(index % 9 + 1) if reward is None else reward)
                       if step == steps - 1 else 0.,
                allowed_actions=copy.deepcopy(actions),
            ))
        episodes.append(EpisodeSnapshot(
            path=path, terminal_key=("candidate", index), composition={"x": index},
            formula=f"dummy-{index}", metadata={"phase": "calibration"},
        ))
    return episodes


def _inputs():
    return dict(
        scaler=StandardScaler().fit(np.array([[0., 0.], [1., 1.]])),
        elem_feats_scaled=np.eye(2), action_values=["0.25", "0.75"],
        device=torch.device("cpu"), gamma=.9,
    )


def _live_state(model):
    return {
        "weights": copy.deepcopy(model.state_dict()),
        "grads": [None if parameter.grad is None else parameter.grad.clone()
                  for parameter in model.parameters()],
        "modes": [module.training for module in model.modules()],
    }


def _initialize_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, .01 * (index + 1))
    optimizer.step()
    # Both populated and absent live gradients must survive the probe.
    next(model.parameters()).grad = None
    model.eval()
    return optimizer


def _forbid_optimizer_steps(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not call an optimizer step")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)


@pytest.mark.parametrize("mode", ["none", "popart"])
@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_dqn_probe_preserves_weights_normalizers_grads_optimizer_rng_and_paths(
    monkeypatch, mode, target_mode,
):
    torch.manual_seed(18)
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1,
                     hidden_dim=8, value_normalization=mode)
    optimizer = _initialize_optimizer(qnet)
    if mode == "popart":
        qnet.update_value_normalization(torch.tensor([2., 5., 8.], dtype=torch.float64))
    target = copy.deepcopy(qnet) if target_mode == "bootstrap" else None
    before = _live_state(qnet)
    target_before = _live_state(target) if target is not None else None
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    episodes = _episodes()
    paths_before = copy.deepcopy(episodes)
    _forbid_optimizer_steps(monkeypatch)

    # Deliberately consume all CPU random streams inside a disposable forward.
    # This makes RNG restoration a real assertion, even for deterministic MLPs.
    original = QRegressor.forward_normalized
    def randomized_forward(self, *args):
        random.random()
        np.random.random()
        torch.rand(1)
        return original(self, *args)
    monkeypatch.setattr(QRegressor, "forward_normalized", randomized_forward)
    rng_before = capture_rng_state()
    controller, report = calibrate_dqn_gradients(
        qnet=qnet, target_net=target, episodes=episodes, loss_fn=torch.nn.SmoothL1Loss(),
        target_mode=target_mode, **_inputs(),
    )
    _assert_equal(capture_rng_state(), rng_before)
    _assert_equal(_live_state(qnet), before)
    if target is not None:
        _assert_equal(_live_state(target), target_before)
    _assert_equal(optimizer.state_dict(), optimizer_before)
    for current, saved in zip(episodes, paths_before):
        _assert_equal(vars(current), vars(saved))
    assert len(controller.calibration_norms) == 100
    assert controller.successful_updates == 0 and controller.history == []
    assert report["optimizer_updates"] == 0
    assert report["additional_predictor_calls"] == 0
    assert report["candidate_count"] == 100
    assert report["target_count"] == 200
    assert report["target_mode"] == target_mode
    assert all(np.isfinite(controller.calibration_norms))
    assert controller.startup_cap > 0


@pytest.mark.parametrize("mode", ["none", "popart"])
def test_a2c_probe_preserves_both_learners_and_pilot_wide_scaling(monkeypatch, mode):
    torch.manual_seed(19)
    policy = PolicyNet(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    critic = ValueNet(state_dim=2, step_dim=2, hidden_dim=8, value_normalization=mode)
    optimizers = [_initialize_optimizer(model) for model in (policy, critic)]
    if mode == "popart":
        critic.update_value_normalization(torch.tensor([1., 4., 7.], dtype=torch.float64))
    before = [_live_state(model) for model in (policy, critic)]
    optimizer_before = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    episodes = _episodes(steps=1)
    paths_before = copy.deepcopy(episodes)
    _forbid_optimizer_steps(monkeypatch)
    original = PolicyNet.forward
    def randomized_forward(self, *args):
        random.random()
        np.random.random()
        torch.rand(1)
        return original(self, *args)
    monkeypatch.setattr(PolicyNet, "forward", randomized_forward)
    rng_before = capture_rng_state()
    controllers, report = calibrate_pg_gradients(
        policy=policy, value_net=critic, episodes=episodes,
        exploration_target=.5, **_inputs(),
    )
    _assert_equal(capture_rng_state(), rng_before)
    for model, saved in zip((policy, critic), before):
        _assert_equal(_live_state(model), saved)
    for optimizer, saved in zip(optimizers, optimizer_before):
        _assert_equal(optimizer.state_dict(), saved)
    for current, saved in zip(episodes, paths_before):
        _assert_equal(vars(current), vars(saved))
    assert set(controllers) == {"actor", "critic"}
    for name, controller in controllers.items():
        assert controller.network == name
        assert len(controller.calibration_norms) == 100
        assert controller.successful_updates == 0 and controller.history == []
        assert controller.startup_cap > 0
    # Single-step paths would all have zero advantage if centered separately.
    assert report["actor_advantage_std"] > 0
    assert any(norm > 0 for norm in controllers["actor"].calibration_norms)
    assert report["optimizer_updates"] == 0
    assert report["target_count"] == 100
    assert report["additional_predictor_calls"] == 0


def test_zero_signal_pilot_stays_unresolved_without_fabricated_cap():
    policy = PolicyNet(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    critic = ValueNet(state_dim=2, step_dim=2, hidden_dim=8, value_normalization="popart")
    with torch.no_grad():
        for model in (policy, critic):
            for parameter in model.parameters():
                parameter.zero_()
    controllers, report = calibrate_pg_gradients(
        policy=policy, value_net=critic, episodes=_episodes(reward=0.),
        exploration_target=0., exploration_adaptive=True, **_inputs(),
    )
    for controller in controllers.values():
        assert controller.calibration_norms == [0.] * 100
        assert controller.startup_cap is None
        assert controller.startup_source == "unresolved"
    assert report["actor_advantage_std"] == 0.
    assert critic.value_normalizer.updates.item() == 0


@pytest.mark.parametrize("bad_pilot", ["short", "duplicate", "nonfinite", "empty_path"])
def test_invalid_pilot_is_rejected_before_touching_live_network(bad_pilot):
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1,
                     hidden_dim=8, value_normalization="popart")
    episodes = _episodes()
    if bad_pilot == "short":
        episodes.pop()
    elif bad_pilot == "duplicate":
        episodes[1].terminal_key = episodes[0].terminal_key
    elif bad_pilot == "nonfinite":
        episodes[0].path[-1].reward = float("nan")
    else:
        episodes[0].path = []
    before, rng_before = _live_state(qnet), capture_rng_state()
    with pytest.raises(ValueError):
        calibrate_dqn_gradients(qnet=qnet, target_net=None, episodes=episodes,
                               loss_fn=torch.nn.MSELoss(), target_mode="mc", **_inputs())
    _assert_equal(_live_state(qnet), before)
    _assert_equal(capture_rng_state(), rng_before)


def test_exception_during_probe_restores_rng_and_live_state(monkeypatch):
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1,
                     hidden_dim=8, value_normalization="popart")
    before, rng_before = _live_state(qnet), capture_rng_state()
    def failed_forward(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("deliberate probe failure")
    monkeypatch.setattr(QRegressor, "forward_normalized", failed_forward)
    with pytest.raises(RuntimeError, match="deliberate probe failure"):
        calibrate_dqn_gradients(qnet=qnet, target_net=None, episodes=_episodes(),
                               loss_fn=torch.nn.MSELoss(), target_mode="mc", **_inputs())
    _assert_equal(_live_state(qnet), before)
    _assert_equal(capture_rng_state(), rng_before)


@pytest.mark.parametrize("method", ["mc", "a2c"])
def test_finite_rewards_with_overflowing_computed_returns_are_numerical(method):
    episodes = _episodes()
    episodes[0].path[0].reward = 1e308
    episodes[0].path[1].reward = 1e308
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    policy = PolicyNet(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    critic = ValueNet(state_dim=2, step_dim=2, hidden_dim=8)
    models = (qnet, policy, critic)
    before, rng_before = [_live_state(model) for model in models], capture_rng_state()
    with pytest.raises(NumericalTrainingError, match="computed.*targets"):
        if method == "mc":
            calibrate_dqn_gradients(qnet=qnet, target_net=None, episodes=episodes,
                                   loss_fn=torch.nn.MSELoss(), target_mode="mc", **_inputs())
        else:
            calibrate_pg_gradients(policy=policy, value_net=critic, episodes=episodes,
                                   exploration_target=.5, **_inputs())
    for model, saved in zip(models, before):
        _assert_equal(_live_state(model), saved)
    _assert_equal(capture_rng_state(), rng_before)


def test_nonfinite_bootstrap_prediction_is_a_numerical_target_failure():
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    target = copy.deepcopy(qnet)
    with torch.no_grad():
        target.net[-1].bias.fill_(float("inf"))
    before, rng_before = _live_state(qnet), capture_rng_state()
    with pytest.raises(NumericalTrainingError, match="computed.*targets"):
        calibrate_dqn_gradients(qnet=qnet, target_net=target, episodes=_episodes(),
                               loss_fn=torch.nn.MSELoss(), target_mode="bootstrap", **_inputs())
    _assert_equal(_live_state(qnet), before)
    _assert_equal(capture_rng_state(), rng_before)


@pytest.mark.parametrize("failure", ["loss", "gradient", "popart"])
def test_dqn_numerical_probe_failures_preserve_live_state_and_rng(failure):
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1,
                     hidden_dim=8, value_normalization="popart")
    episodes = _episodes(reward=1e200 if failure == "popart" else None)

    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, prediction):
            ctx.save_for_backward(prediction)
            return prediction.sum() * 0.

        @staticmethod
        def backward(ctx, output_gradient):
            prediction, = ctx.saved_tensors
            return torch.full_like(prediction, float("nan"))

    def loss(prediction, target):
        random.random()
        np.random.random()
        torch.rand(1)
        if failure == "loss":
            return prediction.mean() + float("inf")
        if failure == "gradient":
            return BadBackward.apply(prediction)
        return torch.nn.functional.mse_loss(prediction, target)

    expected = {"loss": "DQN.*loss", "gradient": "gradient components",
                "popart": "value coordinates"}[failure]
    before, rng_before = _live_state(qnet), capture_rng_state()
    with pytest.raises(NumericalTrainingError, match=expected):
        calibrate_dqn_gradients(qnet=qnet, target_net=None, episodes=episodes,
                               loss_fn=loss, target_mode="mc", **_inputs())
    _assert_equal(_live_state(qnet), before)
    _assert_equal(capture_rng_state(), rng_before)


@pytest.mark.parametrize("failure", ["actor", "critic"])
def test_a2c_nonfinite_losses_are_classified_as_numerical(monkeypatch, failure):
    policy = PolicyNet(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    critic = ValueNet(state_dim=2, step_dim=2, hidden_dim=8)
    if failure == "actor":
        original = PolicyNet.forward
        def nonfinite_forward(self, *args):
            return original(self, *args) * float("nan")
        monkeypatch.setattr(PolicyNet, "forward", nonfinite_forward)
    else:
        with torch.no_grad():
            critic.net[-1].bias.fill_(1e20)
    before = [_live_state(model) for model in (policy, critic)]
    rng_before = capture_rng_state()
    with pytest.raises(NumericalTrainingError, match=f"Nonfinite {failure}.*loss"):
        calibrate_pg_gradients(policy=policy, value_net=critic, episodes=_episodes(),
                               exploration_target=.5, **_inputs())
    for model, saved in zip((policy, critic), before):
        _assert_equal(_live_state(model), saved)
    _assert_equal(capture_rng_state(), rng_before)
