"""Observe gradients at real Adam boundaries in the production trainers."""
import copy

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign import training
from rl_matdesign.gradient_clipping import AdaptiveGradientClipper
from rl_matdesign.exploration import ExplorationController
from rl_matdesign.model import QRegressor, ValueNet
from rl_matdesign.utils.candidate_budget import NumericalTrainingError
from rl_matdesign.utils.training_budget import TrainingBudget
from tests.test_pg_duplicate_budget import _setup


class InspectAdam(torch.optim.Adam):
    """Keep actual Adam behavior while recording its incoming gradients."""

    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_gradients = []

    def step(self, closure=None):
        self.step_gradients.append([
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for group in self.param_groups for parameter in group["params"]
        ])
        return super().step(closure)


def _fixture(normalization, mode):
    torch.manual_seed(31)
    qnet = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1,
                     hidden_dim=8, value_normalization=normalization)
    target = copy.deepcopy(qnet).eval()
    with torch.no_grad():
        target.net[-1].weight.zero_()
        target.net[-1].bias.fill_(5.)
    if normalization == "popart":
        target.update_value_normalization(torch.tensor([15., 20.]))
    row = dict(s_mat_raw=np.array([.1, .2]), s_step=np.array([1., 0.]),
               a_elem_idx=0, a_comp_val=.5, reward=2., mc_return=7.,
               s_mat_next_raw=np.array([.2, .7]), s_step_next=np.array([0., 1.]),
               next_allowed_idx=[(0, 0), (1, 1)], done=False)
    batch = [row, {**row, "done": True, "reward": -3., "mc_return": -6.}]
    return qnet, dict(
        batch=batch, target_net=target if mode == "bootstrap" else None,
        s_mat_scaler=StandardScaler().fit([[0., 1.], [1., 2.]]),
        elem_feats_scaled=np.eye(2), action_values=["0.5", "1.0"],
        loss_fn=torch.nn.SmoothL1Loss(), device=torch.device("cpu"),
        gamma=.5, target_mode=mode,
    )


def _norm(gradients):
    return torch.linalg.vector_norm(torch.cat([
        gradient.double().flatten() for gradient in gradients if gradient is not None
    ])).item()


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["mc", "bootstrap"])
@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_dqn_clips_real_gradients_before_adam_and_records_raw_norm(mode, normalization):
    qnet, kwargs = _fixture(normalization, mode)
    reference = copy.deepcopy(qnet)
    raw_optimizer = InspectAdam(reference.parameters(), lr=.001)
    training._dqn_gradient_step(reference, optimizer=raw_optimizer, **kwargs)
    raw = raw_optimizer.step_gradients[0]
    raw_norm = _norm(raw)
    assert raw_norm > 0
    cap = raw_norm / 5
    controller = AdaptiveGradientClipper([cap] * 100, network="q")
    optimizer = InspectAdam(qnet.parameters(), lr=.001)
    target_before = copy.deepcopy(kwargs["target_net"].state_dict()) if mode == "bootstrap" else None
    diagnostics = {}
    training._dqn_gradient_step(qnet, optimizer=optimizer, gradient_clipper=controller,
                                diagnostics=diagnostics, **kwargs)
    observed = optimizer.step_gradients[0]
    assert _norm(observed) == pytest.approx(cap, rel=1e-6)
    for unclipped, clipped in zip(raw, observed):
        torch.testing.assert_close(clipped, unclipped * .2, rtol=1e-6, atol=1e-8)
    assert controller.history == pytest.approx([raw_norm])
    assert controller.successful_updates == len(optimizer.step_gradients) == 1
    # Adam's first moment proves it consumed the clipped values.
    for parameter, clipped in zip(qnet.parameters(), observed):
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], clipped * .1)
    if target_before is not None:
        _same_state(target_before, kwargs["target_net"].state_dict())


@pytest.mark.parametrize("mode,targets", [("mc", [7., -6.]), ("bootstrap", [4.5, -3.])])
@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_none_clipping_matches_the_original_known_target_adam_update(mode, targets, normalization):
    actual, kwargs = _fixture(normalization, mode)
    reference = copy.deepcopy(actual)
    optimizer = InspectAdam(actual.parameters(), lr=.001)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
    loss_actual = training._dqn_gradient_step(actual, optimizer=optimizer,
                                              gradient_clipper=None, **kwargs)
    inputs = (
        torch.tensor(kwargs["s_mat_scaler"].transform([[.1, .2], [.1, .2]]), dtype=torch.float32),
        torch.tensor([[1., 0.], [1., 0.]]),
        torch.tensor([[1., 0.], [1., 0.]]),
        torch.tensor([[.5], [.5]]),
    )
    expected_targets = torch.tensor(targets, dtype=torch.float64).reshape(-1, 1)
    if normalization == "popart":
        expected_targets = reference.update_value_normalization(expected_targets)
        prediction = reference.forward_normalized(*inputs)
    else:
        expected_targets = expected_targets.float()
        prediction = reference(*inputs)
    loss_expected = torch.nn.functional.smooth_l1_loss(prediction, expected_targets)
    reference_optimizer.zero_grad(set_to_none=True)
    loss_expected.backward()
    reference_optimizer.step()
    assert loss_actual == loss_expected.item()
    _same_state(actual.state_dict(), reference.state_dict())
    _same_state(optimizer.state_dict(), reference_optimizer.state_dict())


@pytest.mark.parametrize("mode", ["mc", "bootstrap"])
@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_dqn_nonfinite_backward_cannot_advance_adam_or_clipping_history(mode, normalization):
    qnet, kwargs = _fixture(normalization, mode)
    optimizer = InspectAdam(qnet.parameters(), lr=.001)
    controller = AdaptiveGradientClipper([1.] * 100, network="q")
    training._dqn_gradient_step(qnet, optimizer=optimizer, gradient_clipper=controller, **kwargs)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    controller_before = copy.deepcopy(controller.state_dict())
    non_head_before = qnet.net[0].weight.detach().clone()
    hook = qnet.net[0].weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(NumericalTrainingError, match="Nonfinite gradient"):
            training._dqn_gradient_step(qnet, optimizer=optimizer, gradient_clipper=controller, **kwargs)
    finally:
        hook.remove()
    assert len(optimizer.step_gradients) == 1
    _same_state(optimizer_before, optimizer.state_dict())
    assert controller.state_dict() == controller_before
    assert torch.equal(qnet.net[0].weight, non_head_before)
    # PopArt may already compensate its head; that is not an Adam update and
    # must not be mistaken for transactional rollback of the whole learner.


@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_a2c_nonfinite_critic_gradient_prevents_both_optimizer_steps(monkeypatch, normalization):
    kwargs, sampled, learned = _setup(monkeypatch, [0, 1], method="a2c")
    kwargs.update(num_iters=1, batch_eps=2)
    if normalization == "popart":
        kwargs["value_net"] = ValueNet(state_dim=kwargs["scaler"].n_features_in_,
                                        step_dim=1, hidden_dim=8, value_normalization="popart")
    value_net, policy = kwargs["value_net"], kwargs["policy"]
    critic_parameter = next(value_net.parameters())
    hook = critic_parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    optimizers = []

    def create_adam(parameters, **options):
        optimizer = InspectAdam(parameters, **options)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", create_adam)
    committed = []
    original_commit = AdaptiveGradientClipper.commit

    def commit(controller, diagnostics):
        committed.append(controller.network)
        return original_commit(controller, diagnostics)

    monkeypatch.setattr(AdaptiveGradientClipper, "commit", commit)
    policy_before = copy.deepcopy(policy.state_dict())
    checkpoint = {
        "is_resume": True,
        "exploration_state": ExplorationController(adaptive=True).state_dict(),
        "training_budget_state": TrainingBudget(count_duplicates=True).state_dict(),
        "gradient_clipping_state": {
            name: AdaptiveGradientClipper([1.] * 100, network=name).state_dict()
            for name in ("actor", "critic")
        },
    }
    try:
        with pytest.raises(NumericalTrainingError, match="Nonfinite gradient"):
            training.train_pg(**kwargs, gradient_clipping="clipping", train_count_duplicates=True,
                              checkpoint_cfg=checkpoint)
    finally:
        hook.remove()
    assert sampled == learned == [0, 1]
    assert len(optimizers) == 2
    assert all(not optimizer.step_gradients and not optimizer.state for optimizer in optimizers)
    assert committed == []
    _same_state(policy_before, policy.state_dict())
