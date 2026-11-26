"""PopArt preserves public values and fits targets in normalized coordinates."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.model import PolicyNet, QRegressor, ValueNet
from rl_matdesign.training import _dqn_gradient_step, _episode_pg_terms
from rl_matdesign.utils.td_target_cache import TDTargetCache


@pytest.mark.parametrize("constructor", [
    lambda mode: QRegressor(
        state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, value_normalization=mode,
    ),
    lambda mode: ValueNet(state_dim=2, step_dim=2, value_normalization=mode),
])
def test_output_preservation_and_mode_off(constructor):
    torch.manual_seed(7)
    plain = constructor("none")
    torch.manual_seed(7)
    model = constructor("popart")
    assert not any("value_normalizer" in key for key in plain.state_dict())
    for key, value in plain.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)

    args = [torch.rand(7, 2), torch.rand(7, 2)]
    if isinstance(model, QRegressor):
        args += [torch.rand(7, 2), torch.rand(7, 1)]
    expected = plain(*args).detach()
    norm = model.value_normalizer
    model.update_value_normalization(torch.zeros(4))
    assert norm.anchor.item() == 0
    assert norm.scale.item() == 1
    model.update_value_normalization(torch.tensor([-5., -3.]))
    assert norm.mean.item() == -4
    assert norm.variance.item() == pytest.approx(17.)
    assert norm.scale.item() == pytest.approx(math.sqrt(17.))
    model.update_value_normalization(torch.tensor([1., 2., 3.]))
    assert norm.mean.item() == pytest.approx(-3.994)
    assert norm.targets_seen.item() == 9
    assert norm.updates.item() == 3
    torch.testing.assert_close(model(*args), expected, rtol=1e-5, atol=1e-6)

    saved = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="Nonfinite PopArt"):
        model.update_value_normalization(torch.tensor([float("nan")]))
    loaded = constructor("popart").eval()
    loaded.load_state_dict(saved)
    assert torch.equal(loaded(*args), model(*args))
    for key, value in saved.items():
        assert torch.equal(value, model.state_dict()[key])
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize("multiplier", [0.01, 1., 100.])
def test_regularized_scale_tracks_units(multiplier):
    model = ValueNet(state_dim=1, step_dim=1, value_normalization="popart")
    normalized = model.update_value_normalization(
        multiplier * torch.tensor([-20., -10.], dtype=torch.float64)
    )
    torch.testing.assert_close(normalized, torch.tensor([-5., 5.]) / math.sqrt(250.))
    assert model.value_normalizer.scale.item() == pytest.approx(multiplier * math.sqrt(250.))


@pytest.mark.parametrize("mode, expected", [("mc", [7., -6.]), ("bootstrap", [4.5, -3.])])
def test_dqn_updates_from_correct_raw_targets(mode, expected):
    model = QRegressor(
        state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8,
        value_normalization="popart",
    )
    target = copy.deepcopy(model).eval()
    with torch.no_grad():
        target.net[-1].weight.zero_()
        target.net[-1].bias.fill_(5.)
    target.update_value_normalization(torch.tensor([15., 20.]))
    frozen = copy.deepcopy(target.state_dict())
    scaler = StandardScaler().fit([[0., 1.], [1., 2.]])
    elems = np.array([[0., 1.], [1., 0.]])
    row = dict(s_mat_raw=np.array([.1, .2]), s_step=np.array([1., 0.]),
               a_elem_idx=0, a_comp_val=.5, reward=2., mc_return=7.,
               s_mat_next_raw=np.array([.2, .7]), s_step_next=np.array([0., 1.]),
               next_allowed_idx=[(0, 0), (1, 1)], done=False)
    batch = [row, {**row, "done": True, "reward": -3., "mc_return": -6.}]
    cache = TDTargetCache()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    got_targets = []

    def loss(predictions, targets):
        got_targets.append(targets.detach())
        return torch.nn.functional.smooth_l1_loss(predictions, targets)

    for _ in range(2):
        _dqn_gradient_step(
            model, batch, target, scaler, elems, ["0.5", "1.0"], opt, loss,
            torch.device("cpu"), .5, target_mode=mode, target_cache=cache,
        )
    expected = torch.tensor(expected, dtype=torch.float64).reshape(-1, 1)
    assert model.value_normalizer.mean.item() == pytest.approx(expected.mean().item())
    torch.testing.assert_close(
        got_targets[-1], model.value_normalizer.normalize(expected, dtype=torch.float32),
        rtol=1e-6, atol=1e-6,
    )
    for key, value in frozen.items():
        assert torch.equal(target.state_dict()[key], value)
    assert len(cache._values) == (1 if mode == "bootstrap" else 0)


def test_a2c_uses_raw_baseline_and_normalized_critic_loss():
    critic = ValueNet(state_dim=1, step_dim=1, value_normalization="popart")
    actor = PolicyNet(state_dim=1, step_dim=1, elem_dim=2, frac_dim=1)
    with torch.no_grad():
        critic.net[-1].weight.zero_()
        critic.net[-1].bias.fill_(2.)
    critic.update_value_normalization(torch.tensor([6., 8.]))
    allowed = [(np.array([1., 0.]), np.array([1.])),
               (np.array([0., 1.]), np.array([1.]))]
    step = SimpleNamespace(
        allowed_actions=allowed, action_elem_onehot=allowed[0][0],
        action_comp_onehot=allowed[0][1], state_material_features=[0.],
        state_step_onehot=[1.],
    )
    _, advantage, _, losses = _episode_pg_terms(
        path=[step], returns=[8.], policy=actor, value_net=critic,
        scaler=StandardScaler().fit([[0.], [1.]]), elem_feats_scaled=np.eye(2),
        action_values=["1.0"], device=torch.device("cpu"),
    )
    assert advantage == pytest.approx([6.])
    assert losses[0].item() == pytest.approx(36 / 50)
    losses[0].backward()
    assert critic.net[-1].bias.grad is not None
    assert all(param.grad is None for param in actor.parameters())
