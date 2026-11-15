"""Batched bootstrap targets preserve masks, Bellman semantics and RNG streams."""
import copy
import random

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.action_encoding import ValueEncoding
from rl_matdesign.action_features import ValueAlphabet
from rl_matdesign.model import QRegressor
from rl_matdesign.training import _compute_td_target, _dqn_gradient_step
from rl_matdesign.utils.td_target_cache import TDTargetCache
from rl_matdesign.utils.td_targets import compute_td_targets


def fixtures(categorical=False):
    torch.manual_seed(19)
    values = (ValueAlphabet(["low", "high", "unused"],
              ValueEncoding(["low", "high", "unused"], kind="categorical"))
              if categorical else ["0.1", "0.5", "1.0"])
    model = QRegressor(state_dim=2, step_dim=3, elem_dim=3,
                       frac_dim=3 if categorical else 1, hidden_dim=8).eval()
    scaler = StandardScaler().fit([[0., 1.], [1., 2.]])
    elems = np.eye(3)
    rows = []
    actions = [(ei, fi) for ei in range(3) for fi in range(3)]
    for index, length in enumerate([1, 7, 2, 0, 5, 8]):
        rows.append(dict(
            s_mat_raw=np.array([index / 10., .2]), s_step=np.array([1., 0., 0.]),
            a_elem_idx=0, a_comp_val=.5, reward=index - 2.75, mc_return=.7,
            s_mat_next_raw=np.array([index / 10., .7]),
            s_step_next=np.array([0., 1., 0.]),
            next_allowed_idx=actions[:length], done=index == 4,
        ))
    # Identical next state with a distinct immediate reward must still get a
    # distinct Bellman target, but it must not launch another target forward.
    rows.append({**copy.deepcopy(rows[1]), "reward": -8.})
    return model, scaler, elems, values, rows


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 8192])
@pytest.mark.parametrize("gamma", [0., .91, 1.])
def test_scalar_parity_for_ragged_terminal_and_chunked_targets(categorical, chunk_size, gamma):
    model, scaler, elems, values, rows = fixtures(categorical)
    expected = [_compute_td_target(row, model, scaler, elems, values,
                                  torch.device("cpu"), gamma) for row in rows]
    actual = compute_td_targets(rows, model, scaler, elems, values,
                                torch.device("cpu"), gamma, _max_action_rows=chunk_size)
    # Matrix batch sizes can change floating-point accumulation; parity here is
    # numerical, deliberately not a claim of old/new bitwise trajectory identity.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual[3] == rows[3]["reward"]  # empty legal-action set
    assert actual[4] == rows[4]["reward"]  # terminal, despite listed actions


def test_chunk_memory_bound_and_dedup_reduce_forward_calls():
    model, scaler, elems, values, rows = fixtures()
    calls = []
    hook = model.register_forward_hook(lambda _model, args, _out: calls.append(len(args[0])))
    compute_td_targets(rows, model, scaler, elems, values, "cpu", .9,
                       _max_action_rows=6)
    assert calls == [6, 6, 6]  # 1 + 7 + 2 + 8 legal actions; duplicate state shared
    calls.clear()
    compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    assert calls == [18]
    hook.remove()


def test_each_minibatch_sees_current_target_network_and_its_own_rewards():
    model, scaler, elems, values, rows = fixtures()
    calls = []
    hook = model.register_forward_hook(lambda *_: calls.append(1))
    original = compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    assert len(calls) == 1
    changed = [{**row, "reward": row["reward"] + 3.} for row in rows]
    updated = compute_td_targets(changed, model, scaler, elems, values, "cpu", .9)
    np.testing.assert_allclose(updated, np.asarray(original) + 3.)
    assert len(calls) == 2
    with torch.no_grad():
        model.net[-1].bias.add_(2.)
    refreshed = compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    assert len(calls) == 3
    for index in [0, 1, 2, 5, 6]:
        assert refreshed[index] == pytest.approx(original[index] + 1.8, abs=2e-6)
    hook.remove()


def test_target_evaluation_does_not_consume_randomness():
    model, scaler, elems, values, rows = fixtures()
    random.seed(91)
    np.random.seed(19)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_only_terminal_empty_batch_and_invalid_chunk_limit():
    model, scaler, elems, values, rows = fixtures()
    def no_forward(*_args):
        pytest.fail("Terminal/no-action targets must not evaluate the target network")
    model.forward = no_forward
    assert compute_td_targets([], model, scaler, elems, values, "cpu", .9) == []
    assert compute_td_targets(rows[3:5], model, scaler, elems, values, "cpu", .9) == [
        rows[3]["reward"], rows[4]["reward"]]
    with pytest.raises(ValueError, match="positive"):
        compute_td_targets(rows, model, scaler, elems, values, "cpu", .9, _max_action_rows=0)


def test_gradient_step_uses_batched_target_and_mc_never_uses_target_network():
    target, scaler, elems, values, rows = fixtures()
    online = copy.deepcopy(target)
    optimizer = torch.optim.Adam(online.parameters(), lr=.001)
    calls = []
    hook = target.register_forward_hook(lambda *_: calls.append(1))
    _dqn_gradient_step(online, rows, target, scaler, elems, values, optimizer,
                       torch.nn.SmoothL1Loss(), torch.device("cpu"), .9)
    assert len(calls) == 1
    _dqn_gradient_step(online, rows, target, scaler, elems, values, optimizer,
                       torch.nn.SmoothL1Loss(), torch.device("cpu"), .9, target_mode="mc")
    assert len(calls) == 1
    hook.remove()


def test_old_cache_contents_cannot_change_new_gradient_or_batch_shapes():
    target, scaler, elems, values, rows = fixtures()
    cache = TDTargetCache()
    # A deliberately stale cache proves the batched path does not reuse it.
    for row in rows:
        cache.put(cache.key(row), 1.e9)
    networks = [copy.deepcopy(target), copy.deepcopy(target)]
    optimizers = [torch.optim.Adam(net.parameters(), lr=.001) for net in networks]
    for _ in range(3):
        losses = [_dqn_gradient_step(net, rows, target, scaler, elems, values, optimizer,
                                    torch.nn.SmoothL1Loss(), torch.device("cpu"), .9,
                                    target_cache=state)
                  for net, optimizer, state in zip(networks, optimizers, [None, cache])]
        assert losses[0] == losses[1]
        for left, right in zip(networks[0].parameters(), networks[1].parameters()):
            assert torch.equal(left, right)


def test_legacy_segment_reduction_fallback(monkeypatch):
    model, scaler, elems, values, rows = fixtures()
    expected = compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    monkeypatch.delattr(torch, "segment_reduce")
    actual = compute_td_targets(rows, model, scaler, elems, values, "cpu", .9)
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available locally")
def test_cuda_parity_and_chunk_boundaries_preserve_rng():
    model, scaler, elems, values, rows = fixtures()
    model = model.cuda()
    before = torch.cuda.get_rng_state().clone()
    expected = [_compute_td_target(row, model, scaler, elems, values,
                                  torch.device("cuda"), .9) for row in rows]
    for limit in [1, 3, 8192]:
        actual = compute_td_targets(rows, model, scaler, elems, values, "cuda", .9,
                                    _max_action_rows=limit)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(before, torch.cuda.get_rng_state())


def test_new_execution_continuation_preserves_network_optimizer_and_rng():
    target, scaler, elems, values, rows = fixtures()
    online = copy.deepcopy(target)
    optimizer = torch.optim.Adam(online.parameters(), lr=.001)
    random.seed(913)

    def update(net, fixed, opt, start, stop):
        losses = []
        for index in range(start, stop):
            minibatch = random.sample(rows, 5)
            losses.append(_dqn_gradient_step(
                net, minibatch, fixed, scaler, elems, values, opt,
                torch.nn.SmoothL1Loss(), torch.device("cpu"), .9))
            if (index + 1) % 3 == 0:
                fixed.load_state_dict(net.state_dict())
        return losses

    update(online, target, optimizer, 0, 5)
    saved = copy.deepcopy({"online": online.state_dict(), "target": target.state_dict(),
                           "optimizer": optimizer.state_dict(), "rng": random.getstate()})
    expected_losses = update(online, target, optimizer, 5, 12)
    expected_rng = random.getstate()
    resumed, resumed_target = copy.deepcopy(online), copy.deepcopy(target)
    resumed.load_state_dict(saved["online"])
    resumed_target.load_state_dict(saved["target"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.001)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    random.setstate(saved["rng"])
    actual_losses = update(resumed, resumed_target, resumed_optimizer, 5, 12)
    assert actual_losses == expected_losses
    assert random.getstate() == expected_rng
    for uninterrupted, recovered in [(online, resumed), (target, resumed_target)]:
        for left, right in zip(uninterrupted.parameters(), recovered.parameters()):
            assert torch.equal(left, right)
    expected_opt, actual_opt = optimizer.state_dict(), resumed_optimizer.state_dict()
    assert expected_opt["param_groups"] == actual_opt["param_groups"]
    for parameter, state in expected_opt["state"].items():
        for field, value in state.items():
            assert torch.equal(value, actual_opt["state"][parameter][field])
