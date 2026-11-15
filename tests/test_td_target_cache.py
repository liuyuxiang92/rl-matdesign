"""Memoization must preserve target values and optimizer trajectories exactly."""
import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign.model import QRegressor
from rl_matdesign.training import _compute_td_target, _dqn_gradient_step
from rl_matdesign.utils.td_target_cache import TDTargetCache


def fixtures():
    torch.manual_seed(19)
    model = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=8)
    scaler = StandardScaler().fit([[0., 1.], [1., 2.]])
    elems = np.array([[0., 1.], [1., 0.]])
    row = dict(s_mat_raw=np.array([.1, .2]), s_step=np.array([1., 0.]),
               a_elem_idx=0, a_comp_val=.5, reward=.2, mc_return=.7,
               s_mat_next_raw=np.array([.2, .7]), s_step_next=np.array([0., 1.]),
               next_allowed_idx=[(0, 0), (1, 1)], done=False)
    return model, scaler, elems, row


def test_same_next_state_reuses_forward_but_not_reward_or_gamma():
    model, scaler, elems, row = fixtures()
    model.eval()
    cache, calls = TDTargetCache(), []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    def target(r, gamma, cache):
        return _compute_td_target(r, model, scaler, elems, ['0.5','1.0'],
                                  torch.device('cpu'), gamma, target_cache=cache)
    expected = target(row, .9, None)
    assert target(row, .9, cache) == expected
    other = {**copy.deepcopy(row), 'reward': -1.}
    expected_other = target(other, .7, None)
    before = len(calls)
    assert target(other, .7, cache) == expected_other
    assert len(calls) == before
    reordered = {**row, 'next_allowed_idx': list(reversed(row['next_allowed_idx']))}
    target(reordered, .9, cache)
    assert len(calls) == before + 1
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.1)
    cache.clear()
    assert target(row, .9, cache) == target(row, .9, None)
    hook.remove()


def test_cache_capacity_and_terminal_rows():
    model, scaler, elems, row = fixtures()
    cache = TDTargetCache(max_entries=1)
    key = cache.key(row)
    cache.put(key, 0.)
    assert cache.get(key) == 0.
    other_key = cache.key({**row, 's_step_next': np.array([1., 0.])})
    cache.put(other_key, 1.)
    assert cache.get(key) is None
    assert len(cache._values) == 1
    tiny = TDTargetCache(max_bytes=1)
    tiny.put(key, 2.)
    assert len(tiny._values) == 0 and tiny._bytes == 0
    result = _compute_td_target({**row,'done':True}, model, scaler, elems,
                               ['0.5','1.0'], torch.device('cpu'), .9, target_cache=cache)
    assert result == row['reward']


def test_cached_gradient_updates_equal_uncached_with_target_copies():
    qnet, scaler, elems, row = fixtures()
    other = copy.deepcopy(qnet)
    targets = [copy.deepcopy(qnet).eval(), copy.deepcopy(qnet).eval()]
    optimizers = [torch.optim.Adam(qnet.parameters(), lr=.001),
                  torch.optim.Adam(other.parameters(), lr=.001)]
    cache = TDTargetCache()
    batch = [row, {**copy.deepcopy(row), 'reward':.5}, {**row, 'done':True}]
    for iteration in range(8):
        losses = [_dqn_gradient_step(net, batch, target, scaler, elems, ['0.5','1.0'],
                  optimizer, torch.nn.SmoothL1Loss(), torch.device('cpu'), .9,
                  target_cache=active_cache)
                  for net,target,optimizer,active_cache in
                  zip([qnet,other], targets, optimizers, [None,cache])]
        assert losses[0] == losses[1]
        for a,b in zip(qnet.parameters(), other.parameters()):
            assert torch.equal(a,b)
        if iteration % 2 == 1:
            for target,net in zip(targets, [qnet,other]):
                target.load_state_dict(net.state_dict())
            cache.clear()
