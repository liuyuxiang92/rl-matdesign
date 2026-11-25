"""Clipping preserves direction and resumes the actual gradient-history policy."""
import copy
import math

import pytest
import torch

from rl_matdesign.gradient_clipping import (
    AdaptiveGradientClipper,
    NonfiniteGradientError,
    global_gradient_norm,
    gradient_clipping_profile,
)


def _parameter(gradient, *, dtype=torch.float64):
    value = torch.tensor(gradient, dtype=dtype)
    parameter = torch.nn.Parameter(torch.zeros_like(value))
    parameter.grad = value.clone()
    return parameter


def _observe(clipper, norm):
    parameter = _parameter([norm])
    diagnostics = clipper.prepare([parameter])
    clipper.commit(diagnostics)
    return diagnostics


def test_global_clipping_spans_parameters_and_preserves_direction():
    first, second = _parameter([3.]), _parameter([4.])
    no_gradient = torch.nn.Parameter(torch.ones(3))
    clipper = AdaptiveGradientClipper([2.] * 100, network="q")
    before = copy.deepcopy(clipper.state_dict())
    diagnostics = clipper.prepare(iter([first, second, no_gradient]))
    assert diagnostics["raw_norm"] == 5
    assert diagnostics["cap"] == 2
    assert diagnostics["coefficient"] == .4
    assert diagnostics["clipped"]
    torch.testing.assert_close(first.grad, torch.tensor([1.2], dtype=torch.float64))
    torch.testing.assert_close(second.grad, torch.tensor([1.6], dtype=torch.float64))
    assert global_gradient_norm([first, second]) == pytest.approx(2)
    assert clipper.state_dict() == before
    clipper.commit(diagnostics)
    assert clipper.history == [5.]  # The raw norm, never the clipped norm.
    assert clipper.state_dict()["calibration_count"] == 100


def test_pilot_positive_percentile_sets_startup_without_training_history():
    clipper = AdaptiveGradientClipper([0., 1., 2., 3., 4.], network="critic")
    assert clipper.startup_cap == pytest.approx(3.7)
    assert clipper.history == []
    assert clipper.successful_updates == 0
    state = clipper.state_dict()
    assert state["calibration_positive_count"] == 4
    diagnostics = _observe(clipper, 10.)
    assert diagnostics["threshold_source"] == "pilot"
    assert diagnostics["history_count"] == 0
    assert clipper.history == [10.]


def test_threshold_excludes_current_outlier_and_uses_previous_successes():
    clipper = AdaptiveGradientClipper([2.], network="actor")
    for norm in range(1, 21):
        diagnostics = _observe(clipper, float(norm))
        assert diagnostics["threshold_source"] == "pilot"
    assert clipper.history == list(range(1, 21))
    diagnostics = _observe(clipper, 1e6)
    assert diagnostics["threshold_source"] == "adaptive_history"
    assert diagnostics["cap"] == pytest.approx(18.1)
    assert diagnostics["positive_history_count"] == 20
    assert clipper.history[-1] == 1e6


def test_zero_updates_age_window_and_restore_pilot_threshold():
    clipper = AdaptiveGradientClipper([7.], network="q")
    for _ in range(20):
        _observe(clipper, 3.)
    assert _observe(clipper, 0.)["threshold_source"] == "adaptive_history"
    for _ in range(200):
        _observe(clipper, 0.)
    assert clipper.history == [0.] * 200
    diagnostics = _observe(clipper, 10.)
    assert diagnostics["cap"] == 7.
    assert diagnostics["threshold_source"] == "pilot"
    assert diagnostics["positive_history_count"] == 0


@pytest.mark.parametrize("pilot", [[], [0.] * 100])
def test_zero_pilot_uses_first_successful_positive_gradient_without_freezing(pilot):
    clipper = AdaptiveGradientClipper(pilot, network="q")
    zero = _observe(clipper, 0.)
    assert zero["cap"] is None
    assert zero["coefficient"] == 1
    assert zero["threshold_source"] == "unresolved_zero"
    assert clipper.startup_cap is None

    parameter = _parameter([3., 4.])
    provisional = clipper.prepare([parameter])
    assert provisional["provisional_startup"]
    assert provisional["cap"] == 5.
    assert not provisional["clipped"]
    assert clipper.startup_cap is None
    torch.testing.assert_close(parameter.grad, torch.tensor([3., 4.], dtype=torch.float64))
    clipper.discard_prepared()  # Failed/skipped optimizer step cannot establish cap.
    assert clipper.startup_cap is None
    assert clipper.successful_updates == 1

    first = _observe(clipper, 8.)
    assert first["provisional_startup"]
    assert clipper.startup_cap == 8.
    assert clipper.startup_established_at_update == 2
    later = _observe(clipper, 16.)
    assert later["cap"] == 8.
    assert later["coefficient"] == .5
    assert not later["provisional_startup"]


def test_zero_gradients_do_not_prevent_adam_momentum_step():
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.1)
    clipper = AdaptiveGradientClipper([.5], network="actor")
    parameter.grad = torch.ones_like(parameter)
    diagnostics = clipper.prepare([parameter])
    optimizer.step()
    clipper.commit(diagnostics)
    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    diagnostics = clipper.prepare([parameter])
    assert diagnostics["raw_norm"] == 0
    assert diagnostics["coefficient"] == 1
    optimizer.step()
    clipper.commit(diagnostics)
    assert not torch.equal(parameter, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradients_fail_before_any_mutation(bad):
    clipper = AdaptiveGradientClipper([1.], network="critic")
    first, second = _parameter([10.]), _parameter([bad])
    before = clipper.state_dict()
    with pytest.raises(NonfiniteGradientError, match="Nonfinite gradient"):
        clipper.prepare([first, second])
    assert first.grad.item() == 10.
    assert clipper.state_dict() == before
    assert _observe(clipper, 2.)["cap"] == 1.


def test_float64_norm_avoids_false_float32_and_float64_overflow():
    first = _parameter([3e30, 4e30], dtype=torch.float32)
    assert global_gradient_norm([first]) == pytest.approx(5e30, rel=1e-6)
    second = _parameter([3e200, 4e200])
    assert global_gradient_norm([second]) == pytest.approx(5e200)
    clipper = AdaptiveGradientClipper([5.], network="q")
    diagnostics = clipper.prepare([second])
    assert math.isfinite(diagnostics["raw_norm"])
    torch.testing.assert_close(second.grad, torch.tensor([3., 4.], dtype=torch.float64))


def test_absent_empty_duplicate_and_unrepresentable_gradients():
    assert global_gradient_norm([]) == 0
    assert global_gradient_norm([torch.nn.Parameter(torch.ones(2))]) == 0
    assert global_gradient_norm([_parameter([])]) == 0
    parameter = _parameter([1.])
    with pytest.raises(ValueError, match="duplicates"):
        global_gradient_norm([parameter, parameter])
    with pytest.raises(NonfiniteGradientError, match="float64 range"):
        global_gradient_norm([_parameter([1.7e308, 1.7e308])])


def test_prepare_commit_transaction_rejects_reuse_and_tampering():
    clipper = AdaptiveGradientClipper([1.], network="q")
    diagnostics = clipper.prepare([_parameter([3.])])
    with pytest.raises(RuntimeError, match="Commit or discard"):
        clipper.prepare([_parameter([4.])])
    altered = {**diagnostics, "raw_norm": 99.}
    with pytest.raises(ValueError, match="does not match"):
        clipper.commit(altered)
    clipper.commit(diagnostics.copy())
    with pytest.raises(ValueError, match="does not match"):
        clipper.commit(diagnostics)
    assert clipper.history == [3.]


def test_actor_critic_histories_do_not_mix():
    actor = AdaptiveGradientClipper([1.], network="actor")
    critic = AdaptiveGradientClipper([100.], network="critic")
    actor_step = actor.prepare([_parameter([10.])])
    critic_step = critic.prepare([_parameter([10.])])
    assert actor_step["coefficient"] == .1
    assert critic_step["coefficient"] == 1
    with pytest.raises(ValueError, match="does not match"):
        critic.commit(actor_step)
    actor.commit(actor_step)
    assert critic.history == []
    critic.commit(critic_step)


@pytest.mark.parametrize("pilot", [[0.] * 100, [0., 1., 3., 5.]])
def test_resume_restores_ordered_window_and_next_clipping_decisions(pilot):
    uninterrupted = AdaptiveGradientClipper(pilot, network="q")
    for norm in [0., 0., 3.] + [float(i % 7) for i in range(220)]:
        _observe(uninterrupted, norm)
    saved = uninterrupted.state_dict()
    restored = AdaptiveGradientClipper.from_state_dict(copy.deepcopy(saved), network="q")
    assert restored.state_dict() == saved
    for norm in [1e4, 0., 3., 10., .01]:
        p, q = _parameter([norm]), _parameter([norm])
        original_decision = uninterrupted.prepare([p])
        resumed_decision = restored.prepare([q])
        assert resumed_decision == original_decision
        assert torch.equal(p.grad, q.grad)
        uninterrupted.commit(original_decision)
        restored.commit(resumed_decision)
    assert restored.state_dict() == uninterrupted.state_dict()
    saved["history"][0] = 999.
    assert restored.history[0] != 999.


@pytest.mark.parametrize("change", [
    lambda state: state.pop("history"),
    lambda state: state.update(unexpected=True),
    lambda state: state.update(version=2),
    lambda state: state.update(version=True),
    lambda state: state["profile"].update(percentile=95.),
    lambda state: state["profile"].update(version=True),
    lambda state: state.update(history=[float("nan")]),
    lambda state: state.update(history=[-1.]),
    lambda state: state.update(history=(3.,)),
    lambda state: state.update(successful_updates=2),
    lambda state: state.update(successful_updates=True),
    lambda state: state.update(calibration_count=99),
    lambda state: state.update(calibration_positive_count=0),
    lambda state: state.update(startup_cap=0.),
    lambda state: state.update(startup_cap=99.),
    lambda state: state.update(startup_source="invented"),
    lambda state: state.update(startup_established_at_update=1),
])
def test_resume_rejects_incompatible_or_corrupt_state(change):
    clipper = AdaptiveGradientClipper([1., 2.], network="q")
    _observe(clipper, 3.)
    state = clipper.state_dict()
    change(state)
    with pytest.raises(ValueError):
        AdaptiveGradientClipper.from_state_dict(state, network="q")


def test_resume_rejects_network_or_provisional_startup_mismatch():
    clipper = AdaptiveGradientClipper([0.], network="q")
    _observe(clipper, 0.)
    _observe(clipper, 5.)
    state = clipper.state_dict()
    with pytest.raises(ValueError, match="network mismatch"):
        AdaptiveGradientClipper.from_state_dict(state, network="actor")
    state["startup_cap"] = 6.
    with pytest.raises(ValueError, match="disagrees with gradient history"):
        AdaptiveGradientClipper.from_state_dict(state)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf"), True, "1"])
def test_invalid_pilot_norms_are_rejected(bad):
    with pytest.raises(ValueError, match="Calibration gradient norm"):
        AdaptiveGradientClipper([bad], network="q")


def test_execution_profile_is_independent_and_fully_describes_fixed_recipe():
    first = gradient_clipping_profile()
    first["percentile"] = 10.
    current = gradient_clipping_profile()
    assert current["percentile"] == 90.
    assert current["window"] == 200
    assert current["min_positive_history"] == 20
    assert current["calibration_candidates"] == 100
    assert current["calibration_budget"] == "outside_training_budget"
