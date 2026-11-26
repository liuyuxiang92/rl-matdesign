"""Exploration invariants, bounded feedback, and exact continuation."""
import copy
import json
import math

import numpy as np
import pytest
import torch

from rl_matdesign.exploration import (
    ExplorationController,
    entropy_probabilities,
    exploration_profile,
    normalized_entropy,
)


def test_endpoints_ties_forced_and_entropy_range():
    np.testing.assert_array_equal(entropy_probabilities([2, 2, 1], 0), [.5, .5, 0])
    np.testing.assert_allclose(entropy_probabilities([2, 2, 1], 1), [1 / 3] * 3)
    np.testing.assert_allclose(entropy_probabilities([2, 2, 2], .2), [1 / 3] * 3)
    np.testing.assert_array_equal(entropy_probabilities([4], .7), [1.])
    assert normalized_entropy([1.]) is None
    assert normalized_entropy([0., 1., 0.]) == 0
    assert normalized_entropy([.5, .5]) == pytest.approx(1)
    # The target is unattainable below the entropy of two tied maximum actions.
    np.testing.assert_array_equal(entropy_probabilities([2, 2, 1], .1), [.5, .5, 0])


@pytest.mark.parametrize("target", [.01, .3, .7, .99])
def test_temperature_hits_requested_entropy_and_respects_score_order(target):
    scores = np.array([-3., -.5, 0., 2., 7.])
    probabilities = entropy_probabilities(scores, target)
    assert normalized_entropy(probabilities) == pytest.approx(target, abs=1e-10)
    assert np.all(np.diff(probabilities) >= 0)
    assert probabilities.sum() == pytest.approx(1)


@pytest.mark.parametrize("factor,offset", [(1e-200, 0), (1e200, 0), (.01, 100), (100, -42)])
def test_affine_score_rescaling_preserves_sampling(factor, offset):
    scores = np.array([-3., -.5, 0., 2., 7.])
    expected = entropy_probabilities(scores, .7, scale=10.)
    actual = entropy_probabilities(factor * scores + offset, .7, scale=10. * factor)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def test_extreme_scores_and_resolution_safeguards_are_finite():
    probabilities = entropy_probabilities([-1.7e308, 0, 1.7e308], .7)
    assert np.isfinite(probabilities).all()
    assert normalized_entropy(probabilities) == pytest.approx(.7)
    np.testing.assert_allclose(entropy_probabilities([1., 1. + 1e-15], .2), [.5, .5])
    np.testing.assert_allclose(entropy_probabilities([0., 1e-10], .2, scale=1), [.5, .5])
    # Explicit greedy requests are honored, even for a very small score gap.
    np.testing.assert_array_equal(entropy_probabilities([0., 1e-10], 0, scale=1), [0, 1])
    tiny = entropy_probabilities([0., np.nextafter(0., 1.)], .7)
    assert normalized_entropy(tiny) == pytest.approx(.7)
    np.testing.assert_array_equal(
        entropy_probabilities([0., 1., np.nextafter(1., 2.)], .01), [0., .5, .5],
    )
    np.testing.assert_array_equal(entropy_probabilities([-1., 0., 1e-300], .01), [0., .5, .5])
    assert normalized_entropy(entropy_probabilities([0., 1.], 1e-200)) == pytest.approx(1e-200, rel=1e-10, abs=0)


@pytest.mark.parametrize("scores,target,scale", [
    ([], .7, None), ([[1, 2]], .7, None), ([np.nan], .7, None),
    ([np.inf], .7, None), ([1, 2], -1, None), ([1, 2], 1.1, None),
    ([1, 2], True, None), ([1, 2], .7, 0), ([1, 2], .7, np.inf),
])
def test_invalid_sampling_inputs_rejected(scores, target, scale):
    with pytest.raises(ValueError):
        entropy_probabilities(scores, target, scale)


@pytest.mark.parametrize("probs", [[], [0., 0.], [-.1, 1.1], [np.nan], [[1.]]])
def test_invalid_probability_vectors_rejected(probs):
    with pytest.raises(ValueError):
        normalized_entropy(probs)


def test_entropy_coefficient_feedback_sign_bounds_and_forced_decisions():
    controller = ExplorationController(adaptive=False)
    start = controller.coefficient
    assert controller.update_entropy(None) == start
    assert controller.entropy_updates == 0
    assert controller.update_entropy(.1) > start
    high = controller.coefficient
    assert controller.update_entropy(.99) < high
    for _ in range(2000):
        controller.update_entropy(0.)
    assert controller.coefficient == pytest.approx(exploration_profile()["coefficient_max"])
    assert controller.target == .7
    controller = ExplorationController(initial_target=0)
    for _ in range(2000):
        controller.update_entropy(1.)
    assert controller.coefficient == pytest.approx(exploration_profile()["coefficient_min"])


def _window(controller, prefix, *, unique=64, feasible=True, quality=1., success=True):
    result = None
    for index in range(exploration_profile()["window_attempts"]):
        result = controller.observe(
            f"{prefix}-{index}" if index < unique else "old",
            quality, feasible=feasible, successful=success,
        )
    assert result["window_completed"]
    return result


def test_ledger_counts_failures_duplicates_and_unknown_feasibility():
    controller = ExplorationController()
    controller.seed_seen(["old"])
    assert controller.attempts == 0
    assert not controller.observe("old", 4.)["new"]
    assert not controller.observe("bad", None, successful=False)["new"]
    assert "bad" not in controller.seen_keys
    assert controller.observe("bad", None, feasible=False)["new"]
    assert controller.observe("unknown", 100, feasible=None)["new"]
    assert controller.observe("good", 3.)["new"]
    assert controller.best_quality == 3.
    state = controller.state_dict()
    assert state["window"] == {"attempts": 5, "successful": 4, "unique": 3,
                               "feasible": 1, "improvements": 1, "qualities": [3.]}


def test_two_stagnant_windows_raise_target_then_cooldown():
    controller = ExplorationController(initial_target=0)
    controller.seed_seen(["old"])
    assert _window(controller, "a", unique=0)["action"] == "hold"
    result = _window(controller, "b", unique=0)
    assert result["action"] == "raise_for_stagnation"
    assert controller.target == .05
    for _ in range(2):
        assert _window(controller, "c", unique=0)["action"] == "cooldown"
        assert controller.target == .05
    assert _window(controller, "d", unique=0)["action"] == "hold"
    assert _window(controller, "e", unique=0)["action"] == "raise_for_stagnation"
    assert controller.target == .1


def test_failures_alone_do_not_raise_exploration_and_fixed_target_stays_fixed():
    controller = ExplorationController()
    for index in range(4):
        result = _window(controller, str(index), success=False)
        assert result["action"] == "evaluation_failure_hold"
    assert controller.target == .7
    for endpoint in (0., 1.):
        controller = ExplorationController(initial_target=endpoint, adaptive=False)
        controller.seed_seen(["old"])
        for _ in range(3):
            assert _window(controller, "same", unique=0)["action"] == "fixed_target"
        assert controller.target == endpoint


def _start_trial(controller):
    controller.seed_seen(["old"])
    assert _window(controller, "base1", unique=32)["action"] == "hold"
    assert _window(controller, "base2", unique=32)["action"] == "lower_target_trial"
    assert controller.target == pytest.approx(.65)
    assert _window(controller, "settle", unique=0)["action"] == "trial_settling"


def test_lower_target_trial_waits_and_reverts_if_useful_yield_degrades():
    controller = ExplorationController()
    _start_trial(controller)
    assert _window(controller, "measure1", unique=16, quality=2)["action"] == "trial_measuring"
    result = _window(controller, "measure2", unique=16, quality=2)
    assert result["action"] == "trial_reverted"
    assert controller.target == .7
    assert controller.cooldown_windows == 2


def test_lower_target_trial_kept_only_for_observed_pareto_improvement():
    controller = ExplorationController()
    _start_trial(controller)
    _window(controller, "measure1", unique=48, quality=1)
    result = _window(controller, "measure2", unique=48, quality=1)
    assert result["action"] == "trial_kept_observationally"
    assert controller.target == pytest.approx(.65)
    assert not result["trial_active"]


def test_changed_target_never_exceeds_endpoints():
    controller = ExplorationController(initial_target=1.)
    controller.seed_seen(["old"])
    _window(controller, "a", unique=0)
    assert _window(controller, "b", unique=0)["action"] == "upper_bound_hold"
    assert controller.target == 1.


def test_checkpoint_roundtrip_inflight_trial_is_exact_and_not_aliased():
    controller = ExplorationController()
    _start_trial(controller)
    _window(controller, "measure1", unique=48, quality=2)
    for index in range(9):
        controller.observe(f"partial-{index}", 3.)
    controller.update_entropy(.2)
    saved = json.loads(json.dumps(controller.state_dict(), allow_nan=False))
    resumed = ExplorationController()
    resumed.load_state_dict(saved)
    assert resumed.state_dict() == saved
    saved["seen_keys"].append("outside")
    assert "outside" not in resumed.seen_keys
    for index in range(200):
        args = (f"continuation-{index}", float(index % 5))
        assert resumed.observe(*args) == controller.observe(*args)
        assert resumed.update_entropy(.5) == controller.update_entropy(.5)
    assert resumed.state_dict() == controller.state_dict()


@pytest.mark.parametrize("change", [
    lambda state: state["profile"].update(window_attempts=2),
    lambda state: state["config"].update(adaptive=False),
    lambda state: state.update(target=np.nan),
    lambda state: state.update(attempts=100),
    lambda state: state.update(seen_keys=["a", "a"]),
    lambda state: state["window"].update(feasible=100),
    lambda state: state.update(previous_window=None),
    lambda state: state["trial"].update(settle_remaining=9),
    lambda state: state["trial"].update(prior_target=.1),
])
def test_invalid_checkpoint_rejected_atomically(change):
    controller = ExplorationController()
    _start_trial(controller)
    before = controller.state_dict()
    bad = copy.deepcopy(before)
    change(bad)
    with pytest.raises(ValueError):
        controller.load_state_dict(bad)
    assert controller.state_dict() == before


def test_invalid_observations_do_not_advance_clock():
    controller = ExplorationController()
    before = controller.state_dict()
    for kwargs in ({"quality": np.nan}, {"key": ""}, {"feasible": "yes"}, {"successful": 1}):
        with pytest.raises(ValueError):
            controller.observe(**{"key": "a", "quality": 1., **kwargs})
    assert controller.state_dict() == before


def test_finite_extreme_quality_mean_roundtrips():
    controller = ExplorationController()
    for index in range(64):
        controller.observe(str(index), 1.7e308 if index % 2 else -1.7e308)
    state = controller.state_dict()
    assert math.isfinite(state["previous_window"]["mean_quality"])
    ExplorationController().load_state_dict(state)


def _block(prefix, size, quality=1.):
    return [{"key": f"{prefix}-{index}", "quality": quality} for index in range(size)]


def test_large_behavior_block_cannot_start_and_score_its_own_target_trial():
    controller = ExplorationController()
    first = controller.observe_block(_block("first", 512))
    assert first["window"]["attempts"] == 512
    assert first["windows"] == 1
    assert first["action"] == "hold"
    second = controller.observe_block(_block("second", 512))
    assert second["action"] == "lower_target_trial"
    assert second["windows"] == 2
    state = controller.state_dict()
    assert state["trial"]["settle_remaining"] == 1
    assert state["trial"]["measurements"] == []
    assert state["trial"]["baseline"]["attempts"] == 1024
    assert controller.observe_block(_block("third", 512))["action"] == "trial_settling"
    assert controller.observe_block(_block("fourth", 512))["action"] == "trial_measuring"
    assert controller.observe_block(_block("fifth", 512))["action"] == "trial_reverted"
    assert controller.windows == 5


def test_variable_windows_and_partial_block_resume_preserve_all_attempts():
    controller = ExplorationController()
    controller.observe_block(_block("large", 100))
    controller.observe_block(_block("partial", 20))
    saved = json.loads(json.dumps(controller.state_dict()))
    assert saved["completed_attempts"] == 100
    assert saved["attempts"] == 120
    resumed = ExplorationController()
    resumed.load_state_dict(saved)
    expected = controller.observe_block(_block("complete", 60))
    assert resumed.observe_block(_block("complete", 60)) == expected
    assert expected["window"]["attempts"] == 80
    assert resumed.state_dict() == controller.state_dict()
    assert resumed.state_dict()["completed_attempts"] == 180


def test_block_validation_is_atomic():
    controller = ExplorationController()
    before = controller.state_dict()
    with pytest.raises(ValueError):
        controller.observe_block([{"key": "first", "quality": 1.}, {"key": "bad", "quality": np.nan}])
    assert controller.state_dict() == before


def test_float32_softmax_entropy_accepts_roundoff_without_mutation():
    generator = torch.Generator().manual_seed(37)
    for _ in range(50):
        probabilities = torch.softmax(torch.randn(161, generator=generator), dim=0).numpy()
        before = probabilities.copy()
        value = normalized_entropy(probabilities)
        assert 0 <= value <= 1
        expected = probabilities.astype(np.float64)
        expected /= expected.sum()
        assert value == pytest.approx(-np.dot(expected, np.log(expected)) / math.log(161))
        np.testing.assert_array_equal(probabilities, before)
