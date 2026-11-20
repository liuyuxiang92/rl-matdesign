"""Exercise the calibrated reward through the scientific predictor adapter."""
from copy import deepcopy
import json
import math

import pytest

from rl_matdesign.predictors.outputs import ModelSamples, OutputSchema, aggregate_samples
from rl_matdesign.predictors.structure_score import (
    StructureScorePredictor,
    _EvaluationRecord,
    _Statistics,
)
from rl_matdesign.reward_calibration import CalibrationPendingError
from rl_matdesign.utils.timing import PredictorTimer


class PhysicalProperty:
    """Read deterministic physical properties supplied by compact fixtures."""

    def __init__(self, cfg, *, seed=None):
        self.quantity = cfg["quantity"]

    def score(self, candidate):
        return candidate[self.quantity]


def _property(name, direction="max", **fields):
    return {"name": name, "predictor": f"{__name__}:PhysicalProperty",
            "quantity": name, "direction": direction, **fields}


def _glassnet_cfg(cte_multiplier=1.0, kelvin_offset=0.0):
    return {
        "properties": [_property("T3"), _property("CTE", "min"), _property("Tg")],
        "reward": {
            "type": "gated",
            "calibration": {"profile": "auto_gated_softplus_v1", "num_candidates": 2},
            "objective": {"terms": {"T3": 1.0}},
            "gates": [
                {"name": "expansion", "score": {"terms": {"CTE": 1.0}},
                 "min": 0.0, "max": 5e-6 * cte_multiplier},
                {"name": "transition", "score": {"terms": {"Tg": 1.0}},
                 "min": 773.0 + kelvin_offset},
            ],
        },
    }


def test_glassnet_all_failed_pilot_and_fixed_gate_scaling():
    cfg = _glassnet_cfg()
    p = StructureScorePredictor(cfg, seed=0)
    a = {"T3": 1200.0, "CTE": 4e-6, "Tg": 753.0}
    b = {"T3": 1400.0, "CTE": 6e-6, "Tg": 793.0}
    with pytest.raises(CalibrationPendingError):
        p.predict(a)
    assert p.cached_outputs(a) is None
    observations = [p.calibration_observation(candidate) for candidate in (a, b)]
    # The same records can cross a pre-model checkpoint without rerunning an oracle.
    observations = json.loads(json.dumps(observations, allow_nan=False))
    assert all("reward" not in row["prediction_record"] for row in observations)
    assert p.cached_outputs(a)["T3"].mean == 1200.0
    assert p.calibration_observation({**a, "T3": 1200})["candidate_key"] == observations[0]["candidate_key"]
    with pytest.raises(ValueError, match="distinct"):
        p.fit_reward_calibration([observations[0], observations[0]])
    artifact = p.fit_reward_calibration(observations)
    assert artifact["quality"]["T3"]["reference"] == pytest.approx(1300)
    assert artifact["quality"]["T3"]["scale"] == pytest.approx(100)
    assert artifact["gates"]["expansion"]["scale"] == pytest.approx(5e-6)
    assert artifact["gates"]["transition"]["scale"] == pytest.approx(20)
    wanted = [-(20 / 20), -(1e-6 / 5e-6)]
    for row, candidate, expected in zip(observations, (a, b), wanted):
        details = p.score_calibration_observation(row)
        assert details["gate_passed"] is False
        assert details["reward"] == pytest.approx(expected)
        assert details["prediction_record"]["obj_T3_mean"] == candidate["T3"]
        assert details["prediction_record"]["reward"] == p.predict(candidate)[0]
    near = {"T3": 2100.0, "CTE": 5e-6, "Tg": 773.0}
    better = {**near, "T3": 3000.0}
    assert p.reward_details(near)["gate_passed"]
    assert p.predict(near)[0] == pytest.approx(1.0 + math.log1p(math.exp(8.0)))
    assert p.predict(better)[0] > p.predict(near)[0]
    both_fail = {"T3": 3000.0, "CTE": 6e-6, "Tg": 763.0}
    assert p.predict(both_fail)[0] == pytest.approx(-0.7)
    assert p.predict({**both_fail, "T3": -4000.0})[0] == pytest.approx(-0.7)
    assert p.predict({**near, "T3": -4000.0})[0] >= 1.0
    assert p.reward_calibration_state_dict() == artifact

    loaded = StructureScorePredictor(cfg, seed=0)
    loaded.load_reward_calibration(json.loads(json.dumps(artifact)))
    assert loaded.predict(near) == p.predict(near)
    changed = deepcopy(cfg)
    changed["reward"]["gates"][1]["min"] = 772.0
    with pytest.raises(ValueError, match="different scientific contract"):
        StructureScorePredictor(changed, seed=0).load_reward_calibration(artifact)
    tampered = deepcopy(artifact)
    tampered["gates"]["expansion"]["scale"] = 1.0
    with pytest.raises(ValueError, match="content hash"):
        loaded.load_reward_calibration(tampered)


def test_calibration_invariant_to_consistent_unit_changes():
    samples = [
        {"T3": 1200.0, "CTE": 4e-6, "Tg": 753.0},
        {"T3": 1400.0, "CTE": 6e-6, "Tg": 793.0},
    ]
    candidates = [
        {"T3": 2500.0, "CTE": 4e-6, "Tg": 800.0},
        {"T3": 2500.0, "CTE": 6e-6, "Tg": 763.0},
    ]

    def evaluate(multiplier, offset):
        def converted(row):
            return {"T3": row["T3"] + offset, "CTE": row["CTE"] * multiplier,
                    "Tg": row["Tg"] + offset}

        p = StructureScorePredictor(_glassnet_cfg(multiplier, offset), seed=0)
        p.fit_reward_calibration([p.calibration_observation(converted(row)) for row in samples])
        return [p.predict(converted(row))[0] for row in candidates]

    assert evaluate(1.0, 0.0) == pytest.approx(evaluate(1e6, -273.15))


def test_zero_quality_spread_requires_an_explicit_physical_floor():
    cfg = _glassnet_cfg()
    rows = [
        {"T3": 1200.0, "CTE": 4e-6, "Tg": 800.0},
        {"T3": 1200.0, "CTE": 6e-6, "Tg": 800.0},
    ]
    p = StructureScorePredictor(cfg)
    with pytest.raises(ValueError, match="zero spread"):
        p.fit_reward_calibration([p.calibration_observation(row) for row in rows])
    cfg["properties"][0]["normalization"] = {"mode": "auto_fixed", "scale_floor": 25.0}
    p = StructureScorePredictor(cfg)
    artifact = p.fit_reward_calibration([p.calibration_observation(row) for row in rows])
    assert artifact["quality"]["T3"]["scale"] == 25.0
    assert artifact["gates"]["transition"]["scale"] == pytest.approx(27.0)


@pytest.mark.parametrize("n_models", [1, 2])
def test_perovskite_gate_and_aggregate_uncertainty(monkeypatch, n_models):
    cfg = {
        "k": 1.0,
        "structure": {"builder": "pipeline", "base_poscar": "unused", "n_random_configs": 1,
                      "steps": [{"type": "substitute", "a_site_symbol": "Fe",
                                 "a_dopant_group": "alloy"}]},
        "properties": [
            {"name": "pure", "predictor": "dp_classification",
             "models": [f"classifier_{i}.pt" for i in range(n_models)],
             "direction": "max", "output_index": 1},
            *[_property(name, objective="mean_plus_kstd") for name in ("delta_400", "delta_500", "delta_600")],
        ],
        "reward": {
            "type": "gated",
            "calibration": {"profile": "auto_gated_softplus_v1", "num_candidates": 2,
                            "quality_mode": "aggregate"},
            "objective": {"terms": {"delta_400": 0.33333, "delta_500": 0.33333,
                                    "delta_600": 0.33333}},
            "gates": [{"name": "synthesis", "score": {"terms": {"pure": 1.0}}, "min": 0.8}],
        },
    }
    if n_models == 2:
        # An ensemble's conservative score is not bounded by the raw
        # probability range. Explicitly supply one probability unit.
        cfg["reward"]["gates"][0]["violation_normalization"] = {"mode": "auto_fixed", "scale_floor": 1.0}
    p = StructureScorePredictor(cfg, seed=0)

    def evaluate(candidate):
        probabilities = candidate["probabilities"]
        output = aggregate_samples(
            OutputSchema("pure", "1"),
            [ModelSamples(f"classifier_{i}", ("sample",), (value,))
             for i, value in enumerate(probabilities)],
        )
        # Equal means with unequal model spread still give distinct utilities.
        values = {"pure": (output.mean, output.config_spread),
                  "delta_400": (1.0, candidate["spread"]),
                  "delta_500": (2.0, 0.5), "delta_600": (3.0, 1.0)}
        stats = _Statistics(values, outputs={"pure": output})
        return _EvaluationRecord(stats, {})

    monkeypatch.setattr(p, "_evaluation", evaluate)
    failed_probabilities = [0.7] if n_models == 1 else [0.75, 0.95]
    pilot = [{"probabilities": failed_probabilities, "spread": spread} for spread in (0.5, 2.5)]
    observations = [p.calibration_observation(row) for row in pilot]
    artifact = p.fit_reward_calibration(observations)
    assert artifact["gates"]["synthesis"]["scale"] == 1.0
    assert artifact["quality"]["__aggregate__"]["reference"] == pytest.approx(0.33333 * 9.0)
    assert artifact["quality"]["__aggregate__"]["scale"] == pytest.approx(0.33333)
    failure = p.score_calibration_observation(observations[-1])
    assert failure["gate_passed"] is False
    assert failure["reward"] == pytest.approx(-0.1 if n_models == 1 else -0.05)
    pass_probabilities = [0.9] if n_models == 1 else [0.9, 0.9]
    assert p.predict({"probabilities": pass_probabilities, "spread": 2.5})[0] == pytest.approx(
        1.0 + math.log1p(math.e)
    )


def test_calibrated_weighted_sum_preserves_declared_directions_and_weights():
    cfg = {
        "properties": [
            _property("quality", normalization={"mode": "auto_fixed"}),
            _property("cost", "min", normalization={"mode": "fixed", "reference": 10.0, "scale": 2.0}),
        ],
        "reward": {"type": "weighted_sum", "calibration": {"num_candidates": 2},
                   "objective": {"terms": {"quality": 2.0, "cost": 0.5}}},
    }
    p = StructureScorePredictor(cfg)
    observations = [p.calibration_observation({"quality": quality, "cost": 10.0})
                    for quality in (100.0, 300.0)]
    p.fit_reward_calibration(observations)
    assert p.predict({"quality": 300.0, "cost": 12.0})[0] == pytest.approx(1.5)


def test_pending_calibration_has_physical_evidence_and_no_reward(tmp_path):
    p = StructureScorePredictor(_glassnet_cfg(), seed=0)
    timer = PredictorTimer(p, eval_seed=0, episode_log_path=tmp_path / "episode_timing.csv")
    timer.set_phase("calibration")
    failed_gate = {"T3": 1200.0, "CTE": 4e-6, "Tg": 753.0}
    observation = timer.calibration_observation(failed_gate)
    row = p.calibration_prediction_record(observation)
    assert row["obj_T3_mean"] == 1200.0
    assert row["gate_passed"] is False
    assert row["gate_transition_violation"] == 20.0
    assert "reward" not in row
    assert "dp_mean" not in row
    assert timer.n_calls == timer.n_unique == 1
    assert timer.best_mean is None
    ledger = json.loads((tmp_path / "evaluations.jsonl").read_text())
    assert ledger["phase"] == "calibration"
    assert ledger["status"] == "ok"
    assert ledger["reward_status"] == "pending"
    assert ledger["reward"] is None
    assert ledger["best_reward"] is None
    assert ledger["outputs"]["T3"]["mean"] == 1200.0
