"""Exercise the calibrated reward through the scientific predictor adapter."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from rl_matdesign.config_schema import load_yaml_config, resolve_config
from rl_matdesign.predictors.outputs import ModelSamples, OutputSchema, aggregate_samples
from rl_matdesign.predictors.structure_score import (
    StructureScorePredictor,
    _EvaluationRecord,
    _Statistics,
)
from rl_matdesign.reward_calibration import CalibrationPendingError
from rl_matdesign.utils.timing import PredictorTimer


def pilot(predictor, candidates):
    return [predictor.calibration_observation({**candidates[i % len(candidates)], "sample_id": i + 1})
            for i in range(100)]


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
        "value_normalization": "popart",
        "properties": [_property("T3"), _property("CTE", "min"), _property("Tg")],
        "reward": {
            "type": "gated",
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
    observations = pilot(p, [a, b])
    # The same records can cross a pre-model checkpoint without rerunning an oracle.
    observations = json.loads(json.dumps(observations, allow_nan=False))
    assert all("reward" not in row["prediction_record"] for row in observations)
    assert p.cached_outputs({**a, "sample_id": 1})["T3"].mean == 1200.0
    assert p.calibration_observation({**a, "T3": 1200, "sample_id": 1})["candidate_key"] == observations[0]["candidate_key"]
    with pytest.raises(ValueError, match="distinct"):
        p.fit_reward_calibration([observations[0], observations[0]])
    artifact = p.fit_reward_calibration(observations)
    assert artifact["spec"]["profile"] == "auto_gated_softplus_v1"
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
        p.fit_reward_calibration(pilot(p, [converted(row) for row in samples]))
        return [p.predict(converted(row))[0] for row in candidates]

    assert evaluate(1.0, 0.0) == pytest.approx(evaluate(1e6, -273.15))


def test_zero_quality_spread_stops_calibration():
    cfg = _glassnet_cfg()
    rows = [
        {"T3": 1200.0, "CTE": 4e-6, "Tg": 800.0},
        {"T3": 1200.0, "CTE": 6e-6, "Tg": 800.0},
    ]
    p = StructureScorePredictor(cfg)
    with pytest.raises(ValueError, match="zero spread"):
        p.fit_reward_calibration(pilot(p, rows))
    assert p.reward_calibration_state_dict() is None


@pytest.mark.parametrize("count", [99, 101])
def test_fitting_requires_exactly_100_distinct_candidates(count):
    p = StructureScorePredictor(_glassnet_cfg())
    observations = [p.calibration_observation({"T3": 1000 + i, "CTE": 4e-6, "Tg": 800})
                    for i in range(count)]
    with pytest.raises(ValueError, match=r"requires 100 distinct valid candidates"):
        p.fit_reward_calibration(observations)


@pytest.mark.parametrize("n_models", [1, 2])
def test_perovskite_gate_and_aggregate_uncertainty(monkeypatch, n_models):
    cfg = {
        "k": 1.0,
        "structure": {"builder": "pipeline", "base_poscar": "unused", "n_random_configs": 1,
                      "steps": [{"type": "substitute", "a_site_symbol": "Fe",
                                 "a_dopant_group": "alloy"}]},
        "value_normalization": "popart",
        "properties": [
            {"name": "pure", "predictor": "dp_classification",
             "models": [f"classifier_{i}.pt" for i in range(n_models)],
             "direction": "max", "output_index": 1},
            *[_property(name, objective="mean_plus_kstd") for name in ("delta_400", "delta_500", "delta_600")],
        ],
        "reward": {
            "type": "gated",
            "objective": {"quality_mode": "aggregate", "terms": {"delta_400": 0.33333, "delta_500": 0.33333,
                                    "delta_600": 0.33333}},
            "gates": [{"name": "synthesis", "score": {"terms": {"pure": 1.0}}, "min": 0.8}],
        },
    }
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
    samples = [{"probabilities": failed_probabilities, "spread": spread} for spread in (0.5, 2.5)]
    observations = pilot(p, samples)
    artifact = p.fit_reward_calibration(observations)
    assert artifact["gates"]["synthesis"]["scale"] == pytest.approx(1.0 if n_models == 1 else 0.05)
    assert artifact["quality"]["__aggregate__"]["reference"] == pytest.approx(0.33333 * 9.0)
    assert artifact["quality"]["__aggregate__"]["scale"] == pytest.approx(0.33333)
    failure = p.score_calibration_observation(observations[-1])
    assert failure["gate_passed"] is False
    assert failure["reward"] == pytest.approx(-0.1 if n_models == 1 else -1.0)
    pass_probabilities = [0.9] if n_models == 1 else [0.9, 0.9]
    assert p.predict({"probabilities": pass_probabilities, "spread": 2.5})[0] == pytest.approx(
        1.0 + math.log1p(math.e)
    )


def test_calibrated_weighted_sum_preserves_declared_directions_and_weights():
    cfg = {
        "value_normalization": "popart",
        "properties": [
            _property("quality"),
            _property("cost", "min"),
        ],
        "reward": {"type": "weighted_sum",
                   "objective": {"terms": {"quality": 2.0, "cost": 0.5}}},
    }
    p = StructureScorePredictor(cfg)
    observations = pilot(p, [{"quality": 100.0, "cost": 8.0}, {"quality": 300.0, "cost": 12.0}])
    artifact = p.fit_reward_calibration(observations)
    assert artifact["spec"]["profile"] == "auto_weighted_sum_v1"
    assert artifact["quality"]["cost"]["reference"] == -10.0
    assert artifact["quality"]["cost"]["scale"] == 2.0
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


@pytest.mark.parametrize("reward_type", ["weighted_sum", "gated"])
def test_value_normalization_selects_pilot(reward_type):
    from rl_matdesign.hpo.artifacts import reward_calibration_requested

    cfg = {"properties": [_property("T3")], "reward": {"type": reward_type}}
    if reward_type == "gated":
        cfg["reward"].update(gates=[{"name": "gate", "score": {"terms": {"T3": 1}}, "min": 0}],
                             failure={"reward": -5})
    assert not reward_calibration_requested(cfg)
    uncalibrated = StructureScorePredictor(cfg)
    assert not uncalibrated.reward_calibration_enabled
    assert uncalibrated.predict({"T3": 5.0})[0] == 5.0
    cfg["value_normalization"] = "popart"
    assert reward_calibration_requested(cfg)
    p = StructureScorePredictor(cfg)
    assert p.reward_calibration_num_candidates == 100
    with pytest.raises(CalibrationPendingError):
        p.predict({"T3": 5.0})


@pytest.mark.parametrize(("scenario", "expected_profile", "examples"), [
    ("glassnet_coarse_2pct", "auto_gated_softplus_v1", [
        ({"forming_temperature": 1500, "thermal_expansion": 4e-6, "glass_transition": 800}, 1500),
        ({"forming_temperature": 1500, "thermal_expansion": 6e-6, "glass_transition": 800}, -11),
        ({"forming_temperature": 1500, "thermal_expansion": 4e-6, "glass_transition": 763}, -10.01),
    ]),
    ("oxides_sinter", "auto_weighted_sum_v1", [({"sinter": 850}, -850), ({"sinter": 875}, -875)]),
])
def test_shipped_yaml_switches_reward_with_the_learner(scenario, expected_profile, examples):
    path = Path(__file__).resolve().parents[1] / "configs" / f"{scenario}.yaml"
    yaml = load_yaml_config(path)
    assert "calibration" not in yaml["reward"]
    # Supply predictions in native units, keeping the scenario's real objective and gates.
    for prop in yaml["properties"]:
        prop.pop("models", None)
        prop.pop("output", None)
        prop.update(predictor=f"{__name__}:PhysicalProperty", quantity=prop["name"])
    raw = StructureScorePredictor(resolve_config(yaml, {"value_normalization": "none"}).config)
    assert raw.reward_calibration_num_candidates == 0
    for candidate, expected in examples:
        assert raw.predict(candidate)[0] == pytest.approx(expected)
    calibrated = StructureScorePredictor(resolve_config(yaml).config)
    assert calibrated.reward_calibration_num_candidates == 100
    first = examples[0][0]
    with pytest.raises(CalibrationPendingError):
        calibrated.predict(first)
    quality = yaml["properties"][0]["name"]
    higher = {**first, quality: first[quality] + 100}
    artifact = calibrated.fit_reward_calibration(pilot(calibrated, [first, higher]))
    assert artifact["spec"]["profile"] == expected_profile
    assert artifact["observation_count"] == 100
    assert calibrated.predict(first)[0] != pytest.approx(examples[0][1])


def test_reuse_previous_explicit_auto_artifact_without_changing_its_hash():
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    cfg = _glassnet_cfg()
    p = StructureScorePredictor(cfg, seed=0)
    samples = [{"T3": t, "CTE": 4e-6, "Tg": 800.0} for t in (1200.0, 1400.0)]
    artifact = p.fit_reward_calibration(pilot(p, samples))
    artifact["spec"]["scientific_identity"]["properties"][0]["normalization"] = {"mode": "auto_fixed"}
    artifact["spec_hash"] = digest(artifact["spec"])
    del artifact["content_hash"]
    artifact["content_hash"] = digest(artifact)
    restored = StructureScorePredictor(cfg, seed=0)
    restored.load_reward_calibration(artifact)
    assert restored.reward_calibration_state_dict() == artifact
    assert restored.predict(samples[0]) == p.predict(samples[0])
