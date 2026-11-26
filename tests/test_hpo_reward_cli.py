"""Public reward-mode configuration and train/generation job integration."""
import json

import pytest
import yaml

from rl_matdesign.config_schema import ConfigError
from rl_matdesign.hpo.artifacts import read_json
from rl_matdesign.hpo.cli import main
from rl_matdesign.hpo.job_runner import JobRunner
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_job_runner import FakeWorker, public_plan_path


def reward_plan_path(tmp_path, method="a2c"):
    source = public_plan_path(tmp_path, threshold=-1550)
    scene_path = tmp_path / "scene.yaml"
    scene = yaml.safe_load(scene_path.read_text())
    scene["properties"] = [
        {"name": "sinter", "predictor": "dummy", "direction": "min"},
        {"name": "calcine", "predictor": "dummy", "direction": "min"},
    ]
    scene["reward"] = {"type": "weighted_sum", "objective": {
        "terms": {"sinter": 1, "calcine": 1}}}
    scene_path.write_text(yaml.safe_dump(scene))
    data = yaml.safe_load(source.read_text())
    data["method"] = method
    data["evaluation"]["metric"] = "reward"
    if method != "a2c":
        data["training"]["parameter_ranges"] = {
            "dqn_batch_size": {"type": "categorical", "choices": [2]},
        }
        if method == "dqn-bootstrap":
            data["training"]["parameter_ranges"]["dqn_target_update_freq"] = {
                "type": "categorical", "choices": [1]}
    source.write_text(yaml.safe_dump(data))
    return source


@pytest.mark.parametrize("method", ["a2c", "dqn-bootstrap", "dqn-mc"])
def test_cli_plan_accepts_two_objectives_as_reward(tmp_path, capsys, method):
    source = reward_plan_path(tmp_path, method)
    for command in ("validate", "plan"):
        assert main([command, "--hpo-config", str(source)]) == 0
        output = json.loads(capsys.readouterr().out)
        assert not output["predictor_evaluated"]
        assert output["budget_grids"]["train"] == [4, 8, 16]
        if command == "plan":
            objective = output["resolved_plan"]["objective"]
            assert objective["metric"] == "reward"
            assert objective["direction"] == "maximize"
            assert objective["threshold"] == -1550
            assert objective["unit"] is None
            assert output["resolved_plan"]["base_config_data"]["reward"]["objective"]["terms"] == {
                "sinter": 1, "calcine": 1}


def test_explicit_property_mode_keeps_original_objective_contract(tmp_path):
    source = public_plan_path(tmp_path)
    original = load_plan(source)
    data = yaml.safe_load(source.read_text())
    data["evaluation"]["metric"] = "property"
    source.write_text(yaml.safe_dump(data))
    explicit = load_plan(source)
    assert explicit["objective"] == original["objective"]
    assert "metric" not in explicit["objective"]
    for key in ("metrics", "seed_schedule", "budget", "search_space"):
        assert explicit[key] == original[key]


@pytest.mark.parametrize("metric", ["reward_objective", "loss", "", None, True, [], {}])
def test_unknown_reward_metric_is_rejected(tmp_path, metric):
    source = reward_plan_path(tmp_path)
    data = yaml.safe_load(source.read_text())
    data["evaluation"]["metric"] = metric
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="evaluation.metric must be property or reward"):
        load_plan(source)


def test_reward_mode_still_validates_scene_reward_terms(tmp_path):
    source = reward_plan_path(tmp_path)
    scene_path = tmp_path / "scene.yaml"
    scene = yaml.safe_load(scene_path.read_text())
    scene["reward"]["objective"]["terms"] = {"unknown_property": 1}
    scene_path.write_text(yaml.safe_dump(scene))
    with pytest.raises(ConfigError, match="unknown property"):
        load_plan(source)


def test_combined_reward_training_generation_and_restored_calibration(tmp_path, monkeypatch):
    source = reward_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)

    def reward_records(rows):
        for index, row in enumerate(rows):
            sinter, calcine = 1000 + 10 * index, 500 + 5 * index
            row["properties"] = {"sinter": sinter, "calcine": calcine}
            row["prediction_record"] = {"reward": -(sinter + calcine)}
        return rows

    worker.archive_transform = reward_records
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    trained = runner({"pg_batch_eps": 2}, 8, 11, "train")
    assert trained["eligible_for_selection"]
    assert trained["raw_metrics"]["best_value"] == -1500
    assert trained["raw_metrics"]["qualified_count"] == 4
    calibration = read_json(runner.out / "quality_calibration.json")
    assert calibration["objective"]["metric"] == "reward"
    generated = runner({"pg_batch_eps": 2, "gen_exploration_target": 1},
                       4, 401, "generation", trained)
    assert generated["eligible_for_selection"]
    assert generated["raw_metrics"]["best_value"] == -1500
    assert generated["raw_metrics"]["qualified_count"] == 4
    assert generated["quality_normalization"] == trained["quality_normalization"]
    restored = JobRunner(load_plan(source), runner.out)
    assert restored({"pg_batch_eps": 2}, 8, 11, "train") == trained
    assert restored({"pg_batch_eps": 2, "gen_exploration_target": 1},
                    4, 401, "generation", trained) == generated
    assert len(worker.calls) == 2
