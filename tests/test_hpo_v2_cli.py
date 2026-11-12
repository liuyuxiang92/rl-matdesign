"""Visible budget feasibility for the simplified public interface."""
import json
from pathlib import Path

import yaml

from rl_matdesign.hpo.cli import main


def test_plan_reports_training_tiers_excluded_by_actual_learning_requirements(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/hpo/v1/dummy_a2c.yaml").read_text())
    config["base_config"] = str(root / "configs/hpo/scenarios/dummy.yaml")
    config["training"]["budget"]["min_unique"] = 8
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump(config))
    for command in ("validate", "plan"):
        assert main([command, "--hpo-config", str(source)]) == 0
        output = json.loads(capsys.readouterr().out)
        # 20 scaler preparation attempts need B > 40 for strict A/B < 1.5.
        assert output["budget_preflight"] == {
            "minimum_training_unique": 41, "omitted_training_budgets": [8, 16, 32]}
        assert output["budget_grids"]["train"] == [64, 96]
        assert output["budget_grids"]["generation"] == [8, 16, 32]
        assert not output["predictor_evaluated"]
