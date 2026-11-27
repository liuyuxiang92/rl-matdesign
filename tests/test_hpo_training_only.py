"""Training-only HPO runs from a fresh directory without generation settings."""
from copy import deepcopy
import json

import pytest
import yaml

from rl_matdesign.hpo.artifacts import read_json
from rl_matdesign.hpo.cli import main, parser
from rl_matdesign.hpo.controller import Campaign, SCOPES
from rl_matdesign.hpo.domain import design_space_info
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_controller import ReusableRunner


def training_source(tmp_path, budget=8):
    scene = {
        "groups": [{"name": "conditions", "kind": "categorical", "choices": [
            {"name": "amount", "values": [1, 2, 3, 4]}]}],
        "properties": [{"name": "quality", "predictor": "dummy", "direction": "max"}],
        "value_normalization": "none", "gradient_clipping": "none",
        "exploration_adaptive": False, "pg_warmup_eps": 0,
    }
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(scene))
    config = {
        "base_config": "scene.yaml", "method": "a2c", "random_seed": 42,
        "evaluation": {"threshold": .5},
        "training": {
            "repeats": 1,
            "budget": {"mode": "absolute_unique", "min_unique": budget,
                       "initial_unique": budget, "max_unique": budget},
            "parameter_ranges": {
                "pg_lr_actor": {"type": "categorical", "choices": [.001]},
                "pg_lr_critic": {"type": "categorical", "choices": [.001]},
                "pg_batch_eps": {"type": "categorical", "choices": [2]},
                "pg_entropy_coef": {"type": "categorical", "choices": [.01]},
                "pg_entropy_min": {"type": "categorical", "choices": [.3]},
            },
            "search": {"max_rounds": 1, "trials_per_round": 1, "startup_random_trials": 1},
        },
    }
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump(config))
    return source


@pytest.mark.parametrize("command", ["validate", "plan"])
@pytest.mark.parametrize("scope", ["training", "ranges", "train-budget", "train-search"])
def test_scoped_preflight_ignores_omitted_generation_for_unknown_space(tmp_path, capsys, command, scope):
    source = training_source(tmp_path)
    assert "generation" not in yaml.safe_load(source.read_text())
    assert main([command, "--hpo-config", str(source), "--stage", scope]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["domain"]["status"] == "unknown"
    assert result["budget_grids"] == {"train": [8]}
    assert not result["predictor_evaluated"]
    if command == "plan":
        assert set(result["schedule"]) == {"train"}
        assert result["stages"] == list(SCOPES[scope])


def test_domain_scope_does_not_mutate_plan_or_relax_training_requirements(tmp_path):
    plan = load_plan(training_source(tmp_path))
    original = deepcopy(plan)
    domain = design_space_info(plan, phases=("train",))
    assert domain["status"] == "unknown"
    assert plan == original
    with pytest.raises(ValueError, match="requires an exact canonical candidate count"):
        design_space_info(plan)
    plan["budget"]["train"] = deepcopy(plan["budget"]["generation"])
    with pytest.raises(ValueError, match="requires an exact canonical candidate count"):
        design_space_info(plan, phases=("train",))


@pytest.mark.parametrize("scope", [None, "all", "search", "generation"])
def test_generation_inclusive_scopes_still_require_generation_budget(tmp_path, capsys, scope):
    source = training_source(tmp_path)
    argv = ["validate", "--hpo-config", str(source)]
    if scope is not None:
        argv += ["--stage", scope]
    assert main(argv) == 2
    assert "fraction_of_space requires an exact" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["validate", "plan", "run"])
def test_training_only_rejects_infeasible_training_budget(tmp_path, capsys, command):
    source = training_source(tmp_path, budget=1)
    argv = [command, "--hpo-config", str(source), "--stage", "training"]
    if command == "run":
        argv += ["--out", str(tmp_path / "run")]
    assert main(argv) == 2
    assert "no legal train budget" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_fresh_training_run_and_resume_never_generate(tmp_path, monkeypatch, capsys, interrupt):
    source = training_source(tmp_path)
    before = source.read_bytes()
    runner = ReusableRunner()
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "training-only-test")
    monkeypatch.setattr("rl_matdesign.hpo.controller.JobRunner", lambda plan, out: runner)
    monkeypatch.setattr(Campaign, "run_generation", lambda self: pytest.fail("generation stage was called"))
    original_budget_stage = Campaign.run_train_budget
    if interrupt:
        def interrupted(self):
            raise KeyboardInterrupt
        monkeypatch.setattr(Campaign, "run_train_budget", interrupted)
    out = tmp_path / "campaign"
    result = main(["run", "--hpo-config", str(source), "--out", str(out), "--stage", "training"])
    assert result == (130 if interrupt else 0)
    state = read_json(out / "state.json")
    assert state["requested_stage"] == "training"
    assert state["completed_stages"] == (["ranges"] if interrupt else list(SCOPES["training"]))
    count = len(runner.calls)
    assert count > 0
    manifest = (out / "manifest.json").read_bytes()
    monkeypatch.setattr(Campaign, "run_train_budget", original_budget_stage)
    capsys.readouterr()
    assert main(["resume", "--out", str(out)]) == 0
    assert read_json(out / "state.json") == {
        "requested_stage": "training", "status": "COMPLETED",
        "completed_stages": ["ranges", "train-budget", "train-search"],
    }
    assert all(call["phase"] == "train" and call["model"] is None for call in runner.calls)
    if not interrupt:
        assert len(runner.calls) == count
    assert (out / "best_training.yaml").is_file()
    assert not (out / "stages/generation.json").exists()
    assert not (out / "best_generation.yaml").exists()
    assert (out / "manifest.json").read_bytes() == manifest
    assert source.read_bytes() == before


def test_all_remains_default_for_each_scoped_command():
    for command in ("validate", "plan", "run"):
        argv = [command, "--hpo-config", "scene.yaml"]
        if command == "run":
            argv += ["--out", "output"]
        assert parser().parse_args(argv).stage == "all"
