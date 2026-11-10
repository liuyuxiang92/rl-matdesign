"""Recorded evidence, rather than return-code optimism, drives benchmark summaries."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from rl_matdesign.predictors.outputs import OutputSchema, ScientificOutput

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_benchmark", ROOT / "scripts/summarize_benchmark.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def configuration():
    return {"groups": [{"name": "glass", "kind": "composition", "species_set": ["SiO2", "B2O3"]}],
            "train_count_duplicates": False, "eval_seed": 11,
            "models": {"glass": {"predictor": "glassnet"}},
            "properties": [
                {"name": "forming", "model_ref": "glass", "output": "T3"},
                {"name": "expansion", "model_ref": "glass", "output": "CTEbelowTg"},
                {"name": "transition", "model_ref": "glass", "output": "Tg"}],
            "reward": {"type": "gated", "objective": {"terms": {"forming": 1}},
                       "gates": [{"name": "cte", "score": {"terms": {"expansion": 1}}, "max": 5e-6},
                                 {"name": "tg", "score": {"terms": {"transition": 1}}, "min": 773}],
                       "failure": {"reward": -10}}}


def observation(key="a", unique=1, reward=-1.5, *, t3=1500, cte=4e-6, tg=800, **kwargs):
    outputs = {name: ScientificOutput(OutputSchema(name, unit), value).to_dict()
               for name, unit, value in [("forming", "K", t3), ("expansion", "K^-1", cte),
                                         ("transition", "K", tg)]}
    return {"phase": "train", "status": "ok", "candidate_key": key, "candidate": {"SiO2": .6},
            "new_successful_candidate": True, "unique": unique, "reward": reward,
            "outputs": outputs, "invocation": "first", **kwargs}


def arm(tmp_path, name, rows, budget=2, cfg=None, seed=11, method="dqn-mc", complete=True):
    directory = tmp_path / name
    directory.mkdir()
    cfg = dict(cfg or configuration())
    cfg.update(train_seed=seed, train_budget=budget, method="a2c" if method == "a2c" else "dqn",
               dqn_target_mode=method.removeprefix("dqn-"))
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
    (directory / "run_config.json").write_text(json.dumps(cfg))
    (directory / "evaluations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = {"config": "configs/glass.yaml", "output": str(directory), "domain_size": 100,
            "budget": budget, "method": method, "seed": seed}
    result = {**plan, "returncode": 0 if complete else 1, "budget_completed": budget,
              "status": "complete" if complete else "incomplete", "stop_reason": "budget_complete"}
    return plan, result


def test_only_successful_unique_training_counts_and_negative_reward_can_be_feasible(tmp_path):
    rows = [observation(wall_elapsed_s=2), observation(new_successful_candidate=False, wall_elapsed_s=3),
            observation("failure", status="failed", new_successful_candidate=False, wall_elapsed_s=4),
            {"phase": "train", "purpose": "reporting", "raw_result": 999},
            observation("generation", phase="generate", reward=1000),
            observation("b", 2, -11, cte=6e-6, wall_elapsed_s=5)]
    plan, result = arm(tmp_path, "complete", rows)
    report, points = summary.summarize_arm(plan, result)
    assert report["comparable"] and report["unique"] == 2
    assert report["requests"] == 4 and report["failed_requests"] == 1
    assert report["duplicate_successes"] == 1 and report["feasible"] == 1
    assert report["best_feasible"]["reward"] == -1.5
    assert report["best_feasible"]["physical"] == {"T3": 1500, "CTEbelowTg": 4e-6, "Tg": 800}
    assert report["feasibility_rate"] == .5
    assert [point["unique"] for point in points] == [1, 2]
    assert [point["wall_elapsed_s"] for point in points] == [2, 5]
    assert points[-1]["best_feasible_T3_K"] == 1500


def test_resumed_invocations_are_separate_and_historical_time_is_not_guessed(tmp_path):
    plan, result = arm(tmp_path, "resumed", [observation(), observation("b", 2, invocation="second",
                          wall_elapsed_s=.25, prediction_elapsed_s=10)])
    report, points = summary.summarize_arm(plan, result)
    assert report["comparable"]
    assert points[0]["wall_elapsed_s"] is None
    assert points[1]["wall_elapsed_s"] == .25
    assert report["invocations"] == {"first": {"wall_elapsed_s": None, "cumulative_prediction_elapsed_s": None},
                                     "second": {"wall_elapsed_s": .25, "cumulative_prediction_elapsed_s": 10}}


@pytest.mark.parametrize("failure", ["missing_result", "incomplete_result", "short_ledger", "duplicate_new", "truncated_resume", "malformed", "regressed_time", "mismatched_result"])
def test_failed_or_inconsistent_evidence_is_excluded(tmp_path, failure):
    rows = [observation(wall_elapsed_s=2), observation("b", 2, wall_elapsed_s=3)]
    if failure == "short_ledger": rows.pop()
    if failure == "duplicate_new": rows[1]["candidate_key"] = "a"
    if failure == "truncated_resume": rows[0]["unique"] = 3
    if failure == "regressed_time": rows[1]["wall_elapsed_s"] = 1
    plan, result = arm(tmp_path, failure, rows)
    if failure == "missing_result": result = None
    if failure == "incomplete_result": result["status"] = "incomplete"
    if failure == "mismatched_result": result["seed"] = 999
    if failure == "malformed":
        with (Path(plan["output"]) / "evaluations.jsonl").open("a") as handle:
            handle.write("{broken\n")
    report, _ = summary.summarize_arm(plan, result)
    assert not report["comparable"] and report["issues"]
    assert all(row["n"] == 0 for row in summary.aggregate([report]))


def test_missing_outputs_keep_reward_curve_but_not_invent_feasibility(tmp_path):
    plan, result = arm(tmp_path, "old", [observation(outputs=None)], budget=1)
    report, points = summary.summarize_arm(plan, result)
    assert report["comparable"] and report["best_reward"] == -1.5
    assert report["unknown_feasibility"] == 1 and report["best_feasible"] is None
    assert points[0]["feasible"] is None and points[0]["best_feasible_T3_K"] is None
    assert report["physical_evidence_issues"]


@pytest.mark.parametrize("mismatch", ["unit", "representation", "selector"])
def test_physical_quantity_requires_selector_units_and_physical_representation(tmp_path, mismatch):
    cfg, row = configuration(), observation()
    if mismatch == "selector": cfg["properties"][0]["output"] = "notT3"
    else: row["outputs"]["forming"]["schema"][mismatch] = "ln" if mismatch == "representation" else "degC"
    plan, result = arm(tmp_path, mismatch, [row], budget=1, cfg=cfg)
    report, points = summary.summarize_arm(plan, result)
    assert report["minimum_feasible_T3"] is None
    assert points[0]["best_feasible_T3_K"] is None


def test_classifier_gate_uses_model_spread_not_native_uncertainty():
    cfg = {"properties": [{"name": "p", "predictor": "dp_classification"}],
           "reward": {"type": "gated", "gates": [{"name": "class", "score": {"terms": {"p": 1}}, "min": .5}],
                      "failure": {"reward": -10}}}
    output = ScientificOutput(OutputSchema("p", "1"), .6, model_spread=.2, native_uncertainty=.01)
    feasible, _, _ = summary.outputs_and_feasibility(cfg, {"outputs": {"p": output.to_dict()}})
    assert feasible is False


def test_optimizer_counts_use_update_rows_only_and_missing_counts_stay_unknown(tmp_path):
    plan, result = arm(tmp_path, "counts", [observation()], budget=1)
    path = Path(plan["output"]) / "training_log.csv"
    path.write_text("phase,optimizer_steps,actor_optimizer_steps,critic_optimizer_steps\ndqn_train,4,,\ndqn_train,5,,\npg_episode,999,999,999\npg_train,,1,1\npg_train,,1,1\n")
    report, _ = summary.summarize_arm(plan, result)
    assert report["optimizer_steps"] == {"q": 9, "actor": 2, "critic": 2}
    path.write_text("phase,optimizer_steps\ndqn_train,4\ndqn_train,\n")
    assert summary.optimizer_counts(path.parent, "dqn-mc")["q"] is None


def test_cli_writes_curves_and_observed_seed_range_excluding_incomplete(tmp_path):
    plans, results = [], []
    for seed, reward, complete in [(1, -1.5, True), (2, -1.3, True), (3, -.1, False)]:
        plan, result = arm(tmp_path, str(seed), [observation(reward=reward, t3=-reward*1000)],
                           budget=1, seed=seed, complete=complete)
        plans.append(plan); results.append(result)
    (tmp_path / "benchmark_plan.json").write_text(json.dumps(plans))
    (tmp_path / "benchmark_results.json").write_text(json.dumps(results))
    report = summary.main([str(tmp_path)])
    row = next(row for row in report["distributions"] if row["metric"] == "best_reward")
    assert (row["n"], row["min"], row["max"], row["median"], row["excluded_arms"]) == (2, -1.5, -1.3, -1.4, 1)
    assert len(list(csv.DictReader((tmp_path / "summary/curves.csv").open()))) == 3
    assert (tmp_path / "summary/aggregate.csv").exists()
    assert json.loads((tmp_path / "summary/summary.json").read_text()) == report


def test_different_scientific_protocols_and_repeated_seed_are_not_pooled(tmp_path):
    reports = []
    for i in range(2):
        cfg = configuration(); cfg["eval_seed"] = i
        plan, result = arm(tmp_path, str(i), [observation()], budget=1, cfg=cfg)
        reports.append(summary.summarize_arm(plan, result)[0])
    assert len(summary.aggregate(reports)) == 8
    with pytest.raises(ValueError, match="Repeated seeds"):
        summary.aggregate([reports[0], reports[0]])


def test_effective_run_config_overrides_authored_yaml(tmp_path):
    plan, result = arm(tmp_path, "effective", [observation()], budget=1)
    cfg = json.loads((Path(plan["output"]) / "run_config.json").read_text()); cfg["reward"]["gates"][0]["max"] = 1e-6
    (Path(plan["output"]) / "run_config.json").write_text(json.dumps(cfg))
    report, _ = summary.summarize_arm(plan, result)
    assert report["feasible"] == 0
    assert report["config_evidence"].endswith("run_config.json")


@pytest.mark.parametrize("key,value", [("k", 2), ("filters", [{"type": "custom"}]), ("n_random_configs", 9), ("custom_plugin_science", "changed")])
def test_scientific_options_are_never_dropped_from_protocol(key, value):
    cfg = configuration()
    assert summary.science_protocol(cfg) != summary.science_protocol({**cfg, key: value})
    assert summary.science_protocol(cfg) == summary.science_protocol({**cfg, "train_seed": 999, "pg_batch_eps": 23})


@pytest.mark.parametrize("key,value", [("method", "a2c"), ("dqn_target_mode", "bootstrap"), ("train_seed", 99), ("train_budget", 99)])
def test_actual_method_seed_and_budget_must_match_plan(tmp_path, key, value):
    plan, result = arm(tmp_path, key, [observation()], budget=1)
    path = Path(plan["output"]) / "run_config.json"
    cfg = json.loads(path.read_text()); cfg[key] = value; path.write_text(json.dumps(cfg))
    report, _ = summary.summarize_arm(plan, result)
    assert not report["comparable"] and any(key in issue for issue in report["issues"])


def test_random_uses_recorded_baseline_protocol_not_rl_method_field(tmp_path):
    plan, result = arm(tmp_path, "random", [observation()], budget=1, method="random")
    directory = Path(plan["output"])
    cfg = json.loads((directory / "run_config.json").read_text()); cfg["method"] = "a2c"
    (directory / "run_config.json").unlink()
    manifest = directory / "manifest.json"; manifest.write_text(json.dumps({"config": cfg}))
    (directory / "benchmark.json").write_text(json.dumps({"method": "uniform_random", "manifest": str(manifest)}))
    report, _ = summary.summarize_arm(plan, result)
    assert report["comparable"] and report["optimizer_steps"] == {"q": 0, "actor": 0, "critic": 0}


def test_authored_yaml_cannot_prove_actual_cli_overrides(tmp_path):
    plan, result = arm(tmp_path, "yaml-only", [observation()], budget=1)
    (Path(plan["output"]) / "run_config.json").unlink()
    report, _ = summary.summarize_arm(plan, result)
    assert not report["comparable"]
    assert any("CLI overrides" in issue for issue in report["issues"])


def test_run_record_paths_do_not_split_independent_seed_distributions(tmp_path):
    reports = []
    for seed in (11, 23):
        plan, result = arm(tmp_path, str(seed), [observation()], budget=1, seed=seed)
        path = Path(plan["output"]) / "run_config.json"
        cfg = json.loads(path.read_text())
        cfg.update(config_file=f"/runs/seed{seed}/config.yaml", config_schema_version=1)
        path.write_text(json.dumps(cfg))
        reports.append(summary.summarize_arm(plan, result)[0])
    rows = summary.aggregate(reports)
    assert len(rows) == 4 and all(row["n"] == 2 for row in rows)


def test_missing_duplicate_accounting_is_not_assumed_unique(tmp_path):
    plan, result = arm(tmp_path, "no-accounting", [observation()], budget=1)
    path = Path(plan["output"]) / "run_config.json"
    cfg = json.loads(path.read_text()); cfg.pop("train_count_duplicates")
    path.write_text(json.dumps(cfg))
    assert not summary.summarize_arm(plan, result)[0]["comparable"]
