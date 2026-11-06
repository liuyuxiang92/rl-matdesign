#!/usr/bin/env python3
"""Summarize recorded benchmark evidence without evaluating models or rerunning arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_matdesign.config_schema import PARAMETERS
from rl_matdesign.predictors.outputs import ScientificOutput
from rl_matdesign.rewards import RewardCombiner

PHYSICAL_OUTPUTS = {"T3": "K", "CTEbelowTg": "K^-1", "Tg": "K"}
CURVE_COLUMNS = ["config", "method", "seed", "budget", "unique", "invocation",
                 "wall_elapsed_s", "cumulative_prediction_elapsed_s", "reward", "best_reward",
                 "feasible", "best_feasible_reward", "best_feasible_T3_K",
                 "best_feasible_CTEbelowTg_K^-1", "best_feasible_Tg_K"]


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def read_config(directory, planned_config):
    # The effective run config captures CLI overrides; an arm YAML is the next
    # best evidence. Never quietly substitute today's original shared config.
    baseline = directory / "benchmark.json"
    if not (directory / "run_config.json").exists() and baseline.exists():
        manifest_path = Path(json.loads(baseline.read_text())["manifest"])
        manifest = json.loads(manifest_path.read_text())
        return manifest["config"], str(manifest_path)
    for name in ("run_config.json", "config.yaml"):
        path = directory / name
        if path.exists():
            value = json.loads(path.read_text()) if name.endswith("json") else yaml.safe_load(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a mapping")
            return value, str(path)
    raise ValueError(f"Missing run_config.json/config.yaml in {directory} (planned {planned_config})")


def science_protocol(cfg):
    """Fingerprint the authored scientific domain/oracle/reward, not optimizer knobs."""
    # Match execution_contract's ownership boundary rather than maintaining a
    # second whitelist of potentially arbitrary plugin scientific options.
    selected = {key: value for key, value in cfg.items()
                if key not in PARAMETERS or key in {"eval_seed", "k", "constrain_training"}}
    for key in ("generated_columns", "predict_cache_size", "config_file", "config_schema_version"):
        selected.pop(key, None)
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def actual_arm_issues(cfg, evidence, plan, directory):
    issues = []
    if plan["method"] not in {"random", "dqn-mc", "dqn-bootstrap", "a2c"}:
        return ["Unsupported benchmark method"]
    if evidence.endswith("config.yaml"):
        return ["Only authored YAML available; actual method, seed and CLI overrides are unverified"]
    expected = {"train_seed": plan["seed"], "train_budget": plan["budget"]}
    if plan["method"] == "random":
        baseline = directory / "benchmark.json"
        if not baseline.exists() or json.loads(baseline.read_text()).get("method") != "uniform_random":
            issues.append("Missing uniform_random execution evidence")
    else:
        expected["method"] = "a2c" if plan["method"] == "a2c" else "dqn"
        if plan["method"].startswith("dqn-"):
            expected["dqn_target_mode"] = plan["method"].removeprefix("dqn-")
    for key, value in expected.items():
        if cfg.get(key) != value:
            issues.append(f"Actual {key} does not match the planned arm")
    return issues


def outputs_and_feasibility(cfg, row):
    """Replay public reward gates only; unknown/custom semantics remain unknown."""
    properties = cfg.get("properties")
    if not isinstance(properties, list) or not properties:
        return None, {}, "No named property protocol available"
    try:
        outputs = {name: ScientificOutput.from_dict(value)
                   for name, value in (row.get("outputs") or {}).items()}
        raw, weights, physical = {}, {}, {}
        for prop in properties:
            name = prop["name"]
            output = outputs[name]
            if output.schema.name != name:
                raise ValueError(f"Output name mismatch for {name}")
            definition = cfg.get("models", {}).get(prop.get("model_ref"), {}) if isinstance(cfg.get("models"), dict) else {}
            predictor = prop.get("predictor", definition.get("predictor"))
            # Mirrors StructureScore's fixed classification gate margin.
            raw[name] = output.mean - ((output.model_spread or 0.0) if predictor == "dp_classification" else 0.0)
            weights[name] = prop.get("weight", 1.0)
            selector = prop.get("output")
            if (selector in PHYSICAL_OUTPUTS and output.schema.representation == "physical"
                    and output.schema.unit == PHYSICAL_OUTPUTS[selector]):
                if selector in physical:
                    raise ValueError(f"Ambiguous multiple properties select {selector}")
                physical[selector] = output.mean
        combiner = RewardCombiner(cfg.get("reward"), weights)
        passed = combiner.evaluate(raw, {name: 0.0 for name in weights})["gate_passed"]
        return passed, physical, None
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return None, {}, str(exc)


def optimizer_counts(directory, method):
    if method == "random":
        return {"q": 0, "actor": 0, "critic": 0}
    path = directory / "training_log.csv"
    if not path.exists():
        return {"q": None, "actor": None, "critic": None}
    counts = {"q": None, "actor": None, "critic": None}
    rules = {"dqn_train": {"q": "optimizer_steps"},
             "pg_train": {"actor": "actor_optimizer_steps", "critic": "critic_optimizer_steps"}}
    missing = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for label, field in rules.get(row.get("phase"), {}).items():
                value = finite(float(row[field])) if row.get(field) else None
                if value is None or value < 0 or not value.is_integer():
                    missing.add(label)
                else:
                    counts[label] = (counts[label] or 0) + int(value)
    for label in missing:
        counts[label] = None
    return counts


def summarize_arm(plan, result):
    directory = Path(plan["output"])
    arm = {key: plan[key] for key in ("config", "method", "seed", "budget", "output", "domain_size")}
    arm.update({"comparable": False, "issues": [], "requests": 0, "failed_requests": 0,
                "duplicate_successes": 0, "unique": 0, "feasible": 0,
                "unknown_feasibility": 0, "best_reward": None, "best_feasible": None,
                "minimum_feasible_T3": None, "invocations": {}, "physical_evidence_issues": []})
    curves, seen = [], set()
    try:
        cfg, arm["config_evidence"] = read_config(directory, plan["config"])
        arm["scientific_protocol_sha256"] = science_protocol(cfg)
        arm["issues"].extend(actual_arm_issues(cfg, arm["config_evidence"], plan, directory))
        if cfg.get("train_count_duplicates") is not False:
            arm["issues"].append("Configuration pays for duplicate candidates; incompatible unique budget")
        arm["optimizer_steps"] = optimizer_counts(directory, plan["method"])
        path = directory / "evaluations.jsonl"
        if not path.exists():
            raise ValueError("Missing evaluations.jsonl")
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("ledger row must be a mapping")
                except ValueError as exc:
                    arm["issues"].append(f"Ledger line {line_no}: {exc}")
                    continue
                if row.get("phase") != "train" or row.get("purpose") == "reporting":
                    continue
                arm["requests"] += 1
                invocation = row.get("invocation") or "unrecorded"
                times = arm["invocations"].setdefault(invocation, {"wall_elapsed_s": None,
                                                                  "cumulative_prediction_elapsed_s": None})
                for field in times:
                    source_field = "prediction_elapsed_s" if field == "cumulative_prediction_elapsed_s" else field
                    value = finite(row.get(source_field))
                    if value is not None:
                        if value < 0 or (times[field] is not None and value < times[field]):
                            arm["issues"].append(f"Ledger line {line_no}: {field} regressed within invocation")
                        times[field] = value
                if row.get("status") == "failed":
                    arm["failed_requests"] += 1
                    continue
                if row.get("status") != "ok":
                    arm["issues"].append(f"Ledger line {line_no}: missing/unknown status")
                    continue
                key = row.get("candidate_key")
                if not isinstance(key, str):
                    arm["issues"].append(f"Ledger line {line_no}: missing candidate identity")
                    continue
                new = row.get("new_successful_candidate")
                if not isinstance(new, bool) or new != (key not in seen):
                    arm["issues"].append(f"Ledger line {line_no}: inconsistent successful-unique history")
                if key in seen:
                    arm["duplicate_successes"] += 1
                    continue
                seen.add(key)
                if new is not True:
                    # A truncated/resumed ledger cannot establish its omitted history.
                    continue
                arm["unique"] += 1
                if row.get("unique") != arm["unique"]:
                    arm["issues"].append(f"Ledger line {line_no}: unique counter does not match complete ledger")
                reward = finite(row.get("reward"))
                if reward is None:
                    arm["issues"].append(f"Ledger line {line_no}: missing/nonfinite reward")
                    continue
                arm["best_reward"] = max(arm["best_reward"], reward) if arm["best_reward"] is not None else reward
                feasible, physical, reason = outputs_and_feasibility(cfg, row)
                if reason and reason not in arm["physical_evidence_issues"]:
                    arm["physical_evidence_issues"].append(reason)
                if feasible is None:
                    arm["unknown_feasibility"] += 1
                elif feasible:
                    arm["feasible"] += 1
                    candidate = {"reward": reward, "candidate_key": key, "candidate": row.get("candidate"),
                                 "physical": physical, "unique": arm["unique"]}
                    if arm["best_feasible"] is None or reward > arm["best_feasible"]["reward"]:
                        arm["best_feasible"] = candidate
                    if "T3" in physical and (arm["minimum_feasible_T3"] is None
                            or physical["T3"] < arm["minimum_feasible_T3"]["physical"]["T3"]):
                        arm["minimum_feasible_T3"] = candidate
                best = arm["best_feasible"] or {}
                best_physical = best.get("physical", {})
                curves.append({**{key: arm[key] for key in ("config", "method", "seed", "budget", "unique")},
                               "invocation": invocation,
                               "wall_elapsed_s": finite(row.get("wall_elapsed_s")),
                               "cumulative_prediction_elapsed_s": finite(row.get("prediction_elapsed_s")),
                               "reward": reward, "best_reward": arm["best_reward"], "feasible": feasible,
                               "best_feasible_reward": best.get("reward"),
                               "best_feasible_T3_K": best_physical.get("T3"),
                               "best_feasible_CTEbelowTg_K^-1": best_physical.get("CTEbelowTg"),
                               "best_feasible_Tg_K": best_physical.get("Tg")})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        arm["issues"].append(str(exc))
    if result is None:
        arm["issues"].append("No benchmark_results.json completion record")
    elif (result.get("status") != "complete" or result.get("returncode") != 0
          or result.get("budget_completed") != plan["budget"] or result.get("stop_reason") != "budget_complete"):
        arm["issues"].append("Run failed or completion record did not prove the requested budget")
    elif any(result.get(key) != plan[key] for key in ("config", "method", "seed", "budget", "domain_size")):
        arm["issues"].append("Completion record does not match the planned arm")
    if arm["unique"] != plan["budget"]:
        arm["issues"].append(f"Ledger has {arm['unique']} successful unique candidates; requested {plan['budget']}")
    arm["comparable"] = not arm["issues"]
    arm["feasibility_rate"] = (arm["feasible"] / arm["unique"] if arm["unique"] and not arm["unknown_feasibility"] else None)
    return arm, curves


def aggregate(arms):
    groups = {}
    for arm in arms:
        key = (arm["config"], arm["method"], arm["budget"], arm["domain_size"], arm.get("scientific_protocol_sha256"))
        groups.setdefault(key, []).append(arm)
    rows = []
    for key, group in groups.items():
        complete = [arm for arm in group if arm["comparable"]]
        seeds = [arm["seed"] for arm in complete]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated seeds cannot be independent replicates: {key[:2]}")
        values = {"best_reward": [arm["best_reward"] for arm in complete],
                  "best_feasible_reward": [(arm["best_feasible"] or {}).get("reward") for arm in complete],
                  "minimum_feasible_T3_K": [((arm["minimum_feasible_T3"] or {}).get("physical", {})).get("T3") for arm in complete],
                  "feasibility_rate": [arm["feasibility_rate"] for arm in complete]}
        for metric, candidates in values.items():
            available = [value for value in candidates if value is not None]
            rows.append({"config": key[0], "method": key[1], "budget": key[2], "domain_size": key[3],
                         "scientific_protocol_sha256": key[4], "metric": metric,
                         "planned_arms": len(group), "complete_arms": len(complete),
                         "excluded_arms": len(group) - len(complete), "n": len(available),
                         "median": median(available) if available else None,
                         "min": min(available) if available else None,
                         "max": max(available) if available else None})
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="Directory containing benchmark_plan.json")
    parser.add_argument("--out", type=Path, help="Default: BENCHMARK/summary")
    args = parser.parse_args(argv)
    plan = json.loads((args.benchmark / "benchmark_plan.json").read_text())
    result_path = args.benchmark / "benchmark_results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else []
    if not isinstance(plan, list) or not isinstance(results, list):
        parser.error("Plan and results must be JSON lists")
    index = {row["output"]: row for row in results}
    if len(index) != len(results) or len({row["output"] for row in plan}) != len(plan):
        parser.error("Duplicate output directories in plan/results")
    arms, curves = [], []
    for row in plan:
        arm, points = summarize_arm(row, index.get(row["output"]))
        arms.append(arm)
        curves.extend(points)
    distribution = aggregate(arms)
    out = args.out or args.benchmark / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "curves.csv", curves, CURVE_COLUMNS)
    fields = ["config", "method", "budget", "domain_size", "scientific_protocol_sha256", "metric",
              "planned_arms", "complete_arms", "excluded_arms", "n", "median", "min", "max"]
    write_csv(out / "aggregate.csv", distribution, fields)
    report = {"schema_version": 1, "plan": str((args.benchmark / "benchmark_plan.json").resolve()),
              "interpretation": ["Best found among paid successful unique training candidates, not a global optimum.",
                  "Ranges are observed seed min/max, not statistical confidence intervals.",
                  "Physical feasibility means configured prediction gates; it does not certify experimental feasibility.",
                  "Wall times end at the last training observation, not full-run completion; never join invocations. Missing timing stays null.",
                  "Cumulative prediction time may include restored checkpoint history; do not sum invocation endpoints.",
                  "Incomplete or failed arms are excluded from distributions but retain diagnostic curves.",
                  "Scientific protocol fingerprints cover declared configuration, not independently rehashed model assets."],
              "arms": arms, "distributions": distribution}
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(arms)} arms; {sum(arm['comparable'] for arm in arms)} complete comparable arms; summary: {out}")
    return report


if __name__ == "__main__":
    main()
