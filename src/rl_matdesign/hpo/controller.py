"""Two-part HPO orchestration, with independent training and generation evidence."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import yaml

from .artifacts import (asset_fingerprints, atomic_json, digest, file_digest,
                        package_digest, parameter_id, read_json)
from .budget_search import budget_grid, budget_schedule, select_budget
from .domain import design_space_info, minimum_training_budget, training_budget_preflight
from .job_runner import JobError, JobRunner
from .range_search import initial_reference, run_range_search
from .scheduling import run_jobs
from ..utils.hpo_resume_compatibility import approved_upgrade

STAGES = ("ranges", "train-budget", "train-search", "generation")
SCOPES = {"all": STAGES, "search": STAGES[1:], **{s: (s,) for s in STAGES}}


class CampaignError(RuntimeError):
    pass


def aggregate_results(rows: list[dict], parameters: dict, budget: int) -> dict:
    if not rows:
        raise ValueError("At least one required seed is necessary")
    eligible = all(row.get("eligible_for_selection") for row in rows)
    keys = set().union(*(row.get("raw_metrics", {}) for row in rows))
    raw = {}
    for key in sorted(keys):
        values = [row.get("raw_metrics", {}).get(key) for row in rows]
        raw[key] = (statistics.mean(values) if all(isinstance(v, (int, float))
                    and not isinstance(v, bool) and math.isfinite(v) for v in values) else None)
    return {"parameters": parameters, "parameter_id": parameter_id(parameters), "budget": budget,
            "status": "COMPLETED" if eligible else next(
                (r.get("status", "INCOMPLETE") for r in rows if not r.get("eligible_for_selection")), "INCOMPLETE"),
            "eligible_for_selection": eligible, "score": statistics.mean(r["score"] for r in rows) if eligible else -1.,
            "raw_metrics": raw, "seed_results": rows,
            "seeds": [r["seed"] for r in rows]}


def rank_results(rows: list[dict], plan: dict) -> list[dict]:
    pending = [r for r in rows if r.get("eligible_for_selection")]
    direction = 1 if plan["objective"]["direction"] == "minimize" else -1
    result = []
    while pending:
        anchor = max(r["score"] for r in pending)
        band = [r for r in pending if anchor - r["score"] <= plan["budget"]["tolerances"]["score"]]
        if plan.get("protocol") == "hpo_v2" or plan.get("scoring_scope") == "score_only":
            # Near ties are equivalent under the one public score tolerance.
            # Never reintroduce component-specific preferences as tie breakers.
            band.sort(key=lambda r: r["parameter_id"])
        else:
            band.sort(key=lambda r: (r["raw_metrics"].get("attempts_per_unique", math.inf),
                      direction * r["raw_metrics"]["best_fraction_mean"]
                      if r["raw_metrics"].get("best_fraction_mean") is not None else math.inf,
                      r["parameter_id"]))
        result.extend(band)
        ids = {id(r) for r in band}
        pending = [r for r in pending if id(r) not in ids]
    return result


def unique_parameters(values) -> list[dict]:
    return list({parameter_id(p): p for p in values}.values())


class Campaign:
    def __init__(self, plan: dict, out: str | Path, *, runner=None, domain=None):
        self.plan = copy.deepcopy(plan)
        self.out = Path(out).resolve()
        self.domain = domain if domain is not None else design_space_info(plan)
        self.manifest_path = self.out / "manifest.json"
        contract = {"plan": plan, "domain": self.domain,
                    "software_digest": package_digest(), "assets": asset_fingerprints(plan)}
        source_paths = {"hpo_yaml": plan["source_path"], "base_yaml": plan["base_config"]}
        if not plan["parameter_profile"].startswith("builtin:"):
            source_paths["profile_yaml"] = plan["parameter_profile"]
        contract["configuration_sources"] = {
            name: {"path": path, "sha256": file_digest(path)} for name, path in source_paths.items()}
        self.contract_digest = digest(contract)
        self.software_migration = None
        if self.manifest_path.exists():
            saved = read_json(self.manifest_path)
            original = {key: saved.get(key) for key in contract}
            if digest(original) != saved.get("contract_digest"):
                raise CampaignError("Campaign manifest contract is damaged or was modified; preserve it for inspection")
            if saved["contract_digest"] != self.contract_digest:
                upgrade = approved_upgrade(original["software_digest"], contract["software_digest"], kind="package")
                comparable = {**original, "software_digest": contract["software_digest"]}
                if upgrade is None or comparable != contract:
                    raise CampaignError("Campaign configuration, software, domain or assets changed; "
                                        "only a registered checkpoint-storage repair can reuse an older campaign")
                self.software_migration = {
                    "migration": upgrade, "from_software_digest": original["software_digest"],
                    "to_software_digest": contract["software_digest"],
                    "original_contract_digest": saved["contract_digest"],
                    "current_contract_digest": self.contract_digest,
                }
        else:
            if self.out.exists() and any(self.out.iterdir()):
                raise CampaignError(f"Output directory is nonempty and is not this HPO campaign: {self.out}")
            self.out.mkdir(parents=True, exist_ok=True)
            atomic_json(self.manifest_path, {**contract, "contract_digest": self.contract_digest,
                        "created_at": datetime.now(timezone.utc).isoformat(), "protocol": plan.get("protocol", "hpo_v1")})
        self.runner = runner or JobRunner(self.plan, self.out)
        self.completed_jobs = 0
        self.known_seconds = []
        self.parallel_jobs = self.plan.get("parallel_jobs", 1)
        ranges = self.stage_path("ranges")
        if ranges.exists() and read_json(ranges).get("status") == "COMPLETED":
            self.plan["search_space"] = read_json(ranges)["ranges"]
        self.plan["_budget_grids"] = {}

    def stage_path(self, stage):
        return self.out / "stages" / f"{stage}.json"

    def progress(self, event: str, **fields):
        row = {"event": event, **fields}
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        atomic_json(self.out / "progress.json", row)

    def evaluate(self, parameters: dict, budget: int, seeds: list[int], phase="train", models=None,
                 pairings=None):
        return self.evaluate_many([dict(parameters=parameters, budget=budget, seeds=seeds,
                                        phase=phase, models=models, pairings=pairings)])[0]

    def evaluate_many(self, requests):
        """Flatten configurations × seeds under one global worker limit."""
        groups, jobs, identities = [], [], {}
        for request in requests:
            parameters, budget, seeds = request["parameters"], request["budget"], request["seeds"]
            phase = request.get("phase", "train")
            models, pairings = request.get("models"), request.get("pairings")
            if models is not None and len(models) != len(seeds):
                raise CampaignError("Each generation seed requires exactly one paired frozen model")
            if pairings is not None and (models is None or len(pairings) != len(seeds)):
                raise CampaignError("Generation pairing metadata must match the models and seeds")
            preflight = training_budget_preflight(self.plan, parameters, budget, self.domain) if phase == "train" else None
            group = dict(parameters=parameters, budget=budget, seeds=seeds, phase=phase,
                         pairings=pairings, preflight=preflight, indices=[], rejected=[])
            if preflight is not None and not preflight["eligible"]:
                self.progress("preflight_rejected", phase=phase, budget=budget,
                              parameter_id=parameter_id(parameters), **preflight)
                for seed in seeds:
                    group["rejected"].append({
                        "status": "INFEASIBLE_LEARNING_BUDGET", "eligible_for_selection": False,
                        "score": -1.0, "seed": seed, "budget": budget, "executed": False,
                        "reason": preflight["reason"], "required_budget": preflight["required_budget"],
                        "failure_kind": preflight["failure_kind"], "wall_seconds": 0.0,
                        "checkpoint_paths": {}, "raw_metrics": {},
                        "ledger": {"attempts_started": 0, "unique_evaluated": 0,
                                   "target_unique": budget, "status": "INFEASIBLE_LEARNING_BUDGET"}})
            else:
                for index, seed in enumerate(seeds):
                    job = dict(parameters=parameters, budget=budget, seed=seed, phase=phase,
                               model=None if models is None else models[index])
                    identity = digest(job)
                    if identity not in identities:
                        identities[identity] = len(jobs)
                        jobs.append(job)
                    group["indices"].append(identities[identity])
            groups.append(group)

        def started(index, job):
            self.progress("job", phase=job["phase"], budget=job["budget"], seed=job["seed"],
                          parameter_id=parameter_id(job["parameters"]), completed_jobs=self.completed_jobs,
                          parallel_jobs=self.parallel_jobs, batch_job_index=index,
                          scheduled_jobs=len(jobs))

        def recorded(index, row):
            self.completed_jobs += 1
            if row.get("wall_seconds") is not None:
                self.known_seconds.append(row["wall_seconds"])
            if row.get("status") in ("SYSTEM_ERROR", "INVALID_RECORDS", "QUALITY_CALIBRATION_REQUIRED"):
                raise CampaignError(f"Job {row.get('job_id', '')} has {row['status']}: "
                                    f"{row.get('reason') or row.get('error') or 'inspect its preserved records'}; "
                                    f"inspect {row.get('directory', '')}")

        def progressed(active, completed, queued):
            self.progress("parallel_batch", parallel_jobs=self.parallel_jobs,
                          scheduled_jobs=len(jobs), completed_batch_jobs=completed,
                          remaining_scheduled_jobs=len(jobs) - completed, queued_jobs=queued,
                          completed_jobs=self.completed_jobs,
                          active_jobs=[{key: jobs[index][key] for key in ("phase", "budget", "seed")}
                                       | {"parameter_id": parameter_id(jobs[index]["parameters"])}
                                       for index in active])

        rows = run_jobs(jobs, self.runner, self.parallel_jobs, on_start=started,
                        on_result=recorded, on_progress=progressed)
        results = []
        for group in groups:
            parameters, budget, seeds, phase = (group[key] for key in ("parameters", "budget", "seeds", "phase"))
            pairings, preflight = group["pairings"], group["preflight"]
            selected = group["rejected"] or [rows[index] for index in group["indices"]]
            if pairings is not None:
                selected = [{**row, **pairing} for row, pairing in zip(selected, pairings)]
            result = aggregate_results(selected, parameters, budget)
            if pairings is not None:
                grouped = {}
                for row in selected:
                    grouped.setdefault(row["model_index"], []).append(row)
                result["model_results"] = [
                    {**aggregate_results(group_rows, parameters, budget),
                     "model_index": model_index, "model_id": group_rows[0]["model_id"],
                     "training_seed": group_rows[0]["training_seed"]}
                    for model_index, group_rows in sorted(grouped.items())]
                result["generation_pairings"] = pairings
            if preflight is not None and not preflight["eligible"]:
                result.update(preflight=preflight, reason=preflight["reason"], required_budget=preflight["required_budget"])
            measurement_key = {"parameters": parameters, "budget": budget, "seeds": seeds, "phase": phase}
            if pairings is not None:
                measurement_key["generation_pairings"] = pairings
            location = self.out / "measurements" / (digest(measurement_key)[:24] + ".json")
            atomic_json(location, result)
            results.append(result)
        return results

    def grids(self, phase):
        minimum = minimum_training_budget(self.plan, self.domain) if phase == "train" else 1
        size = self.domain.get("size")
        grid = budget_grid(self.plan, phase, size, minimum)
        self.plan["_budget_grids"][phase] = grid
        return grid, budget_schedule(self.plan, phase, size, minimum)

    def run_ranges(self):
        grid, schedule = self.grids("train")
        initial = schedule[0]
        def evaluate(params, seeds=None):
            return self.evaluate(params, initial, seeds or self.plan["range_search"]["seeds"])

        def evaluate_many(parameters, seeds=None):
            return self.evaluate_many([dict(parameters=params, budget=initial,
                                           seeds=seeds or self.plan["range_search"]["seeds"])
                                       for params in parameters])

        result = run_range_search(self.plan, evaluate,
                                  **({"evaluate_many": evaluate_many} if self.parallel_jobs > 1 else {}))
        result.update(diagnostic_budget=initial, budget_grid=grid)
        if result["status"] == "COMPLETED":
            self.plan["search_space"] = result["ranges"]
            atomic_json(self.out / "ranges.json", result["ranges"])
            with (self.out / "range_decisions.jsonl").open("w") as stream:
                for decision in result["decisions"]:
                    stream.write(json.dumps(decision, allow_nan=False) + "\n")
        return result

    def panel(self):
        ranges = read_json(self.stage_path("ranges"))
        ranked = rank_results(ranges.get("evaluations", []), self.plan)
        if ranges.get("reference_result", {}).get("eligible_for_selection"):
            reference = ranges["reference"]
        elif ranked:
            # Confirmation seeds may invalidate the original working reference.
            # Keep its history as evidence, but never revive its earlier success.
            reference = ranked[0]["parameters"]
        else:
            return []
        slow = dict(reference)
        for name in ("epsilon_anneal_budget_ratio", "pg_entropy_coef"):
            if name in self.plan["search_space"]:
                spec = self.plan["search_space"][name]
                slow[name] = max(spec["choices"]) if spec["type"] == "categorical" else spec["high"]
        return unique_parameters([reference, *(r["parameters"] for r in ranked[:1]), slow])

    def scan(self, panel, phase, seeds, models=None, pairings=None):
        _, schedule = self.grids(phase)
        results = []
        total = len(schedule) * len(panel) * len(seeds)
        if self.parallel_jobs > 1:
            self.progress("budget_scan", phase=phase, scheduled_jobs=total,
                          remaining_scheduled_jobs=total, estimated_remaining_seconds=None,
                          parallel_jobs=self.parallel_jobs)
            return self.evaluate_many([dict(parameters=parameters, budget=budget, seeds=seeds,
                                            phase=phase, models=models, pairings=pairings)
                                       for budget in schedule for parameters in panel])
        for budget in schedule:
            for parameters in panel:
                self.progress("budget_scan", phase=phase, budget=budget, scheduled_jobs=total,
                              remaining_scheduled_jobs=total - len(results) * len(seeds),
                              estimated_remaining_seconds=(statistics.mean(self.known_seconds) *
                                  (total - len(results) * len(seeds))) if self.known_seconds else None)
                results.append(self.evaluate(parameters, budget, seeds, phase, models, pairings))
        return results

    def run_train_budget(self):
        panel = self.panel()
        results = self.scan(panel, "train", self.plan["range_search"]["seeds"])
        selection = select_budget(results, self.plan, "train")
        return {"status": "COMPLETED" if selection.get("selected_budget") else selection["convergence_status"],
                "panel": panel, "results": results, "selection": selection}

    def train_study(self, budget: int, scope: str, trials: int, startup: int) -> list[dict]:
        import optuna
        from optuna.trial import TrialState
        database = self.out / "study.db"
        name = f"{scope}-b{budget}-{digest(self.plan['search_space'])[:10]}"
        study = optuna.create_study(storage=f"sqlite:///{database}", study_name=name,
                                   direction="maximize", load_if_exists=True)
        results_dir = self.out / "studies" / name
        results_dir.mkdir(parents=True, exist_ok=True)
        # Random startup proposals are independent. Adaptive TPE proposals wait
        # for all preceding scores; only their seed repeats execute concurrently.
        while True:
            frozen = study.get_trials(deepcopy=False)
            running = []
            for trial in frozen:
                if trial.state != TrialState.RUNNING:
                    continue
                if set(trial.params) != set(self.plan["search_space"]):
                    study.tell(trial.number, state=TrialState.FAIL)
                else:
                    running.append((trial.number, dict(trial.params)))
            finished_count = sum(t.state == TrialState.COMPLETE for t in frozen)
            if not running and finished_count >= trials:
                break
            target = min(trials, startup) if finished_count < startup else min(trials, finished_count + 1)
            count = max(0, target - finished_count - len(running))
            if self.parallel_jobs == 1:
                count = min(count, 1)
                running = running[:1]
            for _ in range(count):
                number = len(study.get_trials(deepcopy=False))
                study.sampler = optuna.samplers.TPESampler(
                    seed=(self.plan["training_search"]["seeds"][0] + number) % (2**32),
                    n_startup_trials=startup)
                trial = study.ask()
                parameters = {}
                for key, spec in self.plan["search_space"].items():
                    if spec["type"] == "categorical":
                        parameters[key] = trial.suggest_categorical(key, spec["choices"])
                    elif spec["type"] == "int":
                        parameters[key] = trial.suggest_int(key, spec["low"], spec["high"],
                            step=spec.get("step", 1), log=spec.get("log", False))
                    else:
                        parameters[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
                running.append((trial.number, parameters))
            pending, observed = [], {}
            for number, parameters in running:
                result_path = results_dir / f"trial_{number:05d}.json"
                if result_path.exists():
                    observed[number] = read_json(result_path)
                else:
                    pending.append((number, parameters))
            if pending:
                evaluated = self.evaluate_many([dict(parameters=parameters, budget=budget,
                                                       seeds=self.plan["training_search"]["seeds"])
                                                for _, parameters in pending])
                observed.update({number: result for (number, _), result in zip(pending, evaluated)})
            for number, _ in running:
                result = observed[number]
                atomic_json(results_dir / f"trial_{number:05d}.json", result)
                study.tell(number, result["score"] if result["eligible_for_selection"] else -1.)
        return [read_json(path) for path in sorted(results_dir.glob("trial_*.json"))]

    def run_train_search(self):
        initial = read_json(self.stage_path("train-budget"))
        budget = initial["selection"]["selected_budget"]
        settings = self.plan["training_search"]
        panel = initial["panel"]
        studies = []
        selected = None
        for revision in range(settings["max_budget_revisions"] + 1):
            trials = settings["trials"] if revision == 0 else settings["additional_trials_per_revision"]
            startup = settings["startup_random_trials"] if revision == 0 else settings["additional_random_trials"]
            measured = self.train_study(budget, f"revision-{revision}", trials, startup)
            studies.append({"budget": budget, "results": measured})
            # TPE may propose an identical configuration more than once,
            # especially in a small categorical space. A duplicate trial must
            # not consume another validation slot and exclude a distinct runner-up.
            distinct_ranked = {}
            for row in rank_results(measured, self.plan):
                distinct_ranked.setdefault(row["parameter_id"], row)
            finalists = list(distinct_ranked.values())[:settings["finalists"]]
            if not finalists:
                return {"status": "NO_FEASIBLE_CONFIGURATION", "studies": studies}
            panel = unique_parameters([*panel, *(r["parameters"] for r in finalists)])
            validated = self.scan(panel, "train", settings["validation_seeds"])
            selection = select_budget(validated, self.plan, "train")
            if not selection.get("selected"):
                return {"status": selection["convergence_status"], "studies": studies, "selection": selection}
            selected = selection["selected"]
            if selected["budget"] == budget:
                models = selected["seed_results"]
                if self.plan.get("protocol") != "hpo_v2" and len(models) != len(self.plan["generation_search"]["paired_seeds"]):
                    raise CampaignError("generation paired_seeds must match training validation_seeds count")
                result = {"status": "COMPLETED", "selection": selection, "studies": studies,
                          "results": validated, "models": models, "budget_revisions": revision}
                (self.out / "best_training.yaml").write_text(yaml.safe_dump({
                    "method": self.plan["method"], "train_budget": budget,
                    "parameters": selected["parameters"],
                    "convergence_status": selection["convergence_status"]}, sort_keys=False))
                atomic_json(self.out / "frozen_models.json", models)
                return result
            budget = selected["budget"]
        return {"status": "BUDGET_UNRESOLVED", "selection": selection, "studies": studies,
                "results": validated, "budget_revisions": settings["max_budget_revisions"]}

    def generation_pairs(self, models):
        """Build a reproducible model × sampling-repeat experiment, once.

        Pair identities are independent of temperature/epsilon/top-fraction
        and budget so all settings use the same frozen networks and RNG pairs.
        """
        settings = self.plan["generation_search"]
        if self.plan.get("protocol") != "hpo_v2":
            return settings["paired_seeds"], models, None
        repeats = settings["repeats_per_model"]
        master_seed = settings["master_seed"]
        seeds, expanded_models, pairings = [], [], []
        used = set(self.plan["training_search"]["seeds"])
        used.update(self.plan["training_search"]["validation_seeds"])
        confirmation = self.plan["range_search"].get("confirmation_seed")
        if confirmation is not None:
            used.add(confirmation)
        for model_index, model in enumerate(models):
            model_id = model.get("job_id") or digest({
                "training_seed": model["seed"], "budget": model["budget"],
                "checkpoint_paths": model.get("checkpoint_paths", {}),
                "artifact_hashes": model.get("artifact_hashes", {})})[:24]
            for repeat_index in range(repeats):
                seed = int(digest({"master_seed": master_seed, "phase": "generation",
                                  "model_index": model_index, "repeat_index": repeat_index})[:8], 16)
                while seed in used:
                    seed = (seed + 1) % (2**32)
                used.add(seed)
                seeds.append(seed)
                expanded_models.append(model)
                pairings.append({"seed": seed, "model_index": model_index,
                                 "model_id": model_id, "training_seed": model["seed"],
                                 "generation_repeat_index": repeat_index})
        atomic_json(self.out / "generation_pairings.json", pairings)
        return seeds, expanded_models, pairings

    @staticmethod
    def generation_extensions(parameter, values):
        """Return legal next endpoint probes for a selected sampling method."""
        if parameter == "gen_temperature":
            positives = [value for value in values if value > 0]
            if not positives:
                return []
            return [("low", positives[0], positives[0] / 10),
                    ("high", positives[-1], positives[-1] * 10)]
        if parameter not in {"gen_epsilon", "gen_top_frac"}:
            raise CampaignError(f"Unsupported generation parameter: {parameter}")
        # These parameters have exact finite domains; reach the legal boundary
        # directly rather than inventing an unbounded multiplicative scale.
        return [("low", values[0], 0.0), ("high", values[-1], 1.0)]

    def run_generation(self):
        trained = read_json(self.stage_path("train-search"))
        params = trained["selection"]["selected"]["parameters"]
        models = trained["models"]
        settings = self.plan["generation_search"]
        parameter = settings.get("parameter", "gen_temperature")
        values = sorted(set(settings["initial_values"]))
        seeds, paired_models, pairings = self.generation_pairs(models)
        extensions = {"low": 0, "high": 0}
        results = []
        measured = set()
        while True:
            added = [value for value in values if value not in measured]
            if added:
                results += self.scan([{**params, parameter: value} for value in added],
                                     "generation", seeds, paired_models, pairings)
                measured.update(added)
            proposed = []
            by_budget = {}
            for row in results:
                by_budget.setdefault(row["budget"], []).append(row)
            winners = [rank_results(rows, self.plan)[0] for rows in by_budget.values()
                       if rank_results(rows, self.plan)]
            for direction, endpoint, next_value in self.generation_extensions(parameter, values):
                boundary_wins = any(row["parameters"][parameter] == endpoint for row in winners)
                all_failed = not winners and direction == "high"
                if (boundary_wins or all_failed) and extensions[direction] < settings["max_extra_values_per_direction"]:
                    if math.isfinite(next_value) and next_value >= 0 and next_value not in measured:
                        extensions[direction] += 1
                        proposed.append(next_value)
            if not proposed:
                break
            values = sorted(set([*values, *proposed]))
        selection = select_budget(results, self.plan, "generation")
        status = "COMPLETED" if selection.get("selected") else selection["convergence_status"]
        if status == "COMPLETED":
            best = selection["selected"]
            if self.plan.get("protocol") == "hpo_v2":
                # A standalone export should express the one selected method,
                # without exposing implementation-only disabled alternatives.
                output = {"num_gen_eps": best["budget"], parameter: best["parameters"][parameter],
                          "convergence_status": selection["convergence_status"]}
            else:
                output = {"num_gen_eps": best["budget"], "gen_temperature": best["parameters"]["gen_temperature"],
                          "gen_epsilon": 0., "gen_top_frac": 0., "exploration_gen_eps": 0,
                          "convergence_status": selection["convergence_status"]}
            (self.out / "best_generation.yaml").write_text(yaml.safe_dump(output, sort_keys=False))
        result = {"status": status, "selection": selection, "results": results,
                  "parameter": parameter, "parameter_values": values}
        if parameter == "gen_temperature":
            result["temperatures"] = values  # Preserve readers of v1 reports.
        if pairings is not None:
            result["generation_pairings"] = pairings
        return result

    def record_software_migration(self):
        if self.software_migration is None:
            return
        path = self.out / "software_migrations.json"
        history = read_json(path) if path.exists() else {"migrations": []}
        entries = history.get("migrations")
        if not isinstance(entries, list):
            raise CampaignError("Invalid software migration history; preserve it for inspection")
        if not any(all(row.get(key) == value for key, value in self.software_migration.items())
                   for row in entries):
            entries.append({**self.software_migration,
                            "recorded_at": datetime.now(timezone.utc).isoformat()})
            atomic_json(path, history)

    def run(self, scope="all"):
        if scope not in SCOPES:
            raise CampaignError(f"Unknown stage {scope}")
        self.record_software_migration()
        state = {"requested_stage": scope, "status": "RUNNING", "completed_stages": []}
        atomic_json(self.out / "state.json", state)
        try:
            for stage in SCOPES[scope]:
                index = STAGES.index(stage)
                if index:
                    predecessor = self.stage_path(STAGES[index - 1])
                    if not predecessor.exists() or read_json(predecessor).get("status") != "COMPLETED":
                        raise CampaignError(f"Stage {stage} requires completed {STAGES[index - 1]}; "
                                            f"run --stage {STAGES[index - 1]} first")
                path = self.stage_path(stage)
                result = read_json(path) if path.exists() else None
                if result is None or result.get("status") not in ("COMPLETED", "NO_FEASIBLE_CONFIGURATION", "NO_FEASIBLE_MATERIAL_EVIDENCE", "BUDGET_UNRESOLVED"):
                    self.progress("stage_start", stage=stage)
                    result = getattr(self, "run_" + stage.replace("-", "_"))()
                    atomic_json(path, result)
                if result["status"] != "COMPLETED":
                    state.update(status=result["status"], current_stage=stage)
                    atomic_json(self.out / "state.json", state)
                    return state
                state["completed_stages"].append(stage)
                atomic_json(self.out / "state.json", state)
            state["status"] = "COMPLETED"
        except KeyboardInterrupt:
            state["status"] = "INTERRUPTED"
            atomic_json(self.out / "state.json", state)
            raise
        except Exception as exc:
            state.update(status="SYSTEM_ERROR", reason=str(exc))
            atomic_json(self.out / "state.json", state)
            raise
        finally:
            atomic_json(self.out / "state.json", state)
            try:
                self.export_reports()
            except Exception as report_error:
                # Keep the original run outcome, including KeyboardInterrupt;
                # failed reporting must not hide the causal job diagnostic.
                state["report_error"] = str(report_error)
                atomic_json(self.out / "state.json", state)
        atomic_json(self.out / "state.json", state)
        return state

    def export_reports(self):
        from .reporting import export_reports
        export_reports(self.out)
