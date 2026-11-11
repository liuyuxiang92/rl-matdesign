"""Isolated train/generate jobs with durable cost and complete raw records."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .artifacts import atomic_json, checkpoint_files, digest, file_digest, read_json
from .scoring import score_records
from .spec import phase_efficiency
from .algorithm_profiles import resolve_parameters


class JobError(RuntimeError):
    """A job cannot safely be executed or resumed."""


class JobRunner:
    def __init__(self, plan: dict, out: str | Path):
        self.plan, self.out = plan, Path(out)

    def __call__(self, parameters: dict, budget: int, seed: int, phase: str,
                 model: dict | None = None) -> dict:
        if phase not in ("train", "generation"):
            raise ValueError(f"Unknown phase {phase}")
        if phase == "generation" and model is None:
            raise JobError("Generation requires a frozen training model")
        input_models = {} if model is None else {
            name: file_digest(path) for name, path in model["checkpoint_paths"].items()
            if name in ("policy.pt", "qnet.pt", "std_scaler.bin", "value_net.pt")}
        if model is not None:
            required = {"std_scaler.bin", "policy.pt" if self.plan["method"] == "a2c" else "qnet.pt"}
            if not required.issubset(input_models):
                raise JobError("Frozen model is missing its policy/Q network or scaler")
            expected_hashes = model.get("artifact_hashes", {})
            for name, observed in input_models.items():
                source = model["checkpoint_paths"][name]
                if expected_hashes.get(source) != observed:
                    raise JobError(f"Frozen model artifact changed or has no recorded fingerprint: {source}")
        key = {"parameters": parameters, "budget": budget, "seed": seed,
               "phase": phase, "input_models": input_models}
        job_id = digest(key)[:24]
        directory = self.out / "jobs" / job_id
        result_path = directory / "result.json"
        previous = read_json(result_path) if result_path.exists() else None
        if previous and previous["status"] not in ("INTERRUPTED", "INCOMPLETE", "SYSTEM_ERROR"):
            for path, expected in previous.get("artifact_hashes", {}).items():
                if not Path(path).is_file() or file_digest(path) != expected:
                    raise JobError(f"Completed job artifact changed: {path}")
            return previous
        directory.mkdir(parents=True, exist_ok=True)
        ledger_path = directory / "candidate_budget.json"
        resume = False
        recover_terminal = False
        if ledger_path.exists():
            old_ledger = read_json(ledger_path)
            recover_terminal = old_ledger.get("status") in {
                "COMPLETED", "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}
            if not recover_terminal and not old_ledger.get("attempts_started", 0):
                raise JobError(f"Job {job_id} has initialized accounting artifacts but no consumed attempts; "
                               "inspect the preserved initialization failure before starting a new campaign. "
                               "Existing ledgers are never silently erased.")
            if not recover_terminal and old_ledger.get("attempts_started", 0):
                resume_checkpoint = directory / ("generation_checkpoint.pt" if phase == "generation" else "checkpoint.pt")
                if not resume_checkpoint.exists():
                    raise JobError(f"Job {job_id} consumed attempts without a resumable checkpoint; "
                                   "cost is preserved and will not be reset. See candidate_budget.json.")
                resume = True

        train_budget = budget if model is None else model["budget"]
        cfg = copy.deepcopy(self.plan["base_config_data"])
        training_parameters = {name: value for name, value in parameters.items() if name != "gen_temperature"}
        cfg.update(resolve_parameters(self.plan, training_parameters, train_budget))
        efficiency = phase_efficiency(self.plan, phase)
        cfg.update(hpo_recording=True, hpo_phase=phase, hpo_target_unique=budget,
                   hpo_ratio_limit=efficiency["max_attempts_per_unique"],
                   train_count_duplicates=False,
                   train_max_duplicate_attempts=max(1000, 10 * budget),
                   exploration_gen_eps=0)
        if phase == "generation":
            cfg.update(num_gen_eps=budget, gen_epsilon=0.0, gen_top_frac=0.0,
                       gen_temperature=parameters["gen_temperature"],
                       max_gen_attempts=max(1000, 10 * budget))
            if resume:
                cfg["hpo_resume_generation"] = True
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        atomic_json(directory / "job.json", key)
        args = [sys.executable, "-m", "rl_matdesign.hpo.worker", "--config", str(config_path),
                "--out", str(directory), "--train-seed",
                str(seed if model is None else model["seed"]), "--gen-seed", str(seed)]
        if phase == "train":
            args += ["--skip-generation", "--save-checkpoint-freq", "1"]
            if resume:
                args.append("--resume-training")
        else:
            args.append("--only-generate")
            for name, flag in (("qnet.pt", "--load-qnet"), ("policy.pt", "--load-policy"),
                               ("std_scaler.bin", "--load-scaler"), ("value_net.pt", "--load-value-net")):
                if name in model["checkpoint_paths"]:
                    args += [flag, model["checkpoint_paths"][name]]
        environment = os.environ.copy()
        package_parent = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = package_parent + os.pathsep + environment.get("PYTHONPATH", "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment.setdefault(name, "1")
        started = time.monotonic()
        interrupted = False
        if recover_terminal:
            # A process may finish and persist its ledger before the parent
            # persists result.json. Reconstruct evidence, never buy it twice.
            returncode = None
            elapsed = None
        else:
            atomic_json(directory / "execution.json", {"status": "RUNNING", "argv": args,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "resume": resume, "previous": previous})
            with (directory / "stdout.log").open("a") as stdout, (directory / "stderr.log").open("a") as stderr:
                process = subprocess.Popen(args, cwd=self.plan["base_config_dir"], env=environment,
                                           stdout=stdout, stderr=stderr)
                try:
                    returncode = process.wait()
                except (KeyboardInterrupt, SystemExit):
                    interrupted = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    returncode = process.returncode
            elapsed = time.monotonic() - started
        ledger = read_json(ledger_path) if ledger_path.exists() else {
            "status": "INTERRUPTED" if interrupted else "SYSTEM_ERROR",
            "attempts_started": 0, "unique_evaluated": 0, "target_unique": budget}
        record_path = directory / "candidate_records.jsonl"
        records = []
        if record_path.exists():
            try:
                records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise JobError(f"Incomplete record archive preserved at {record_path}: {exc}") from exc
        scored = score_records(records, ledger, self.plan, phase)
        if interrupted:
            scored.update(status="INTERRUPTED", eligible_for_selection=False, score=-1.0)
        elif returncode and scored.get("status") not in ("INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"):
            scored.update(status="SYSTEM_ERROR", eligible_for_selection=False, score=-1.0)
        paths = checkpoint_files(directory) if phase == "train" else {}
        if phase == "train" and scored.get("eligible_for_selection"):
            required = {"std_scaler.bin", "policy.pt" if self.plan["method"] == "a2c" else "qnet.pt"}
            if not required.issubset(paths):
                scored.update(status="SYSTEM_ERROR", eligible_for_selection=False, score=-1.0,
                              reason="Completed training did not save all required model artifacts")
        if model:
            after = {name: file_digest(path) for name, path in model["checkpoint_paths"].items()
                     if name in input_models}
            if after != input_models:
                raise JobError("Frozen generation modified its input model or scaler")
        prior_time_unknown = resume and (previous is None or previous.get("wall_seconds") is None)
        result = {**scored, "job_id": job_id, **key, "parameters": parameters,
                  "checkpoint_paths": paths, "ledger": ledger, "directory": str(directory),
                  "records_path": str(record_path), "returncode": returncode,
                  "wall_seconds": None if elapsed is None or prior_time_unknown else elapsed + ((previous or {}).get("wall_seconds") or 0),
                  "current_execution_seconds": elapsed,
                  "recovered_from_terminal_artifacts": recover_terminal,
                  "timing_status": ("unknown_after_parent_interruption" if recover_terminal else
                                    "partial_after_resume" if prior_time_unknown else "measured"),
                  "artifact_hashes": {p: file_digest(p) for p in
                      [*paths.values(), str(record_path), str(ledger_path)] if Path(p).is_file()}}
        atomic_json(result_path, result)
        atomic_json(directory / "execution.json", {"status": result["status"], "argv": args,
                                                   "returncode": returncode, "wall_seconds": result["wall_seconds"]})
        if interrupted:
            raise KeyboardInterrupt
        return result
