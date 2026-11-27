"""Run paired seeds crossed with training budgets in isolated Python processes.

The scheduler owns no scientific RNG or predictor. Each child executes the same
single-run entry point with a frozen scalar configuration in its own directory.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import yaml

from .config_schema import ConfigError, ResolvedRunConfigs
from .utils.candidate_budget import CandidateBudget


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _child_command(args, config: Path, output: Path) -> list[str]:
    command = [sys.executable, "-m", "rl_matdesign.experiment",
               "--config", str(config), "--out", str(output)]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.save_checkpoint_freq:
        command.extend(["--save-checkpoint-freq", str(args.save_checkpoint_freq)])
    if args.skip_generation:
        command.append("--skip-generation")
    return command


def _completion_status(row: dict, config: dict) -> str:
    """A strict HPO budget can be infeasible even when the child exits zero."""
    if config.get("hpo_recording"):
        try:
            budget = json.loads((Path(row["out"]) / "candidate_budget.json").read_text())
            status = budget["status"]
            row["budget_status"] = status
            if row["returncode"] != 0:
                return "failed"
            target = config["hpo_target_unique"]
            limit = CandidateBudget(target, config["hpo_ratio_limit"])
            if (budget.get("phase") != config["hpo_phase"]
                    or budget.get("target_unique") != target
                    or budget.get("ratio_limit") != float(config["hpo_ratio_limit"])):
                raise ValueError("Candidate budget identity differs from the requested config")
            if status in {"INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL", "INCOMPLETE"}:
                return status.lower()
            if status != "COMPLETED":
                return "failed"
            attempts = budget.get("attempts_started")
            if (type(attempts) is not int or not target <= attempts <= limit.max_attempts
                    or type(budget.get("unique_evaluated")) is not int
                    or budget["unique_evaluated"] != target or budget.get("pending") != 0):
                raise ValueError("Completed candidate budget does not meet the requested unique/attempt limits")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["error"] = f"Missing or invalid candidate budget: {exc}"
            return "failed"
    return "completed" if row["returncode"] == 0 else "failed"


def _stop_children(active: dict) -> None:
    # Each child owns a session, so its predictor subprocesses also receive the
    # shutdown signal. The child's own cleanup releases separately owned workers.
    for proc, _stream in active.values():
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for proc, stream in active.values():
        try:
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait()
        finally:
            stream.close()


@contextmanager
def _defer_launch_signals():
    """Register a spawned process before allowing interruption to unwind.

    Python handlers are deferred, rather than blocking OS signal masks (which
    a child would inherit). Caught handlers reset normally when the child execs.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    pending = []

    def defer(signum, frame):
        pending.append((signum, frame))

    try:
        for sig, handler in saved.items():
            if handler != signal.SIG_IGN:
                signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        if pending:
            signum, frame = pending[0]
            handler = saved[signum]
            if callable(handler):
                handler(signum, frame)
            else:
                raise SystemExit(128 + signum)


def run_seed_sweep(args, plan: ResolvedRunConfigs) -> int:
    """Execute every list entry once; retain all outcomes and return 1 on failure.

    Bulk resume/generation-from-artifacts is deliberately explicit: use one
    saved scalar config with its own output directory for those operations.
    """
    incompatible = [name for name in ("only_generate", "resume_training", "load_qnet",
                    "load_policy", "load_scaler", "load_value_net", "load_reward_calibration")
                    if getattr(args, name, None)]
    if incompatible:
        flags = ", ".join("--" + name.replace("_", "-") for name in incompatible)
        raise ConfigError(f"Seed/budget-list runs do not support {flags}. Use the saved scalar config.yaml "
                          "and --out for the individual run to load or resume its artifacts.")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".seed_runs.lock"
    try:
        reservation = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ConfigError(f"Seed batch already reserved: {output}. Choose a new --out directory.") from exc
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    manifest = None
    with reservation:
        reservation.write(str(os.getpid()) + "\n")
        reservation.flush()
        try:
            if any(path != lock for path in output.iterdir()):
                raise ConfigError(f"Seed/budget-list runs require an empty --out directory: {output}. "
                                  "Choose a new directory; existing results are never overwritten.")
            rows = []
            for index, resolved in enumerate(plan.runs, 1):
                cfg = resolved.config
                seeds = {key: cfg[key + "_seed"] for key in ("eval", "train", "gen")}
                name = (f"run_{index:03d}_eval_{seeds['eval']}_train_{seeds['train']}"
                        f"_gen_{seeds['gen']}")
                parent = output / f"budget_{cfg['train_budget']}" if plan.train_budget_values else output
                directory = parent / name
                row = {"index": index, "seeds": seeds, "out": str(directory),
                       "config": str(directory / "config.yaml"),
                       "log": str(directory / "run.log"), "status": "pending",
                       "returncode": None}
                if plan.train_budget_values:
                    row["train_budget"] = cfg["train_budget"]
                rows.append(row)
            manifest = {"version": 1, "status": "running", "run_count": len(rows),
                        "parallel_jobs": plan.parallel_jobs,
                        "source_config": str(Path(args.config).resolve()),
                        "seed_list_keys": list(plan.seed_list_keys), "runs": rows}
            if plan.train_budget_values:
                manifest["train_budget_values"] = list(plan.train_budget_values)
            summary = output / "seed_runs.json"
            _write_json(summary, manifest)
            # Freeze all effective settings before launching any child. Relative
            # scientific paths retain the caller's working-directory semantics.
            for row, resolved in zip(rows, plan.runs):
                directory = Path(row["out"])
                directory.mkdir(parents=True)
                Path(row["config"]).write_text(
                    yaml.safe_dump(resolved.config, sort_keys=False), encoding="utf-8")
                _write_json(directory / "seed_config_sources.json", resolved.sources)
            env = os.environ.copy()
            source = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            next_index = 0
            label = "budget/seed combinations" if plan.train_budget_values else "paired runs"
            print(f"[SEEDS] {len(rows)} {label}; at most {plan.parallel_jobs} simultaneously; "
                  f"summary={summary}", flush=True)
            while next_index < len(rows) or active:
                while next_index < len(rows) and len(active) < plan.parallel_jobs:
                    position = next_index
                    next_index += 1
                    row = rows[position]
                    command = _child_command(args, Path(row["config"]), Path(row["out"]))
                    row["command"] = command
                    with _defer_launch_signals():
                        stream = open(row["log"], "w", encoding="utf-8")
                        try:
                            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                    env=env, start_new_session=(os.name == "posix"))
                        except OSError as exc:
                            stream.close()
                            row.update(status="failed", error=f"Could not start child: {exc}")
                            _write_json(summary, manifest)
                            continue
                        active[position] = (proc, stream)
                        row.update(status="running", pid=proc.pid)
                    _write_json(summary, manifest)
                finished = []
                for position, (proc, stream) in active.items():
                    code = proc.poll()
                    if code is None:
                        continue
                    stream.close()
                    row = rows[position]
                    row["returncode"] = code
                    row["status"] = _completion_status(row, plan.runs[position].config)
                    finished.append(position)
                    print(f"[SEEDS] run {row['index']}/{len(rows)}: {row['status']} "
                          f"(log: {row['log']})", flush=True)
                for position in finished:
                    del active[position]
                if finished:
                    _write_json(summary, manifest)
                elif active:
                    time.sleep(0.05)
            complete = sum(row["status"] == "completed" for row in rows)
            manifest.update(status="completed" if complete == len(rows) else "failed",
                            completed_runs=complete, unsuccessful_runs=len(rows) - complete)
            _write_json(summary, manifest)
            return 0 if complete == len(rows) else 1
        except BaseException as exc:
            _stop_children(active)
            if manifest is not None:
                for position, (proc, _stream) in active.items():
                    manifest["runs"][position].update(status="interrupted", returncode=proc.returncode)
                for row in manifest["runs"]:
                    if row["status"] == "pending":
                        row["status"] = "cancelled"
                manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                else "failed", error=f"{type(exc).__name__}: {exc}")
                _write_json(output / "seed_runs.json", manifest)
            raise
        finally:
            lock.unlink(missing_ok=True)
