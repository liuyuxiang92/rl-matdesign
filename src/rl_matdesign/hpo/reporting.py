"""Recomputable summaries; physical job costs are counted once, not per reuse."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .artifacts import atomic_json, read_json


def _csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def export_reports(out: str | Path) -> dict:
    out = Path(out)
    manifest_path = out / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    space_size = manifest.get("domain", {}).get("size")
    def fraction(budget):
        return budget / space_size if isinstance(space_size, int) and space_size > 0 else None
    jobs = [read_json(path) for path in sorted((out / "jobs").glob("*/result.json"))]
    rows = []
    for job in jobs:
        ledger = job.get("ledger", {})
        rows.append({"job_id": job["job_id"], "phase": job["phase"], "status": job["status"],
            "budget": job["budget"], "budget_fraction_of_space": fraction(job["budget"]),
            "seed": job["seed"], "parameters": job["parameters"],
            "eligible_for_selection": job["eligible_for_selection"], "score": job["score"],
            **job.get("raw_metrics", {}), "attempts_started": ledger.get("attempts_started", 0),
            "unique_evaluated": ledger.get("unique_evaluated", 0),
            "wall_seconds": job.get("wall_seconds"),
            "timing_status": job.get("timing_status", "complete" if job.get("wall_seconds") is not None else "unknown")})
    _csv(out / "trials.csv", [r for r in rows if r["phase"] == "train"])
    _csv(out / "generation_trials.csv", [r for r in rows if r["phase"] == "generation"])
    budgets, selections = [], {}
    for stage in ("train-budget", "train-search", "generation"):
        path = out / "stages" / f"{stage}.json"
        if not path.exists():
            continue
        value = read_json(path)
        selections[stage] = {"status": value["status"], "selection": value.get("selection")}
        for r in value.get("results", []):
            budgets.append({"stage": stage, "parameter_id": r["parameter_id"], "budget": r["budget"],
                "budget_fraction_of_space": fraction(r["budget"]),
                "seeds": r["seeds"], "score": r["score"], "eligible_for_selection": r["eligible_for_selection"],
                **r["raw_metrics"]})
    _csv(out / "budget_results.csv", budgets)
    # A killed process may have durable counters but no final result.json.
    known = {j["job_id"] for j in jobs}
    unfinished = []
    for path in sorted((out / "jobs").glob("*/candidate_budget.json")):
        if path.parent.name not in known:
            unfinished.append({"job_id": path.parent.name, "ledger": read_json(path)})
    cost = {}
    for phase in ("train", "generation"):
        completed_rows = [row for row in rows if row["phase"] == phase]
        pending_rows = [row for row in unfinished if row["ledger"].get("phase") == phase]
        known_seconds = sum(row["wall_seconds"] for row in completed_rows if row["wall_seconds"] is not None)
        unknown_time_jobs = sum(row["wall_seconds"] is None for row in completed_rows) + len(pending_rows)
        cost[phase] = {
            "jobs": len(completed_rows) + len(pending_rows),
            "unfinished_jobs": len(pending_rows),
            "attempts_started": sum(row["attempts_started"] for row in completed_rows)
                + sum(row["ledger"].get("attempts_started", 0) for row in pending_rows),
            "unique_evaluated": sum(row["unique_evaluated"] for row in completed_rows)
                + sum(row["ledger"].get("unique_evaluated", 0) for row in pending_rows),
            "wall_seconds": known_seconds if not unknown_time_jobs else None,
            "known_wall_seconds": known_seconds,
            "jobs_with_unknown_total_time": unknown_time_jobs,
        }
    summary = {"selections": selections, "physical_cost": cost,
               "design_space_size": space_size,
               "unfinished_jobs_with_cost": unfinished,
               "cost_scope": "Distinct job records; candidate credits are phase-local, not campaign-global novelty"}
    atomic_json(out / "selection.json", summary)
    return summary
