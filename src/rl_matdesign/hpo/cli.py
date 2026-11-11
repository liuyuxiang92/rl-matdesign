"""Installed, scenario-independent HPO command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..config_schema import ConfigError
from .artifacts import read_json
from .spec import load_plan


def parser():
    root = argparse.ArgumentParser(prog="rl-matdesign hpo", description="Two-part, YAML-configured material-design HPO")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--hpo-config", required=True)
        command.add_argument("--method", choices=("a2c", "dqn-bootstrap", "dqn-mc"))
        if name == "run":
            command.add_argument("--out", required=True)
            command.add_argument("--stage", choices=("all", "ranges", "search", "train-budget", "train-search", "generation"), default="all")
    for name in ("status", "resume"):
        commands.add_parser(name).add_argument("--out", required=True)
    return root


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    from .controller import Campaign, CampaignError
    from .job_runner import JobError
    try:
        if args.command == "status":
            directory = Path(args.out)
            _print({name: read_json(directory / f"{name}.json") for name in
                    ("state", "progress") if (directory / f"{name}.json").exists()})
            if not (directory / "manifest.json").exists():
                raise CampaignError("No HPO campaign manifest exists in --out")
            return 0
        if args.command == "resume":
            manifest = read_json(Path(args.out) / "manifest.json")
            state = read_json(Path(args.out) / "state.json")
            plan = load_plan(manifest["plan"]["source_path"], method=manifest["plan"]["method"])
            campaign = Campaign(plan, args.out)
            result = campaign.run(state["requested_stage"])
            _print(result)
            return 0 if result["status"] == "COMPLETED" else 3
        plan = load_plan(args.hpo_config, method=args.method)
        from .domain import design_space_info, minimum_training_budget
        from .budget_search import budget_grid, budget_schedule
        domain = design_space_info(plan)
        minimum = minimum_training_budget(plan, domain)
        grids = {phase: budget_grid(plan, phase, domain.get("size"), minimum if phase == "train" else 1)
                 for phase in ("train", "generation")}
        if args.command == "validate":
            _print({"valid": True, "method": plan["method"], "domain": domain,
                    "budget_grids": grids, "predictor_evaluated": False})
        elif args.command == "plan":
            _print({"resolved_plan": plan, "domain": domain, "budget_grids": grids,
                    "schedule": {phase: budget_schedule(plan, phase, domain.get("size"), minimum if phase == "train" else 1)
                                 for phase in ("train", "generation")},
                    "stages": ["ranges", "train-budget", "train-search", "generation"],
                    "eta": "Available after measured jobs; adaptive range/budget additions remain bounded but conditional",
                    "predictor_evaluated": False})
        else:
            result = Campaign(plan, args.out, domain=domain).run(args.stage)
            _print(result)
            return 0 if result["status"] == "COMPLETED" else 3
        return 0
    except (ValueError, ConfigError, CampaignError, JobError, FileNotFoundError) as exc:
        print(f"HPO error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("HPO interrupted; recorded candidate costs are preserved.", file=sys.stderr)
        return 130
