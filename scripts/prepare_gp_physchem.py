"""Prepare the external GP catalog and write a usable RL configuration.

Run this with the RL Python; --gp-python points to the isolated GP environment.
The GP's chemistry/descriptor code is used by integrations/gp_physchem/worker.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import yaml


def make_config(prepared: Path, gp_python: Path, worker: Path, *, smoke=False) -> dict:
    """Use only the prepared original catalog; never union training monomers."""
    return {
        "groups": [
            {"name": "n_terminal", "kind": "categorical", "choices": [
                {"name": "treatment", "element": "treatment", "values": ["H", "Ac"]}
            ]},
            {"name": "peptide", "kind": "sequence", "length": 11,
             "tokens": {"csv": str(prepared / "catalog.csv"), "id_column": "monomer"},
             "allow_repeats": True,
             "count_constraints": [
                 {"where": {"column": "has_free_thiol", "equals": "1"}, "min": 2, "max": 2}
             ]},
        ],
        "properties": [
            {"name": "pIC50",
             "predictor": "rl_matdesign.predictors.json_process:JsonProcessPredictor",
             "command": [str(gp_python), "-u", str(worker), "serve", "--prepared", str(prepared)],
             "startup_timeout": 600, "request_timeout": 120,
             "direction": "max", "objective": "mean_minus_kstd"},
        ],
        "k": 1.0,
        "method": "a2c",
        "pg_warmup_eps": 2 if smoke else 100,
        "pg_num_iters": 2 if smoke else 100,
        "pg_batch_eps": 2 if smoke else 16,
        "pg_lr_actor": 0.001, "pg_lr_critic": 0.001, "pg_entropy_coef": 0.05,
        "dqn_warmup_eps": 2 if smoke else 100,
        "dqn_num_train_eps": 3 if smoke else 1600,
        "dqn_buffer_size": 50000, "dqn_batch_size": 32 if smoke else 256,
        "dqn_grad_steps_per_ep": 1, "dqn_target_update_freq": 100,
        "dqn_eps_anneal_eps": 3 if smoke else 1000, "dqn_eps_min": 0.05,
        "dqn_augment_permutations": 0,
        "gamma": 0.99, "gen_temperature": 1.0,
        "num_gen_eps": 3 if smoke else 100,
        "exploration_gen_eps": 1 if smoke else 20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp-root", type=Path, required=True)
    parser.add_argument("--gp-python", type=Path, required=True)
    parser.add_argument("--new-monomers", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/gp_physchem/prepared"))
    parser.add_argument("--config", type=Path, default=Path("configs/peptide_gp.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Use tiny validation budgets")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    worker = root / "integrations/gp_physchem/worker.py"
    gp_root, gp_python = args.gp_root.expanduser().resolve(), args.gp_python.expanduser().absolute()
    if not gp_python.is_file():
        parser.error(f"GP interpreter does not exist: {gp_python}")
    prepared = args.out.resolve()
    command = [str(gp_python), "-u", str(worker), "prepare",
               "--gp-root", str(gp_root), "--out", str(prepared)]
    if args.new_monomers:
        command += ["--new-monomers", str(args.new_monomers.resolve())]
    subprocess.run(command, check=True)
    with (prepared / "manifest.json").open() as stream:
        manifest = json.load(stream)
    cfg = make_config(prepared, gp_python, worker, smoke=args.smoke)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(
        "# Generated from the actual gp_physchem catalog. Budgets are starting settings, not tuned.\n"
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    print(f"Prepared {manifest['n_tokens']} monomers; wrote {args.config.resolve()}")
    print("Reward: raw pIC50 mean - posterior_sd (k=1); edit objective/k in YAML as needed.")


if __name__ == "__main__":
    main()
