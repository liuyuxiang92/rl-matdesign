"""Summarize complete DQN replay trajectories using the current design schema.

FIFO eviction and deduplicated augmentation can leave partial trajectories.
Only consecutive paths starting at step zero and ending at done are decoded.
Typed action indices are replayed through the configured groups; no formula
parsing or normalization is used to reconstruct scientific candidates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class Episode:
    formula: str
    candidate: dict
    terminal_reward: float
    n_rows: int


def _split_episodes(buffer):
    current = []
    for row in buffer:
        step = int(np.argmax(row["s_step"]))
        if step == 0:
            current = []
        if step != len(current):
            current = []
            continue
        current.append(row)
        if row["done"]:
            yield current
            current = []


def _decode_episodes(buffer, *, env):
    result = []
    original_reward = env.reward_fn
    try:
        env.reward_fn = lambda _: 0.0
        for rows in _split_episodes(buffer):
            env.initialize()
            for row in rows:
                label, value = int(row["a_elem_idx"]), int(row["a_comp_idx"])
                action = (tuple(float(i == label) for i in range(len(env.species_set))),
                          tuple(float(i == value) for i in range(len(env.action_values))))
                env.step(action)
            if env.is_terminal:
                result.append(Episode(env.terminal_formula, deepcopy(env.terminal_cation_fractions()),
                                      float(sum(row["reward"] for row in rows)), len(rows)))
    finally:
        env.reward_fn = original_reward
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-csv")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    import torch
    from rl_matdesign.configuration import build_env, build_predictor
    from rl_matdesign.utils.model_contract import environment_contract, require_contract
    from rl_matdesign.utils.composition import composition_key

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    env, _, _ = build_env(cfg, object())
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("type") != "dqn" or "buffer" not in saved:
        parser.error("checkpoint must contain a DQN replay buffer")
    require_contract(saved.get("environment_contract"), environment_contract(env), str(checkpoint))
    episodes = _decode_episodes(saved["buffer"], env=env)
    if args.max_rows is not None:
        if args.max_rows < 1:
            parser.error("--max-rows must be positive")
        episodes = episodes[:args.max_rows]
    if not episodes:
        parser.error("No complete consecutive trajectories in this replay buffer")
    predictor = (build_predictor(cfg, seed=args.seed if args.seed is not None else cfg["eval_seed"])
                 if args.recompute else None)
    output = Path(args.out_csv) if args.out_csv else run_dir / "replay_buffer_summary.csv"
    cache = {}
    fields = ["formula", "candidate_json", "terminal_reward", "n_buffer_rows"]
    if predictor is not None:
        fields.extend(["pred_mean", "pred_std"])
    try:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode in episodes:
                row = dict(formula=episode.formula, candidate_json=json.dumps(episode.candidate, sort_keys=True),
                           terminal_reward=episode.terminal_reward, n_buffer_rows=episode.n_rows)
                if predictor is not None:
                    key = composition_key(episode.candidate)
                    if key not in cache:
                        cache[key] = predictor.predict(episode.candidate)
                    row["pred_mean"], row["pred_std"] = cache[key]
                writer.writerow(row)
    finally:
        if predictor is not None and callable(getattr(predictor, "close", None)):
            predictor.close()
    print(f"Wrote {len(episodes)} complete trajectories to {output}")


if __name__ == "__main__":
    main()
