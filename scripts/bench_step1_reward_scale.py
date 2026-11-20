"""Finite fixture check for learner sensitivity to whole reward units."""
import contextlib
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from rl_matdesign.action_features import value_alphabet
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.model import PolicyNet, ValueNet
from rl_matdesign.training import _fit_scaler_from_warmup, generate_candidates
from rl_matdesign.training import train_dqn_online, train_pg
from rl_matdesign.utils.seeding import set_global_seed


LABELS = ["Na", "K", "Ca", "Sr", "Mg", "Ba", "Li", "Cs"]
FEATURES = np.eye(len(LABELS))
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


class Fixture:
    def __init__(self, family, rare):
        self.family = family
        self.rare = rare

    def scientific(self, payload):
        i = LABELS.index(next(iter(payload["choice"])))
        if self.family == "weighted_sum":
            raw = (100 * i, i + (i % 2))
            quality = (raw[0] / 100 + raw[1]) / 2
            return quality, True, quality
        if self.family == "continuous_gate":
            quality = 1000 + 100 * i
            expansion = (4 if (i >= (7 if self.rare else 4)) else 6) * 1e-6
            transition = 770 + 3 * i
            violation = max(0, expansion - 5e-6) / 5e-6 + max(0, 773 - transition) / 20
            utility = (quality - 1350) / 230
        else:
            probability = 0.87 if (i >= (7 if self.rare else 4)) else 0.5 + i * .025
            violation = max(0, 0.8 - probability)
            deltas = [i + j * (i % 3) / 10 for j in range(3)]
            quality = sum(deltas) / len(deltas)
            utility = (quality - 3.5) / 2.3
        reward = 1 + math.log1p(math.exp(utility)) if violation == 0 else -violation
        return reward, violation == 0, quality

    def predict(self, payload):
        return self.scientific(payload)[0], None


def run(family, rare, method, mode, multiplier, seed):
    set_global_seed(seed, deterministic=True)
    oracle = Fixture(family, rare)
    env = MultiGroupEnv(groups=[{
        "name": "choice", "kind": "composition", "species_set": LABELS,
        "amount_set": [1.0], "n_steps": 1, "unique": True, "total": 1,
    }], reward_fn=lambda candidate: oracle.predict(candidate)[0] * multiplier,
        state_featurizer=lambda _: np.array([0.0]))
    values = value_alphabet(env)
    policy = qnet = None
    common = dict(env=env, elem_feats_scaled=FEATURES, action_values=values, device=DEVICE)
    if method == "a2c":
        scaler = _fit_scaler_from_warmup(env, 4)
        policy = PolicyNet(
            state_dim=scaler.n_features_in_, step_dim=env.max_steps,
            elem_dim=len(LABELS), frac_dim=1, hidden_dim=8,
        )
        critic = ValueNet(
            state_dim=scaler.n_features_in_, step_dim=env.max_steps,
            hidden_dim=8, value_normalization=mode,
        )
        rows = train_pg(
            **common, policy=policy, value_net=critic, scaler=scaler,
            num_iters=6, batch_eps=4, train_count_duplicates=True,
        )
        losses = [row["critic_loss"] for row in rows if row["phase"] == "pg_train"]
    else:
        qnet, scaler, rows = train_dqn_online(
            **common, hidden_dim=8, n_warmup_eps=4, dqn_num_train_eps=20,
            dqn_buffer_size=32, batch_size=4, dqn_grad_steps_per_ep=1,
            dqn_target_update_freq=4, dqn_eps_anneal_eps=16,
            train_count_duplicates=True, dqn_target_mode=method,
            value_normalization=mode,
        )
        losses = [row["train_loss"] for row in rows if row["phase"] == "dqn_train"]
    assert losses and all(math.isfinite(loss) for loss in losses)
    env.reward_fn = lambda candidate: oracle.predict(candidate)[0]
    set_global_seed(seed + 1000, deterministic=True)
    generated = generate_candidates(
        **common, scaler=scaler, qnet=qnet, policy=policy, predictor=oracle,
        num_gen_eps=4, max_attempts=48, gen_temperature=0, gen_epsilon=.35,
    )
    qualities = []
    for row in generated:
        reward, passed, quality = oracle.scientific(json.loads(row["candidate_json"]))
        assert row["reward"] == reward
        if passed:
            qualities.append(quality)
    return len(qualities), max(qualities) if qualities else None


def main():
    configurations = [
        ("weighted_sum", False), ("continuous_gate", False),
        ("continuous_gate", True), ("classifier_gate", False), ("classifier_gate", True),
    ]
    aggregate = defaultdict(list)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        for (family, rare), method, mode, multiplier, seed in itertools.product(
            configurations, ("bootstrap", "mc", "a2c"), ("none", "popart"),
            (0.01, 1.0, 100.0), range(10),
        ):
            aggregate[family, rare, method, mode, multiplier].append(
                run(family, rare, method, mode, multiplier, seed)
            )
    result = []
    for (family, rare, method, mode, multiplier), observations in aggregate.items():
        best = [best for _, best in observations if best is not None]
        result.append(dict(
            family=family, rare=rare, method=method, mode=mode, multiplier=multiplier,
            runs=len(observations), with_passing=len(best),
            mean_passing=float(np.mean([count for count, _ in observations])),
            mean_best_if_passing=float(np.mean(best)) if best else None,
        ))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
