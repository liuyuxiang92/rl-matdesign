"""Method-specific, backend-free HPO spaces and budget-dependent parameters."""
from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping

from ..config_schema import ConfigError, PARAMETERS

METHODS = ("a2c", "dqn-bootstrap", "dqn-mc")

_DQN_SPACE = {
    "dqn_lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "dqn_batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "warmup_budget_fraction": {"type": "float", "low": 0.02, "high": 0.20},
    "dqn_eps_min": {"type": "float", "low": 0.02, "high": 0.40},
    "epsilon_anneal_budget_ratio": {"type": "float", "low": 0.25, "high": 2.0, "log": True},
}
_A2C_SPACE = {
    "pg_lr_actor": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "pg_lr_critic": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "pg_batch_eps": {"type": "categorical", "choices": [8, 16, 32, 64]},
    "pg_entropy_coef": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "pg_entropy_min": {"type": "float", "low": 0.0, "high": 0.8},
}
_EXTENSIONS = {
    "dqn_batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dqn_target_update_freq": [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000],
    "pg_batch_eps": [1, 2, 4, 8, 16, 32, 64, 128, 256],
}


def get_profile(method: str) -> dict:
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}, got {method!r}")
    space = deepcopy(_A2C_SPACE if method == "a2c" else _DQN_SPACE)
    if method == "dqn-bootstrap":
        space["dqn_target_update_freq"] = {"type": "categorical", "choices": [10, 25, 50, 100, 200]}
    return {
        "method": method,
        "search_space": space,
        "extension_choices": {name: deepcopy(values) for name, values in _EXTENSIONS.items() if name in space},
    }


def validate_parameter_value(method: str, name: str, value):
    """Check a value's mathematical domain, including proposed outward probes."""
    if name not in get_profile(method)["search_space"]:
        raise ConfigError(f"search_space parameter {name!r} is not active for {method}; fixed parameters cannot be searched")
    if name in {"warmup_budget_fraction", "epsilon_anneal_budget_ratio"}:
        if isinstance(value, bool):
            raise ConfigError(f"{name} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigError(f"{name} must be a finite number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{name} must be finite and > 0")
        if name == "warmup_budget_fraction" and value >= 1:
            raise ConfigError("warmup_budget_fraction must be < 1 to leave online training budget")
        return value
    return PARAMETERS[name].normalize(name, value)


def validate_search_space(method: str, space: Mapping) -> dict:
    if not isinstance(space, Mapping):
        raise ConfigError("search_space must be a mapping")
    result = {}
    for name, raw in space.items():
        if name not in get_profile(method)["search_space"]:
            raise ConfigError(f"search_space parameter {name!r} is not active for {method}")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"search_space.{name} must be a mapping")
        entry = dict(raw)
        kind = entry.get("type")
        allowed = {"type", "choices"} if kind == "categorical" else {"type", "low", "high", "log", "step"}
        if set(entry) - allowed:
            raise ConfigError(f"Unknown fields in search_space.{name}: {sorted(set(entry) - allowed)}")
        if kind == "categorical":
            choices = entry.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ConfigError(f"search_space.{name}.choices must be a nonempty list")
            choices = [validate_parameter_value(method, name, value) for value in choices]
            if len(set(choices)) != len(choices):
                raise ConfigError(f"search_space.{name}.choices contains duplicates")
            result[name] = {"type": kind, "choices": sorted(choices)}
            continue
        if kind not in {"float", "int"}:
            raise ConfigError(f"search_space.{name}.type must be float, int or categorical")
        if "low" not in entry or "high" not in entry:
            raise ConfigError(f"search_space.{name} requires low and high")
        low = validate_parameter_value(method, name, entry["low"])
        high = validate_parameter_value(method, name, entry["high"])
        if kind == "int" and (isinstance(low, bool) or not isinstance(low, int) or not isinstance(high, int)):
            raise ConfigError(f"search_space.{name} integer bounds must be integers")
        if name in PARAMETERS and PARAMETERS[name].kind is int and kind != "int":
            raise ConfigError(f"search_space.{name} requires type int or categorical")
        if low >= high:
            raise ConfigError(f"search_space.{name} requires low < high")
        log = entry.get("log", False)
        if not isinstance(log, bool):
            raise ConfigError(f"search_space.{name}.log must be boolean")
        if log and low <= 0:
            raise ConfigError(f"search_space.{name} logarithmic bounds must be positive")
        step = entry.get("step", 1 if kind == "int" else None)
        if step is not None:
            if kind != "int" or isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ConfigError(f"search_space.{name}.step must be a positive integer for type int")
            if log and step != 1:
                raise ConfigError(f"search_space.{name} log integer distributions require step=1")
            if (high - low) % step:
                raise ConfigError(f"search_space.{name} high must lie on its integer step grid")
        result[name] = {"type": kind, "low": low, "high": high, "log": log}
        if step is not None:
            result[name]["step"] = step
    return result


def initial_parameters(plan: Mapping) -> dict:
    """Derive the reproducible working reference, never an extra user standard."""
    out = {}
    for name, entry in plan["search_space"].items():
        if entry["type"] == "categorical":
            choices = sorted(entry["choices"])
            value = choices[(len(choices) - 1) // 2]
        else:
            low, high = entry["low"], entry["high"]
            value = math.sqrt(low * high) if entry.get("log", False) else (low + high) / 2
            if entry["type"] == "int":
                step = entry.get("step", 1)
                value = min(high, low + round((value - low) / step) * step)
        out[name] = value
    return out


def resolve_parameters(plan: Mapping, params: Mapping, budget: int) -> dict:
    """Resolve one training configuration to flat runtime overrides.

    Domain validation deliberately allows range-search probes outside initial
    bounds. Full configurations start from the initial reference; a supplied
    partial mapping is an explicit override, not a missing runtime default.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ConfigError("budget must be a positive integer")
    method = plan["method"]
    values = initial_parameters(plan)
    for name, value in params.items():
        values[name] = validate_parameter_value(method, name, value)
    base = plan.get("base_config_data", {})
    overrides = {name: value for name, value in values.items() if name in PARAMETERS}
    overrides.update({
        "method": "a2c" if method == "a2c" else "dqn",
        "train_budget": budget,
        "train_count_duplicates": False,
        "pg_repeat_penalty_coef": 0.0,
        "gen_epsilon": 0.0,
        "gen_top_frac": 0.0,
        "exploration_gen_eps": 0,
        "num_gen_eps": 0,
    })
    if method != "a2c":
        warmup = max(1, round(values["warmup_budget_fraction"] * budget))
        if warmup >= budget:
            raise ConfigError("budget must leave at least one online DQN candidate after warmup")
        overrides.update({
            "dqn_target_mode": method.split("-", 1)[1],
            "dqn_warmup_eps": warmup,
            "dqn_num_train_eps": budget - warmup,
            "dqn_eps_anneal_eps": max(1, round(values["epsilon_anneal_budget_ratio"] * (budget - warmup))),
            "dqn_grad_steps_per_ep": base.get("dqn_grad_steps_per_ep", PARAMETERS["dqn_grad_steps_per_ep"].default),
        })
    else:
        # The common runtime validates dormant DQN totals too. Zeroing these
        # inactive counters prevents an irrelevant default warmup rejecting B.
        overrides.update({"dqn_warmup_eps": 0, "dqn_num_train_eps": budget})
    pg_batch = overrides.get("pg_batch_eps", base.get("pg_batch_eps", PARAMETERS["pg_batch_eps"].default))
    overrides["pg_num_iters"] = (budget + pg_batch - 1) // pg_batch
    return overrides
