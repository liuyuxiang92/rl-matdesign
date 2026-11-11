"""Read design-space cardinality without evaluating scientific properties."""
from __future__ import annotations

from contextlib import redirect_stdout
import sys

from .artifacts import digest


def design_space_info(plan: dict) -> dict:
    cfg = plan["base_config_data"]
    groups = cfg.get("groups", [])
    info = {"status": "unknown", "size": None, "episode_steps": 1,
            "reason": "No exact cardinality adapter for this configured domain",
            "domain_digest": digest({k: cfg.get(k) for k in
                ("groups", "coupling_constraints", "constraint_filter", "constraints", "anion_formula")})}
    # Counting paths would overcount order-equivalent compositions. Only use the
    # existing DP when its canonical-composition and filter contracts apply.
    if len(groups) == 1 and groups[0].get("kind", "composition") == "composition":
        from ..configuration import build_env
        from ..utils.design_count import sampler_from_env
        # Existing scene builders emit human diagnostics. Keep HPO validate /
        # plan stdout parseable as JSON without suppressing those diagnostics.
        with redirect_stdout(sys.stderr):
            env, _, _ = build_env(cfg, object())
        info["episode_steps"] = int(getattr(env, "max_steps", getattr(env, "n_components", 1)))
        if getattr(env, "phase_filter", None) is not None or cfg.get("filters") or groups[0].get("filters"):
            info["reason"] = "Configured action filters have no exact counted-domain adapter"
        else:
            try:
                sampler = sampler_from_env(env)
                info.update(status="exact", size=sampler.size, reason=None)
            except ValueError as exc:
                info["reason"] = str(exc)
    if plan["budget"]["mode"] == "fraction_of_space" and info["status"] != "exact":
        raise ValueError(f"fraction_of_space requires an exact canonical candidate count: {info['reason']}. "
                         "Select budget.mode=absolute_unique for this scenario.")
    return info


def minimum_training_budget(plan: dict, domain: dict) -> int:
    """One common diagnostic minimum, never an unreported per-trial allowance."""
    import math
    from .budget_search import minimum_unique_for_preparation
    from .spec import phase_efficiency
    cfg, space = plan["base_config_data"], plan["search_space"]
    def largest(name):
        spec = space[name]
        return max(spec["choices"]) if spec["type"] == "categorical" else spec["high"]
    if plan["method"] == "a2c":
        warmup = cfg.get("pg_warmup_eps", 0)
        limit = phase_efficiency(plan, "train")["max_attempts_per_unique"]
        batch = largest("pg_batch_eps")
        return max(batch, minimum_unique_for_preparation(warmup, limit))
    batch = largest("dqn_batch_size")
    minimum = math.ceil(batch / max(1, domain.get("episode_steps", 1))) + 1
    if plan["method"] == "dqn-bootstrap":
        freq = largest("dqn_target_update_freq")
        warm_fraction = largest("warmup_budget_fraction")
        minimum = max(minimum, math.ceil((freq + 1) / (1 - warm_fraction)))
    return minimum


def training_budget_preflight(plan: dict, parameters: dict, budget: int, domain: dict) -> dict:
    """Check the actual trial, including newly expanded parameter boundaries.

    The shared grid is still calibrated from the diagnosed range. A probe may
    require more learning work than that grid initially anticipated; reject it
    explicitly at the unchanged comparison budget rather than claim that an
    untrained network is a successful learning experiment.
    """
    from .algorithm_profiles import initial_parameters, validate_parameter_value

    values = initial_parameters(plan)
    for name, value in parameters.items():
        values[name] = validate_parameter_value(plan["method"], name, value)
    specific_plan = {**plan, "search_space": {
        name: {"type": "categorical", "choices": [value]} for name, value in values.items()}}
    required = minimum_training_budget(specific_plan, domain)
    result = {"eligible": True, "required_budget": required, "reason": None}
    if plan["method"] != "a2c":
        capacity = plan["base_config_data"]["dqn_buffer_size"]
        batch = values["dqn_batch_size"]
        if batch > capacity:
            result.update(eligible=False, required_budget=None, failure_kind="replay_capacity",
                          reason=f"dqn_batch_size={batch} exceeds fixed dqn_buffer_size={capacity}; "
                                 "no finite sampling budget can enable a gradient update")
            return result
    if budget < required:
        result.update(eligible=False, failure_kind="insufficient_learning_budget",
                      reason=f"Training budget {budget} is below required budget {required} for this parameter configuration; "
                             "the shared comparison budget is unchanged and no sampling was started")
    return result
