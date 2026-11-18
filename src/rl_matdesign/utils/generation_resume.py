"""Evaluation-state snapshots shared by training and generation checkpoints."""
from __future__ import annotations

import copy

from .seeding import restore_rng_state

# Evaluation-result caches only. Calculators, models and worker pools are rebuilt
# from the scenario and must never be serialized as if they were prediction data.
_RESULT_CACHES = ("_cache", "_stats_cache", "_batch_stats")


def predictor_snapshot(predictor):
    ledger = getattr(predictor, "state_dict", None)
    if not callable(ledger):
        return None
    inner = getattr(predictor, "_inner", predictor)
    caches = {name: copy.deepcopy(getattr(inner, name)) for name in _RESULT_CACHES
              if isinstance(getattr(inner, name, None), dict)}
    hook = getattr(inner, "generation_state_dict", None)
    extra = copy.deepcopy(hook()) if callable(hook) else None
    return {"ledger": ledger(), "caches": caches, "extra": extra}


def predictor_checkpoint_fields(predictor):
    """Capture ledger and result caches at one fully consumed training boundary."""
    state = predictor_snapshot(predictor)
    if state is None:
        return {"predictor_ledger": None, "predictor_caches": {}, "predictor_extra": None}
    return {"predictor_ledger": state["ledger"], "predictor_caches": state["caches"],
            "predictor_extra": state["extra"]}


def restore_predictor_state(predictor, saved):
    """Restore paid-evaluation history and cached results without changing RNG."""
    load_ledger = getattr(predictor, "load_state_dict", None)
    if not callable(load_ledger):
        raise ValueError("Predictor cannot restore its cost ledger")
    load_ledger(saved["predictor_ledger"])
    inner = getattr(predictor, "_inner", predictor)
    for name, value in saved.get("predictor_caches", {}).items():
        cache = getattr(inner, name, None)
        if name not in _RESULT_CACHES or not isinstance(cache, dict):
            raise ValueError(f"Predictor cache contract changed: {name}")
        cache.clear()
        cache.update(copy.deepcopy(value))
    extra = saved.get("predictor_extra")
    if extra is not None:
        hook = getattr(inner, "load_generation_state_dict", None)
        if not callable(hook):
            raise ValueError("Predictor cannot restore its custom state")
        hook(copy.deepcopy(extra))


def restore_generation(predictor, saved, contract):
    from .source_compatibility import generation_contracts_equal

    if not generation_contracts_equal(saved.get("generation_contract"), contract):
        raise ValueError("Generation resume changed its frozen weights, scenario, seed or sampling settings")
    restore_predictor_state(predictor, saved)
    # Model/scaler loading and environment construction may draw randomness.
    # Restore only after all setup, immediately before resumed action sampling.
    restore_rng_state(saved["rng_state"])
