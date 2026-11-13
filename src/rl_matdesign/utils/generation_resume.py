"""Generation checkpoints at fully consumed evaluation-wave boundaries."""
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


def restore_generation(predictor, saved, contract):
    from .execution_contract import contracts_equal

    if not contracts_equal(saved.get("generation_contract"), contract):
        raise ValueError("Generation resume changed its frozen weights, scenario, seed or sampling settings")
    load_ledger = getattr(predictor, "load_state_dict", None)
    if not callable(load_ledger):
        raise ValueError("Generation predictor cannot restore its cost ledger")
    load_ledger(saved["predictor_ledger"])
    inner = getattr(predictor, "_inner", predictor)
    for name, value in saved.get("predictor_caches", {}).items():
        cache = getattr(inner, name, None)
        if name not in _RESULT_CACHES or not isinstance(cache, dict):
            raise ValueError(f"Generation predictor cache contract changed: {name}")
        cache.clear()
        cache.update(copy.deepcopy(value))
    extra = saved.get("predictor_extra")
    if extra is not None:
        hook = getattr(inner, "load_generation_state_dict", None)
        if not callable(hook):
            raise ValueError("Generation predictor cannot restore its custom state")
        hook(copy.deepcopy(extra))
    # Model/scaler loading and environment construction may draw randomness.
    # Restore only after all setup, immediately before resumed action sampling.
    restore_rng_state(saved["rng_state"])
