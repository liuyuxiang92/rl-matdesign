"""Version the scientific meaning of RL inputs, not merely tensor shapes."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = 3


def environment_contract(env) -> dict[str, Any]:
    """Describe state and action labels so equal-sized incompatible files fail."""
    state_schema = getattr(env, "state_feature_schema", None)
    result = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "state": state_schema() if callable(state_schema) else {"kind": "descriptor-v1"},
        "action_encoding": "label-identity-descriptor-and-typed-value-v2",
        # Training APIs also accept minimal external environments. Record their
        # missing alphabet explicitly; never invent a compatible material domain.
        "species": list(env.species_set) if hasattr(env, "species_set") else None,
        "values": list(env.fraction_set) if hasattr(env, "fraction_set") else None,
        "complete": hasattr(env, "species_set") and hasattr(env, "fraction_set"),
        "max_steps": int(env.n_components),
        "episode_style": getattr(env, "episode_style", "integer_accumulation"),
        "anion_formula": getattr(env, "anion_formula", ""),
        "total_units": getattr(env, "_total_units", None),
        "bounds": getattr(env, "_element_unit_bounds_by_sym", None),
        "design_spec": getattr(env, "design_spec", None),
    }
    for name in ("design_schema", "action_feature_schema"):
        describe = getattr(env, name, None)
        if callable(describe):
            result[name] = describe()
    if result["complete"]:
        from ..action_features import value_alphabet
        result["value_features"] = value_alphabet(env).to_schema()
    if hasattr(env, "_inners"):
        result["groups"] = [
            {"name": name, "sites": sites, "contract": environment_contract(inner)}
            for name, sites, inner in zip(env.group_names, env._sites, env._inners)
        ]
    # Tuples must compare identically after JSON/Torch serialization.
    return json.loads(json.dumps(result, sort_keys=True))


def require_contract(actual, expected: dict, source: str = "artifact") -> None:
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True) or not expected.get("complete", False):
        reason = "missing state/action contract" if actual is None else "different state/action contract"
        raise ValueError(
            f"{source}: {reason}; this model/scaler/replay is incompatible with "
            "the current design space. Retrain with this version, or use the "
            "archived code/environment that produced the artifact."
        )


def bind_scaler(scaler, env):
    scaler.rl_matdesign_contract_ = environment_contract(env)
    return scaler


def validate_scaler(scaler, env, source: str = "scaler") -> None:
    require_contract(getattr(scaler, "rl_matdesign_contract_", None),
                     environment_contract(env), source)



def scaler_fingerprint(scaler) -> str:
    """Hash fitted normalization values, independent of joblib serialization.

    Numeric dtype, shape and C-order bytes are explicit. The semantic environment
    contract is also bound, while execution-only options such as ``copy`` are not.
    A missing contract remains explicit for low-level isolated training tests;
    production runners separately require validate_scaler before artifact loading.
    """
    import numpy as np

    if not hasattr(scaler, 'n_features_in_'):
        raise ValueError('Cannot fingerprint an unfitted scaler')
    metadata = {
        'schema_version': 1,
        'scaler_class': f'{type(scaler).__module__}.{type(scaler).__qualname__}',
        'environment_contract': getattr(scaler, 'rl_matdesign_contract_', None),
        'with_mean': getattr(scaler, 'with_mean', None),
        'with_std': getattr(scaler, 'with_std', None),
    }
    if hasattr(scaler, 'feature_names_in_'):
        names = list(scaler.feature_names_in_)
        if any(not isinstance(name, str) for name in names):
            raise ValueError('Scaler feature_names_in_ must contain strings')
        metadata['feature_names_in_'] = names
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8'))
    for name in ('mean_', 'scale_', 'var_', 'n_features_in_', 'n_samples_seen_'):
        if not hasattr(scaler, name):
            raise ValueError(f'Fitted scaler is missing {name}')
        value = getattr(scaler, name)
        if value is None:
            descriptor, payload = {'name': name, 'value': None}, b''
        else:
            array = np.asarray(value)
            if array.dtype.kind not in 'biuf' or not np.all(np.isfinite(array)):
                raise ValueError(f'Scaler {name} must contain finite real numeric values')
            descriptor = {'name': name, 'dtype': array.dtype.str, 'shape': list(array.shape)}
            payload = array.tobytes(order='C')
        header = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)
    return digest.hexdigest()


def require_scaler_fingerprint(scaler, expected, source: str = 'checkpoint') -> None:
    if not isinstance(expected, str) or expected != scaler_fingerprint(scaler):
        raise ValueError(f'{source}: fitted scaler fingerprint differs or is missing; '
                         'use the exact normalization fitted for this model/checkpoint')

def save_weights(path, model, env, role: str, *, scaler=None) -> None:
    import torch

    torch.save({"environment_contract": environment_contract(env),
                "scaler_fingerprint": scaler_fingerprint(scaler) if scaler is not None else None,
                "evaluation_contract": getattr(env, "evaluation_contract", None),
                f"{role}_state": model.state_dict()}, path)


def load_artifact(path, env, *, device="cpu", checkpoint: bool = False, scaler=None):
    import torch

    raw = torch.load(path, map_location=device, weights_only=not checkpoint)
    require_contract(raw.get("environment_contract") if isinstance(raw, dict) else None,
                     environment_contract(env), str(Path(path)))
    if scaler is not None:
        require_scaler_fingerprint(scaler, raw.get('scaler_fingerprint'), str(Path(path)))
    expected = getattr(env, "evaluation_contract", None)
    if expected is not None:
        actual = (raw.get("execution_contract") or {}).get("evaluation") if checkpoint else raw.get("evaluation_contract")
        from .hpo_resume_compatibility import evaluation_contracts_equal
        if not evaluation_contracts_equal(
                actual, expected, allow_source_upgrade=getattr(env, "hpo_recorder", None) is not None):
            raise ValueError(f"{path}: evaluation contract differs or is missing; model assets, code, seeds or runtime changed")
    return raw
