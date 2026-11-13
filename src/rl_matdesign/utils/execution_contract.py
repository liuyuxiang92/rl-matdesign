"""Checkpoint compatibility includes the oracle and learning protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import csv
import re


def numerical_runtime_contract(runtime: dict) -> dict:
    """Freeze numerical dependencies/settings, excluding invocation-only state.

    All installed distribution versions are conservatively compared: changing an
    unrelated tool also requires a new run. This is evidence compatibility, not a
    claim that package versions alone lock external native libraries or plugins.
    """
    packages = {}
    for name, version in runtime.get("packages", {}).items():
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in packages and packages[canonical] != version:
            raise ValueError(f"Conflicting installed distribution versions for {canonical}")
        packages[canonical] = version
    torch_state = {key: value for key, value in runtime.get("torch", {}).items()
                   if key != "cuda_initialized"}
    driver = runtime.get("nvidia_smi", {})
    if driver.get("status") == "available":
        columns = [value.strip() for value in driver["columns"].split(",")]
        devices = []
        for values in csv.reader(driver.get("output", "").splitlines()):
            if not values:
                continue
            if len(values) != len(columns):
                raise ValueError("Malformed NVIDIA driver evidence")
            fields = dict(zip(columns, (value.strip() for value in values)))
            devices.append({key: fields[key] for key in ("index", "name", "driver_version")})
        driver = {"status": "available", "devices": devices,
                  "truncated": bool(driver.get("truncated", False))}
    else:
        driver = {"status": driver.get("status", "unknown")}
    value = {"schema_version": 1, "python": runtime.get("python"),
             "platform": runtime.get("platform"), "machine": runtime.get("machine"),
             "packages": packages, "torch": torch_state, "driver": driver,
             "determinism_environment": runtime.get("determinism_environment", {})}
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def execution_contract(config, predictor, manifest_path, device) -> dict:
    from ..config_schema import PARAMETERS
    from .provenance import sha256_file
    import numpy as np
    import torch

    manifest = json.loads(Path(manifest_path).read_text())
    # The input config and explicitly loaded RL artifacts are invocation
    # evidence, not physical model assets. Their contents legitimately differ
    # when switching methods or loading a checkpoint.
    assets = [item for item in manifest["input_files"]
              if item["field"] not in {"config_path", "explicit_input"}]
    source = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    describe = getattr(predictor, "oracle_identity", None)
    source_identity = getattr(predictor, "source_identity", None)
    scientific = {key: value for key, value in config.items()
                  if key not in PARAMETERS or key in {"eval_seed", "k"}}
    scientific.pop("generated_columns", None)
    scientific.pop("predict_cache_size", None)
    # This is resolved sampling provenance, not a predictor input. In
    # particular, choosing another generation policy must not invalidate a
    # frozen model's oracle contract. Exact generation resume locks the actual
    # selected policy and values separately in generation_sampling_contract.
    scientific.pop("generation_selection", None)
    evaluation = {"schema_version": 1, "source_sha256": digest.hexdigest(),
                  "configuration": scientific, "assets": assets,
                  "adapter_identity": describe() if callable(describe) else None,
                  "sources": source_identity() if callable(source_identity) else None,
                  "numerical_runtime": numerical_runtime_contract(manifest.get("runtime", {})),
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "device": str(device),
                              "platform": platform.platform(),
                              "cuda_devices": [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]}}
    active_prefix = "dqn_" if config["method"] == "dqn" else "pg_"
    adjustable_budgets = {"dqn_num_train_eps", "pg_num_iters"}
    learning = {key: value for key, value in config.items()
                if key in {"method", "train_seed", "constrain_training", "train_count_duplicates", "train_max_duplicate_attempts"}
                or (key.startswith(active_prefix) and key not in adjustable_budgets)}
    # Saving limits DQN candidate prefetch waves. Changing this frequency
    # changes when the policy is updated relative to subsequent sampling.
    learning["checkpoint_frequency"] = manifest.get("runtime_args", {}).get("save_checkpoint_freq", 0)
    return json.loads(json.dumps({"evaluation": evaluation, "learning": learning}, sort_keys=True))


def generation_sampling_contract(config) -> dict:
    """Freeze effective sampling semantics without binding their source spelling.

    The strategy identity remains explicit even when every numeric policy value
    is zero. The snapshot's YAML/override provenance is deliberately omitted:
    saving and reloading the same effective policy must remain compatible.
    """
    from ..config_schema import selected_generation_parameter
    return {
        "parameter": selected_generation_parameter(config),
        **{key: config[key] for key in (
            "gen_seed", "gen_temperature", "gen_top_frac", "gen_epsilon",
            "num_gen_eps", "exploration_gen_eps")},
    }


def contracts_equal(actual, expected) -> bool:
    """Compare serialized contracts exactly, without bool/int coercion or NaN."""
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _contract_difference_paths(actual, expected, *, limit: int = 8) -> list[str]:
    """Identify incompatible fields without exposing recorded values."""
    paths = []

    def visit(left, right, path):
        if len(paths) >= limit or contracts_equal(left, right):
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    paths.append(child)
                else:
                    visit(left[key], right[key], child)
                if len(paths) >= limit:
                    break
        elif isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                child = f"{path}[{index}]"
                if index >= len(left) or index >= len(right):
                    paths.append(child)
                else:
                    visit(left[index], right[index], child)
                if len(paths) >= limit:
                    break
        else:
            paths.append(path or "<root>")

    visit(actual, expected, "")
    return paths


def require_execution_contract(actual, expected) -> None:
    if not contracts_equal(actual, expected):
        changed = ", ".join(_contract_difference_paths(actual, expected))
        raise ValueError("Checkpoint oracle, runtime or training protocol differs "
                         f"(first differing fields: {changed}). Exact resume requires "
                         "the same code, model/template bytes, conditions, seeds and learning settings; "
                         "use the archived version or start a new run.")
