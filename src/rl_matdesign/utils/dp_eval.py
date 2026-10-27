"""DeepMD evaluation helpers shared across predictors.

Three evaluation paths share structure preparation and model-major results:

* **ASE calculator** (``deepmd.calculator.DP``) — wraps a DP model as an ASE
  Calculator. Use ``atoms.get_potential_energy()`` for one scalar per
  (structure, model) pair. This is the path HEA / perovskite / Ti-alloy-style
  bulk-energy predictors take.

* **Raw DeepProperty** (``deepmd.pt.infer.deep_eval.DeepProperty``) — direct
  inference call that returns a property vector (possibly multi-output). Pick
  one scalar via ``output_index``. This is the path the OOH overpotential
  pipeline takes (multiple binding energies → Sabatier formula).

* **Raw DeepClassification** — calls ``predict_proba`` directly, then selects
  a class probability via ``output_index``. Probability conversion belongs to
  DeepMD, rather than this module.

All paths are kept lazy so importing this module does not require deepmd-kit.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def load_ase_calculators(
    dp_models: List[str], *, head: Optional[str] = None
) -> List[Any]:
    """Load a list of ASE-compatible DeepMD calculators (one per checkpoint).

    ``head`` selects which output head of a multi-task DP checkpoint to use
    (e.g. ``"Omat24"`` for the energy head of an Omat24+property model). Leave
    as ``None`` for single-task checkpoints — DPCalculator's own default.
    """
    try:
        from deepmd.calculator import DP as DPCalculator
    except ImportError as exc:
        raise ImportError(
            "The energy backend requires deepmd-kit (ASE calculator): "
            "`pip install deepmd-kit`."
        ) from exc
    kwargs = {"head": head} if head is not None else {}
    return [DPCalculator(model=p, **kwargs) for p in dp_models]


def eval_energy_ase(
    atoms_list: List[Any],
    calc_list: List[Any],
    *,
    energy_per_atom: bool = True,
) -> np.ndarray:
    """Evaluate every (structure, calculator) pair.

    Returns a 2-D array shaped **(n_models, n_structures)** — the repo-wide
    convention shared with :func:`eval_property_ensemble`, so callers can fold the
    two axes separately and never have to know which loop ran outermost. Axis 1
    (structures) is *configurational* spread and gets averaged away; axis 0
    (models) is *epistemic uncertainty* and is what a ``mean_minus_kstd``
    objective penalises. See ``structure_score._score``.
    """
    out = np.empty((len(calc_list), len(atoms_list)), dtype=float)
    for j, atoms in enumerate(atoms_list):
        for i, calc in enumerate(calc_list):
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            e = atoms_copy.get_potential_energy()
            if energy_per_atom:
                e /= len(atoms_copy)
            out[i, j] = float(e)
    return out


def pick_scalar(
    vector: np.ndarray,
    *,
    output_index: int = 0,
    output_aggregator: str = "index",
) -> float:
    """Pick one scalar from a possibly-multi-output prediction vector.

    Parameters
    ----------
    vector:
        1-D array of model outputs.
    output_index:
        Used only when ``output_aggregator == "index"``.
    output_aggregator:
        ``"index"`` (default) selects ``vector[output_index]``; ``"mean"``
        returns the mean over all outputs; ``"max"`` returns the maximum.
    """
    v = np.asarray(vector).reshape(-1)
    if v.size == 0:
        raise ValueError("Empty prediction vector.")
    if output_aggregator == "index":
        if output_index < 0 or output_index >= v.size:
            raise IndexError(
                f"output_index={output_index} is out of range for a length-{v.size} output vector."
            )
        return float(v[output_index])
    if output_aggregator == "mean":
        return float(np.mean(v))
    if output_aggregator == "max":
        return float(np.max(v))
    raise ValueError(
        f"Unknown output_aggregator {output_aggregator!r}; expected one of: index, mean, max."
    )


def _broadcast_frame_param(fparam: Any, nframes: int) -> np.ndarray:
    """Broadcast a scalar / length-d vector / (nframes, d) array to (nframes, d)."""
    arr = np.atleast_1d(np.asarray(fparam, dtype=np.float64))
    if arr.ndim == 1:
        return np.tile(arr.reshape(1, -1), (nframes, 1))
    if arr.ndim == 2:
        if arr.shape[0] == nframes:
            return arr
        if arr.shape[0] == 1:
            return np.tile(arr, (nframes, 1))
    raise ValueError(
        f"fparam must be a scalar, a length-d vector, or an (nframes, d) array; "
        f"got shape {arr.shape} for nframes={nframes}."
    )


def _broadcast_atomic_param(aparam: Any, nframes: int, natoms: int) -> np.ndarray:
    """Broadcast atomic params to (nframes, natoms, d)."""
    arr = np.asarray(aparam, dtype=np.float64)
    if arr.ndim == 1:                       # per-atom scalar or single d-vector
        if arr.shape[0] == natoms:
            arr = arr.reshape(natoms, 1)
        else:
            arr = np.tile(arr.reshape(1, -1), (natoms, 1))
        return np.tile(arr.reshape(1, natoms, -1), (nframes, 1, 1))
    if arr.ndim == 2 and arr.shape[0] == natoms:
        return np.tile(arr.reshape(1, natoms, -1), (nframes, 1, 1))
    if arr.ndim == 3 and arr.shape[:2] == (nframes, natoms):
        return arr
    raise ValueError(
        f"aparam must broadcast to (nframes={nframes}, natoms={natoms}, d); "
        f"got shape {arr.shape}."
    )


def _prepare_dp_inputs(
    structures: List[Any],
    elem_to_type: dict,
    *,
    fparam: Optional[Any],
    aparam: Optional[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prepare a batch shared by property and classification evaluators."""
    nframes = len(structures)
    if not nframes:
        raise ValueError("At least one structure is required for DP evaluation.")
    natoms = len(structures[0])
    if any(len(atoms) != natoms for atoms in structures):
        raise ValueError("A DP evaluation batch must have the same atom count in every frame.")
    coords = np.zeros((nframes, natoms, 3), dtype=np.float64)
    cells = np.zeros((nframes, 3, 3), dtype=np.float64)
    atom_types = np.zeros((nframes, natoms), dtype=np.int32)
    for i, atoms in enumerate(structures):
        coords[i] = atoms.get_positions()
        cells[i] = atoms.get_cell().array
        syms = atoms.get_chemical_symbols()
        unknown = sorted({s for s in syms if s not in elem_to_type})
        if unknown:
            raise RuntimeError(
                f"Elements {unknown} are not in the DP model type_map "
                f"({sorted(elem_to_type)}). Check the checkpoint covers every species."
            )
        atom_types[i] = np.fromiter(
            (elem_to_type[s] for s in syms), dtype=np.int32, count=natoms,
        )

    eval_kwargs: dict = {"mixed_type": True}
    if fparam is not None:
        eval_kwargs["fparam"] = _broadcast_frame_param(fparam, nframes)
    if aparam is not None:
        eval_kwargs["aparam"] = _broadcast_atomic_param(aparam, nframes, natoms)
    return coords, cells, atom_types, eval_kwargs


def _property_output_matrix(result: Any, nframes: int) -> np.ndarray:
    """Unwrap DeepProperty's tuple while retaining legacy ndarray outputs."""
    if isinstance(result, tuple):
        if not result:
            raise ValueError("DeepProperty.eval returned an empty tuple.")
        result = result[0]
    values = np.asarray(result)
    if values.ndim == 0 and nframes == 1:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if nframes == 1:
            values = values.reshape(1, -1)
        elif values.size == nframes:
            values = values.reshape(nframes, 1)
    if values.ndim != 2 or values.shape[0] != nframes or values.shape[1] == 0:
        raise ValueError(
            "DeepProperty.eval must return one output vector per frame with "
            f"shape ({nframes}, n_outputs); got {values.shape}."
        )
    return values


def eval_property_ensemble(
    structures: List[Any],
    models: List[Any],
    elem_to_type: dict,
    *,
    output_index: int = 0,
    output_aggregator: str = "index",
    fparam: Optional[Any] = None,
    aparam: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate a ``DeepProperty`` ensemble on *structures*.

    Returns a 2-D array shaped **(n_models, n_structures)** — the same convention
    as :func:`eval_energy_ase` — each entry collapsed from that model's output
    vector via :func:`pick_scalar`. Used by the ``structure_score`` predictor's
    ``property`` backend, which averages axis 1 (structures) to define the
    property and takes mean/std over axis 0 (models) as the uncertainty.

    ``fparam`` / ``aparam`` supply frame / atomic parameters for models trained
    with them (``numb_fparam > 0`` / ``numb_aparam > 0``). ``fparam`` is broadcast
    to one row per frame; pass a scalar, a length-``numb_fparam`` vector, or a
    full ``(nframes, numb_fparam)`` array.
    """
    coords, cells, atom_types, eval_kwargs = _prepare_dp_inputs(
        structures, elem_to_type, fparam=fparam, aparam=aparam,
    )
    nframes = len(structures)
    out = np.empty((len(models), nframes), dtype=float)
    for i, dp in enumerate(models):
        values = _property_output_matrix(
            dp.eval(coords=coords, cells=cells, atom_types=atom_types, **eval_kwargs),
            nframes,
        )
        for j, frame_vec in enumerate(values):
            out[i, j] = pick_scalar(
                frame_vec, output_index=output_index, output_aggregator=output_aggregator
            )
    return out


def eval_classification_ensemble(
    structures: List[Any],
    models: List[Any],
    elem_to_type: dict,
    *,
    output_index: int = 0,
    output_aggregator: str = "index",
    fparam: Optional[Any] = None,
    aparam: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate class probabilities as **(n_models, n_structures)** scalars.

    Each model must provide DeepMD's ``predict_proba`` interface returning
    ``(nframes, n_classes)`` with at least two classes. Binary and multiclass
    models follow the same path; this helper never transforms logits. Models
    in an ensemble must share class count and class ordering (the latter is a
    caller-provided semantic contract). ``index`` selects the requested class;
    ``mean`` and ``max`` aggregate probabilities across classes.

    Returned values retain the separate model and configuration axes, just as
    :func:`eval_property_ensemble`. They are not yet averaged across either.
    """
    coords, cells, atom_types, eval_kwargs = _prepare_dp_inputs(
        structures, elem_to_type, fparam=fparam, aparam=aparam,
    )
    nframes = len(structures)
    out = np.empty((len(models), nframes), dtype=float)
    ensemble_classes: Optional[int] = None
    for i, dp in enumerate(models):
        predict_proba = getattr(dp, "predict_proba", None)
        if not callable(predict_proba):
            raise TypeError(
                "The DP classification backend requires a DeepMD model with "
                "predict_proba(); use a runtime supporting classification models."
            )
        probabilities = np.asarray(
            predict_proba(coords=coords, cells=cells, atom_types=atom_types, **eval_kwargs),
            dtype=float,
        )
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != nframes
            or probabilities.shape[1] < 2
        ):
            raise ValueError(
                "DeepMD predict_proba must return shape "
                f"({nframes}, n_classes) with n_classes >= 2; got {probabilities.shape}."
            )
        nclasses = probabilities.shape[1]
        get_num_classes = getattr(dp, "get_num_classes", None)
        if callable(get_num_classes) and get_num_classes() != nclasses:
            raise ValueError(
                f"Model {i} class count disagrees with its predict_proba output "
                f"containing {nclasses} classes."
            )
        if ensemble_classes is None:
            ensemble_classes = nclasses
        elif ensemble_classes != nclasses:
            raise ValueError(
                "Classification ensemble models must have the same class count; "
                f"expected {ensemble_classes}, got {nclasses} for model {i}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("DeepMD predict_proba returned non-finite probabilities.")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("DeepMD predict_proba probabilities must lie in [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
            raise ValueError("DeepMD predict_proba probabilities must sum to 1 for every frame.")
        for j, frame_vec in enumerate(probabilities):
            out[i, j] = pick_scalar(
                frame_vec, output_index=output_index, output_aggregator=output_aggregator
            )
    return out
