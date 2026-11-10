"""PropertyPredictor Protocol — the reward interface for the RL framework.

Any object that implements ``predict(composition) -> (mean, std)`` satisfies
this interface via structural subtyping.  No inheritance required.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class PropertyPredictor(Protocol):
    """Protocol for property prediction models used as RL reward oracles.

    The agent calls ``predict()`` at the terminal step to obtain the reward.
    Higher ``mean`` values indicate better candidates (framework always
    maximises reward; negate energies as needed in your predictor).

    Parameters
    ----------
    composition:
        Dict mapping element symbol → fractional occupancy, e.g.
        ``{"Co": 0.20, "Cr": 0.20, "Fe": 0.20, "Mn": 0.20, "Ni": 0.20}``.
        Fractions must sum to 1.0 (guaranteed by :class:`CompositionEnv`).

    Returns
    -------
    (mean, std):
        Scalar mean reward and ensemble standard deviation.
        ``std`` is used for uncertainty-driven exploration and Pareto tracking.

    Optional training-sample diagnostics
    ------------------------------------
    Predictors with transformed rewards may expose
    ``prediction_record(candidate, prediction, *, k=1.0) -> dict``. This hook
    must only read the completed evaluation, without inference, random draws,
    or timing mutation. Return generated.csv property fields such as dp_mean
    or obj_<name>_mean/std. A batch predictor must retain every result in its
    current wave until these records have been captured. Plain predict-only
    plug-ins need no hook: the returned tuple is recorded directly.
    """

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        ...

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        """Batch variant.  Default loops over ``predict``; override for speed."""
        return [self.predict(c) for c in compositions]
