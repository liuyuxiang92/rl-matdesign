"""PropertyPredictor Protocol — the reward interface for the RL framework.

Any object that implements ``predict(composition) -> (mean, std)`` satisfies
this interface via structural subtyping.  No inheritance required.
"""
from __future__ import annotations

from typing import Any, List, Protocol, Tuple, runtime_checkable


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
        Multi-group inputs map group names to composition/categorical mappings
        or ordered lists of token IDs for sequence groups. External adapters own
        domain-specific encodings, molecular assembly and position conventions.

    Returns
    -------
    (mean, std):
        Scalar mean reward and ensemble standard deviation.
        ``std`` is used for uncertainty-driven exploration and Pareto tracking.
    """

    def predict(self, composition: Any) -> Tuple[float, float]:
        ...

    def batch_predict(
        self, compositions: List[Any]
    ) -> List[Tuple[float, float]]:
        """Batch variant.  Default loops over ``predict``; override for speed."""
        return [self.predict(c) for c in compositions]
