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
        An unavailable uncertainty is None, including composite rewards whose
        physical outputs have different units; never replace it with zero.

    Optional training-export hook:
        ``prediction_record(candidate, prediction, *, k=1.0) -> dict`` returns
        generated.csv-compatible fields from the completed evaluation only.
        It must not invoke inference, rebuild a structure, consume randomness,
        or change timing state. Preserve absent scientific uncertainties as
        None and retain metadata from the same evaluation. Native scientific
        adapters can use their cached outputs; composite adapters must also
        retain the selected uncertainty, reward gates, and paired metadata.
        The hook is optional and is not part of structural protocol checks.
    """

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float | None]:
        ...

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float | None]]:
        """Batch variant.  Default loops over ``predict``; override for speed."""
        return [self.predict(c) for c in compositions]
