"""RFMagpiePredictor — generic sklearn-regressor-on-composition predictor.

Loads a pretrained sklearn model (``RandomForestRegressor`` or any regressor
saved with joblib) and predicts a scalar property from a matminer/Magpie feature
vector of the composition. The same featurizer the rest of the framework uses
(:data:`rl_matdesign.featurization.feature_calculators`) is reused so the feature
order matches the model's training-time order exactly.

This is the composition-space analog of a DeepMD property head: give it a model
file, it returns ``(value, std)`` for a composition. It has **no opinion** about
whether higher or lower is better — that is the reward engine's ``direction:``
job. So a sintering-temperature model returns the raw temperature ``T`` (not
``-T``); pair it with ``direction: min`` to drive ``T`` down.

``std`` is the spread across the forest's trees when available (``estimators_``),
else ``None``. Tree spread is descriptive, not calibrated predictive uncertainty.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np


class RFMagpiePredictor:
    """Predict a scalar property from a composition via an sklearn model.

    Parameters
    ----------
    model_path:
        Path to the pretrained sklearn joblib checkpoint.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._cache = OrderedDict()

    # ------------------------------------------------------------------
    # Lazy model + featurizer loaders
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            import joblib

            self._model = joblib.load(self.model_path)
        return self._model

    @staticmethod
    def _get_featurizer():
        from ..featurization import feature_calculators

        if feature_calculators is None:
            raise ImportError(
                "matminer featurizer unavailable — RFMagpiePredictor requires "
                "matminer + pymatgen. Install with `pip install matminer pymatgen`."
            )
        return feature_calculators

    # ------------------------------------------------------------------
    # PropertyPredictor protocol
    # ------------------------------------------------------------------

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float | None]:
        """Return ``(value, std)`` for *composition*.

        *composition* is a dict of ``{element: fraction}`` (fractions need not
        sum to 1 — pymatgen ``Composition`` renormalises internally). The value
        is the **raw** model output; the reward engine applies the sign.
        """
        key = tuple(sorted((str(el), float(f)) for el, f in composition.items()))
        if (not key or any(not np.isfinite(f) or f < 0 for _, f in key)
                or not any(f > 0 for _, f in key)):
            raise ValueError("RF composition needs finite nonnegative amounts and a positive total")
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        from pymatgen.core.composition import Composition

        model = self._load_model()
        feat_calc = self._get_featurizer()

        comp_obj = Composition(dict(key))
        features = np.asarray(feat_calc.featurize(comp_obj), dtype=float).reshape(1, -1)
        if not np.all(np.isfinite(features)):
            raise ValueError("RF featurizer returned nonfinite features")

        value = float(model.predict(features)[0])
        if not np.isfinite(value):
            raise ValueError("RF model returned a nonfinite prediction")

        # Preserve the historical population tree spread when available. A
        # missing/singleton ensemble cannot establish a spread estimate.
        estimators = getattr(model, "estimators_", ())
        std = None
        if len(estimators) >= 2:
            per_tree = np.asarray([est.predict(features)[0] for est in estimators])
            if not np.all(np.isfinite(per_tree)):
                raise ValueError("RF tree returned a nonfinite prediction")
            std = float(per_tree.std())

        result = (value, std)
        self._cache[key] = result
        while len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return result

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float | None]]:
        return [self.predict(c) for c in compositions]
