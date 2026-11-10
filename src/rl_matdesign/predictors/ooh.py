"""OOH scientific oracle with one shared evaluation and a legacy reward view."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import math
from threading import RLock
from typing import Dict, List, Tuple

from .outputs import ScientificOutput
from ..utils.rollout_errors import EvaluationProtocolError


_CACHE_CAPACITY = 4096


class _OutputLRU(OrderedDict):
    """Bound insertion and checkpoint update alike; keys include oracle identity."""

    def __init__(self, capacity=_CACHE_CAPACITY):
        self.capacity = capacity
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > getattr(self, 'capacity', _CACHE_CAPACITY):
            self.popitem(last=False)

    def update(self, other=(), **kwargs):
        pairs = other.items() if isinstance(other, Mapping) else other
        for key, value in pairs:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value


class OOHCatalystPredictor:
    """Positive-direction overpotential with explicit, optional spread estimates.

    ``predict_outputs`` is the scientific interface and never applies an objective.
    Unit is explicitly unknown because the original model lacks a verified unit
    declaration. ``predict_raw`` selects models/configs/total descriptive spread;
    total means the derived pooled population spread, not native uncertainty.
    ``predict`` is a flat compatibility reward interface: it negates the mean
    because smaller overpotential is better, then applies the requested objective.

    Model/template assets must already exist. They are hashed before model imports,
    then monitored for changes before cache reuse. No asset is downloaded or edited.
    """

    def __init__(self, base_poscar: str, dp_models: List[str], *, objective: str = 'mean',
                 k: float = 1.0, n_random_configs: int = 10, ads_height: float = 1.9,
                 ads_dz: float = 1.0, geo_opt: bool = False,
                 geo_opt_model: str = './DPA-3.1-3M_1.pt', rng_seed: int = 123,
                 uncertainty: str = 'models', output_index: int = 0, adsorbates=None) -> None:
        from abcde_ooh.dp_predictor import (
            ADSORBATE_MODES, DPConfig, DeepMDOverpotentialPredictor, normalize_adsorbates,
        )
        self.objective = objective
        self.k = k
        self.uncertainty = uncertainty
        self.output_index = output_index
        self.adsorbates = ADSORBATE_MODES if adsorbates is None else normalize_adsorbates(adsorbates)
        self._validate_reward_view()
        cfg = DPConfig(base_poscar=base_poscar, model_files=tuple(dp_models),
                       n_random_configs=n_random_configs, ads_height=ads_height, ads_dz=ads_dz,
                       seed=rng_seed, geo_opt=geo_opt, geo_opt_model=geo_opt_model,
                       output_index=output_index, adsorbates=self.adsorbates)
        self._predictor = DeepMDOverpotentialPredictor(cfg)
        self._identity = self._predictor.oracle_identity()
        self._settings_key = self._identity_key(self._identity)
        self._cache = _OutputLRU()
        self._lock = RLock()

    @staticmethod
    def _identity_key(identity: dict) -> str:
        payload = json.dumps(identity, sort_keys=True, allow_nan=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _validate_reward_view(self) -> None:
        if self.objective not in ('mean', 'mean_minus_kstd', 'mean_plus_kstd'):
            raise EvaluationProtocolError('Unknown OOH legacy reward objective')
        if self.uncertainty not in ('models', 'configs', 'total'):
            raise EvaluationProtocolError('OOH uncertainty must be models, configs, or total')
        if isinstance(self.k, bool) or not isinstance(self.k, (int, float)) or not math.isfinite(self.k) or self.k < 0:
            raise EvaluationProtocolError('OOH k must be finite and nonnegative')

    @property
    def output_schema(self):
        return self._predictor.output_schema

    def oracle_identity(self) -> dict:
        identity = self._predictor.oracle_identity()
        if (self._identity_key(identity) != self._settings_key
                or self.output_index != self._predictor.cfg.output_index
                or tuple(self.adsorbates) != self._predictor.cfg.adsorbates):
            raise EvaluationProtocolError('OOH oracle configuration changed; create a new predictor')
        return identity

    def _comp_key(self, composition: Dict[str, float]) -> tuple:
        from abcde_ooh.dp_predictor import _normalize_composition
        from ..utils.composition import composition_key
        return self._settings_key, composition_key(_normalize_composition(composition))

    def predict_outputs(self, composition: Dict[str, float]) -> dict[str, ScientificOutput]:
        """Use a single cached model/config matrix for every scientific view."""
        with self._lock:
            self.oracle_identity()  # Includes fast metadata checks before a cache hit.
            key = self._comp_key(composition)
            output = self._cache.get(key)
            # Historical checkpoint caches contain tuples, which cannot carry
            # the named scientific schema or raw samples. Never reinterpret them.
            if isinstance(output, ScientificOutput) and output.schema == self.output_schema['overpotential']:
                self._cache.move_to_end(key)
                return {'overpotential': output}
            output = self._predictor.predict_outputs(composition)['overpotential']
            self._cache[key] = output
            return {'overpotential': output}

    def predict_raw(self, composition: Dict[str, float]) -> Tuple[float, float | None]:
        from abcde_ooh.dp_predictor import select_output_spread
        output = self.predict_outputs(composition)['overpotential']
        return output.mean, select_output_spread(output, self.uncertainty)

    def cached_outputs(self, composition: Dict[str, float]) -> dict[str, ScientificOutput] | None:
        """Read a completed scientific output only; a cache miss never infers."""
        with self._lock:
            output = self._cache.get(self._comp_key(composition))
            if not isinstance(output, ScientificOutput) or output.schema != self.output_schema['overpotential']:
                return None
            return {'overpotential': output}

    def prediction_record(self, composition, prediction, *, k: float = 1.0) -> dict:
        """Expose the exact cached positive overpotential and selected spread."""
        from abcde_ooh.dp_predictor import select_output_spread

        outputs = self.cached_outputs(composition)
        if outputs is None:
            raise KeyError("Completed OOH prediction missing from the scientific output cache")
        output = outputs['overpotential']
        std = select_output_spread(output, self.uncertainty)
        phase_ok, phase_label = self.check_phase(composition)
        return {
            "dp_mean": output.mean, "dp_std": std,
            "dp_mean_minus_std": None if std is None else output.mean - k * std,
            "primary_ok": bool(phase_ok), "primary_label": phase_label or '',
        }

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float | None]:
        self._validate_reward_view()
        mean, spread = self.predict_raw(composition)
        if self.objective == 'mean' or self.k == 0:
            return -mean, spread
        if spread is None:
            raise EvaluationProtocolError(
                f'OOH objective {self.objective} requires {self.uncertainty} spread, '
                'but its sample axis has insufficient observations')
        reward = -mean + (-1 if self.objective == 'mean_minus_kstd' else 1) * self.k * spread
        if not math.isfinite(reward):
            raise EvaluationProtocolError('OOH legacy reward is not finite')
        return reward, spread

    def check_phase(self, composition: Dict[str, float]) -> Tuple[bool, str]:
        from abcde_ooh.constraints.primary_phase import check_primary_phase
        ok, label = check_primary_phase(composition)
        return bool(ok), label or ''

    def source_identity(self) -> dict:
        from ..structure_sources import SOURCE_SEED_VERSION
        self._predictor.validate_assets()
        return {'eval_seed': self._predictor.cfg.seed, 'seed_version': SOURCE_SEED_VERSION,
                'sources': [self._predictor.structure_source.to_schema()],
                'property_sources': {'overpotential': 'ooh_slab'}}

    def batch_predict_outputs(self, compositions):
        return [self.predict_outputs(composition) for composition in compositions]

    def batch_predict(self, compositions):
        return [self.predict(composition) for composition in compositions]
