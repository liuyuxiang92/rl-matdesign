"""Shared episode records and strict one-hot validation for design domains."""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float
    allowed_actions: list = field(default_factory=list)


def one_hot_index(vector, size: int) -> int:
    """Decode exactly one finite unit entry, rejecting malformed actions."""
    try:
        values = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Action must contain valid one-hot vectors") from exc
    if (values.shape != (size,) or not np.all((values == 0.0) | (values == 1.0))
            or np.count_nonzero(values == 1.0) != 1):
        raise ValueError("Action must contain valid one-hot vectors")
    return int(values.argmax())


def step_one_hot(step: int, max_steps: int) -> np.ndarray:
    """Encode a one-based action number in the fixed episode horizon."""
    if not 1 <= step <= max_steps:
        raise ValueError("Step out of range")
    result = np.zeros(max_steps, dtype=float)
    result[step - 1] = 1.0
    return result
