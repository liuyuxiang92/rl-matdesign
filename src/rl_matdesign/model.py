from __future__ import annotations

import torch
from torch import nn

from .value_normalization import POPART_PROFILE, PopArt


class _ValueRegressor(nn.Module):
    def __init__(self, value_normalization: str, value_normalization_profile: str) -> None:
        super().__init__()
        if value_normalization not in ("none", "popart"):
            raise ValueError(f"Unknown value normalization: {value_normalization!r}")
        if value_normalization_profile != POPART_PROFILE:
            raise ValueError(f"Unknown value normalization profile: {value_normalization_profile!r}")
        self.value_normalizer = PopArt() if value_normalization == "popart" else None

    def update_value_normalization(self, targets: torch.Tensor) -> torch.Tensor:
        if self.value_normalizer is None:
            return targets
        return self.value_normalizer.update(targets, self.net[-1])

    def _decode_value(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.value_normalizer is None:
            return prediction
        return self.value_normalizer.decode(prediction)


class QRegressor(_ValueRegressor):
    """Scalar Q(s,a) regressor with a public prediction in return units."""

    def __init__(
        self,
        *,
        state_dim: int,
        step_dim: int,
        elem_dim: int,
        frac_dim: int,
        hidden_dim: int = 128,
        value_normalization: str = "none",
        value_normalization_profile: str = POPART_PROFILE,
    ) -> None:
        super().__init__(value_normalization, value_normalization_profile)
        in_dim = int(state_dim + step_dim + elem_dim + frac_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_material, s_step, a_elem, a_comp):
        return self._decode_value(self.forward_normalized(s_material, s_step, a_elem, a_comp))

    def forward_normalized(self, s_material, s_step, a_elem, a_comp):
        x = torch.cat(
            (
                s_material.float(),
                s_step.float(),
                a_elem.float(),
                a_comp.float(),
            ),
            dim=1,
        )
        return self.net(x)


class PolicyNet(nn.Module):
    """Actor network for policy gradient methods. Same architecture as QRegressor; outputs scalar logit."""

    def __init__(
        self,
        *,
        state_dim: int,
        step_dim: int,
        elem_dim: int,
        frac_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        in_dim = int(state_dim + step_dim + elem_dim + frac_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_material, s_step, a_elem, a_comp):
        x = torch.cat(
            (
                s_material.float(),
                s_step.float(),
                a_elem.float(),
                a_comp.float(),
            ),
            dim=1,
        )
        return self.net(x)


class ValueNet(_ValueRegressor):
    """Critic network for A2C. Takes state features only (material + step encoding)."""

    def __init__(
        self,
        *,
        state_dim: int,
        step_dim: int,
        hidden_dim: int = 128,
        value_normalization: str = "none",
        value_normalization_profile: str = POPART_PROFILE,
    ) -> None:
        super().__init__(value_normalization, value_normalization_profile)
        in_dim = int(state_dim + step_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_material, s_step):
        return self._decode_value(self.forward_normalized(s_material, s_step))

    def forward_normalized(self, s_material, s_step):
        x = torch.cat(
            (
                s_material.float(),
                s_step.float(),
            ),
            dim=1,
        )
        return self.net(x)
