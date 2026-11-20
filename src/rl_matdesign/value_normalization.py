"""Adaptive target scaling with output preservation for scalar Q/V networks."""
from __future__ import annotations

import torch
from torch import nn

POPART_PROFILE = "popart_v1"
_EMA_WEIGHT = 0.001
_SCALE_FLOOR_RATIO = 1e-6


class PopArt(nn.Module):
    """Running target statistics saved alongside the network's linear head."""

    def __init__(self) -> None:
        super().__init__()
        for name, initial in (("mean", 0.), ("variance", 0.), ("scale", 1.), ("anchor", 0.)):
            self.register_buffer(name, torch.tensor(initial, dtype=torch.float64))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("targets_seen", torch.zeros((), dtype=torch.int64))

    def normalize(self, targets: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return ((targets.to(self.mean) - self.mean) / self.scale).to(dtype)

    def decode(self, prediction: torch.Tensor) -> torch.Tensor:
        return (prediction.to(self.mean) * self.scale + self.mean).to(prediction.dtype)

    @torch.no_grad()
    def update(self, targets: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        """Fit one minibatch and compensate the head before building loss graphs."""
        values = targets.detach().to(self.mean)
        if values.numel() == 0:
            raise ValueError("PopArt needs at least one target")
        batch_variance, batch_mean = torch.var_mean(values, unbiased=False)
        rms = (batch_variance + batch_mean.square()).sqrt()
        started = self.anchor > 0
        next_anchor = torch.where(started, self.anchor, rms)
        ready = next_anchor > 0
        beta = _EMA_WEIGHT
        variance = ((1 - beta) * self.variance + beta * batch_variance
                    + beta * (1 - beta) * (batch_mean - self.mean).square())
        next_mean = torch.where(started, (1 - beta) * self.mean + beta * batch_mean, batch_mean)
        next_variance = torch.where(started, variance, rms.square())
        next_scale = torch.where(
            ready,
            torch.maximum(next_variance.sqrt(), _SCALE_FLOOR_RATIO * next_anchor),
            self.scale,
        )

        # Compute and validate the full reparameterization before mutating any
        # state, including the values as represented in the head's precision.
        weight = (head.weight.to(self.mean) * (self.scale / next_scale)).to(head.weight.dtype)
        bias = ((head.bias.to(self.mean) * self.scale + self.mean - next_mean)
                / next_scale).to(head.bias.dtype)
        normalized = ((values - next_mean) / next_scale).to(head.weight.dtype)
        stats = torch.stack((next_mean, next_variance, next_scale, next_anchor))
        if not torch.cat((stats, weight.flatten(), bias, normalized.flatten())).isfinite().all().item():
            raise ValueError("Nonfinite PopArt targets, statistics or compensated head")
        if next_scale <= 0:
            raise ValueError("PopArt scale must be positive")

        head.weight.copy_(weight)
        head.bias.copy_(bias)
        self.mean.copy_(next_mean)
        self.variance.copy_(next_variance)
        self.scale.copy_(next_scale)
        self.anchor.copy_(next_anchor)
        self.updates.add_(1)
        self.targets_seen.add_(values.numel())
        return normalized

    def diagnostics(self, prefix: str) -> dict:
        mean, variance, scale, anchor = torch.stack(
            (self.mean, self.variance, self.scale, self.anchor)
        ).tolist()
        spread = variance ** .5
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_spread": spread,
            f"{prefix}_scale": scale,
            f"{prefix}_initialized": anchor > 0,
            f"{prefix}_floor_active": anchor > 0 and scale > spread,
            f"{prefix}_updates": int(self.updates.item()),
            f"{prefix}_targets_seen": int(self.targets_seen.item()),
        }
