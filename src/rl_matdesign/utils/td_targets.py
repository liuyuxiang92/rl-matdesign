"""Bounded batched Bellman targets for ragged sets of legal next actions.

Batching preserves the Bellman equation and action masks, but changes matrix
shapes. Floating-point results need not be bitwise identical to scalar target
forwards; this is an explicit execution change, not an exact-resume shortcut.
"""
from __future__ import annotations

import numpy as np
import torch

from ..action_features import value_features
from .td_target_cache import TDTargetCache


# Limit action rows, rather than replay rows: a state can have many legal actions.
# This is an implementation memory bound, not a user-facing learning parameter.
_TARGET_ACTION_ROWS_PER_CHUNK = 8192


def _segment_max(values, lengths):
    """Reduce contiguous state segments without one host synchronization each."""
    if hasattr(torch, "segment_reduce"):
        return torch.segment_reduce(
            values, "max", lengths=torch.tensor(lengths, device=values.device),
            unsafe=True,
        )
    # Retain support for older Torch versions without changing legal-action masks.
    return torch.stack([part.max() for part in values.split(lengths)])


def compute_td_targets(
    batch, target_net, s_mat_scaler, elem_feats_scaled, fraction_set, device,
    gamma, *, _max_action_rows=None,
):
    """Return targets in replay order, evaluating only recorded legal actions.

Terminal rows (including nonterminal rows with no legal next action) retain
their immediate reward. Repeated next states share a target evaluation; reward
and gamma are applied separately for every replay row, after one host transfer.
    Deduplication is scoped to this minibatch. No transient cross-update cache
    may change forward/chunk shapes after restart.
"""
    limit = (_TARGET_ACTION_ROWS_PER_CHUNK if _max_action_rows is None
             else int(_max_action_rows))
    if limit < 1:
        raise ValueError("Target action chunk limit must be positive")
    y = [float(row["reward"]) for row in batch]
    missing = []
    by_key = {}
    references = []
    for index, row in enumerate(batch):
        if row["done"] or not row["next_allowed_idx"]:
            continue
        key = TDTargetCache.key(row)
        state_index = by_key.get(key)
        if state_index is None:
            state_index = len(missing)
            by_key[key] = state_index
            missing.append((key, row))
        references.append((index, state_index))
    if not missing:
        return y

    # StandardScaler remains fitted/frozen by the caller. Transform one row per
    # distinct next state, preserving its float64 -> float32 conversion order.
    scaled = s_mat_scaler.transform(np.asarray(
        [row["s_mat_next_raw"] for _, row in missing], dtype=float))
    steps = np.asarray([row["s_step_next"] for _, row in missing], dtype=float)
    maxima = torch.full((len(missing),), -float("inf"), dtype=torch.float32,
                        device=device)
    state_index = offset = 0
    with torch.no_grad():
        while state_index < len(missing):
            owners, actions, segment_owners, lengths = [], [], [], []
            remaining = limit
            while remaining and state_index < len(missing):
                allowed = missing[state_index][1]["next_allowed_idx"]
                count = min(remaining, len(allowed) - offset)
                owners.extend([state_index] * count)
                actions.extend(allowed[offset:offset + count])
                segment_owners.append(state_index)
                lengths.append(count)
                remaining -= count
                offset += count
                if offset == len(allowed):
                    state_index += 1
                    offset = 0
            owner_array = np.asarray(owners, dtype=int)
            action_array = np.asarray(actions, dtype=int)
            q_next = target_net(
                torch.tensor(scaled[owner_array], dtype=torch.float32, device=device),
                torch.tensor(steps[owner_array], dtype=torch.float32, device=device),
                torch.tensor(elem_feats_scaled[action_array[:, 0]],
                             dtype=torch.float32, device=device),
                torch.tensor(value_features(fraction_set, action_array[:, 1]),
                             dtype=torch.float32, device=device),
            ).reshape(-1)
            segment_values = _segment_max(q_next, lengths)
            indices = torch.tensor(segment_owners, dtype=torch.long, device=device)
            maxima[indices] = torch.maximum(maxima[indices], segment_values)

    # Match scalar code's Python float reward + gamma * max_q arithmetic rather
    # than introducing an extra float32/FMA target computation on the device.
    maximum_values = maxima.cpu().tolist()
    for index, state_index in references:
        y[index] += gamma * maximum_values[state_index]
    return y
