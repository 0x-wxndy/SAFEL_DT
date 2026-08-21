"""Trimmed-mean Byzantine-robust aggregator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def trimmed_mean_aggregate(
    deltas: Sequence[np.ndarray],
    sample_counts: Sequence[int] | None = None,
    *,
    beta: float = 0.1,
) -> np.ndarray:
    """Coordinate-wise trimmed mean: drop ``floor(beta * n)`` extremes each side."""
    del sample_counts
    if not deltas:
        raise ValueError("trimmed_mean requires at least one client delta.")
    if not (0.0 <= beta < 0.5):
        raise ValueError(f"beta must be in [0, 0.5), got {beta}")
    stacked = np.stack([np.asarray(d, dtype=np.float64) for d in deltas], axis=0)
    n = stacked.shape[0]
    k = int(np.floor(beta * n))
    if k == 0:
        return stacked.mean(axis=0)
    if 2 * k >= n:
        raise ValueError(f"beta={beta} trims everything for n={n}")
    sorted_vals = np.sort(stacked, axis=0)
    kept = sorted_vals[k : n - k]
    return kept.mean(axis=0)
