"""Coordinate-wise median aggregator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def median_aggregate(
    deltas: Sequence[np.ndarray],
    sample_counts: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the coordinate-wise median of client deltas."""
    del sample_counts
    if not deltas:
        raise ValueError("median requires at least one client delta.")
    stacked = np.stack([np.asarray(d, dtype=np.float64) for d in deltas], axis=0)
    return np.median(stacked, axis=0)
