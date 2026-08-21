"""Sample-weighted FedAvg aggregator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fedavg_aggregate(
    deltas: Sequence[np.ndarray],
    sample_counts: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the sample-weighted mean of client deltas."""
    if not deltas:
        raise ValueError("fedavg requires at least one client delta.")
    stacked = np.stack([np.asarray(d, dtype=np.float64) for d in deltas], axis=0)
    if sample_counts is None:
        return stacked.mean(axis=0)
    weights = np.asarray(sample_counts, dtype=np.float64)
    if weights.shape[0] != stacked.shape[0]:
        raise ValueError("sample_counts length must match number of deltas")
    total = float(weights.sum())
    if total <= 0.0:
        return stacked.mean(axis=0)
    return (stacked * weights[:, None]).sum(axis=0) / total
