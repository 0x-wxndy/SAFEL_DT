"""Krum and Multi-Krum byzantine-robust aggregators.

Krum (Blanchard et al., 2017): for each candidate delta ``i``, sum the
squared L2 distances to its ``n - f - 1`` closest neighbours (including
itself, distance 0); pick the candidate with the smallest sum.

Multi-Krum picks the top-``m`` candidates by Krum score and averages them.

Quorum & fallback
-----------------
Both Krum variants require ``n - f - 2 >= 0`` (at least ``f + 2`` clients
to discriminate). When the upstream selection (e.g. aggressive fog policy
+ stragglers) shrinks the cohort below quorum, the function returns a
plain unweighted mean over the surviving deltas instead of raising.

This matches the "graceful degradation" behaviour expected in HFL: a
cohort with only 1-2 fogs cannot afford a byzantine *and* a robust
median, so we fall back to the most-information-preserving option
(``FedAvg`` on the survivors) rather than crashing the simulation.
Callers that need strict Krum semantics should check the cohort size
themselves and skip the aggregation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np


def _pairwise_sqdists(deltas: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([d.astype(np.float64).ravel() for d in deltas], axis=0)
    diff = stacked[:, None, :] - stacked[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _mean_fallback(deltas: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(d, dtype=np.float64) for d in deltas], axis=0)
    return stacked.mean(axis=0)


def krum_aggregate(
    deltas: Sequence[np.ndarray],
    sample_counts: Sequence[int] | None = None,
    *,
    f: int = 1,
) -> np.ndarray:
    """Return the Krum-selected delta, or the unweighted mean if below quorum.

    Parameters
    ----------
    deltas
        List of plaintext, flat client deltas.
    sample_counts
        Ignored (Krum is unweighted). Accepted for API symmetry with
        `fedavg_aggregate`.
    f
        Upper bound on the number of byzantine clients (so we sum distances
        over ``n - f - 1`` closest neighbours).
    """
    del sample_counts  # unused
    n = len(deltas)
    if n == 0:
        raise ValueError("krum requires at least one client delta.")
    if n - f - 2 < 0:
        warnings.warn(
            f"krum below quorum (n={n}, f={f}); falling back to unweighted mean.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _mean_fallback(deltas)
    sq = _pairwise_sqdists(list(deltas))
    scores = np.zeros(n)
    keep = n - f - 1  # number of closest neighbours summed (incl. self at 0)
    for i in range(n):
        sorted_d = np.sort(sq[i])
        scores[i] = float(sorted_d[:keep].sum())
    winner = int(np.argmin(scores))
    return np.asarray(deltas[winner], dtype=np.float64).copy()


def multi_krum_aggregate(
    deltas: Sequence[np.ndarray],
    sample_counts: Sequence[int] | None = None,
    *,
    f: int = 1,
    m: int = 2,
) -> np.ndarray:
    """Average the ``m`` lowest-scoring deltas under Krum.

    Falls back to unweighted mean over ``min(m, n)`` deltas when the cohort
    is below Krum quorum (``n - f - 2 < 0``).
    """
    del sample_counts
    n = len(deltas)
    if n == 0:
        raise ValueError("multi_krum requires at least one client delta.")
    if n - f - 2 < 0:
        warnings.warn(
            f"multi_krum below quorum (n={n}, f={f}); falling back to unweighted mean.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _mean_fallback(deltas)
    if m <= 0:
        raise ValueError(f"m must be > 0, got m={m}")
    m_eff = min(m, n)
    sq = _pairwise_sqdists(list(deltas))
    keep = n - f - 1
    scores = np.array([float(np.sort(sq[i])[:keep].sum()) for i in range(n)])
    winners = np.argsort(scores)[:m_eff]
    selected = np.stack([np.asarray(deltas[i], dtype=np.float64) for i in winners], axis=0)
    return selected.mean(axis=0)
