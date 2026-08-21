"""Registry of Byzantine-robust / FedAvg aggregation strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from safel_dt.fl.strategies.fedavg import fedavg_aggregate
from safel_dt.fl.strategies.krum import krum_aggregate, multi_krum_aggregate
from safel_dt.fl.strategies.median import median_aggregate
from safel_dt.fl.strategies.trimmed_mean import trimmed_mean_aggregate

StrategyFn = Callable[..., np.ndarray]


def _krum(deltas: Sequence[np.ndarray], sample_counts: Sequence[int] | None = None, **kw: Any) -> np.ndarray:
    return krum_aggregate(deltas, sample_counts, **kw)


def _multi_krum(
    deltas: Sequence[np.ndarray], sample_counts: Sequence[int] | None = None, **kw: Any
) -> np.ndarray:
    return multi_krum_aggregate(deltas, sample_counts, **kw)


STRATEGY_REGISTRY: dict[str, StrategyFn] = {
    "fedavg": fedavg_aggregate,
    "krum": _krum,
    "multi_krum": _multi_krum,
    "median": median_aggregate,
    "trimmed_mean": trimmed_mean_aggregate,
}


def get_strategy(name: str) -> StrategyFn:
    """Return a strategy callable; raise ``KeyError`` for unknown names."""
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown aggregation strategy: {name!r}") from exc


__all__ = [
    "STRATEGY_REGISTRY",
    "fedavg_aggregate",
    "get_strategy",
    "krum_aggregate",
    "median_aggregate",
    "multi_krum_aggregate",
    "trimmed_mean_aggregate",
]
