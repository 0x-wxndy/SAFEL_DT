"""Fog capacity / workload constraint helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def current_workload(
    selection: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
) -> float:
    """``sum_i s_i * lambda_i``."""
    s = np.asarray(selection, dtype=np.float64)
    lam = np.asarray(lambdas, dtype=np.float64)
    if s.shape != lam.shape:
        raise ValueError(f"shape mismatch: selection={s.shape}, lambdas={lam.shape}")
    return float(np.sum(s * lam))


def g_cap(workload: float, mu_fog: float) -> float:
    """Hinge violation ``max(0, workload - mu_fog)``."""
    return float(max(0.0, workload - mu_fog))
