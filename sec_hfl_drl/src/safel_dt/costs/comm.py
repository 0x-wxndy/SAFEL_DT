"""Communication cost (paper eq. for C_comm)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cohort_size(selection: Sequence[float] | np.ndarray) -> int:
    """Count selected clients (entries with selection > 0)."""
    s = np.asarray(selection, dtype=np.float64)
    return int(np.sum(s > 0.0))


def comm_cost(
    selection: Sequence[float] | np.ndarray,
    n_samples: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray,
) -> float:
    """``sum_i s_i * n_i * sigma_i`` (record-size weighted payload)."""
    s = np.asarray(selection, dtype=np.float64)
    n = np.asarray(n_samples, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    if s.shape != n.shape or s.shape != sig.shape:
        raise ValueError(
            f"shape mismatch: selection={s.shape}, n_samples={n.shape}, sigma={sig.shape}"
        )
    return float(np.sum(s * n * sig))
