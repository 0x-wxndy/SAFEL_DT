"""Security / crypto cost helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def per_device_sec_cost(c_enc: float, c_auth: float, c_verify: float) -> float:
    """Per-device security cost ``c_enc + c_auth + c_verify``."""
    return float(c_enc + c_auth + c_verify)


def total_sec_cost(
    selection: Sequence[float] | np.ndarray,
    per_device_sec: Sequence[float] | np.ndarray,
) -> float:
    """Masked sum of per-device security costs."""
    s = np.asarray(selection, dtype=np.float64)
    p = np.asarray(per_device_sec, dtype=np.float64)
    if s.shape != p.shape:
        raise ValueError(f"shape mismatch: selection={s.shape}, per_device={p.shape}")
    return float(np.sum(s * p))
