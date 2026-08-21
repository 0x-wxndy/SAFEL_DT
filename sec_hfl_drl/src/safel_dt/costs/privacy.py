"""Privacy proxy and DP noise (paper eqs. (8)-(9)).

The exact MI ``I(D_s; W_s)`` is intractable for high-dimensional models, so we
upper-bound it using the standard DP -> MI inequality::

    I(D_s; W_s)  <=  epsilon * |D_s|   (loose, paper section III)
                 <=  epsilon^2 / 2     (tighter, Renyi DP)

and treat the slack ``g_priv = max(0, I_est - eta)`` as a soft constraint
penalised in the Lagrangian reward.

Gaussian DP noise on aggregate updates is provided here as a helper; it is
applied to the decrypted aggregate before broadcast (paper section III,
"operational privacy realisation", item 1).
"""

from __future__ import annotations

import numpy as np


def mi_upper_bound(epsilon: float, n_selected: int, *, tight: bool = False) -> float:
    """Conservative upper bound on ``I(D_s; W_s)``.

    ``n_selected`` is the *count of selected clients* (``|D_s|``), not the
    total number of training samples. The loose bound is ``epsilon * |D_s|``
    from group privacy composition over k mechanisms; the tight bound is the
    Renyi-DP form ``epsilon^2 / 2``.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if n_selected < 0:
        raise ValueError(f"n_selected must be >= 0, got {n_selected}")
    if tight:
        return float(0.5 * epsilon * epsilon)
    return float(epsilon * n_selected)


def g_priv(mi_estimate: float, eta: float) -> float:
    """Hinge violation ``max(0, mi_estimate - eta)``."""
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")
    return float(max(0.0, mi_estimate - eta))


def gaussian_dp_noise(scale: float, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Sample ``N(0, scale^2 I)`` Gaussian noise."""
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    return rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float64)
