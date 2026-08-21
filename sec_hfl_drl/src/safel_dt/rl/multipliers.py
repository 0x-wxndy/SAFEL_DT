"""Dual-ascent helpers for Lagrangian multipliers."""

from __future__ import annotations


def dual_ascent_step(nu: float, g_plus: float, eta: float, nu_max: float) -> float:
    """Projected dual step ``clip(nu + eta * g_plus, 0, nu_max)``."""
    if eta < 0.0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    if nu_max < 0.0:
        raise ValueError(f"nu_max must be >= 0, got {nu_max}")
    updated = float(nu) + float(eta) * float(g_plus)
    return float(min(max(updated, 0.0), nu_max))
