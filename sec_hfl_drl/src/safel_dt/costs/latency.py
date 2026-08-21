"""Round latency and soft deadline constraint."""

from __future__ import annotations


def t_round(t_comm: float, t_sec: float, t_ml: float) -> float:
    """Sum of communication, security, and ML time stages."""
    return float(t_comm + t_sec + t_ml)


def g_lat(t: float, delta: float) -> float:
    """Hinge violation ``max(0, t - delta)``."""
    return float(max(0.0, t - delta))
