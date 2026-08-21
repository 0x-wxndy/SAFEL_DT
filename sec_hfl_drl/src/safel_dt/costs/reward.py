"""Lagrangian reward terms (paper eq. (5))."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostMaxes:
    """Normalisers for the three cost channels."""

    comm_max: float
    train_max: float
    sec_max: float


@dataclass(frozen=True)
class Multipliers:
    """Dual variables ``(nu_lat, nu_cap, nu_priv)``."""

    lat: float = 0.0
    cap: float = 0.0
    priv: float = 0.0


@dataclass(frozen=True)
class PenaltyConstraints:
    """Soft constraint thresholds referenced by the Lagrangian."""

    delta: float
    mu_fog: float
    eta: float


@dataclass(frozen=True)
class RewardWeights:
    """Scalar weights on utility and normalised costs."""

    omega: float = 1.0
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.2


def utility_augmented_lagrangian(
    *,
    utility: float,
    cost_comm: float,
    cost_train: float,
    cost_sec: float,
    cost_max: CostMaxes,
    g_lat_plus: float,
    g_cap_plus: float,
    g_priv_plus: float,
    constraints: PenaltyConstraints,
    multipliers: Multipliers,
    weights: RewardWeights | None = None,
    reward_clip: float | None = None,
) -> float:
    """Augmented Lagrangian reward used by the fog policies.

    ``constraints`` is accepted for API symmetry with the paper notation
    (``delta``, ``mu_fog``, ``eta``) but the hinge violations are already
    computed upstream and passed as ``g_*_plus``.
    """
    del constraints  # violations already materialised as g_*_plus
    w = weights if weights is not None else RewardWeights()
    if cost_max.comm_max <= 0.0 or cost_max.train_max <= 0.0 or cost_max.sec_max <= 0.0:
        raise ValueError("cost_max entries must be > 0")

    cost_term = (
        w.alpha * (cost_comm / cost_max.comm_max)
        + w.beta * (cost_train / cost_max.train_max)
        + w.gamma * (cost_sec / cost_max.sec_max)
    )
    penalty = (
        multipliers.lat * g_lat_plus
        + multipliers.cap * g_cap_plus
        + multipliers.priv * g_priv_plus
    )
    reward = float(w.omega * utility - cost_term - penalty)
    if reward_clip is not None:
        clip = float(reward_clip)
        reward = float(min(max(reward, -clip), clip))
    return reward
