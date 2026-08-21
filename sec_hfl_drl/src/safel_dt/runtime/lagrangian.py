"""Lagrangian reward + dual-ascent multiplier loop (paper section IV).

Two pieces:

* :class:`LagrangianConfig` -- static configuration: cost normalisers,
  constraint thresholds, reward weights, dual step sizes, multiplier
  caps.
* :class:`LagrangianState` -- the *mutable* part: current multipliers
  ``(nu_lat, nu_cap, nu_priv)``. Updated once per global round via
  :meth:`LagrangianState.step`, which averages each constraint
  violation across all fogs (matches the paper's global-multiplier
  treatment).

The per-fog Lagrangian reward (paper eq. (5)) is computed by
:func:`compute_fog_reward` given a :class:`CostBreakdown` and the current
multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safel_dt.costs.reward import (
    CostMaxes,
    Multipliers,
    PenaltyConstraints,
    RewardWeights,
    utility_augmented_lagrangian,
)
from safel_dt.rl.multipliers import dual_ascent_step
from safel_dt.runtime.cost_accounting import CostBreakdown


@dataclass(frozen=True)
class DualStepConfig:
    """Per-multiplier dual step size + clip."""

    eta_lat: float = 0.05
    eta_cap: float = 0.05
    eta_priv: float = 0.05
    nu_max_lat: float = 10.0
    nu_max_cap: float = 10.0
    nu_max_priv: float = 10.0


@dataclass(frozen=True)
class LagrangianConfig:
    """All static knobs for the Lagrangian reward + dual ascent."""

    cost_maxes: CostMaxes
    constraints: PenaltyConstraints
    weights: RewardWeights = field(default_factory=RewardWeights)
    dual_steps: DualStepConfig = field(default_factory=DualStepConfig)
    initial_multipliers: Multipliers = field(default_factory=Multipliers)
    reward_clip: float | None = 10.0


@dataclass
class LagrangianState:
    """Mutable Lagrangian state -- holds the current multipliers."""

    cfg: LagrangianConfig
    multipliers: Multipliers = field(init=False)

    def __post_init__(self) -> None:
        self.multipliers = self.cfg.initial_multipliers

    def step(self, breakdowns: list[CostBreakdown]) -> Multipliers:
        """Project + clip dual-ascent step using the round's violations.

        Constraint violations are *averaged* across all participating
        fogs (this matches global-multiplier semantics; per-fog
        multipliers come for free if the caller maintains one
        ``LagrangianState`` per fog).
        """
        if not breakdowns:
            return self.multipliers
        n = float(len(breakdowns))
        g_lat_mean = sum(b.g_lat for b in breakdowns) / n
        g_cap_mean = sum(b.g_cap for b in breakdowns) / n
        g_priv_mean = sum(b.g_priv for b in breakdowns) / n

        ds = self.cfg.dual_steps
        self.multipliers = Multipliers(
            lat=dual_ascent_step(
                self.multipliers.lat, g_lat_mean, ds.eta_lat, ds.nu_max_lat
            ),
            cap=dual_ascent_step(
                self.multipliers.cap, g_cap_mean, ds.eta_cap, ds.nu_max_cap
            ),
            priv=dual_ascent_step(
                self.multipliers.priv, g_priv_mean, ds.eta_priv, ds.nu_max_priv
            ),
        )
        return self.multipliers


def compute_fog_reward(
    *,
    breakdown: CostBreakdown,
    utility: float,
    state: LagrangianState,
) -> float:
    """Per-fog Lagrangian reward (paper eq. (5))."""
    return utility_augmented_lagrangian(
        utility=utility,
        cost_comm=breakdown.cost_comm,
        cost_train=breakdown.cost_train,
        cost_sec=breakdown.cost_sec,
        cost_max=state.cfg.cost_maxes,
        g_lat_plus=breakdown.g_lat,
        g_cap_plus=breakdown.g_cap,
        g_priv_plus=breakdown.g_priv,
        constraints=state.cfg.constraints,
        multipliers=state.multipliers,
        weights=state.cfg.weights,
        reward_clip=state.cfg.reward_clip,
    )
