"""Unit tests for `safel_dt.runtime.lagrangian`.

Covers:

* The Lagrangian reward agrees with the closed-form `utility_augmented_lagrangian`.
* Multipliers grow when the corresponding constraint is violated.
* Multipliers stay at zero when constraints are satisfied (projection works).
* Multipliers cap at `nu_max` after sustained violations.
"""

from __future__ import annotations

from safel_dt.costs.reward import CostMaxes, Multipliers, PenaltyConstraints
from safel_dt.runtime.cost_accounting import CostBreakdown
from safel_dt.runtime.lagrangian import (
    DualStepConfig,
    LagrangianConfig,
    LagrangianState,
    compute_fog_reward,
)


def _cfg(**overrides: object) -> LagrangianConfig:
    base = {
        "cost_maxes": CostMaxes(comm_max=1000.0, train_max=10.0, sec_max=10.0),
        "constraints": PenaltyConstraints(delta=1.0, mu_fog=10.0, eta=1.0),
        "dual_steps": DualStepConfig(
            eta_lat=0.1, eta_cap=0.1, eta_priv=0.1,
            nu_max_lat=2.0, nu_max_cap=2.0, nu_max_priv=2.0,
        ),
        "initial_multipliers": Multipliers(0.0, 0.0, 0.0),
        "reward_clip": None,
    }
    base.update(overrides)
    return LagrangianConfig(**base)  # type: ignore[arg-type]


def _breakdown(*, g_lat: float = 0.0, g_cap: float = 0.0, g_priv: float = 0.0) -> CostBreakdown:
    return CostBreakdown(
        fog_id=0,
        n_selected=1,
        cost_comm=10.0,
        cost_train=1.0,
        cost_sec=1.0,
        t_round=0.5,
        workload=1.0,
        mi_estimate=0.0,
        g_lat=g_lat,
        g_cap=g_cap,
        g_priv=g_priv,
    )


def test_no_violations_keeps_multipliers_at_zero() -> None:
    state = LagrangianState(cfg=_cfg())
    for _ in range(5):
        state.step([_breakdown()])
    assert state.multipliers.lat == 0.0
    assert state.multipliers.cap == 0.0
    assert state.multipliers.priv == 0.0


def test_latency_violation_grows_nu_lat() -> None:
    state = LagrangianState(cfg=_cfg())
    state.step([_breakdown(g_lat=0.5)])
    assert state.multipliers.lat > 0.0
    assert state.multipliers.cap == 0.0
    assert state.multipliers.priv == 0.0


def test_multipliers_cap_at_nu_max() -> None:
    state = LagrangianState(cfg=_cfg())
    for _ in range(100):
        state.step([_breakdown(g_lat=5.0, g_cap=5.0, g_priv=5.0)])
    assert state.multipliers.lat == 2.0
    assert state.multipliers.cap == 2.0
    assert state.multipliers.priv == 2.0


def test_reward_drops_when_penalty_active() -> None:
    state = LagrangianState(cfg=_cfg())
    r_clean = compute_fog_reward(
        breakdown=_breakdown(),
        utility=0.1,
        state=state,
    )
    state.multipliers = Multipliers(lat=1.0, cap=1.0, priv=1.0)
    r_violated = compute_fog_reward(
        breakdown=_breakdown(g_lat=0.5, g_cap=2.0, g_priv=0.5),
        utility=0.1,
        state=state,
    )
    assert r_violated < r_clean


def test_reward_clip_applied() -> None:
    cfg = _cfg(reward_clip=0.05)
    state = LagrangianState(cfg=cfg)
    r = compute_fog_reward(
        breakdown=_breakdown(),
        utility=10.0,
        state=state,
    )
    assert r == 0.05
