"""Tests for the A4 D3QN epsilon schedule + exponential decay flag."""

from __future__ import annotations

import pytest

from safel_dt.rl.d3qn import D3qnConfig, D3qnController


def _ctrl(**cfg_overrides: object) -> D3qnController:
    base = dict(epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_steps=100, seed=0)
    base.update(cfg_overrides)
    return D3qnController(obs_dim=4, n_actions=2, cfg=D3qnConfig(**base))


def test_default_nu_max_paper_consistent() -> None:
    """A2: bumped from 5.0 to 10.0 (paper Table 1).

    Re-asserted here because a downstream module imports the default.
    """
    from safel_dt.runtime.lagrangian import DualStepConfig
    assert DualStepConfig().nu_max_lat == 10.0
    assert DualStepConfig().nu_max_cap == 10.0
    assert DualStepConfig().nu_max_priv == 10.0


def test_linear_decay_default_endpoints() -> None:
    c = _ctrl()
    assert c.epsilon == pytest.approx(1.0)
    c._step = 50
    assert c.epsilon == pytest.approx(0.51, abs=1e-6)
    c._step = 100
    assert c.epsilon == pytest.approx(0.02)
    c._step = 200
    assert c.epsilon == pytest.approx(0.02)


def test_exponential_decay_drops_faster_than_linear() -> None:
    """At 30% of the decay window, exponential should be far below linear."""
    lin = _ctrl(epsilon_exponential=False)
    exp = _ctrl(epsilon_exponential=True)
    lin._step = 30
    exp._step = 30
    assert exp.epsilon < lin.epsilon
    # Sanity: exponential is still strictly above the floor at step 30.
    assert exp.epsilon > exp.cfg.epsilon_end


def test_exponential_decay_hits_floor_by_decay_steps() -> None:
    exp = _ctrl(epsilon_exponential=True)
    exp._step = 99
    near_floor = exp.epsilon
    exp._step = 100
    at_floor = exp.epsilon
    assert at_floor == pytest.approx(exp.cfg.epsilon_end)
    assert near_floor >= at_floor


def test_lower_epsilon_end_is_respected() -> None:
    c = _ctrl(epsilon_end=0.005, epsilon_decay_steps=50)
    c._step = 50
    assert c.epsilon == pytest.approx(0.005)
