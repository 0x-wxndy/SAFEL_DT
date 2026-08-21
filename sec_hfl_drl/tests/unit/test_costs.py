"""Tests for the cost equations in `safel_dt.costs.*`."""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.costs import (
    CostMaxes,
    Multipliers,
    PenaltyConstraints,
    RewardWeights,
    cohort_size,
    comm_cost,
    current_workload,
    g_cap,
    g_lat,
    g_priv,
    mi_upper_bound,
    per_device_sec_cost,
    phi_cpu,
    t_round,
    total_sec_cost,
    train_cost,
    utility_augmented_lagrangian,
)

# --- comm -------------------------------------------------------------------


def test_comm_cost_linear_in_selection() -> None:
    n = np.array([10, 20, 30])
    sigma = np.array([1.0, 1.0, 1.0])
    assert comm_cost([1, 0, 0], n, sigma) == 10.0
    assert comm_cost([1, 1, 1], n, sigma) == 60.0
    assert comm_cost([0, 0, 0], n, sigma) == 0.0


def test_comm_cost_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        comm_cost([1, 0], [10, 20, 30], [1.0, 1.0, 1.0])


# --- sec --------------------------------------------------------------------


def test_per_device_sec_cost_sums() -> None:
    assert per_device_sec_cost(0.01, 0.02, 0.03) == pytest.approx(0.06)


def test_total_sec_cost_masked() -> None:
    sel = np.array([1, 0, 1])
    psec = np.array([0.1, 0.2, 0.3])
    assert total_sec_cost(sel, psec) == pytest.approx(0.4)


# --- train ------------------------------------------------------------------


def test_phi_cpu_baseline_and_saturation() -> None:
    """phi_cpu is the M/M/1 form clipped at ``cap`` (default 100).

    The cap keeps ``cost_train`` a smooth, comparable signal; capacity
    *overflow* (``workload >= mu_fog``) is reported by ``g_cap`` instead
    of being baked into the cost.
    """
    assert phi_cpu(0.0, 10.0) == pytest.approx(1.0)
    assert phi_cpu(5.0, 10.0) == pytest.approx(2.0)
    assert phi_cpu(10.0, 10.0) == pytest.approx(100.0)
    assert phi_cpu(20.0, 10.0) == pytest.approx(100.0)
    assert phi_cpu(9.0, 10.0) == pytest.approx(10.0)


def test_phi_cpu_custom_cap() -> None:
    assert phi_cpu(10.0, 10.0, cap=5.0) == pytest.approx(5.0)
    assert phi_cpu(9.0, 10.0, cap=5.0) == pytest.approx(5.0)
    assert phi_cpu(5.0, 10.0, cap=5.0) == pytest.approx(2.0)


def test_phi_cpu_bad_args() -> None:
    with pytest.raises(ValueError):
        phi_cpu(0.0, 0.0)
    with pytest.raises(ValueError):
        phi_cpu(-1.0, 10.0)
    with pytest.raises(ValueError):
        phi_cpu(0.0, 10.0, cap=0.0)


def test_train_cost_scales_with_phi_cpu() -> None:
    base = train_cost(c_sec=1.0, c_ml=1.0, workload=0.0, mu_fog=10.0)
    half = train_cost(c_sec=1.0, c_ml=1.0, workload=5.0, mu_fog=10.0)
    over = train_cost(c_sec=1.0, c_ml=1.0, workload=20.0, mu_fog=10.0)
    assert base == pytest.approx(2.0)
    assert half == pytest.approx(4.0)
    assert over == pytest.approx(200.0)


# --- latency ----------------------------------------------------------------


def test_t_round_sums() -> None:
    assert t_round(0.5, 1.0, 2.5) == pytest.approx(4.0)


def test_g_lat_hinge() -> None:
    assert g_lat(3.0, 5.0) == 0.0
    assert g_lat(7.0, 5.0) == 2.0


# --- privacy ----------------------------------------------------------------


def test_mi_upper_bound_loose_vs_tight() -> None:
    """``n_selected`` is the count of *selected clients*, not samples."""
    loose = mi_upper_bound(epsilon=2.0, n_selected=5)
    tight = mi_upper_bound(epsilon=2.0, n_selected=5, tight=True)
    assert loose == pytest.approx(10.0)
    assert tight == pytest.approx(2.0)


def test_g_priv_hinge() -> None:
    assert g_priv(1.0, 2.0) == 0.0
    assert g_priv(5.0, 2.0) == 3.0


# --- capacity ---------------------------------------------------------------


def test_current_workload_and_g_cap() -> None:
    sel = np.array([1, 1, 0])
    lambdas = np.array([3.0, 7.0, 1.0])
    assert current_workload(sel, lambdas) == pytest.approx(10.0)
    assert g_cap(10.0, 8.0) == 2.0
    assert g_cap(5.0, 8.0) == 0.0


def test_cohort_size_counts_selection() -> None:
    assert cohort_size([1, 0, 0.5, 0]) == 2
    assert cohort_size([0, 0, 0]) == 0


# --- reward -----------------------------------------------------------------


def test_reward_zero_violations_is_utility_minus_costs() -> None:
    r = utility_augmented_lagrangian(
        utility=1.0,
        cost_comm=10.0,
        cost_train=10.0,
        cost_sec=10.0,
        cost_max=CostMaxes(comm_max=100.0, train_max=100.0, sec_max=100.0),
        g_lat_plus=0.0,
        g_cap_plus=0.0,
        g_priv_plus=0.0,
        constraints=PenaltyConstraints(delta=5.0, mu_fog=50.0, eta=2.0),
        multipliers=Multipliers(lat=1.0, cap=1.0, priv=1.0),
        weights=RewardWeights(omega=1.0, alpha=0.3, beta=0.3, gamma=0.2),
    )
    expected = 1.0 - (0.3 * 0.1 + 0.3 * 0.1 + 0.2 * 0.1)
    assert r == pytest.approx(expected)


def test_reward_violations_penalised_by_multipliers() -> None:
    r_no_viol = utility_augmented_lagrangian(
        utility=1.0,
        cost_comm=0.0,
        cost_train=0.0,
        cost_sec=0.0,
        cost_max=CostMaxes(1.0, 1.0, 1.0),
        g_lat_plus=0.0,
        g_cap_plus=0.0,
        g_priv_plus=0.0,
        constraints=PenaltyConstraints(1.0, 1.0, 1.0),
        multipliers=Multipliers(),
    )
    r_viol = utility_augmented_lagrangian(
        utility=1.0,
        cost_comm=0.0,
        cost_train=0.0,
        cost_sec=0.0,
        cost_max=CostMaxes(1.0, 1.0, 1.0),
        g_lat_plus=0.5,
        g_cap_plus=0.0,
        g_priv_plus=0.0,
        constraints=PenaltyConstraints(1.0, 1.0, 1.0),
        multipliers=Multipliers(lat=2.0),
    )
    assert r_viol < r_no_viol
    assert r_no_viol - r_viol == pytest.approx(1.0)


def test_reward_clip_is_applied() -> None:
    r = utility_augmented_lagrangian(
        utility=100.0,
        cost_comm=0.0,
        cost_train=0.0,
        cost_sec=0.0,
        cost_max=CostMaxes(1.0, 1.0, 1.0),
        g_lat_plus=0.0,
        g_cap_plus=0.0,
        g_priv_plus=0.0,
        constraints=PenaltyConstraints(1.0, 1.0, 1.0),
        multipliers=Multipliers(),
        reward_clip=5.0,
    )
    assert r == 5.0
