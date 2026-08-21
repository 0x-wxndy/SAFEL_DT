"""Tests for the byzantine-robust strategies in `safel_dt.fl.strategies`."""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.fl.strategies import (
    STRATEGY_REGISTRY,
    get_strategy,
    krum_aggregate,
    median_aggregate,
    multi_krum_aggregate,
    trimmed_mean_aggregate,
)


def test_registry_has_expected_keys() -> None:
    assert set(STRATEGY_REGISTRY) >= {"fedavg", "krum", "trimmed_mean", "median", "multi_krum"}


def test_get_strategy_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_strategy("not-a-strategy")


# --- krum -------------------------------------------------------------------


def test_krum_picks_benign_when_one_outlier() -> None:
    # 4 clients, 1 attacker (gamma boost). With f=1, Krum should select
    # one of the benign clients (delta near [0, 0]).
    benign = [np.array([0.0, 0.0]), np.array([0.01, -0.01]), np.array([-0.02, 0.01])]
    attacker = np.array([100.0, 100.0])
    deltas = [*benign, attacker]
    chosen = krum_aggregate(deltas, None, f=1)
    # the chosen delta should be one of the benign ones (not the attacker)
    assert not np.allclose(chosen, attacker)


def test_krum_below_quorum_falls_back_to_mean() -> None:
    """Below-quorum cohorts should warn and return the unweighted mean."""
    deltas = [np.zeros(2), np.ones(2)]
    with pytest.warns(RuntimeWarning, match="below quorum"):
        out = krum_aggregate(deltas, None, f=5)
    assert np.allclose(out, [0.5, 0.5])


def test_krum_rejects_empty_cohort() -> None:
    with pytest.raises(ValueError, match="at least one"):
        krum_aggregate([], None, f=1)


def test_multi_krum_averages_top_m() -> None:
    deltas = [np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([100.0])]
    out = multi_krum_aggregate(deltas, None, f=1, m=3)
    assert np.allclose(out, [1.0])


def test_multi_krum_below_quorum_falls_back_to_mean() -> None:
    deltas = [np.array([1.0]), np.array([3.0])]
    with pytest.warns(RuntimeWarning, match="below quorum"):
        out = multi_krum_aggregate(deltas, None, f=5, m=2)
    assert np.allclose(out, [2.0])


def test_multi_krum_clips_m_to_cohort_size() -> None:
    """``m > n`` is now silently clipped (used to raise ValueError)."""
    deltas = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
    out = multi_krum_aggregate(deltas, None, f=0, m=10)
    assert np.allclose(out, [1.0])


def test_multi_krum_rejects_bad_m() -> None:
    with pytest.raises(ValueError, match="m must be"):
        multi_krum_aggregate([np.zeros(2), np.ones(2), np.zeros(2)], None, f=0, m=0)


# --- trimmed mean -----------------------------------------------------------


def test_trimmed_mean_drops_extremes() -> None:
    deltas = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([100.0, -100.0]),
        np.array([-100.0, 100.0]),
    ]
    out = trimmed_mean_aggregate(deltas, None, beta=0.2)
    assert np.allclose(out, [0.0, 0.0])


def test_trimmed_mean_beta_zero_is_mean() -> None:
    deltas = [np.array([1.0]), np.array([3.0])]
    out = trimmed_mean_aggregate(deltas, None, beta=0.0)
    assert np.allclose(out, [2.0])


def test_trimmed_mean_rejects_bad_beta() -> None:
    with pytest.raises(ValueError):
        trimmed_mean_aggregate([np.zeros(2)], None, beta=0.5)
    with pytest.raises(ValueError):
        trimmed_mean_aggregate([np.zeros(2)], None, beta=-0.1)


# --- median -----------------------------------------------------------------


def test_median_is_robust_to_single_outlier() -> None:
    deltas = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1000.0, 1000.0]),
    ]
    out = median_aggregate(deltas, None)
    assert np.allclose(out, [0.0, 0.0])


def test_median_empty_raises() -> None:
    with pytest.raises(ValueError):
        median_aggregate([], None)
