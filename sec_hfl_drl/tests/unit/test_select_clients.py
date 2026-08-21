"""Tests for the SelectClients procedure (paper Algorithm 1)."""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.rl.select_clients import SelectionConfig, select_clients


def test_threshold_drops_low_weights() -> None:
    w = np.array([0.1, 0.6, 0.4, 0.8])
    chosen = select_clients(w, SelectionConfig(tau=0.5, m_min=0))
    assert chosen == [1, 3]


def test_capacity_takes_top_k() -> None:
    w = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    chosen = select_clients(w, SelectionConfig(tau=0.0, mu_fog=2, m_min=0))
    assert chosen == [0, 1]


def test_minimum_cohort_promotes_best_rejected() -> None:
    w = np.array([0.1, 0.2, 0.3, 0.4])
    chosen = select_clients(w, SelectionConfig(tau=0.9, m_min=2))
    # No one survives tau=0.9; m_min=2 forces the top-2 by weight.
    assert chosen == [2, 3]


def test_ties_broken_by_index() -> None:
    w = np.array([0.5, 0.5, 0.5, 0.5])
    chosen = select_clients(w, SelectionConfig(tau=0.0, mu_fog=2, m_min=0))
    assert chosen == [0, 1]


def test_empty_input() -> None:
    chosen = select_clients(np.array([]), SelectionConfig())
    assert chosen == []


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(tau=1.5)
    with pytest.raises(ValueError):
        SelectionConfig(mu_fog=0)
    with pytest.raises(ValueError):
        SelectionConfig(m_min=-1)
