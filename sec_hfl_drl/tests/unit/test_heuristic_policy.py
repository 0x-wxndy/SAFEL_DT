"""Unit tests for `safel_dt.rl.heuristic_policy`."""

from __future__ import annotations

import pytest

from safel_dt.rl.heuristic_policy import HeuristicConfig, HeuristicFogPolicy
from safel_dt.rl.policy import RoundFeedback


def _make(num_clients: int = 4, mu_fog: int = 2, seed: int = 0) -> HeuristicFogPolicy:
    return HeuristicFogPolicy(
        num_clients=num_clients,
        client_ids=list(range(num_clients)),
        n_samples_per_client=[100] * num_clients,
        rounds_total=10,
        cfg=HeuristicConfig(mu_fog=mu_fog, seed=seed),
    )


def test_select_returns_mu_fog_distinct_indices() -> None:
    p = _make(num_clients=5, mu_fog=3)
    chosen = p.select(round_idx=0)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    assert all(0 <= i < 5 for i in chosen)


def test_invalid_mu_fog_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicFogPolicy(
            num_clients=4,
            client_ids=[0, 1, 2, 3],
            n_samples_per_client=[100] * 4,
            rounds_total=5,
            cfg=HeuristicConfig(mu_fog=10),
        )


def test_high_loss_clients_get_deselected() -> None:
    """After feeding back a high loss on client 0, the heuristic should
    stop selecting it (assuming mu_fog < num_clients)."""
    p = _make(num_clients=4, mu_fog=2)
    p.select(round_idx=0)
    p.observe_feedback(
        RoundFeedback(
            round_idx=0,
            selected_local_indices=[0, 1, 2, 3],
            client_losses={0: 10.0, 1: 0.1, 2: 0.1, 3: 0.1},
            reward=0.0,
        )
    )
    chosen = p.select(round_idx=1)
    assert 0 not in chosen


def test_more_samples_beats_fewer_when_loss_equal() -> None:
    """With equal (zero) losses, the higher-sample client should win."""
    p = HeuristicFogPolicy(
        num_clients=3,
        client_ids=[10, 11, 12],
        n_samples_per_client=[10, 200, 50],
        rounds_total=5,
        cfg=HeuristicConfig(mu_fog=1, m_min=1, seed=0),
    )
    chosen = p.select(round_idx=0)
    assert chosen == [1]  # the 200-sample client (local idx 1)


def test_explore_prob_one_picks_random() -> None:
    """With explore_prob=1.0 selection should be random (not deterministic)."""
    p = _make(num_clients=4, mu_fog=2)
    p.cfg.__dict__  # touch
    p2 = HeuristicFogPolicy(
        num_clients=4,
        client_ids=[0, 1, 2, 3],
        n_samples_per_client=[100, 100, 100, 100],
        rounds_total=5,
        cfg=HeuristicConfig(mu_fog=2, explore_prob=1.0, seed=42),
    )
    seen: set[tuple[int, ...]] = set()
    for _ in range(20):
        seen.add(tuple(p2.select(round_idx=0)))
    assert len(seen) > 1
