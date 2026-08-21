"""Tests for the non-SAC fog policies (AllPolicy, RandomPolicy)."""

from __future__ import annotations

import pytest

from safel_dt.rl.policy import AllPolicy, RandomPolicy, RoundFeedback


def test_all_policy_selects_everyone() -> None:
    p = AllPolicy(num_clients=4)
    assert p.select(round_idx=0) == [0, 1, 2, 3]
    p.observe_feedback(
        RoundFeedback(
            round_idx=0,
            selected_local_indices=[0, 1, 2, 3],
            client_losses={},
            reward=0.1,
        )
    )


def test_random_policy_size_and_determinism() -> None:
    p = RandomPolicy(num_clients=5, k=3, seed=42)
    a = p.select(round_idx=0)
    b = p.select(round_idx=1)
    assert len(a) == 3
    assert len(b) == 3
    assert all(0 <= i < 5 for i in a)
    p_same = RandomPolicy(num_clients=5, k=3, seed=42)
    assert p_same.select(round_idx=0) == a


def test_random_policy_no_duplicates() -> None:
    p = RandomPolicy(num_clients=10, k=5, seed=0)
    chosen = p.select(round_idx=0)
    assert len(chosen) == len(set(chosen))


def test_random_policy_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        RandomPolicy(num_clients=3, k=0)
    with pytest.raises(ValueError):
        RandomPolicy(num_clients=3, k=4)
