"""Unit tests for `safel_dt.rl.binary_rl_policy`."""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.rl.binary_rl_policy import BinaryRLConfig, BinaryRLFogPolicy
from safel_dt.rl.policy import RoundFeedback


def _make(
    num_clients: int = 4,
    mu_fog: int = 2,
    seed: int = 0,
    **overrides: object,
) -> BinaryRLFogPolicy:
    base = {
        "mu_fog": mu_fog,
        "eps_start": 0.0,  # deterministic argmax to make tests predictable
        "eps_end": 0.0,
        "alpha": 0.5,
        "seed": seed,
    }
    base.update(overrides)
    return BinaryRLFogPolicy(
        num_clients=num_clients,
        client_ids=list(range(num_clients)),
        n_samples_per_client=[100] * num_clients,
        rounds_total=10,
        cfg=BinaryRLConfig(**base),  # type: ignore[arg-type]
    )


def test_select_size_equals_mu_fog() -> None:
    p = _make(num_clients=5, mu_fog=3)
    chosen = p.select(round_idx=0)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_invalid_mu_fog_rejected() -> None:
    with pytest.raises(ValueError):
        BinaryRLFogPolicy(
            num_clients=4,
            client_ids=[0, 1, 2, 3],
            n_samples_per_client=[100] * 4,
            rounds_total=5,
            cfg=BinaryRLConfig(mu_fog=99),
        )


def test_q_values_grow_for_low_loss_clients() -> None:
    p = _make(num_clients=4, mu_fog=4)
    p.select(round_idx=0)
    p.observe_feedback(
        RoundFeedback(
            round_idx=0,
            selected_local_indices=[0, 1, 2, 3],
            client_losses={0: 0.1, 1: 5.0, 2: 0.1, 3: 5.0},
            reward=0.0,
        )
    )
    q = p.q_values
    assert q[0] > q[1]
    assert q[2] > q[3]


def test_deterministic_argmax_selects_top_q() -> None:
    p = _make(num_clients=4, mu_fog=2)
    p.select(round_idx=0)
    p.observe_feedback(
        RoundFeedback(
            round_idx=0,
            selected_local_indices=[0, 1, 2, 3],
            client_losses={0: 0.1, 1: 5.0, 2: 0.1, 3: 5.0},
            reward=0.0,
        )
    )
    chosen = p.select(round_idx=1)
    assert set(chosen) == {0, 2}


def test_fog_reward_mode_updates_q_uniformly_for_selected() -> None:
    p = _make(num_clients=4, mu_fog=4, use_fog_reward=True)
    p.select(round_idx=0)
    p.observe_feedback(
        RoundFeedback(
            round_idx=0,
            selected_local_indices=[0, 1, 2, 3],
            client_losses={i: float(i) for i in range(4)},
            reward=0.7,
        )
    )
    q = p.q_values
    assert np.allclose(q, q[0])
    assert q[0] > 0.0


def test_epsilon_decays_over_rounds() -> None:
    p = BinaryRLFogPolicy(
        num_clients=4,
        client_ids=[0, 1, 2, 3],
        n_samples_per_client=[100] * 4,
        rounds_total=20,
        cfg=BinaryRLConfig(
            mu_fog=2, eps_start=1.0, eps_end=0.1, eps_decay_rounds=10, seed=0
        ),
    )
    eps0 = p.epsilon
    for _ in range(20):
        p.observe_feedback(
            RoundFeedback(
                round_idx=0,
                selected_local_indices=[0, 1],
                client_losses={0: 0.1, 1: 0.2},
                reward=0.0,
            )
        )
    assert p.epsilon < eps0
    assert abs(p.epsilon - 0.1) < 1e-6
