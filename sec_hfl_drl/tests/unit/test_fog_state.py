"""Tests for the per-fog observation extractor."""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.rl.state_extraction import FEATURES_PER_CLIENT, FogTraceBuffer, obs_dim


def test_obs_dim() -> None:
    assert obs_dim(3) == 3 * FEATURES_PER_CLIENT


def test_initial_observation_is_finite() -> None:
    buf = FogTraceBuffer(
        client_ids=[10, 20, 30],
        n_samples_per_client=[100, 200, 300],
        rounds_total=10,
    )
    obs = buf.build_observation(round_idx=0)
    assert obs.shape == (obs_dim(3),)
    assert np.all(np.isfinite(obs))


def test_loss_ema_updates_for_selected_clients_only() -> None:
    buf = FogTraceBuffer(
        client_ids=[0, 1, 2], n_samples_per_client=[100, 100, 100], rounds_total=10
    )
    buf.update_after_round(
        selected_client_ids=[0],
        client_losses={0: 1.0, 1: 0.5, 2: 0.2},
    )
    # Only client 0 should have its loss_ema updated.
    assert buf.traces[0].loss_ema == pytest.approx(1.0)
    assert buf.traces[1].loss_ema == 0.0
    assert buf.traces[2].loss_ema == 0.0


def test_participation_last_resets_each_round() -> None:
    buf = FogTraceBuffer(client_ids=[0, 1], n_samples_per_client=[10, 10], rounds_total=5)
    buf.update_after_round(selected_client_ids=[0], client_losses={0: 0.5})
    assert buf.traces[0].participated_last == 1.0
    assert buf.traces[1].participated_last == 0.0
    buf.update_after_round(selected_client_ids=[1], client_losses={1: 0.4})
    assert buf.traces[0].participated_last == 0.0
    assert buf.traces[1].participated_last == 1.0


def test_round_progress_increases() -> None:
    buf = FogTraceBuffer(client_ids=[0], n_samples_per_client=[10], rounds_total=10)
    early = buf.build_observation(round_idx=0)
    late = buf.build_observation(round_idx=8)
    # progress is the 4th feature for each client
    assert late[3] > early[3]


def test_ema_alpha_invalid() -> None:
    with pytest.raises(ValueError):
        FogTraceBuffer(client_ids=[0], n_samples_per_client=[10], rounds_total=5, ema_alpha=0.0)
    with pytest.raises(ValueError):
        FogTraceBuffer(client_ids=[0], n_samples_per_client=[10], rounds_total=5, ema_alpha=1.5)


def test_length_mismatch_invalid() -> None:
    with pytest.raises(ValueError):
        FogTraceBuffer(client_ids=[0, 1], n_samples_per_client=[10], rounds_total=5)
