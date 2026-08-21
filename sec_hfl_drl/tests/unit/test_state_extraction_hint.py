"""Tests for the A5 heuristic-hint dimension in :class:`FogTraceBuffer`."""

from __future__ import annotations

import numpy as np

from safel_dt.fl.adversary_features import ClientAdversaryFeatures
from safel_dt.rl.state_extraction import (
    FogTraceBuffer,
    features_per_client,
    obs_dim,
)


def test_features_per_client_includes_hint_when_enabled() -> None:
    assert features_per_client() == 4
    assert features_per_client(adversary_features=True) == 7
    assert features_per_client(heuristic_hint=True) == 5
    assert features_per_client(adversary_features=True, heuristic_hint=True) == 8


def test_obs_dim_scales_with_clients() -> None:
    assert obs_dim(3, adversary_features=True, heuristic_hint=True) == 24
    assert obs_dim(10, adversary_features=False, heuristic_hint=True) == 50


def test_trace_buffer_emits_hint_value_per_client() -> None:
    buf = FogTraceBuffer(
        client_ids=[0, 1, 2],
        n_samples_per_client=[100, 100, 100],
        rounds_total=10,
        adversary_features=True,
        heuristic_hint=True,
    )
    obs = buf.build_observation(round_idx=0)
    # 3 clients * (4 base + 3 adv + 1 hint) = 24
    assert obs.shape == (24,)


def test_hint_value_lowers_when_adversary_features_get_worse() -> None:
    """An adversarial-looking client (high cos_dist, |ratio-1| >> 0) should
    get a *lower* hint score than a benign one."""
    buf = FogTraceBuffer(
        client_ids=[0, 1],
        n_samples_per_client=[100, 100],
        rounds_total=10,
        adversary_features=True,
        heuristic_hint=True,
    )
    # round 0: both clients participate, one with clean features, the
    # other with adversarial-looking features.
    feats = {
        0: ClientAdversaryFeatures(
            delta_norm_ratio=1.0, cos_dist_to_mean=0.05, loss_zscore=0.0,
        ),
        1: ClientAdversaryFeatures(
            delta_norm_ratio=4.0, cos_dist_to_mean=1.2, loss_zscore=0.0,
        ),
    }
    buf.update_after_round(
        selected_client_ids=[0, 1],
        client_losses={0: 0.3, 1: 0.3},
        client_features=feats,
    )
    obs = buf.build_observation(round_idx=1)
    # 2 clients * 8 features = 16. Hint is the 8th feature in each block.
    hint_c0 = obs[7]
    hint_c1 = obs[15]
    assert hint_c0 > hint_c1, (
        f"benign client hint {hint_c0:.3f} should exceed adversarial {hint_c1:.3f}"
    )


def test_hint_default_off_is_legacy_shape() -> None:
    buf = FogTraceBuffer(
        client_ids=[0, 1, 2],
        n_samples_per_client=[100, 100, 100],
        rounds_total=10,
        adversary_features=False,
    )
    obs = buf.build_observation(round_idx=0)
    assert obs.shape == (12,)  # 3 clients * 4 base features


def test_hint_without_adversary_features_falls_back_to_base_score() -> None:
    """The hint should still produce *some* informative score even when
    adversary features are off; verify it is finite and matches the
    base-only branch."""
    buf = FogTraceBuffer(
        client_ids=[0, 1],
        n_samples_per_client=[200, 50],
        rounds_total=10,
        adversary_features=False,
        heuristic_hint=True,
    )
    obs = buf.build_observation(round_idx=0)
    # 2 clients * (4 base + 1 hint) = 10. Hint at indices 4 and 9.
    assert np.isfinite(obs).all()
    assert obs[4] > obs[9]  # more samples = higher score
