"""Pure-function tests for the shared heuristic score (A5).

The deterministic :class:`HeuristicFogPolicy` and the SAC ``heuristic_hint``
feature both call :func:`safel_dt.rl.heuristic_score.heuristic_scores`. We
test the score in isolation here, plus a parity test that confirms the
policy delegates correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from safel_dt.rl.heuristic_policy import HeuristicConfig, HeuristicFogPolicy
from safel_dt.rl.heuristic_score import HeuristicScoreCfg, heuristic_scores


def test_heuristic_score_more_samples_ranks_higher() -> None:
    loss = np.array([1.0, 1.0, 1.0])
    samples = np.array([100.0, 50.0, 25.0])
    out = heuristic_scores(loss_ema=loss, samples=samples)
    assert out[0] > out[1] > out[2]


def test_heuristic_score_low_loss_ranks_higher() -> None:
    loss = np.array([0.1, 1.0, 3.0])
    samples = np.array([100.0, 100.0, 100.0])
    out = heuristic_scores(loss_ema=loss, samples=samples)
    assert out[0] > out[1] > out[2]


def test_heuristic_score_high_cos_dist_ranks_lower() -> None:
    """When adversary features are passed, large cos_dist suppresses score."""
    loss = np.array([1.0, 1.0, 1.0])
    samples = np.array([100.0, 100.0, 100.0])
    cos_dist = np.array([0.0, 0.5, 1.5])
    norm_ratio = np.array([1.0, 1.0, 1.0])
    out = heuristic_scores(
        loss_ema=loss, samples=samples,
        cos_dist_ema=cos_dist, delta_norm_ratio_ema=norm_ratio,
    )
    assert out[0] > out[1] > out[2]


def test_heuristic_score_norm_outlier_penalised_both_ways() -> None:
    """``delta_norm_ratio_ema`` ~1 is benign; both <<1 and >>1 are penalised."""
    loss = np.array([1.0, 1.0, 1.0])
    samples = np.array([100.0, 100.0, 100.0])
    cos_dist = np.array([0.0, 0.0, 0.0])
    norm_ratio = np.array([1.0, 5.0, 0.2])
    out = heuristic_scores(
        loss_ema=loss, samples=samples,
        cos_dist_ema=cos_dist, delta_norm_ratio_ema=norm_ratio,
    )
    assert out[0] > out[1]  # benign > inflated
    assert out[0] > out[2]  # benign > shrunken


def test_heuristic_score_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        heuristic_scores(
            loss_ema=np.array([1.0, 1.0]),
            samples=np.array([100.0]),
        )
    with pytest.raises(ValueError):
        heuristic_scores(
            loss_ema=np.array([1.0, 1.0]),
            samples=np.array([100.0, 100.0]),
            cos_dist_ema=np.array([0.5]),
            delta_norm_ratio_ema=np.array([1.0, 1.0]),
        )


def test_heuristic_policy_delegates_to_shared_score() -> None:
    """``HeuristicFogPolicy._scores()`` must equal :func:`heuristic_scores`."""
    cfg = HeuristicConfig(mu_fog=2, m_min=1, w_samples=1.0, w_loss=1.0,
                          w_divergence=0.5, w_cos_dist=2.0, w_norm_outlier=1.0)
    pol = HeuristicFogPolicy(
        num_clients=3,
        client_ids=[0, 1, 2],
        n_samples_per_client=[100, 50, 200],
        rounds_total=10,
        cfg=cfg,
        adversary_features=True,
    )
    # Seed the trace with some non-trivial values.
    pol._trace.traces[0].loss_ema = 0.5
    pol._trace.traces[0].cos_dist_to_mean_ema = 0.1
    pol._trace.traces[0].delta_norm_ratio_ema = 1.0
    pol._trace.traces[1].loss_ema = 1.5
    pol._trace.traces[1].cos_dist_to_mean_ema = 0.8
    pol._trace.traces[1].delta_norm_ratio_ema = 3.0
    pol._trace.traces[2].loss_ema = 0.7
    pol._trace.traces[2].cos_dist_to_mean_ema = 0.2
    pol._trace.traces[2].delta_norm_ratio_ema = 1.1

    direct = heuristic_scores(
        loss_ema=np.array([0.5, 1.5, 0.7]),
        samples=np.array([100.0, 50.0, 200.0]),
        cos_dist_ema=np.array([0.1, 0.8, 0.2]),
        delta_norm_ratio_ema=np.array([1.0, 3.0, 1.1]),
        cfg=HeuristicScoreCfg(
            w_samples=cfg.w_samples, w_loss=cfg.w_loss,
            w_divergence=cfg.w_divergence, w_cos_dist=cfg.w_cos_dist,
            w_norm_outlier=cfg.w_norm_outlier, max_loss=cfg.max_loss,
        ),
    )
    np.testing.assert_allclose(pol._scores(), direct, rtol=1e-12)
