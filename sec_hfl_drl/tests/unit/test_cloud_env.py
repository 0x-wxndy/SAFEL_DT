"""Unit tests for `safel_dt.rl.cloud_env`."""

from __future__ import annotations

import numpy as np

from safel_dt.costs.reward import Multipliers
from safel_dt.fl.cloud_server import CloudRoundOutcome
from safel_dt.rl.cloud_env import CloudObsConfig, CloudObservation


def _outcome(
    per_fog_participants: dict[int, list[int]],
    per_client_losses: dict[int, float],
    n_accepted: int = 3,
    n_rejected: int = 0,
    aggregator: str = "fedavg",
) -> CloudRoundOutcome:
    return CloudRoundOutcome(
        round_idx=0,
        accuracy=0.8,
        loss=0.5,
        n_clients_accepted=n_accepted,
        n_clients_rejected=n_rejected,
        n_fogs_accepted=len(per_fog_participants),
        aggregator=aggregator,
        per_client_losses=per_client_losses,
        per_fog_participants=per_fog_participants,
    )


def test_obs_dim_matches_layout() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1, 2))
    assert cfg.obs_dim == 4 * 3 + 3 + 5
    assert cfg.n_actions == 5


def test_reset_returns_zero_vector_of_correct_shape() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1))
    obs_builder = CloudObservation(cfg=cfg)
    obs = obs_builder.reset()
    assert obs.shape == (cfg.obs_dim,)
    assert obs.dtype == np.float32
    assert np.all(obs == 0.0)


def test_action_recorded_into_one_hot_block() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1))
    builder = CloudObservation(cfg=cfg)
    builder.reset()
    builder.record_action(2)
    vec = builder.build(outcome=None, multipliers=Multipliers(), per_fog_reward=None)
    onehot_start = 4 * cfg.n_fogs + 3
    assert vec[onehot_start + 2] == 1.0
    assert vec[onehot_start + 0] == 0.0


def test_multipliers_reflected_in_obs() -> None:
    cfg = CloudObsConfig(fog_ids=(0,))
    builder = CloudObservation(cfg=cfg)
    builder.reset()
    vec = builder.build(
        outcome=None,
        multipliers=Multipliers(lat=1.5, cap=2.5, priv=0.25),
        per_fog_reward=None,
    )
    base = 4 * cfg.n_fogs
    assert vec[base + 0] == 1.5
    assert vec[base + 1] == 2.5
    assert vec[base + 2] == 0.25


def test_fog_features_from_outcome() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1), max_loss=4.0)
    builder = CloudObservation(cfg=cfg)
    builder.reset()
    outcome = _outcome(
        per_fog_participants={0: [10, 11], 1: [20]},
        per_client_losses={10: 2.0, 11: 2.0, 20: 1.0},
        n_accepted=3,
        n_rejected=1,
    )
    vec = builder.build(
        outcome=outcome,
        multipliers=Multipliers(),
        per_fog_reward={0: 0.5, 1: -0.25},
    )
    assert vec[0] == 2.0 / 4.0  # mean loss fog0 normalized
    assert vec[1] == 1.0 / 4.0  # rejection_rate = 1 / (3+1)
    assert vec[4] == 1.0 / 4.0  # mean loss fog1 normalized
    assert abs(vec[7] - (-0.25)) < 1e-6  # last_reward for fog1
