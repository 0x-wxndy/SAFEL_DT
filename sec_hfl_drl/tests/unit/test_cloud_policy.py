"""Unit tests for `safel_dt.rl.cloud_policy`."""

from __future__ import annotations

from safel_dt.costs.reward import Multipliers
from safel_dt.fl.cloud_server import CloudRoundOutcome
from safel_dt.rl.cloud_env import CloudObsConfig
from safel_dt.rl.cloud_policy import (
    CloudFeedback,
    D3qnCloudPolicy,
    StaticCloudPolicy,
)
from safel_dt.rl.d3qn import D3qnConfig


def _outcome(aggregator: str = "fedavg") -> CloudRoundOutcome:
    return CloudRoundOutcome(
        round_idx=0,
        accuracy=0.5,
        loss=0.5,
        n_clients_accepted=2,
        n_clients_rejected=0,
        n_fogs_accepted=2,
        aggregator=aggregator,
        per_client_losses={0: 0.5, 1: 0.5},
        per_fog_participants={0: [0], 1: [1]},
    )


def test_static_policy_returns_fixed_name() -> None:
    p = StaticCloudPolicy(name="krum")
    assert p.select(round_idx=0) == "krum"
    assert p.select(round_idx=5) == "krum"
    assert "krum" in p.aggregators


def test_static_policy_ignores_feedback() -> None:
    p = StaticCloudPolicy(name="median")
    fb = CloudFeedback(
        round_idx=0,
        chosen_aggregator="median",
        outcome=_outcome(),
        multipliers=Multipliers(),
        per_fog_reward={0: 0.0, 1: 0.0},
        reward=0.0,
    )
    p.observe_feedback(fb)


def test_d3qn_policy_select_returns_valid_aggregator_name() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1))
    policy = D3qnCloudPolicy(
        obs_cfg=cfg,
        d3qn_cfg=D3qnConfig(learning_starts=2, batch_size=2, buffer_size=16, seed=0),
    )
    a = policy.select(round_idx=0)
    assert a in cfg.aggregators


def test_d3qn_policy_learns_from_feedback() -> None:
    cfg = CloudObsConfig(fog_ids=(0, 1))
    policy = D3qnCloudPolicy(
        obs_cfg=cfg,
        d3qn_cfg=D3qnConfig(learning_starts=2, batch_size=2, buffer_size=32, seed=0),
    )
    for r in range(6):
        agg = policy.select(round_idx=r)
        fb = CloudFeedback(
            round_idx=r,
            chosen_aggregator=agg,
            outcome=_outcome(aggregator=agg),
            multipliers=Multipliers(),
            per_fog_reward={0: 0.1, 1: 0.05},
            reward=0.075,
        )
        policy.observe_feedback(fb)
    assert policy.controller._buf._size == 6  # one transition per round
