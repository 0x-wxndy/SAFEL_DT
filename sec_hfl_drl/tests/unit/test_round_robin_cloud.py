"""Unit tests for `safel_dt.rl.cloud_policy.RoundRobinCloudPolicy`."""

from __future__ import annotations

import pytest

from safel_dt.costs.reward import Multipliers
from safel_dt.fl.cloud_server import CloudRoundOutcome
from safel_dt.rl.cloud_policy import CloudFeedback, RoundRobinCloudPolicy


def test_cycles_through_menu_deterministically() -> None:
    p = RoundRobinCloudPolicy(aggregators=("fedavg", "krum", "median"))
    seq = [p.select(round_idx=r) for r in range(7)]
    assert seq == ["fedavg", "krum", "median", "fedavg", "krum", "median", "fedavg"]


def test_empty_aggregator_menu_rejected() -> None:
    with pytest.raises(ValueError):
        RoundRobinCloudPolicy(aggregators=())


def test_observe_feedback_is_a_noop() -> None:
    p = RoundRobinCloudPolicy()
    p.observe_feedback(
        CloudFeedback(
            round_idx=0,
            chosen_aggregator="fedavg",
            outcome=CloudRoundOutcome(
                round_idx=0,
                accuracy=0.5,
                loss=0.5,
                n_clients_accepted=1,
                n_clients_rejected=0,
                n_fogs_accepted=1,
                aggregator="fedavg",
            ),
            multipliers=Multipliers(),
            per_fog_reward={0: 0.0},
            reward=0.0,
        )
    )
