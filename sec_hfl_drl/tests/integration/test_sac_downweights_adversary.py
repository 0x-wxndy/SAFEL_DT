"""SAC learns to down-weight an adversary in the participation cohort.

Scenario
--------
* Single fog of 4 clients on a learnable synthetic tabular task.
* Client #3 is a fixed adversary running ``LabelFlipAttack`` every round.
* SAC must pick ``m_min=2..mu_fog=3`` clients per round.
* Reward = utility gain (Δ test accuracy) minus a tiny selection penalty.

Pass criterion
--------------
The adversary's *participation rate* over the last third of training must
be **strictly lower** than its rate over the first third. We don't
require zero -- SAC is exploration-heavy and 40 rounds is short -- but we
require a clear downward trend, which is the paper's central claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Subset

from safel_dt.attacks import FixedAdversary, LabelFlipAttack
from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.rl.policy import SacPolicy
from safel_dt.rl.sac_controller import SacControllerConfig
from safel_dt.rl.select_clients import SelectionConfig
from safel_dt.runtime.simulator import RewardConfig, SimulatorConfig, run_simulation


@pytest.mark.slow
def test_sac_downweights_adversary_over_rounds() -> None:
    rng_master = np.random.default_rng(0)
    num_clients = 4
    in_features = 16
    num_classes = 4
    rounds = 40
    proj_seed = 7777

    full = SyntheticTabularDataset(
        n_samples=num_clients * 200,
        in_features=in_features,
        num_classes=num_classes,
        seed=11,
        projection_seed=proj_seed,
    )
    parts = iid_partition(len(full), num_clients, rng_master)
    client_train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=300,
        in_features=in_features,
        num_classes=num_classes,
        seed=22,
        projection_seed=proj_seed,
    )

    attacker_id = 3
    schedule = FixedAdversary(
        malicious_ids=[attacker_id],
        attack=LabelFlipAttack(num_classes=num_classes, shift=2),
    )
    client_to_fog = assign_clients_to_fogs(num_clients, 1)

    sac_policy = SacPolicy(
        num_clients=num_clients,
        client_ids=client_to_fog[0],
        n_samples_per_client=[len(s) for s in client_train_sets],
        rounds_total=rounds,
        selection=SelectionConfig(tau=0.5, mu_fog=3, m_min=2),
        sac_cfg=SacControllerConfig(
            buffer_size=512,
            batch_size=8,
            learning_starts=4,
            gradient_steps=2,
            seed=0,
        ),
    )

    participation_log: list[set[int]] = []

    class TrackingPolicy:
        """Decorator over SacPolicy that records which globals participated."""

        num_clients = sac_policy.num_clients

        def select(self, *, round_idx: int) -> list[int]:
            chosen_local = sac_policy.select(round_idx=round_idx)
            participation_log.append(
                {sac_policy.client_ids[i] for i in chosen_local}
            )
            return chosen_local

        def observe_feedback(self, feedback) -> None:  # type: ignore[no-untyped-def]
            sac_policy.observe_feedback(feedback)

    cfg = SimulatorConfig(
        seed=0,
        rounds=rounds,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=16, num_classes=num_classes
        ),
        client_train_sets=client_train_sets,  # type: ignore[arg-type]
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=32, lr=0.1),
        malicious_schedule=schedule,
        fog_policies={0: TrackingPolicy()},  # type: ignore[dict-item]
        reward_cfg=RewardConfig(selection_penalty=0.02),
    )
    outcomes = run_simulation(cfg)
    assert len(outcomes) == rounds
    assert len(participation_log) == rounds

    # Split rounds into thirds; compare early vs late adversary participation.
    third = rounds // 3
    early = participation_log[:third]
    late = participation_log[-third:]
    rate_early = sum(attacker_id in s for s in early) / max(len(early), 1)
    rate_late = sum(attacker_id in s for s in late) / max(len(late), 1)
    # The agent must show *some* downward trend on the adversary.
    assert rate_late < rate_early, (
        f"SAC failed to learn: adversary participation went from "
        f"{rate_early:.2f} (first {third} rounds) to {rate_late:.2f} "
        f"(last {third} rounds)."
    )
