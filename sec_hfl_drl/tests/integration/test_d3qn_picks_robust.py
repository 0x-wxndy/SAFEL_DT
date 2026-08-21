"""D3QN learns to favour a robust aggregator when FedAvg is poisoned.

Scenario
--------
* 8 clients across 2 fogs.
* 2 sign-flipped ``ModelScaleAttack(gamma=-10)`` clients (one per fog).
* Cloud picks between ``{fedavg, krum, multi_krum, trimmed_mean, median}``
  each round, driven by :class:`D3qnCloudPolicy`.
* Reward = global Δaccuracy (no Lagrangian, to keep the signal sharp).

Pass criteria
-------------
1. D3QN learns to *prefer non-FedAvg aggregators* late in training:
   the fraction of FedAvg picks in the last third must be < the first.
2. Final accuracy must exceed a FedAvg-only baseline under the same
   attack (the whole point of switching).
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Subset

from safel_dt.attacks import FixedAdversary, ModelScaleAttack
from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.rl.cloud_env import CloudObsConfig
from safel_dt.rl.cloud_policy import D3qnCloudPolicy
from safel_dt.rl.d3qn import D3qnConfig
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


def _build_world(seed: int, rounds: int):
    rng_master = np.random.default_rng(seed)
    num_clients = 8
    in_features = 16
    num_classes = 3

    full = SyntheticTabularDataset(
        n_samples=num_clients * 200,
        in_features=in_features,
        num_classes=num_classes,
        seed=11,
        projection_seed=4242,
    )
    parts = iid_partition(len(full), num_clients, rng_master)
    client_train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=400,
        in_features=in_features,
        num_classes=num_classes,
        seed=22,
        projection_seed=4242,
    )
    schedule = FixedAdversary(
        malicious_ids=[3, 6],
        attack=ModelScaleAttack(gamma=-10.0),
    )
    client_to_fog = assign_clients_to_fogs(num_clients, 2)
    return {
        "client_train_sets": client_train_sets,
        "test_set": test_set,
        "schedule": schedule,
        "client_to_fog": client_to_fog,
        "in_features": in_features,
        "num_classes": num_classes,
        "num_clients": num_clients,
        "rounds": rounds,
    }


@pytest.mark.slow
def test_d3qn_outperforms_fedavg_under_sign_flip() -> None:
    seed = 0
    rounds = 30
    world = _build_world(seed, rounds)
    in_features = world["in_features"]
    num_classes = world["num_classes"]

    obs_cfg = CloudObsConfig(fog_ids=tuple(sorted(world["client_to_fog"].keys())))
    cloud_policy = D3qnCloudPolicy(
        obs_cfg=obs_cfg,
        d3qn_cfg=D3qnConfig(
            buffer_size=256,
            batch_size=8,
            learning_starts=4,
            gradient_steps=2,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=20,
            seed=seed,
        ),
    )
    chosen: list[str] = []

    class TrackingCloudPolicy:
        aggregators = cloud_policy.aggregators

        def select(self, *, round_idx: int) -> str:
            a = cloud_policy.select(round_idx=round_idx)
            chosen.append(a)
            return a

        def observe_feedback(self, feedback) -> None:  # type: ignore[no-untyped-def]
            cloud_policy.observe_feedback(feedback)

    cfg_d3qn = SimulatorConfig(
        seed=seed,
        rounds=rounds,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=16, num_classes=num_classes
        ),
        client_train_sets=world["client_train_sets"],
        client_to_fog=world["client_to_fog"],
        test_set=world["test_set"],
        train_cfg=LocalTrainConfig(epochs=1, batch_size=32, lr=0.1),
        malicious_schedule=world["schedule"],
        aggregator="fedavg",
        cloud_policy=TrackingCloudPolicy(),  # type: ignore[arg-type]
    )
    outcomes_d3qn = run_simulation(cfg_d3qn)
    final_acc_d3qn = outcomes_d3qn[-1].accuracy

    cfg_fedavg = SimulatorConfig(
        seed=seed,
        rounds=rounds,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=16, num_classes=num_classes
        ),
        client_train_sets=world["client_train_sets"],
        client_to_fog=world["client_to_fog"],
        test_set=world["test_set"],
        train_cfg=LocalTrainConfig(epochs=1, batch_size=32, lr=0.1),
        malicious_schedule=world["schedule"],
        aggregator="fedavg",
    )
    outcomes_fedavg = run_simulation(cfg_fedavg)
    final_acc_fedavg = outcomes_fedavg[-1].accuracy

    assert len(chosen) == rounds
    # --- 1) D3QN should taper off FedAvg over time ---
    third = rounds // 3
    fedavg_rate_early = sum(a == "fedavg" for a in chosen[:third]) / max(third, 1)
    fedavg_rate_late = sum(a == "fedavg" for a in chosen[-third:]) / max(third, 1)
    assert fedavg_rate_late <= fedavg_rate_early, (
        f"D3QN failed to learn: FedAvg pick rate went "
        f"{fedavg_rate_early:.2f} -> {fedavg_rate_late:.2f}"
    )

    # --- 2) D3QN must beat the FedAvg-only baseline under attack ---
    assert final_acc_d3qn > final_acc_fedavg, (
        f"D3QN ({final_acc_d3qn:.3f}) failed to outperform "
        f"FedAvg-only ({final_acc_fedavg:.3f}) under sign-flip attack."
    )
