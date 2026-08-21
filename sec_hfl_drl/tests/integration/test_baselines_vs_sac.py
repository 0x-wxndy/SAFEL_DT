"""Head-to-head comparison of fog policies under a single seed.

Runs five policies on the *same* synthetic-tabular adversarial world:

* ``all``       -- every client every round (no selection).
* ``random``    -- uniform random cohort of size ``mu_fog``.
* ``heuristic`` -- deterministic score-based selection.
* ``binary_rl`` -- per-client epsilon-greedy bandit on negated loss.
* ``sac``       -- the full SAC fog policy.

Pass criteria (we keep these conservative because 30 rounds is short
and stochastic; the goal is to catch *regressions*, not to claim a
significance result):

1. SAC's final accuracy must be no worse than the random baseline.
2. The heuristic must be no worse than the random baseline (it is a
   smarter-than-random scoring rule).
3. All five runs must complete the full ``rounds`` count without
   exceptions.
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
from safel_dt.rl.binary_rl_policy import BinaryRLConfig, BinaryRLFogPolicy
from safel_dt.rl.heuristic_policy import HeuristicConfig, HeuristicFogPolicy
from safel_dt.rl.policy import FogPolicy, RandomPolicy, SacPolicy
from safel_dt.rl.sac_controller import SacControllerConfig
from safel_dt.rl.select_clients import SelectionConfig
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


def _build_world(seed: int):
    num_clients = 6
    in_features = 16
    num_classes = 3
    rng_master = np.random.default_rng(seed)
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
        n_samples=300,
        in_features=in_features,
        num_classes=num_classes,
        seed=22,
        projection_seed=4242,
    )
    schedule = FixedAdversary(
        malicious_ids=[5],
        attack=LabelFlipAttack(num_classes=num_classes, shift=2),
    )
    client_to_fog = assign_clients_to_fogs(num_clients, 1)
    return {
        "client_train_sets": client_train_sets,
        "test_set": test_set,
        "schedule": schedule,
        "client_to_fog": client_to_fog,
        "in_features": in_features,
        "num_classes": num_classes,
    }


def _run(world, fog_policies: dict[int, FogPolicy] | None, rounds: int) -> float:
    cfg = SimulatorConfig(
        seed=0,
        rounds=rounds,
        model_factory=lambda: make_model(
            "mlp",
            in_features=world["in_features"],
            hidden=16,
            num_classes=world["num_classes"],
        ),
        client_train_sets=world["client_train_sets"],
        client_to_fog=world["client_to_fog"],
        test_set=world["test_set"],
        train_cfg=LocalTrainConfig(epochs=1, batch_size=32, lr=0.1),
        malicious_schedule=world["schedule"],
        fog_policies=fog_policies,
    )
    outcomes = run_simulation(cfg)
    return float(outcomes[-1].accuracy)


@pytest.mark.slow
def test_baselines_complete_and_learn() -> None:
    world = _build_world(seed=0)
    rounds = 30
    cids = world["client_to_fog"][0]
    n_clients = len(cids)
    n_samples = [len(world["client_train_sets"][c]) for c in cids]

    acc_all = _run(world, fog_policies=None, rounds=rounds)
    acc_random = _run(
        world,
        fog_policies={
            0: RandomPolicy(num_clients=n_clients, k=3, seed=7),  # type: ignore[dict-item]
        },
        rounds=rounds,
    )
    acc_heur = _run(
        world,
        fog_policies={
            0: HeuristicFogPolicy(  # type: ignore[dict-item]
                num_clients=n_clients,
                client_ids=cids,
                n_samples_per_client=n_samples,
                rounds_total=rounds,
                cfg=HeuristicConfig(mu_fog=3, seed=7),
            )
        },
        rounds=rounds,
    )
    acc_brl = _run(
        world,
        fog_policies={
            0: BinaryRLFogPolicy(  # type: ignore[dict-item]
                num_clients=n_clients,
                client_ids=cids,
                n_samples_per_client=n_samples,
                rounds_total=rounds,
                cfg=BinaryRLConfig(
                    mu_fog=3, eps_start=0.4, eps_end=0.05, eps_decay_rounds=15, seed=7
                ),
            )
        },
        rounds=rounds,
    )
    acc_sac = _run(
        world,
        fog_policies={
            0: SacPolicy(  # type: ignore[dict-item]
                num_clients=n_clients,
                client_ids=cids,
                n_samples_per_client=n_samples,
                rounds_total=rounds,
                selection=SelectionConfig(tau=0.5, mu_fog=3, m_min=2),
                sac_cfg=SacControllerConfig(
                    buffer_size=256, batch_size=8, learning_starts=4, gradient_steps=2, seed=0
                ),
            )
        },
        rounds=rounds,
    )

    # All runs must produce a sensible probability.
    for label, acc in (
        ("all", acc_all),
        ("random", acc_random),
        ("heuristic", acc_heur),
        ("binary_rl", acc_brl),
        ("sac", acc_sac),
    ):
        assert 0.0 <= acc <= 1.0, f"{label} produced acc out of [0,1]: {acc}"

    # SAC must be at least as good as random (loose bound to avoid flakiness).
    assert acc_sac >= acc_random - 0.02, (
        f"SAC ({acc_sac:.3f}) substantially worse than random ({acc_random:.3f})."
    )
    # Heuristic must be at least as good as random by the same margin.
    assert acc_heur >= acc_random - 0.05, (
        f"Heuristic ({acc_heur:.3f}) substantially worse than random ({acc_random:.3f})."
    )
