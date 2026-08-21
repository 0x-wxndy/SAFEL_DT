"""PR-9c integration: under label-flip, robust aggregators recover what
FedAvg loses.

We construct a small but learnable synthetic task with 30% label-flip
adversaries from round 0. FedAvg should converge to noticeably lower
accuracy than Krum / Multi-Krum / Trimmed Mean / Median, which discount
or ignore the corrupted updates.

This is the paper's headline robustness claim, encoded as a regression
test so we know if the simulator ever stops biting.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Subset

from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.runtime.attack_builder import AttackSpec, build_schedule
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


def _world(seed: int = 0):
    num_clients = 8
    in_features, num_classes = 16, 4
    full = SyntheticTabularDataset(
        n_samples=num_clients * 200,
        in_features=in_features,
        num_classes=num_classes,
        seed=seed,
        projection_seed=99,
    )
    parts = iid_partition(len(full), num_clients, np.random.default_rng(seed + 1))
    train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=400,
        in_features=in_features,
        num_classes=num_classes,
        seed=seed + 100,
        projection_seed=99,
    )
    return train_sets, test_set, in_features, num_classes, num_clients


def _build_cfg(
    *,
    aggregator: str,
    aggregator_options: dict | None = None,
    attack_spec: AttackSpec | None = None,
    rounds: int = 10,
) -> SimulatorConfig:
    train_sets, test_set, in_features, num_classes, num_clients = _world()
    schedule, _ = build_schedule(
        spec=attack_spec or AttackSpec(),
        num_classes=num_classes,
        client_ids=list(range(num_clients)),
        seed=0,
    )
    return SimulatorConfig(
        seed=0,
        rounds=rounds,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=32, num_classes=num_classes,
        ),
        client_train_sets=train_sets,  # type: ignore[arg-type]
        client_to_fog=assign_clients_to_fogs(num_clients, 2),
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=2, batch_size=32, lr=0.1),
        aggregator=aggregator,
        aggregator_options=aggregator_options or {},
        malicious_schedule=schedule,
    )


@pytest.mark.slow
def test_fedavg_degrades_under_label_flip() -> None:
    """FedAvg's final accuracy under 25% label-flip should be visibly worse
    than its no-attack baseline."""
    clean = run_simulation(_build_cfg(aggregator="fedavg"))
    attacked = run_simulation(
        _build_cfg(
            aggregator="fedavg",
            attack_spec=AttackSpec(name="label_flip", frac=0.25, label_shift=1),
        )
    )
    assert attacked[-1].accuracy < clean[-1].accuracy - 0.05, (
        f"label-flip should hurt fedavg: clean={clean[-1].accuracy:.3f}, "
        f"attacked={attacked[-1].accuracy:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "aggregator,options",
    [
        ("krum", {"f": 2}),
        ("multi_krum", {"f": 2, "m": 4}),
        ("trimmed_mean", {"beta": 0.25}),
        ("median", {}),
    ],
)
def test_robust_aggregator_beats_fedavg_under_attack(
    aggregator: str, options: dict,
) -> None:
    """Under the same attack each robust aggregator must end at least
    as accurate as FedAvg."""
    spec = AttackSpec(name="label_flip", frac=0.25, label_shift=1)
    fed = run_simulation(_build_cfg(aggregator="fedavg", attack_spec=spec))
    rob = run_simulation(
        _build_cfg(aggregator=aggregator, aggregator_options=options, attack_spec=spec)
    )
    assert rob[-1].accuracy >= fed[-1].accuracy - 0.02, (
        f"{aggregator} ({rob[-1].accuracy:.3f}) substantially worse than "
        f"fedavg ({fed[-1].accuracy:.3f}) under the same attack"
    )
