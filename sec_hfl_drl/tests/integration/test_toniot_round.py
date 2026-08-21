"""Integration test: 1 FL round on real TON_IoT (network-flow slice).

Skips cleanly if the cache is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from safel_dt.data.datasets import load_toniot
from safel_dt.data.partition import assign_clients_to_fogs
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


@pytest.mark.slow
@pytest.mark.requires_toniot
def test_toniot_one_round_smoke(toniot_csv: Path, data_dir: Path) -> None:
    client_train_sets, test_set, meta = load_toniot(
        data_dir=data_dir,
        mode="binary",
        num_clients=9,
        max_samples=50_000,  # ~ a quarter of the dataset; tiny for a smoke test
        test_fraction=0.2,
        seed=0,
    )
    num_clients = len(client_train_sets)
    assert num_clients == 9
    in_features = int(meta["in_features"])
    num_classes = int(meta["num_classes"])
    assert num_classes == 2

    cfg = SimulatorConfig(
        seed=0,
        rounds=1,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=32, num_classes=num_classes
        ),
        client_train_sets=client_train_sets,
        client_to_fog=assign_clients_to_fogs(num_clients, 3, strategy="contiguous"),
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=128, lr=0.05),
    )
    outcomes = run_simulation(cfg)
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.n_clients_accepted == 9
    assert o.n_fogs_accepted == 3
    assert 0.0 <= o.accuracy <= 1.0
    assert o.loss > 0.0
