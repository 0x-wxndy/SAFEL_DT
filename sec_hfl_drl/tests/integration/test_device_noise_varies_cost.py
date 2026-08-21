"""PR-9a integration: with device_noise_sigma > 0, per-round costs vary."""

from __future__ import annotations

import json

import numpy as np
from torch.utils.data import Subset

from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


def _world(n_clients: int = 4):
    in_features, num_classes = 16, 3
    full = SyntheticTabularDataset(
        n_samples=n_clients * 64,
        in_features=in_features,
        num_classes=num_classes,
        seed=11,
        projection_seed=22,
    )
    parts = iid_partition(len(full), n_clients, np.random.default_rng(33))
    train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=128,
        in_features=in_features,
        num_classes=num_classes,
        seed=44,
        projection_seed=22,
    )
    return train_sets, test_set, in_features, num_classes


def _run(sigma: float, tmp_path) -> list[dict]:
    train_sets, test_set, in_features, num_classes = _world()
    n_clients = len(train_sets)
    trace = tmp_path / f"trace_sigma_{sigma}.jsonl"

    cfg = SimulatorConfig(
        seed=0,
        rounds=4,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=8, num_classes=num_classes,
        ),
        client_train_sets=train_sets,  # type: ignore[arg-type]
        client_to_fog=assign_clients_to_fogs(n_clients, 2),
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=8, lr=0.05),
        trace_path=trace,
        device_noise_sigma=sigma,
    )
    run_simulation(cfg)
    return [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines() if ln]


def _per_round_comm(rows: list[dict]) -> list[float]:
    return [sum(v["comm"] for v in r["costs"].values()) for r in rows]


def test_no_noise_keeps_costs_constant(tmp_path) -> None:
    """With sigma=0 and all clients selected, per-round comm cost must be identical."""
    rows = _run(0.0, tmp_path)
    comms = _per_round_comm(rows)
    assert len(comms) == 4
    # all clients each round, identical -> same comm cost every round
    for c in comms[1:]:
        assert abs(c - comms[0]) < 1e-9


def test_noise_varies_per_round_comm(tmp_path) -> None:
    """With sigma=0.3 comm cost must vary round to round."""
    rows = _run(0.3, tmp_path)
    comms = _per_round_comm(rows)
    assert len(comms) == 4
    # at least two distinct values
    assert len({round(c, 6) for c in comms}) >= 2
    # spread is non-trivial relative to mean
    mean = float(np.mean(comms))
    spread = float(np.std(comms))
    assert spread > 0.01 * mean
