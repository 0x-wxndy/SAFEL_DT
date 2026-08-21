"""PR-9b integration: drops actually exclude clients from aggregation.

These tests exercise the end-to-end simulator with random + late drops
enabled and verify the JSONL trace records them. Drop_prob is forced
post-build (DeviceDTState is mutable) so we don't depend on the
stochastic profile draw.
"""

from __future__ import annotations

import json

import numpy as np
from torch.utils.data import Subset

import safel_dt.runtime.simulator as sim_mod
from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.runtime.simulator import SimulatorConfig, run_simulation


def _make_world(num_clients: int = 4):
    in_features, num_classes = 16, 3
    full = SyntheticTabularDataset(
        n_samples=num_clients * 64,
        in_features=in_features,
        num_classes=num_classes,
        seed=11,
        projection_seed=22,
    )
    parts = iid_partition(len(full), num_clients, np.random.default_rng(33))
    train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=128, in_features=in_features, num_classes=num_classes,
        seed=44, projection_seed=22,
    )
    return train_sets, test_set, in_features, num_classes


def _force_drop_prob(p: float):
    """Monkey-patch ``_build_device_states`` to set every client's drop_prob."""
    real = sim_mod._build_device_states

    def patched(**kw):
        devs = real(**kw)
        for d in devs.values():
            d.drop_prob = p
        return devs

    return real, patched


def _restore(real) -> None:
    sim_mod._build_device_states = real  # type: ignore[assignment]


def test_drop_prob_one_drops_everyone(tmp_path) -> None:
    train_sets, test_set, in_features, num_classes = _make_world()
    num_clients = len(train_sets)
    client_to_fog = assign_clients_to_fogs(num_clients, 2)

    trace = tmp_path / "trace.jsonl"
    cfg = SimulatorConfig(
        seed=0,
        rounds=2,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=8, num_classes=num_classes,
        ),
        client_train_sets=train_sets,  # type: ignore[arg-type]
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=8, lr=0.05),
        trace_path=trace,
        enable_random_drops=True,
    )
    real, patched = _force_drop_prob(1.0)
    sim_mod._build_device_states = patched  # type: ignore[assignment]
    try:
        run_simulation(cfg)
    finally:
        _restore(real)

    lines = trace.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        dr = row["dropped_random"]
        total = sum(len(v) for v in dr.values())
        assert total == num_clients, f"expected all {num_clients} dropped, got {dr}"


def test_drop_prob_zero_logs_empty_drops(tmp_path) -> None:
    train_sets, test_set, in_features, num_classes = _make_world()
    num_clients = len(train_sets)
    client_to_fog = assign_clients_to_fogs(num_clients, 2)

    trace = tmp_path / "trace.jsonl"
    cfg = SimulatorConfig(
        seed=0,
        rounds=2,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=8, num_classes=num_classes,
        ),
        client_train_sets=train_sets,  # type: ignore[arg-type]
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=8, lr=0.05),
        trace_path=trace,
        enable_random_drops=True,
        drop_late=False,
    )
    real, patched = _force_drop_prob(0.0)
    sim_mod._build_device_states = patched  # type: ignore[assignment]
    try:
        run_simulation(cfg)
    finally:
        _restore(real)

    lines = trace.read_text(encoding="utf-8").strip().split("\n")
    for line in lines:
        row = json.loads(line)
        dr_total = sum(len(v) for v in row.get("dropped_random", {}).values())
        dl_total = sum(len(v) for v in row.get("dropped_late", {}).values())
        assert dr_total == 0
        assert dl_total == 0


def test_drop_late_kicks_in(tmp_path) -> None:
    """With a tiny deadline, the late-drop branch should evict clients."""
    train_sets, test_set, in_features, num_classes = _make_world()
    num_clients = len(train_sets)
    client_to_fog = assign_clients_to_fogs(num_clients, 2)

    from safel_dt.types import FogDTState
    fog_states = {
        fid: FogDTState(fog_id=fid, device_ids=list(cids), delta=0.001)
        for fid, cids in client_to_fog.items()
    }

    trace = tmp_path / "trace.jsonl"
    cfg = SimulatorConfig(
        seed=0,
        rounds=2,
        model_factory=lambda: make_model(
            "mlp", in_features=in_features, hidden=8, num_classes=num_classes,
        ),
        client_train_sets=train_sets,  # type: ignore[arg-type]
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(epochs=1, batch_size=8, lr=0.05),
        trace_path=trace,
        fog_states=fog_states,
        enable_random_drops=False,
        drop_late=True,
    )
    real, patched = _force_drop_prob(0.0)
    sim_mod._build_device_states = patched  # type: ignore[assignment]
    try:
        run_simulation(cfg)
    finally:
        _restore(real)

    lines = trace.read_text(encoding="utf-8").strip().split("\n")
    n_late_total = 0
    for line in lines:
        row = json.loads(line)
        n_late_total += sum(len(v) for v in row.get("dropped_late", {}).values())
    assert n_late_total > 0
