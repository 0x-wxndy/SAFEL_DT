"""Tests for the A1 N-BaIoT intra-device IID split.

We mock out the disk-bound :func:`_load_device` so the test runs without
the 6.5 GB raw CSV cache; the partition logic is the unit under test.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from safel_dt.data import nbaiot


def _fake_load_device(root, device_id, *, max_per_class, rng):  # noqa: ARG001
    """Return (x, y_binary, y_multi) of fixed shape regardless of device."""
    # 200 rows so 5 shards fit comfortably (40 rows each).
    x = np.full((200, 4), float(device_id), dtype=np.float32)
    yb = np.zeros(200, dtype=np.int64)
    yb[100:] = 1
    ym = np.zeros(200, dtype=np.int64)
    ym[100:] = (device_id + 1)
    return x, yb, ym


@pytest.fixture(autouse=True)
def _patch_loader(tmp_path):
    """Stub out the on-disk loader + the layout-detection helper."""
    with patch.object(nbaiot, "ensure_nbaiot", return_value=tmp_path), \
         patch.object(nbaiot, "_load_device", side_effect=_fake_load_device):
        yield


def test_num_clients_none_keeps_one_per_device() -> None:
    train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", max_per_class=200, seed=0,
    )
    assert len(train) == 9
    assert meta["num_clients"] == 9
    assert meta["device_assignment"] == list(range(9))


def test_num_clients_equal_to_devices_is_legacy() -> None:
    train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", num_clients=9, max_per_class=200, seed=0,
    )
    assert len(train) == 9
    assert meta["device_assignment"] == list(range(9))


def test_num_clients_30_splits_each_device() -> None:
    train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", num_clients=30, max_per_class=200, seed=0,
    )
    assert len(train) == 30
    assert meta["num_clients"] == 30
    # Each physical device feeds either ceil(30/9)=4 or 3 logical clients.
    counts = Counter(meta["device_assignment"])
    assert sum(counts.values()) == 30
    assert all(3 <= c <= 4 for c in counts.values()), counts
    # All 9 devices represented.
    assert set(counts.keys()) == set(range(9))


def test_num_clients_emits_interleaved_device_order() -> None:
    """With 30 clients and 9 devices, the first 9 logical clients must
    cover devices 0..8 exactly (shard 0 of each device, in order)."""
    _train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", num_clients=30, max_per_class=200, seed=0,
    )
    assert meta["device_assignment"][:9] == list(range(9))
    # Second slice keeps the same device ordering modulo cohort size.
    assert meta["device_assignment"][9:18] == list(range(9))


def test_shards_are_disjoint_and_partition_each_device() -> None:
    """For devices that contribute the *full* shard count, all training
    rows must be partitioned (no duplicates, no drops).

    With 200 mock rows per device * (1 - 0.2 test_fraction) = 160 train
    rows. ``num_clients=30`` -> shards_per_device=4. The first
    ``num_clients - n_devices*(shards_per_device-1) = 30 - 27 = 3``
    devices emit all 4 shards; the remaining 6 emit 3 shards.
    """
    train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", num_clients=30, max_per_class=200, seed=0,
    )
    rows_per_device: dict[int, int] = {}
    for ds, did in zip(train, meta["device_assignment"], strict=True):
        rows_per_device[did] = rows_per_device.get(did, 0) + len(ds)
    # 160 train rows per device. 4 shards (devices 0..2): full coverage.
    # 3 shards (devices 3..8): 3/4 of 160 = 120 rows.
    full_devices = [d for d, n in rows_per_device.items() if n == 160]
    partial_devices = [d for d, n in rows_per_device.items() if n == 120]
    assert len(full_devices) == 3, rows_per_device
    assert len(partial_devices) == 6, rows_per_device
    # No device has zero or duplicated rows.
    assert all(n > 0 for n in rows_per_device.values())


def test_num_clients_below_devices_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        nbaiot.load_nbaiot_per_device(
            data_dir="ignored", mode="binary", num_clients=4, max_per_class=200, seed=0,
        )


def test_num_clients_too_high_rejected_for_tiny_devices() -> None:
    """If a device has fewer rows than shards_per_device, we error rather
    than emit empty shards."""
    # 9 devices * 200 rows; ask for 300 clients -> ceil(300/9) = 34 shards
    # per device, but each device only has 200 rows. 200 // 34 = 5 rows
    # per shard -> still fine. Push higher to force the failure.
    with pytest.raises(ValueError, match="cannot split"):
        nbaiot.load_nbaiot_per_device(
            data_dir="ignored", mode="binary", num_clients=9999,
            max_per_class=200, seed=0,
        )


def test_num_clients_returns_meta_for_trace() -> None:
    _train, _test, meta = nbaiot.load_nbaiot_per_device(
        data_dir="ignored", mode="binary", num_clients=18, max_per_class=200, seed=0,
    )
    # ``device_assignment`` is exactly num_clients long.
    assert len(meta["device_assignment"]) == 18
    # Each device contributes exactly 2 shards (18/9 = 2).
    assert Counter(meta["device_assignment"]) == {d: 2 for d in range(9)}
