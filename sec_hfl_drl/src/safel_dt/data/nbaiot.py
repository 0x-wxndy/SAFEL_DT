"""N-BaIoT loader (paper section V): 9 real IoT devices, 10 attack classes
+ benign, 115 numerical flow features per record.

Supports both common distributions of the dataset:

* **Nested (UCI) layout** -- per-device folders with ``benign_traffic.csv``
  and ``gafgyt/`` / ``mirai/`` subfolders. Auto-downloadable from the UCI
  archive (~600 MB).
* **Flat (Kaggle) layout** -- single folder with files named
  ``<device_id>.<family>.<subtype>.csv`` (and ``<device_id>.benign.csv``),
  where ``device_id`` is 1-indexed per the UCI device list.

The loader inspects the cache and dispatches to the right reader, so you
can drop either layout under ``<data_dir>/`` and it just works. Devices
3 (Ennio_Doorbell) and 7 (Samsung_SNH_1011_N_Webcam) only carry gafgyt
attacks in UCI; missing mirai files are tolerated silently.

Labels are either:

* ``mode="binary"`` -- 0=benign, 1=attack (default; matches the paper's
  binary IDS framing).
* ``mode="multi"`` -- 11 classes (benign + 10 attack subtypes).
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

NBaIoTMode = Literal["binary", "multi"]

NBAIOT_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/442/"
    "detection+of+iot+botnet+attacks+n+baiot.zip"
)

# Canonical device names as published by UCI. Index 0..8 maps to UCI's
# 1-indexed device IDs (the Kaggle flat layout uses these 1-indexed IDs
# in the filenames; we expose 0-indexed IDs everywhere in code).
DEVICE_NAMES: Final[tuple[str, ...]] = (
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)

# (family, subtype) pairs in their stable label order.
ATTACK_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("gafgyt", "combo"),
    ("gafgyt", "junk"),
    ("gafgyt", "scan"),
    ("gafgyt", "tcp"),
    ("gafgyt", "udp"),
    ("mirai", "ack"),
    ("mirai", "scan"),
    ("mirai", "syn"),
    ("mirai", "udp"),
    ("mirai", "udpplain"),
)

# Folder names we accept as "this directory holds N-BaIoT data".
_NBAIOT_DIR_ALIASES: Final[tuple[str, ...]] = (
    "nbaiot",
    "n-baiot",
    "n_baiot",
    "n-baiot-dataset",
    "detection_of_iot_botnet_attacks_n_baiot",
)


def _looks_like_flat(root: Path) -> bool:
    """Flat layout sentinel: any ``<id>.benign.csv`` at the top level."""
    return bool(list(root.glob("[1-9].benign.csv")))


def _looks_like_nested(root: Path) -> bool:
    """Nested layout sentinel: a device folder with ``benign_traffic.csv``."""
    return bool(list(root.glob("*/benign_traffic.csv")))


def _resolve_root(data_dir: Path) -> Path | None:
    """Find a directory under ``data_dir`` that holds N-BaIoT data.

    Tries (in order):
    1. ``data_dir`` itself.
    2. Any subdirectory whose lowercased name matches a known alias.
    3. Any subdirectory containing either layout's sentinel files.
    """
    if not data_dir.is_dir():
        return None
    if _looks_like_flat(data_dir) or _looks_like_nested(data_dir):
        return data_dir
    aliases = {a.lower() for a in _NBAIOT_DIR_ALIASES}
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() in aliases and (
            _looks_like_flat(child) or _looks_like_nested(child)
        ):
            return child
    for child in data_dir.iterdir():
        if child.is_dir() and (_looks_like_flat(child) or _looks_like_nested(child)):
            return child
    return None


def ensure_nbaiot(data_dir: Path | str, *, force: bool = False) -> Path:
    """Return the directory that contains N-BaIoT, downloading if needed.

    The download path creates ``<data_dir>/nbaiot/`` and extracts the UCI
    ZIP into it. If the data is already available under any supported
    layout / folder name, that existing directory is reused.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = _resolve_root(data_dir)
        if existing is not None:
            return existing

    target = data_dir / "nbaiot"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[nbaiot] downloading from {NBAIOT_URL} (~600 MB; one-time)...")
    with urllib.request.urlopen(NBAIOT_URL, timeout=600) as resp:
        data = resp.read()
    print(f"[nbaiot] downloaded {len(data) / 1e6:.0f} MB; extracting...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(target)
    for inner in list(target.rglob("*.zip")):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        os.remove(inner)
    resolved = _resolve_root(target) or _resolve_root(data_dir)
    if resolved is None:
        raise RuntimeError(
            f"N-BaIoT download completed but no recognisable layout found under {target}"
        )
    return resolved


def _read_csv_fast(path: Path) -> np.ndarray:
    """Read a feature CSV into a float32 ndarray (header row skipped)."""
    import pandas as pd

    df = pd.read_csv(path, dtype=np.float32, engine="c")
    return df.to_numpy(dtype=np.float32, copy=False)


def _device_dir_nested(root: Path, device_id: int) -> Path | None:
    """Resolve UCI's per-device directory (handles a couple of naming variants)."""
    name = DEVICE_NAMES[device_id]
    for c in (root / name, root / f"{device_id + 1}.{name}"):
        if c.is_dir():
            return c
    matches = [m for m in root.glob(f"*{name}*") if m.is_dir()]
    return matches[0] if matches else None


def _load_device_nested(
    root: Path, device_id: int, *, max_per_class: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ddir = _device_dir_nested(root, device_id)
    if ddir is None:
        raise FileNotFoundError(
            f"device {DEVICE_NAMES[device_id]!r} not found under {root}"
        )

    xs: list[np.ndarray] = []
    ybin: list[np.ndarray] = []
    ymul: list[np.ndarray] = []

    def _append(path: Path, label_bin: int, label_multi: int) -> None:
        arr = _read_csv_fast(path)
        if max_per_class is not None and arr.shape[0] > max_per_class:
            idx = rng.choice(arr.shape[0], max_per_class, replace=False)
            arr = arr[idx]
        xs.append(arr)
        ybin.append(np.full(arr.shape[0], label_bin, dtype=np.int64))
        ymul.append(np.full(arr.shape[0], label_multi, dtype=np.int64))

    benign = ddir / "benign_traffic.csv"
    if not benign.is_file():
        raise FileNotFoundError(f"no benign_traffic.csv in {ddir}")
    _append(benign, label_bin=0, label_multi=0)

    for i, (family, subtype) in enumerate(ATTACK_FILES, start=1):
        for path in (
            ddir / family / f"{subtype}.csv",
            ddir / f"{family}_attacks" / f"{subtype}.csv",
        ):
            if path.is_file():
                _append(path, label_bin=1, label_multi=i)
                break
    return np.concatenate(xs), np.concatenate(ybin), np.concatenate(ymul)


def _load_device_flat(
    root: Path, device_id: int, *, max_per_class: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat layout: ``<root>/<device_id_1based>.<family>.<subtype>.csv``."""
    did1 = device_id + 1
    xs: list[np.ndarray] = []
    ybin: list[np.ndarray] = []
    ymul: list[np.ndarray] = []

    def _append(path: Path, label_bin: int, label_multi: int) -> None:
        arr = _read_csv_fast(path)
        if max_per_class is not None and arr.shape[0] > max_per_class:
            idx = rng.choice(arr.shape[0], max_per_class, replace=False)
            arr = arr[idx]
        xs.append(arr)
        ybin.append(np.full(arr.shape[0], label_bin, dtype=np.int64))
        ymul.append(np.full(arr.shape[0], label_multi, dtype=np.int64))

    benign = root / f"{did1}.benign.csv"
    if not benign.is_file():
        raise FileNotFoundError(
            f"missing {benign.name} for device {DEVICE_NAMES[device_id]!r}"
        )
    _append(benign, label_bin=0, label_multi=0)

    for i, (family, subtype) in enumerate(ATTACK_FILES, start=1):
        path = root / f"{did1}.{family}.{subtype}.csv"
        if path.is_file():
            _append(path, label_bin=1, label_multi=i)
        # devices 3 + 7 legitimately have no mirai files -> just skip
    return np.concatenate(xs), np.concatenate(ybin), np.concatenate(ymul)


def _load_device(
    root: Path, device_id: int, *, max_per_class: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _looks_like_flat(root):
        return _load_device_flat(root, device_id, max_per_class=max_per_class, rng=rng)
    if _looks_like_nested(root):
        return _load_device_nested(root, device_id, max_per_class=max_per_class, rng=rng)
    raise FileNotFoundError(f"no recognisable N-BaIoT layout under {root}")


class NBaIoTDataset(Dataset):
    """Per-device N-BaIoT slice as a torch Dataset."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self._x = x.astype(np.float32)
        self._y = y.astype(np.int64)
        if mean is not None and std is not None:
            self._x = ((self._x - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
        self.in_features = int(self._x.shape[1])

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._x[idx]),
            torch.tensor(int(self._y[idx]), dtype=torch.long),
        )


def load_nbaiot_per_device(
    *,
    data_dir: Path | str,
    mode: NBaIoTMode = "binary",
    device_ids: list[int] | None = None,
    max_per_class: int | None = 5000,
    test_fraction: float = 0.2,
    seed: int = 0,
    normalise: bool = True,
    num_clients: int | None = None,
) -> tuple[list[NBaIoTDataset], NBaIoTDataset, dict[str, object]]:
    """Load N-BaIoT into one dataset per logical client + a shared test set.

    When ``num_clients`` is ``None`` or equal to the number of physical
    devices, each device becomes one client (legacy behaviour). When
    ``num_clients > n_devices``, each device's training rows are split
    into ``ceil(num_clients / n_devices)`` shards and emitted in
    interleaved device order (shard 0 of every device, then shard 1, ...).
    """
    if not (0.0 < test_fraction < 0.5):
        raise ValueError(f"test_fraction must be in (0, 0.5), got {test_fraction}")
    if device_ids is None:
        device_ids = list(range(9))
    n_devices = len(device_ids)
    if num_clients is not None and num_clients < n_devices:
        raise ValueError(
            f"unsupported num_clients={num_clients}: must be >= n_devices={n_devices}"
        )
    root = ensure_nbaiot(data_dir)
    rng = np.random.default_rng(seed)

    per_device_train_x: list[np.ndarray] = []
    per_device_train_y: list[np.ndarray] = []
    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []

    for did in device_ids:
        x, yb, ym = _load_device(root, did, max_per_class=max_per_class, rng=rng)
        y = yb if mode == "binary" else ym
        n = x.shape[0]
        idx = rng.permutation(n)
        n_test = int(n * test_fraction)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        per_device_train_x.append(x[train_idx])
        per_device_train_y.append(y[train_idx])
        test_xs.append(x[test_idx])
        test_ys.append(y[test_idx])

    mean = std = None
    if normalise:
        full_train = np.concatenate(per_device_train_x)
        mean = full_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = full_train.std(axis=0, dtype=np.float64).astype(np.float32)

    target_clients = n_devices if num_clients is None else int(num_clients)
    device_assignment: list[int] = []
    train_datasets: list[NBaIoTDataset] = []

    if target_clients == n_devices:
        for did, x, y in zip(device_ids, per_device_train_x, per_device_train_y, strict=True):
            train_datasets.append(NBaIoTDataset(x=x, y=y, mean=mean, std=std))
            device_assignment.append(int(did))
    else:
        shards_per_device = int(np.ceil(target_clients / n_devices))
        # How many devices emit the full shard count vs one fewer.
        n_full = target_clients - n_devices * (shards_per_device - 1)
        # Always carve each device into ``shards_per_device`` equal pieces so
        # partial devices drop the leftover shard (3/4 coverage) instead of
        # being re-split into fewer larger pieces.
        device_shards: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for d_idx, (x, y) in enumerate(zip(per_device_train_x, per_device_train_y, strict=True)):
            n_emit = shards_per_device if d_idx < n_full else shards_per_device - 1
            if n_emit <= 0:
                raise ValueError(
                    f"cannot split device {device_ids[d_idx]} into {n_emit} shards"
                )
            if x.shape[0] < shards_per_device:
                raise ValueError(
                    f"cannot split device {device_ids[d_idx]}: "
                    f"{x.shape[0]} rows < {shards_per_device} shards"
                )
            cuts = np.array_split(np.arange(x.shape[0]), shards_per_device)
            pieces = [(x[c], y[c]) for c in cuts if len(c) > 0]
            if len(pieces) < shards_per_device:
                raise ValueError(
                    f"cannot split device {device_ids[d_idx]} into "
                    f"{shards_per_device} non-empty shards"
                )
            device_shards.append(pieces[:n_emit])

        # Interleave: shard k of each device in device order.
        for shard_idx in range(shards_per_device):
            for d_idx, did in enumerate(device_ids):
                shards = device_shards[d_idx]
                if shard_idx >= len(shards):
                    continue
                sx, sy = shards[shard_idx]
                train_datasets.append(NBaIoTDataset(x=sx, y=sy, mean=mean, std=std))
                device_assignment.append(int(did))

        if len(train_datasets) != target_clients:
            raise ValueError(
                f"internal error: built {len(train_datasets)} clients, expected {target_clients}"
            )

    test_set = NBaIoTDataset(
        x=np.concatenate(test_xs), y=np.concatenate(test_ys), mean=mean, std=std
    )
    meta = {
        "device_ids": device_ids,
        "device_names": [DEVICE_NAMES[d] for d in device_ids],
        "device_assignment": device_assignment,
        "num_clients": len(train_datasets),
        "mode": mode,
        "num_classes": 2 if mode == "binary" else 11,
        "in_features": int(test_set.in_features),
        "max_per_class": max_per_class,
        "test_fraction": test_fraction,
        "root": str(root),
    }
    return train_datasets, test_set, meta
