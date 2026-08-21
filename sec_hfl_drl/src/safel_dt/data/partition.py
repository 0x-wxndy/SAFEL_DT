"""Federated partitioning helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def iid_partition(
    n: int,
    num_clients: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Split ``0..n-1`` into ``num_clients`` nearly-equal IID shards."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if n < num_clients:
        raise ValueError(f"n={n} < num_clients={num_clients}")
    idx = rng.permutation(n)
    return [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(idx, num_clients)]


def assign_clients_to_fogs(num_clients: int, num_fogs: int) -> dict[int, list[int]]:
    """Round-robin assignment of client ids to fog ids."""
    if num_clients <= 0 or num_fogs <= 0:
        raise ValueError("num_clients and num_fogs must be > 0")
    out: dict[int, list[int]] = {fid: [] for fid in range(num_fogs)}
    for cid in range(num_clients):
        out[cid % num_fogs].append(cid)
    return out


def dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Label-Dirichlet non-IID partition (standard FL helper)."""
    y = np.asarray(labels, dtype=np.int64)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cuts)
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())
    return [np.asarray(sorted(ix), dtype=np.int64) for ix in client_indices]


def natural_per_device_partition(
    device_ids: Sequence[int],
) -> list[np.ndarray]:
    """Group sample indices by their native device id."""
    devices = np.asarray(device_ids, dtype=np.int64)
    unique = sorted(set(int(d) for d in devices.tolist()))
    return [np.where(devices == d)[0].astype(np.int64) for d in unique]
