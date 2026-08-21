"""Per-client adversary-detection features for fog policies (PR-14)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClientAdversaryFeatures:
    """Compact attack-aware descriptors for one client update."""

    delta_norm_ratio: float
    cos_dist_to_mean: float
    loss_zscore: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def compute_client_adversary_features(
    *,
    deltas: dict[int, np.ndarray],
    losses: dict[int, float],
) -> dict[int, ClientAdversaryFeatures]:
    """Compute features relative to the cohort mean delta / loss stats."""
    if not deltas:
        return {}
    ids = list(deltas.keys())
    stacked = np.stack([np.asarray(deltas[i], dtype=np.float64).ravel() for i in ids], axis=0)
    mean = stacked.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean)) + 1e-12
    norms = np.linalg.norm(stacked, axis=1)
    # cosine distance to mean
    dots = stacked @ mean
    cos_sim = dots / ((norms + 1e-12) * mean_norm)
    cos_dist = 1.0 - cos_sim

    loss_vals = np.array([float(losses.get(i, 0.0)) for i in ids], dtype=np.float64)
    loss_mean = float(loss_vals.mean()) if loss_vals.size else 0.0
    loss_std = float(loss_vals.std()) if loss_vals.size > 1 else 1.0
    if loss_std < 1e-12:
        loss_std = 1.0

    out: dict[int, ClientAdversaryFeatures] = {}
    for j, cid in enumerate(ids):
        out[int(cid)] = ClientAdversaryFeatures(
            delta_norm_ratio=float(norms[j] / mean_norm),
            cos_dist_to_mean=float(cos_dist[j]),
            loss_zscore=float((loss_vals[j] - loss_mean) / loss_std),
        )
    return out
