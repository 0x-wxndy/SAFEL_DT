"""Synthetic tabular datasets for hermetic unit / integration tests."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTabularDataset(Dataset):
    """Deterministic linear-projection classification dataset.

    Features are drawn from a seeded RNG; labels come from an affine
    projection (also seeded) so train / test splits that share
    ``projection_seed`` stay consistent.
    """

    def __init__(
        self,
        *,
        n_samples: int,
        in_features: int = 32,
        num_classes: int = 4,
        seed: int = 0,
        projection_seed: int = 0,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if in_features <= 0 or num_classes <= 1:
            raise ValueError("in_features > 0 and num_classes > 1 required")
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, size=(n_samples, in_features)).astype(np.float32)
        proj_rng = np.random.default_rng(projection_seed)
        w = proj_rng.normal(0.0, 1.0, size=(in_features, num_classes)).astype(np.float32)
        logits = x @ w
        y = np.argmax(logits, axis=1).astype(np.int64)
        self._x = torch.from_numpy(x)
        self._y = torch.from_numpy(y)
        self.in_features = in_features
        self.num_classes = num_classes

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]
