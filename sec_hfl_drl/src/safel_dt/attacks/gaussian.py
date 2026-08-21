"""Gaussian-noise attack: add ``N(0, sigma^2 I)`` noise to the local delta."""

from __future__ import annotations

import numpy as np


class GaussianNoiseAttack:
    """Add Gaussian noise to the weight delta before encryption."""

    name: str = "gaussian"

    def __init__(self, sigma: float = 1.5) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)

    def transform_label(self, label: int) -> int:
        return int(label)

    def transform_delta(self, delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64)
        noise = rng.normal(0.0, self.sigma, size=d.shape)
        return d + noise
