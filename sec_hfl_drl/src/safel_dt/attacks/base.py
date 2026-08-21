"""Attack interface shared by every malicious transform."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Attack(Protocol):
    """Minimal duck-typed attack contract used by clients and schedules."""

    name: str

    def transform_label(self, label: int) -> int: ...

    def transform_delta(self, delta: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


class NoAttack:
    """Identity attack (benign client behaviour)."""

    name: str = "none"

    def transform_label(self, label: int) -> int:
        return int(label)

    def transform_delta(self, delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        del rng
        return np.asarray(delta, dtype=np.float64)
