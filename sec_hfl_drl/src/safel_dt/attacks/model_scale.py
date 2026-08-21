"""Model-scale attack: multiply the local weight delta by gamma before encryption.

Two modes:

* Fixed scalar (``gamma`` set, ``gamma_range`` None): multiplies the
  delta by a constant. Single-attack baseline; reproducible.
* Random per-instance (``gamma_range=(lo, hi)``): the *constructor*
  draws ``self.gamma`` once from ``Uniform[lo, hi]`` using ``rng_seed``;
  the value is then fixed for the lifetime of this attack instance.
  Used by the paper's mixed-attack mode, where each malicious client
  commits to its own random scaling factor at cohort construction.

With ``gamma > 1`` the malicious client *boosts* its update (so its
"gradient direction" dominates the aggregate). With ``gamma < 0`` it
reverses the direction (a sign-flip attack).
"""

from __future__ import annotations

import numpy as np


class ModelScaleAttack:
    """Multiply the delta by a per-instance scalar (constant or U[lo, hi])."""

    name: str = "model_scale"

    def __init__(
        self,
        gamma: float = 10.0,
        *,
        gamma_range: tuple[float, float] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        if gamma_range is not None:
            lo, hi = float(gamma_range[0]), float(gamma_range[1])
            if lo > hi:
                raise ValueError(f"gamma_range lo > hi: {gamma_range}")
            if lo == 0.0 and hi == 0.0:
                raise ValueError("gamma_range cannot be (0, 0).")
            rng = np.random.default_rng(rng_seed)
            sampled = float(rng.uniform(lo, hi))
            if sampled == 0.0:
                sampled = hi if hi != 0.0 else lo
            self.gamma = sampled
        else:
            if gamma == 0.0:
                raise ValueError("gamma must be non-zero.")
            self.gamma = float(gamma)

    def transform_label(self, label: int) -> int:
        return int(label)

    def transform_delta(self, delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(delta, dtype=np.float64) * self.gamma
