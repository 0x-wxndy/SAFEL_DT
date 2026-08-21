"""Per-round simulation context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safel_dt.transport.timing import SimClock


@dataclass
class RoundContext:
    """Bundle of per-round handles shared across simulator helpers."""

    round_idx: int
    rng: np.random.Generator
    clock: SimClock
