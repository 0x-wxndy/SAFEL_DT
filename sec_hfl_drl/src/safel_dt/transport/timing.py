"""Simulation clock and wall-time measurement helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class SimClock:
    """Lightweight simulated-time tracker (seconds)."""

    now: float = 0.0
    history: list[float] = field(default_factory=list)

    def advance(self, dt: float) -> float:
        self.now += float(dt)
        self.history.append(self.now)
        return self.now


@contextmanager
def measure() -> Iterator[Callable[[], float]]:
    """Yield a zero-arg callable returning elapsed wall seconds since enter."""
    t0 = time.perf_counter()

    def elapsed() -> float:
        return float(time.perf_counter() - t0)

    yield elapsed
