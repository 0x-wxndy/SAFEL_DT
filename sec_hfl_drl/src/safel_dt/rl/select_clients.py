"""``SelectClients`` procedure (paper Algorithm 1).

Given continuous SAC participation weights ``a_i in [0, 1]`` for the
clients attached to a fog, returns the **set of selected client indices**
subject to three constraints:

* ``tau`` -- per-client inclusion threshold; clients with ``a_i < tau``
  are dropped first.
* ``mu_fog`` -- per-fog capacity; if more than ``mu_fog`` clients survive
  the threshold, take the top ``mu_fog`` by weight (deterministic).
* ``m_min`` -- minimum cohort; if *fewer* than ``m_min`` survive (or even
  zero), top up by promoting the highest-weight rejected clients until
  the cohort reaches ``m_min``.

Ties are broken by client index (stable / reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    """Parameters of the SelectClients procedure."""

    tau: float = 0.5
    mu_fog: int | None = None
    m_min: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.mu_fog is not None and self.mu_fog <= 0:
            raise ValueError(f"mu_fog must be > 0, got {self.mu_fog}")
        if self.m_min < 0:
            raise ValueError(f"m_min must be >= 0, got {self.m_min}")


def select_clients(weights: np.ndarray, cfg: SelectionConfig) -> list[int]:
    """Return the sorted list of selected client indices.

    Indices refer to positions inside ``weights`` (i.e. *local* to the
    caller's fog, not global client IDs).
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = w.size
    if n == 0:
        return []

    m_min = min(cfg.m_min, n)
    mu_fog = cfg.mu_fog if cfg.mu_fog is not None else n

    # Ranking: descending by weight, ties broken by ascending index.
    order = np.lexsort((np.arange(n), -w))

    survivors = [int(i) for i in order if w[i] >= cfg.tau]

    if len(survivors) > mu_fog:
        survivors = survivors[:mu_fog]

    if len(survivors) < m_min:
        already = set(survivors)
        for idx in order:
            i = int(idx)
            if i in already:
                continue
            survivors.append(i)
            already.add(i)
            if len(survivors) >= m_min:
                break

    return sorted(survivors)
