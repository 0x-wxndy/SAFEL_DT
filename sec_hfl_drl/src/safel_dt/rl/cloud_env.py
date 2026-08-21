"""Cloud-level observation construction.

Unlike the fog SAC, the cloud action is discrete (pick an aggregator
out of a fixed registry), so we don't need a full Gymnasium ``Env`` --
the simulator drives transitions directly. This module just defines:

* :class:`CloudObsConfig` -- the aggregator menu + observation knobs.
* :class:`CloudObservation` -- per-round feature builder.

Observation vector layout (size ``4 * n_fogs + 3 + n_aggregators``):

* Per fog (size 4): ``mean_loss / max_loss``, ``rejection_rate``,
  ``participation_fraction``, ``last_reward``.
* Global (size 3 + ``n_aggregators``): multipliers
  ``(nu_lat, nu_cap, nu_priv)`` followed by a one-hot of the previously
  chosen aggregator (zero one-hot on the very first round).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safel_dt.costs.reward import Multipliers
from safel_dt.fl.cloud_server import CloudRoundOutcome

DEFAULT_AGGREGATORS: tuple[str, ...] = (
    "fedavg",
    "krum",
    "multi_krum",
    "trimmed_mean",
    "median",
)


@dataclass(frozen=True)
class CloudObsConfig:
    """Static config for cloud observation/action space."""

    fog_ids: tuple[int, ...]
    aggregators: tuple[str, ...] = DEFAULT_AGGREGATORS
    max_loss: float = 5.0

    @property
    def n_fogs(self) -> int:
        return len(self.fog_ids)

    @property
    def n_actions(self) -> int:
        return len(self.aggregators)

    @property
    def obs_dim(self) -> int:
        return 4 * self.n_fogs + 3 + self.n_actions


@dataclass
class CloudObservation:
    """Stateful builder that turns each round outcome into a numpy obs."""

    cfg: CloudObsConfig
    _last_aggregator_idx: int = field(init=False, default=-1)
    _last_per_fog_reward: dict[int, float] = field(init=False, default_factory=dict)

    def reset(self) -> np.ndarray:
        self._last_aggregator_idx = -1
        self._last_per_fog_reward = {fid: 0.0 for fid in self.cfg.fog_ids}
        return self.build(
            outcome=None,
            multipliers=Multipliers(),
            per_fog_reward=None,
        )

    def record_action(self, action_idx: int) -> None:
        self._last_aggregator_idx = int(action_idx)

    def build(
        self,
        *,
        outcome: CloudRoundOutcome | None,
        multipliers: Multipliers,
        per_fog_reward: dict[int, float] | None,
    ) -> np.ndarray:
        """Construct the observation vector for the *next* decision step."""
        vec = np.zeros(self.cfg.obs_dim, dtype=np.float32)
        if per_fog_reward is not None:
            self._last_per_fog_reward.update(per_fog_reward)

        if outcome is not None:
            participants = outcome.per_fog_participants
            losses = outcome.per_client_losses
            n_total = max(outcome.n_clients_accepted + outcome.n_clients_rejected, 1)
            rejection_rate = outcome.n_clients_rejected / n_total
            for k, fid in enumerate(self.cfg.fog_ids):
                fog_participants = participants.get(fid, [])
                fog_losses = [losses[cid] for cid in fog_participants if cid in losses]
                mean_loss = (
                    float(np.mean(fog_losses)) if fog_losses else 0.0
                )
                norm_loss = min(mean_loss, self.cfg.max_loss) / self.cfg.max_loss
                part_frac = (
                    len(fog_participants) / max(n_total, 1)
                )
                last_r = self._last_per_fog_reward.get(fid, 0.0)
                base = 4 * k
                vec[base + 0] = float(norm_loss)
                vec[base + 1] = float(rejection_rate)
                vec[base + 2] = float(part_frac)
                vec[base + 3] = float(last_r)

        off = 4 * self.cfg.n_fogs
        vec[off + 0] = float(multipliers.lat)
        vec[off + 1] = float(multipliers.cap)
        vec[off + 2] = float(multipliers.priv)
        if 0 <= self._last_aggregator_idx < self.cfg.n_actions:
            vec[off + 3 + self._last_aggregator_idx] = 1.0
        return vec
