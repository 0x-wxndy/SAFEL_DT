"""Fog-level participation policy abstraction.

The simulator queries one ``FogPolicy`` per fog at the start of each
round to decide which child clients participate. After the round it
hands back the realised utility so the policy can learn (no-ops for
the non-RL policies).

Concrete policies:

* :class:`AllPolicy` -- every client participates (the PR-3 / PR-4 default).
* :class:`RandomPolicy` -- pick ``k`` clients uniformly at random.
* :class:`SacPolicy` -- SB3 SAC; uses the SelectClients procedure.

All policies operate on *local* client indices inside the fog (0 .. N-1).
The simulator translates those back to global client IDs before
invoking the fog server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from safel_dt.rl.sac_controller import SacController, SacControllerConfig
from safel_dt.rl.select_clients import SelectionConfig, select_clients
from safel_dt.rl.state_extraction import FogTraceBuffer


@dataclass(frozen=True)
class RoundFeedback:
    """Per-round signal the simulator hands back to a fog policy."""

    round_idx: int
    selected_local_indices: list[int]
    client_losses: dict[int, float]
    reward: float
    done: bool = False
    client_features: dict[int, object] | None = None


class FogPolicy(Protocol):
    """Pre-round selection + post-round learning hook."""

    num_clients: int

    def select(self, *, round_idx: int) -> list[int]:
        """Return the *local* indices (0..N-1) of selected clients."""
        ...

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        """Update internal traces (and replay buffer, for SAC)."""
        ...


# --- concrete policies ----------------------------------------------------


@dataclass
class AllPolicy:
    """Every client participates every round."""

    num_clients: int

    def select(self, *, round_idx: int) -> list[int]:
        return list(range(self.num_clients))

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        del feedback


@dataclass
class RandomPolicy:
    """Uniformly pick ``k`` clients each round."""

    num_clients: int
    k: int
    seed: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not (0 < self.k <= self.num_clients):
            raise ValueError(f"k must be in (0, num_clients], got {self.k}")
        self._rng = np.random.default_rng(self.seed)

    def select(self, *, round_idx: int) -> list[int]:
        return sorted(
            self._rng.choice(self.num_clients, size=self.k, replace=False).tolist()
        )

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        del feedback


@dataclass
class SacPolicy:
    """SB3 SAC + SelectClients (paper Algorithm 1).

    The trace buffer records per-client loss / participation; the
    controller outputs continuous weights; the selector applies
    ``tau / mu_fog / m_min`` to produce the participation cohort.
    """

    num_clients: int
    client_ids: list[int]
    n_samples_per_client: list[int]
    rounds_total: int
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    sac_cfg: SacControllerConfig = field(default_factory=SacControllerConfig)
    deterministic: bool = False
    adversary_features: bool = False
    heuristic_hint: bool = False

    _controller: SacController = field(init=False)
    _trace: FogTraceBuffer = field(init=False)
    _last_obs: np.ndarray = field(init=False)
    _last_action: np.ndarray = field(init=False)
    _initialised: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.num_clients != len(self.client_ids):
            raise ValueError("num_clients must match len(client_ids)")
        self._controller = SacController(
            num_clients=self.num_clients,
            cfg=self.sac_cfg,
            adversary_features=self.adversary_features,
            heuristic_hint=self.heuristic_hint,
        )
        self._trace = FogTraceBuffer(
            client_ids=list(self.client_ids),
            n_samples_per_client=list(self.n_samples_per_client),
            rounds_total=self.rounds_total,
            adversary_features=self.adversary_features,
            heuristic_hint=self.heuristic_hint,
        )
        shape = self._controller.env.observation_space.shape
        if shape is None or len(shape) == 0:
            raise RuntimeError("FogEnv observation_space.shape must be defined")
        self._last_obs = np.zeros(shape[0], dtype=np.float32)
        self._last_action = np.zeros(self.num_clients, dtype=np.float32)

    @property
    def controller(self) -> SacController:
        return self._controller

    def select(self, *, round_idx: int) -> list[int]:
        obs = self._trace.build_observation(round_idx=round_idx)
        action = self._controller.act(obs, deterministic=self.deterministic)
        self._last_obs = obs
        self._last_action = action
        self._initialised = True
        return select_clients(action, self.selection)

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        if not self._initialised:
            return
        selected_global = [self.client_ids[i] for i in feedback.selected_local_indices]
        feats = None
        if feedback.client_features:
            from safel_dt.fl.adversary_features import ClientAdversaryFeatures

            feats = {
                int(k): v
                for k, v in feedback.client_features.items()
                if isinstance(v, ClientAdversaryFeatures)
            }
        self._trace.update_after_round(
            selected_client_ids=selected_global,
            client_losses=feedback.client_losses,
            client_features=feats,
        )
        next_obs = self._trace.build_observation(round_idx=feedback.round_idx + 1)
        self._controller.learn(
            obs=self._last_obs,
            action=self._last_action,
            reward=float(feedback.reward),
            next_obs=next_obs,
            done=feedback.done,
        )
