"""Per-fog Gymnasium environment.

This is a *thin* wrapper that exposes:

* ``observation_space = Box(-inf, +inf, (obs_dim,))`` -- per-client features
  stacked across the ``N`` clients of the fog.
* ``action_space = Box(0, 1, (N,))`` -- continuous participation weights.

The env's `step()` / `reset()` are never used to drive an FL round; the
SAFEL-DT simulator owns the round loop and pushes ``(s, a, r, s')`` tuples
into the SAC agent's replay buffer directly via
:class:`safel_dt.rl.sac_controller.SacController`. The env exists solely
to satisfy Stable-Baselines3's constructor signature.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from safel_dt.rl.state_extraction import obs_dim as default_obs_dim


class FogEnv(gym.Env[np.ndarray, np.ndarray]):
    """Observation / action space holder for a fog-level SAC agent."""

    def __init__(
        self,
        num_clients: int,
        *,
        adversary_features: bool = False,
        heuristic_hint: bool = False,
    ) -> None:
        super().__init__()
        if num_clients <= 0:
            raise ValueError(f"num_clients must be > 0, got {num_clients}")
        self.num_clients = num_clients
        self._obs_dim = default_obs_dim(
            num_clients,
            adversary_features=adversary_features,
            heuristic_hint=heuristic_hint,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_clients,),
            dtype=np.float32,
        )
        self._zeros = np.zeros(self._obs_dim, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self._zeros.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Never invoked by the simulator; provided for API compliance only.
        return self._zeros.copy(), 0.0, True, False, {}
