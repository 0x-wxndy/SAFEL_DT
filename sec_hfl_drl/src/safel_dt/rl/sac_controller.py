"""Thin Stable-Baselines3 SAC wrapper with an explicit ``act`` / ``learn`` API.

We drive the FL loop ourselves rather than letting SB3 own it
(``model.learn()`` assumes the env transitions are cheap, which is not
the case for an FL round). Instead, this controller:

* exposes ``act(obs) -> action`` for the simulator's pre-round hook,
* exposes ``learn(obs, action, reward, next_obs, done)`` for the
  post-round hook -- it pushes the transition into SAC's replay buffer
  and, once warm-started, calls ``model.train(gradient_steps)``.

No SAC algorithm logic lives here -- we only wire the buffer + train
call. Hyperparameters mirror the SB3 defaults except for ``learning_starts``
which we lower (1 FL round is expensive; we cannot afford a 100-round warm-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from safel_dt.rl.fog_env import FogEnv


@dataclass(frozen=True)
class SacControllerConfig:
    """Hyperparameters forwarded to ``stable_baselines3.SAC``."""

    learning_rate: float = 3e-4
    buffer_size: int = 1_000
    batch_size: int = 32
    tau: float = 0.005
    gamma: float = 0.99
    learning_starts: int = 8
    train_freq: int = 1
    gradient_steps: int = 1
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    verbose: int = 0
    seed: int = 0


class SacController:
    """SAC wrapper. Holds one SB3 model + its replay buffer."""

    def __init__(
        self,
        num_clients: int,
        cfg: SacControllerConfig | None = None,
        *,
        adversary_features: bool = False,
        heuristic_hint: bool = False,
    ) -> None:
        from stable_baselines3 import SAC
        from stable_baselines3.common.logger import Logger

        self.num_clients = num_clients
        self.cfg = cfg or SacControllerConfig()
        self.env = FogEnv(
            num_clients=num_clients,
            adversary_features=adversary_features,
            heuristic_hint=heuristic_hint,
        )
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.cfg.learning_rate,
            buffer_size=self.cfg.buffer_size,
            batch_size=self.cfg.batch_size,
            tau=self.cfg.tau,
            gamma=self.cfg.gamma,
            learning_starts=self.cfg.learning_starts,
            train_freq=self.cfg.train_freq,
            gradient_steps=self.cfg.gradient_steps,
            policy_kwargs=dict(self.cfg.policy_kwargs),
            verbose=self.cfg.verbose,
            seed=self.cfg.seed,
            device="cpu",
        )
        # SB3 normally initialises the logger inside ``learn()``; we drive
        # the training loop manually, so install a no-op logger up front
        # so that ``model.train()`` can record metrics without crashing.
        self.model.set_logger(Logger(folder=None, output_formats=[]))
        self._n_transitions: int = 0

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Return the continuous action vector for one fog observation."""
        a, _state = self.model.predict(obs.astype(np.float32), deterministic=deterministic)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    def remember(
        self,
        *,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer (no training)."""
        buf = self.model.replay_buffer
        if buf is None:
            raise RuntimeError("SAC replay buffer is unexpectedly None")
        buf.add(
            obs.astype(np.float32).reshape(1, -1),
            next_obs.astype(np.float32).reshape(1, -1),
            action.astype(np.float32).reshape(1, -1),
            np.array([float(reward)], dtype=np.float32),
            np.array([bool(done)], dtype=bool),
            [{}],
        )
        self._n_transitions += 1

    def learn(
        self,
        *,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> None:
        """Push transition + train (when warmed up)."""
        self.remember(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        if self._n_transitions >= self.cfg.learning_starts:
            self.model.train(
                gradient_steps=self.cfg.gradient_steps,
                batch_size=min(self.cfg.batch_size, self._n_transitions),
            )
