"""Dueling + Double DQN controller (paper's cloud-level RL agent).

A minimal, well-typed D3QN with the same ``act()`` / ``learn()`` per-step
API as :class:`safel_dt.rl.sac_controller.SacController`. No external RL
framework -- the network is small and the replay loop is trivial, so a
focused PyTorch implementation is easier to inspect than a Tianshou
wrapper and avoids the Collector/Trainer overhead.

Architecture
------------
Standard dueling head::

    Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') )

Double DQN target::

    a_next = argmax_a Q_online(s', a)
    y      = r + gamma * Q_target(s', a_next) * (1 - done)

Target network is soft-updated each gradient step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class D3qnConfig:
    """Hyperparameters for :class:`D3qnController`."""

    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 5_000
    learning_starts: int = 16
    gradient_steps: int = 1
    target_tau: float = 0.01  # soft-update coefficient
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200
    epsilon_exponential: bool = False
    seed: int = 0


_DEFAULT_CFG: D3qnConfig = D3qnConfig()


class DuelingQNet(nn.Module):
    """Tiny dueling Q-network (2-layer trunk + value/advantage heads)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class _ReplayBuffer:
    """Numpy-backed ring buffer of fixed-size transitions."""

    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        self._cap = int(capacity)
        self._size = 0
        self._idx = 0
        self._obs = np.zeros((self._cap, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((self._cap, obs_dim), dtype=np.float32)
        self._act = np.zeros(self._cap, dtype=np.int64)
        self._rew = np.zeros(self._cap, dtype=np.float32)
        self._done = np.zeros(self._cap, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._act[i] = int(action)
        self._rew[i] = float(reward)
        self._done[i] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=n)
        return (
            self._obs[idx],
            self._act[idx],
            self._rew[idx],
            self._next_obs[idx],
            self._done[idx],
        )


class D3qnController:
    """Dueling + Double DQN with the SAC controller's per-step learning API."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        cfg: D3qnConfig = _DEFAULT_CFG,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {n_actions}")
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        torch.manual_seed(cfg.seed)
        self._device = torch.device("cpu")
        self._online = DuelingQNet(obs_dim, n_actions, cfg.hidden_dim).to(self._device)
        self._target = DuelingQNet(obs_dim, n_actions, cfg.hidden_dim).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        for p in self._target.parameters():
            p.requires_grad_(False)
        self._opt = torch.optim.Adam(self._online.parameters(), lr=cfg.learning_rate)
        self._buf = _ReplayBuffer(cfg.buffer_size, obs_dim, cfg.seed)
        self._rng = np.random.default_rng(cfg.seed + 1)
        self._step = 0

    @property
    def epsilon(self) -> float:
        if self._step >= self.cfg.epsilon_decay_steps:
            return self.cfg.epsilon_end
        frac = self._step / max(self.cfg.epsilon_decay_steps, 1)
        if self.cfg.epsilon_exponential:
            # Geometric decay from start toward end over decay_steps.
            # At step=0 -> start; at step=decay_steps -> end.
            ratio = self.cfg.epsilon_end / max(self.cfg.epsilon_start, 1e-12)
            return float(self.cfg.epsilon_start * (ratio ** frac))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Return an integer action (epsilon-greedy unless ``deterministic``)."""
        if not deterministic and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        q = self._online(x)
        return int(torch.argmax(q, dim=-1).item())

    def learn(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Push one transition, take ``gradient_steps`` gradient steps if warm."""
        self._buf.add(obs, action, reward, next_obs, done)
        self._step += 1
        if len(self._buf) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return {"loss": 0.0, "epsilon": self.epsilon, "buffer": float(len(self._buf))}

        last_loss = 0.0
        for _ in range(max(1, self.cfg.gradient_steps)):
            last_loss = self._gradient_step()
            self._soft_update_target()
        return {
            "loss": float(last_loss),
            "epsilon": self.epsilon,
            "buffer": float(len(self._buf)),
        }

    def _gradient_step(self) -> float:
        obs_b, act_b, rew_b, next_obs_b, done_b = self._buf.sample(self.cfg.batch_size)
        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(act_b, dtype=torch.long, device=self._device)
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=self._device)
        next_obs_t = torch.as_tensor(next_obs_b, dtype=torch.float32, device=self._device)
        done_t = torch.as_tensor(done_b, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            next_q_online = self._online(next_obs_t)
            next_a = torch.argmax(next_q_online, dim=-1, keepdim=True)
            next_q_target = self._target(next_obs_t).gather(-1, next_a).squeeze(-1)
            y = rew_t + self.cfg.gamma * next_q_target * (1.0 - done_t)

        q_pred = self._online(obs_t).gather(-1, act_t.unsqueeze(-1)).squeeze(-1)
        loss = nn.functional.mse_loss(q_pred, y)
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._opt.step()
        return cast(float, loss.item())

    def _soft_update_target(self) -> None:
        tau = self.cfg.target_tau
        with torch.no_grad():
            for p_online, p_target in zip(
                self._online.parameters(), self._target.parameters(), strict=True
            ):
                p_target.mul_(1.0 - tau).add_(p_online, alpha=tau)
