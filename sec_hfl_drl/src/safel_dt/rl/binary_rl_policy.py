"""Per-client epsilon-greedy bandit (paper's "binary RL" baseline).

For each client *i* the policy maintains a scalar Q-value
``Q[i] = E[reward_signal | i selected]`` and selects, every round,
the top ``mu_fog`` clients by ``Q`` with probability ``1 - eps`` and a
uniform random cohort with probability ``eps``.

Update rule (one transition per selected client per round)::

    Q[i] <- Q[i] + alpha * (reward_signal_i - Q[i])

The reward signal is the *negated* per-client loss observed in
:class:`safel_dt.rl.policy.RoundFeedback.client_losses` (lower loss =
higher utility). Clients with low / corrupt local losses (label flip,
sign-flipped model scale, etc.) accumulate a high apparent reward and
get over-selected -- which is exactly what the heavy-weight SAC agent
is supposed to *fix*. The "binary RL" baseline therefore mirrors what
a simpler per-client bandit would do without a global utility view.

Configurable so that, in adversarial settings, the experimenter can
also feed the fog reward in instead of the per-client loss (set
``use_fog_reward=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safel_dt.rl.policy import RoundFeedback


@dataclass(frozen=True)
class BinaryRLConfig:
    """Bandit knobs."""

    alpha: float = 0.2  # Q-value learning rate
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_rounds: int = 50
    mu_fog: int = 3
    m_min: int = 1
    initial_q: float = 0.0
    use_fog_reward: bool = False
    max_loss_clip: float = 5.0
    seed: int = 0


@dataclass
class BinaryRLFogPolicy:
    """Conforms to :class:`safel_dt.rl.policy.FogPolicy`."""

    num_clients: int
    client_ids: list[int]
    n_samples_per_client: list[int]
    rounds_total: int
    cfg: BinaryRLConfig = field(default_factory=BinaryRLConfig)

    _q: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)
    _step: int = field(init=False, default=0)
    _last_selected_local: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.num_clients != len(self.client_ids):
            raise ValueError("num_clients must match len(client_ids).")
        if not (0 < self.cfg.mu_fog <= self.num_clients):
            raise ValueError(f"mu_fog must be in (0, num_clients], got {self.cfg.mu_fog}.")
        if not (0 < self.cfg.m_min <= self.cfg.mu_fog):
            raise ValueError(
                f"m_min must be in (0, mu_fog], got "
                f"m_min={self.cfg.m_min}, mu_fog={self.cfg.mu_fog}."
            )
        self._q = np.full(self.num_clients, self.cfg.initial_q, dtype=np.float64)
        self._rng = np.random.default_rng(self.cfg.seed)

    @property
    def epsilon(self) -> float:
        if self._step >= self.cfg.eps_decay_rounds:
            return self.cfg.eps_end
        frac = self._step / max(self.cfg.eps_decay_rounds, 1)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @property
    def q_values(self) -> np.ndarray:
        return self._q.copy()

    def select(self, *, round_idx: int) -> list[int]:
        del round_idx
        k = self.cfg.mu_fog
        if self._rng.random() < self.epsilon:
            chosen = self._rng.choice(self.num_clients, size=k, replace=False).tolist()
        else:
            order = np.argsort(-self._q)
            chosen = order[:k].tolist()
        chosen = sorted(int(i) for i in chosen[: max(self.cfg.m_min, k)])
        self._last_selected_local = chosen
        return chosen

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        self._step += 1
        if not feedback.selected_local_indices:
            return
        if self.cfg.use_fog_reward:
            signal = float(feedback.reward)
            for i in feedback.selected_local_indices:
                if 0 <= i < self.num_clients:
                    self._q[i] += self.cfg.alpha * (signal - self._q[i])
            return
        for i in feedback.selected_local_indices:
            if not (0 <= i < self.num_clients):
                continue
            cid = self.client_ids[i]
            loss = float(feedback.client_losses.get(cid, self.cfg.max_loss_clip))
            loss = min(loss, self.cfg.max_loss_clip)
            r_i = -loss
            self._q[i] += self.cfg.alpha * (r_i - self._q[i])
