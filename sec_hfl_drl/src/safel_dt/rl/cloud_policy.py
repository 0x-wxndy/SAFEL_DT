"""Cloud-level aggregator-selection policy.

Mirrors the :class:`safel_dt.rl.policy.FogPolicy` pattern but at the
cloud: ``select(round_idx)`` returns an aggregator name; the simulator
hands back a :class:`CloudFeedback` after the round so the policy can
update its replay buffer / take a gradient step.

Concrete policies
-----------------
* :class:`StaticCloudPolicy` -- always picks the same aggregator.
* :class:`D3qnCloudPolicy` -- Dueling + Double DQN that learns which
  aggregator to use given the per-fog state and current multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from safel_dt.costs.reward import Multipliers
from safel_dt.fl.cloud_server import CloudRoundOutcome
from safel_dt.rl.cloud_env import CloudObsConfig, CloudObservation
from safel_dt.rl.d3qn import D3qnConfig, D3qnController


@dataclass(frozen=True)
class CloudFeedback:
    """Post-round signal handed back to the cloud policy."""

    round_idx: int
    chosen_aggregator: str
    outcome: CloudRoundOutcome
    multipliers: Multipliers
    per_fog_reward: dict[int, float]
    reward: float
    done: bool = False


class CloudPolicy(Protocol):
    """Pre-round aggregator pick + post-round learning hook."""

    aggregators: tuple[str, ...]

    def select(self, *, round_idx: int) -> str:
        """Return the aggregator name to use this round."""
        ...

    def observe_feedback(self, feedback: CloudFeedback) -> None:
        """Update internal state (replay buffer + gradient step for D3QN)."""
        ...


@dataclass
class StaticCloudPolicy:
    """Always pick the same aggregator (the legacy / non-RL baseline)."""

    name: str = "fedavg"
    aggregators: tuple[str, ...] = field(default=("fedavg",))

    def __post_init__(self) -> None:
        if self.name not in self.aggregators:
            self.aggregators = (self.name, *self.aggregators)

    def select(self, *, round_idx: int) -> str:
        del round_idx
        return self.name

    def observe_feedback(self, feedback: CloudFeedback) -> None:
        del feedback

    def debug_info(self) -> dict[str, object]:
        return {"policy": "static", "aggregator": self.name}


@dataclass
class RoundRobinCloudPolicy:
    """Cycle through the aggregator menu deterministically.

    Useful as a deterministic comparison point against
    :class:`D3qnCloudPolicy`: any RL policy worth its weight should
    outperform a blind cycle.
    """

    aggregators: tuple[str, ...] = field(
        default=("fedavg", "krum", "multi_krum", "trimmed_mean", "median")
    )
    _idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.aggregators:
            raise ValueError("aggregators must be non-empty.")

    def select(self, *, round_idx: int) -> str:
        name = self.aggregators[round_idx % len(self.aggregators)]
        self._idx = round_idx
        return name

    def observe_feedback(self, feedback: CloudFeedback) -> None:
        del feedback

    def debug_info(self) -> dict[str, object]:
        return {
            "policy": "round_robin",
            "idx": int(self._idx),
            "n_aggregators": len(self.aggregators),
        }


@dataclass
class D3qnCloudPolicy:
    """D3QN-driven cloud policy.

    Builds its observation from :class:`CloudObservation`, picks an
    aggregator via epsilon-greedy ``argmax_a Q(s, a)``, and learns from
    one transition per round via :class:`D3qnController`.
    """

    obs_cfg: CloudObsConfig
    d3qn_cfg: D3qnConfig = field(default_factory=D3qnConfig)
    deterministic: bool = False

    _obs_builder: CloudObservation = field(init=False)
    _controller: D3qnController = field(init=False)
    _last_obs: np.ndarray = field(init=False)
    _last_action: int = field(init=False, default=-1)
    _ready: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._obs_builder = CloudObservation(cfg=self.obs_cfg)
        self._controller = D3qnController(
            obs_dim=self.obs_cfg.obs_dim,
            n_actions=self.obs_cfg.n_actions,
            cfg=self.d3qn_cfg,
        )
        self._last_obs = self._obs_builder.reset()

    @property
    def aggregators(self) -> tuple[str, ...]:
        return self.obs_cfg.aggregators

    @property
    def controller(self) -> D3qnController:
        return self._controller

    def select(self, *, round_idx: int) -> str:
        del round_idx
        action_idx = self._controller.act(self._last_obs, deterministic=self.deterministic)
        self._last_action = int(action_idx)
        self._obs_builder.record_action(action_idx)
        self._ready = True
        return self.aggregators[action_idx]

    def observe_feedback(self, feedback: CloudFeedback) -> None:
        if not self._ready:
            return
        next_obs = self._obs_builder.build(
            outcome=feedback.outcome,
            multipliers=feedback.multipliers,
            per_fog_reward=feedback.per_fog_reward,
        )
        self._controller.learn(
            obs=self._last_obs,
            action=self._last_action,
            reward=float(feedback.reward),
            next_obs=next_obs,
            done=feedback.done,
        )
        self._last_obs = next_obs

    def debug_info(self) -> dict[str, object]:
        return {
            "policy": "d3qn",
            "epsilon": float(self._controller.epsilon),
            "last_action": int(self._last_action),
            "last_aggregator": (
                self.aggregators[self._last_action]
                if 0 <= self._last_action < len(self.aggregators)
                else None
            ),
            "buffer": int(len(self._controller._buf)),
        }
