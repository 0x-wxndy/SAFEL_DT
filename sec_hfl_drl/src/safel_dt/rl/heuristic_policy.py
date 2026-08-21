"""Deterministic score-based fog policy.

The paper's first non-RL baseline: pick the top ``mu_fog`` clients by a
hand-tuned linear score over per-client features observed from the
trace buffer. Scoring is delegated to
:func:`safel_dt.rl.heuristic_score.heuristic_scores` so SAC's hint
feature stays bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safel_dt.fl.adversary_features import ClientAdversaryFeatures
from safel_dt.rl.heuristic_score import HeuristicScoreCfg, heuristic_scores
from safel_dt.rl.policy import RoundFeedback
from safel_dt.rl.state_extraction import FogTraceBuffer


@dataclass(frozen=True)
class HeuristicConfig:
    """Weights and selection-cardinality settings."""

    w_samples: float = 1.0
    w_loss: float = 1.0
    w_divergence: float = 0.5
    w_cos_dist: float = 2.0
    w_norm_outlier: float = 1.0
    max_loss: float = 5.0
    mu_fog: int = 3
    m_min: int = 1
    explore_prob: float = 0.0
    seed: int = 0


@dataclass
class HeuristicFogPolicy:
    """Score-based fog selection. Conforms to :class:`FogPolicy`."""

    num_clients: int
    client_ids: list[int]
    n_samples_per_client: list[int]
    rounds_total: int
    cfg: HeuristicConfig = field(default_factory=HeuristicConfig)
    adversary_features: bool = False
    _trace: FogTraceBuffer = field(init=False)
    _rng: np.random.Generator = field(init=False)

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
        self._trace = FogTraceBuffer(
            client_ids=list(self.client_ids),
            n_samples_per_client=list(self.n_samples_per_client),
            rounds_total=self.rounds_total,
            adversary_features=self.adversary_features,
        )
        self._rng = np.random.default_rng(self.cfg.seed)

    def _scores(self) -> np.ndarray:
        samples = np.asarray(self.n_samples_per_client, dtype=np.float64)
        losses = np.asarray(
            [self._trace.traces[cid].loss_ema for cid in self.client_ids], dtype=np.float64
        )
        score_cfg = HeuristicScoreCfg(
            w_samples=self.cfg.w_samples,
            w_loss=self.cfg.w_loss,
            w_divergence=self.cfg.w_divergence,
            w_cos_dist=self.cfg.w_cos_dist,
            w_norm_outlier=self.cfg.w_norm_outlier,
            max_loss=self.cfg.max_loss,
        )
        if self.adversary_features:
            cos = np.asarray(
                [self._trace.traces[cid].cos_dist_to_mean_ema for cid in self.client_ids],
                dtype=np.float64,
            )
            ratios = np.asarray(
                [self._trace.traces[cid].delta_norm_ratio_ema for cid in self.client_ids],
                dtype=np.float64,
            )
            return heuristic_scores(
                loss_ema=losses,
                samples=samples,
                cos_dist_ema=cos,
                delta_norm_ratio_ema=ratios,
                cfg=score_cfg,
            )
        return heuristic_scores(loss_ema=losses, samples=samples, cfg=score_cfg)

    def select(self, *, round_idx: int) -> list[int]:
        del round_idx
        if self._rng.random() < self.cfg.explore_prob:
            k = self.cfg.mu_fog
            return sorted(
                self._rng.choice(self.num_clients, size=k, replace=False).tolist()
            )
        scores = self._scores()
        order = np.argsort(-scores)
        chosen = order[: self.cfg.mu_fog].tolist()
        return sorted(int(i) for i in chosen[: max(self.cfg.m_min, self.cfg.mu_fog)])

    def observe_feedback(self, feedback: RoundFeedback) -> None:
        selected_global = [self.client_ids[i] for i in feedback.selected_local_indices]
        feats = None
        if feedback.client_features:
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
