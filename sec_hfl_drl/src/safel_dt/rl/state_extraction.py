"""Per-fog observation vector construction.

For each client in a fog we stack a feature block; the full fog
observation is the concatenation across clients in a deterministic order.

Base features per client (always on):

* ``loss_ema`` -- exponentially smoothed local loss
* ``participated_last`` -- 1.0 if selected in round t-1
* ``samples_norm`` -- ``log1p(n_samples) / log1p(max_n_samples_in_fog)``
* ``round_progress`` -- ``round_idx / max(rounds, 1)``

Optional adversary features (PR-14, ``adversary_features=True``):

* ``delta_norm_ratio_ema``
* ``cos_dist_to_mean_ema``
* ``loss_zscore_ema``

Optional heuristic hint (A5, ``heuristic_hint=True``):

* scalar score from :func:`safel_dt.rl.heuristic_score.heuristic_scores`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safel_dt.fl.adversary_features import ClientAdversaryFeatures
from safel_dt.rl.heuristic_score import HeuristicScoreCfg, heuristic_scores

FEATURES_PER_CLIENT: int = 4
_ADV_FEATURES: int = 3
_HINT_FEATURES: int = 1


def features_per_client(
    *,
    adversary_features: bool = False,
    heuristic_hint: bool = False,
) -> int:
    """Number of scalar features emitted per client."""
    n = FEATURES_PER_CLIENT
    if adversary_features:
        n += _ADV_FEATURES
    if heuristic_hint:
        n += _HINT_FEATURES
    return n


def obs_dim(
    num_clients: int,
    *,
    adversary_features: bool = False,
    heuristic_hint: bool = False,
) -> int:
    """Dimension of the observation vector for a fog of size ``N``."""
    return features_per_client(
        adversary_features=adversary_features,
        heuristic_hint=heuristic_hint,
    ) * int(num_clients)


@dataclass
class ClientTrace:
    """Per-client running state that the observation extractor consumes."""

    loss_ema: float = 0.0
    participated_last: float = 0.0
    n_participations: int = 0
    delta_norm_ratio_ema: float = 1.0
    cos_dist_to_mean_ema: float = 0.0
    loss_zscore_ema: float = 0.0


@dataclass
class FogTraceBuffer:
    """Per-fog rolling traces used to build the SAC observation each round."""

    client_ids: list[int]
    n_samples_per_client: list[int]
    rounds_total: int
    traces: dict[int, ClientTrace] = field(default_factory=dict)
    ema_alpha: float = 0.3
    adversary_features: bool = False
    heuristic_hint: bool = False
    hint_cfg: HeuristicScoreCfg = field(default_factory=HeuristicScoreCfg)

    def __post_init__(self) -> None:
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if len(self.client_ids) != len(self.n_samples_per_client):
            raise ValueError(
                "client_ids and n_samples_per_client must have the same length"
            )
        for cid in self.client_ids:
            self.traces.setdefault(cid, ClientTrace())

    def update_after_round(
        self,
        *,
        selected_client_ids: list[int],
        client_losses: dict[int, float],
        client_features: dict[int, ClientAdversaryFeatures] | None = None,
    ) -> None:
        selected = set(selected_client_ids)
        for cid in self.client_ids:
            trace = self.traces[cid]
            trace.participated_last = 1.0 if cid in selected else 0.0
            if cid in selected and cid in client_losses:
                new_loss = float(client_losses[cid])
                trace.loss_ema = (
                    (1.0 - self.ema_alpha) * trace.loss_ema + self.ema_alpha * new_loss
                    if trace.n_participations > 0
                    else new_loss
                )
                trace.n_participations += 1
            if (
                self.adversary_features
                and client_features is not None
                and cid in client_features
                and cid in selected
            ):
                feat = client_features[cid]
                a = self.ema_alpha
                if trace.n_participations <= 1:
                    trace.delta_norm_ratio_ema = float(feat.delta_norm_ratio)
                    trace.cos_dist_to_mean_ema = float(feat.cos_dist_to_mean)
                    trace.loss_zscore_ema = float(feat.loss_zscore)
                else:
                    trace.delta_norm_ratio_ema = (
                        (1.0 - a) * trace.delta_norm_ratio_ema + a * float(feat.delta_norm_ratio)
                    )
                    trace.cos_dist_to_mean_ema = (
                        (1.0 - a) * trace.cos_dist_to_mean_ema + a * float(feat.cos_dist_to_mean)
                    )
                    trace.loss_zscore_ema = (
                        (1.0 - a) * trace.loss_zscore_ema + a * float(feat.loss_zscore)
                    )

    def build_observation(self, *, round_idx: int) -> np.ndarray:
        progress = float(round_idx) / max(float(self.rounds_total), 1.0)
        max_samples = max(self.n_samples_per_client) if self.n_samples_per_client else 1
        denom = float(np.log1p(max_samples))
        if denom <= 0:
            denom = 1.0

        hint_vals: np.ndarray | None = None
        if self.heuristic_hint:
            losses = np.asarray(
                [self.traces[cid].loss_ema for cid in self.client_ids], dtype=np.float64
            )
            samples = np.asarray(self.n_samples_per_client, dtype=np.float64)
            if self.adversary_features:
                cos = np.asarray(
                    [self.traces[cid].cos_dist_to_mean_ema for cid in self.client_ids],
                    dtype=np.float64,
                )
                ratios = np.asarray(
                    [self.traces[cid].delta_norm_ratio_ema for cid in self.client_ids],
                    dtype=np.float64,
                )
                hint_vals = heuristic_scores(
                    loss_ema=losses,
                    samples=samples,
                    cos_dist_ema=cos,
                    delta_norm_ratio_ema=ratios,
                    cfg=self.hint_cfg,
                )
            else:
                hint_vals = heuristic_scores(
                    loss_ema=losses, samples=samples, cfg=self.hint_cfg
                )

        feats: list[float] = []
        for i, (cid, n_samples) in enumerate(
            zip(self.client_ids, self.n_samples_per_client, strict=True)
        ):
            trace = self.traces[cid]
            block = [
                float(trace.loss_ema),
                float(trace.participated_last),
                float(np.log1p(n_samples)) / denom,
                progress,
            ]
            if self.adversary_features:
                block.extend(
                    [
                        float(trace.delta_norm_ratio_ema),
                        float(trace.cos_dist_to_mean_ema),
                        float(trace.loss_zscore_ema),
                    ]
                )
            if self.heuristic_hint and hint_vals is not None:
                block.append(float(hint_vals[i]))
            feats.extend(block)
        return np.asarray(feats, dtype=np.float32)
