"""Pure heuristic scoring used by both :class:`HeuristicFogPolicy` and (when
``heuristic_hint=True``) the SAC observation extractor.

The score is the paper's hand-tuned linear function over per-client
features observed from the trace buffer::

    score_i = w_n * samples_norm_i
            - w_l * loss_ema_i / max_loss
            - w_d * divergence_proxy_i

with an optional adversary-aware penalty when the per-client cos-distance
and delta-norm-ratio EMAs are available::

    + (- w_cos_dist * cos_dist_ema
       - w_norm_outlier * |delta_norm_ratio_ema - 1|)

Both the deterministic policy and the SAC hint feature must produce the
*same* score: extracting the function here avoids the two call sites
drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeuristicScoreCfg:
    """Linear-combination weights for the heuristic score.

    Defaults match :class:`safel_dt.rl.heuristic_policy.HeuristicConfig`
    so SAC's hint feature is informationally identical to what the
    heuristic policy uses internally for ranking.
    """

    w_samples: float = 1.0
    w_loss: float = 1.0
    w_divergence: float = 0.5
    w_cos_dist: float = 2.0
    w_norm_outlier: float = 1.0
    max_loss: float = 5.0


def heuristic_scores(
    *,
    loss_ema: np.ndarray,
    samples: np.ndarray,
    cos_dist_ema: np.ndarray | None = None,
    delta_norm_ratio_ema: np.ndarray | None = None,
    cfg: HeuristicScoreCfg = HeuristicScoreCfg(),
) -> np.ndarray:
    """Return one score per client (same length as ``loss_ema``).

    Higher = "more useful / more trustworthy". The result is bounded
    roughly in ``[-(w_cos_dist*2 + w_norm_outlier*1 + w_loss + w_divergence),
    w_samples]`` -- callers that want to feed the score into an SAC
    observation should normalise / clip according to their network
    initialisation.
    """
    loss_ema = np.asarray(loss_ema, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if loss_ema.shape != samples.shape:
        raise ValueError(
            f"loss_ema {loss_ema.shape} and samples {samples.shape} "
            "must have the same shape"
        )
    samples_norm = samples / max(samples.max() if samples.size > 0 else 1.0, 1.0)
    loss_norm = np.minimum(loss_ema, cfg.max_loss) / cfg.max_loss
    mean_loss = float(loss_ema.mean()) if loss_ema.size > 0 else 0.0
    divergence = np.abs(loss_ema - mean_loss) / max(cfg.max_loss, 1e-9)
    base = (
        cfg.w_samples * samples_norm
        - cfg.w_loss * loss_norm
        - cfg.w_divergence * divergence
    )
    if cos_dist_ema is None or delta_norm_ratio_ema is None:
        return base
    cos_dist_ema = np.asarray(cos_dist_ema, dtype=np.float64)
    delta_norm_ratio_ema = np.asarray(delta_norm_ratio_ema, dtype=np.float64)
    if cos_dist_ema.shape != loss_ema.shape:
        raise ValueError("cos_dist_ema and loss_ema must have the same shape")
    if delta_norm_ratio_ema.shape != loss_ema.shape:
        raise ValueError("delta_norm_ratio_ema and loss_ema must have the same shape")
    norm_outlier = np.abs(delta_norm_ratio_ema - 1.0)
    return base - cfg.w_cos_dist * cos_dist_ema - cfg.w_norm_outlier * norm_outlier


__all__ = ["HeuristicScoreCfg", "heuristic_scores"]
