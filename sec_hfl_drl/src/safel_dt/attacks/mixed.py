"""Mixed-attack cohort builder.

Wires concrete attack classes (:class:`LabelFlipAttack`,
:class:`ModelScaleAttack`, :class:`GaussianNoiseAttack`) into the
cohort-id -> attack assignment consumed by
:class:`safel_dt.attacks.schedule.MixedAdversary`.

The assignment is *fixed once at cohort construction*: each malicious
client commits to a single attack family for the whole run (paper
adversarial-schedule paragraph). Family weights default to uniform
over the three families; the per-attack hyperparameters match the
paper text (random target class for label-flip, U[10, 50] gamma for
model-scale, sigma=1.5 for Gaussian).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from safel_dt.attacks.base import Attack
from safel_dt.attacks.gaussian import GaussianNoiseAttack
from safel_dt.attacks.label_flip import LabelFlipAttack
from safel_dt.attacks.model_scale import ModelScaleAttack

DEFAULT_FAMILY_WEIGHTS: Mapping[str, float] = {
    "label_flip": 1.0,
    "model_scale": 1.0,
    "gaussian": 1.0,
}


def build_mixed_attacks(
    cohort: list[int],
    num_classes: int,
    *,
    rng_seed: int,
    family_weights: Mapping[str, float] | None = None,
    gamma_range: tuple[float, float] = (10.0, 50.0),
    gaussian_sigma: float = 1.5,
) -> dict[int, Attack]:
    """Assign each cohort member one attack from {flip, scale, gaussian}.

    Parameters
    ----------
    cohort:
        Client ids that may attack at some point. Order is preserved
        for reproducibility (each id gets a deterministic family given
        the seed).
    num_classes:
        Number of label classes; needed by :class:`LabelFlipAttack`.
    rng_seed:
        Seeds the family-assignment RNG. Distinct per-attack instance
        RNGs are derived from this seed so each
        :class:`LabelFlipAttack` / :class:`ModelScaleAttack` instance
        is independently reproducible.
    family_weights:
        Probability weights for ``{"label_flip", "model_scale",
        "gaussian"}``. Defaults to uniform. Unknown keys raise.
    gamma_range:
        ``(lo, hi)`` for :class:`ModelScaleAttack`. Each model-scale
        attacker independently samples its gamma from this interval.
    gaussian_sigma:
        Stddev for :class:`GaussianNoiseAttack`. Single fixed value
        across all gaussian attackers (no per-client sigma in the
        paper).

    Returns
    -------
    dict
        ``{client_id: Attack}`` covering every member of ``cohort``.
    """
    weights = dict(family_weights or DEFAULT_FAMILY_WEIGHTS)
    valid = {"label_flip", "model_scale", "gaussian"}
    unknown = set(weights) - valid
    if unknown:
        raise ValueError(f"unknown family weights {sorted(unknown)}; valid: {sorted(valid)}")
    for k in valid:
        weights.setdefault(k, 0.0)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"family_weights must sum to > 0, got {weights}")
    families = ("label_flip", "model_scale", "gaussian")
    probs = np.array([weights[f] for f in families], dtype=np.float64) / total

    rng = np.random.default_rng(rng_seed)
    assignment: dict[int, Attack] = {}
    for i, cid in enumerate(sorted(int(c) for c in cohort)):
        family = families[int(rng.choice(len(families), p=probs))]
        instance_seed = int(rng.integers(0, 2**31 - 1))
        if family == "label_flip":
            assignment[cid] = LabelFlipAttack(
                num_classes=num_classes,
                target_strategy="random",
                rng_seed=instance_seed,
            )
        elif family == "model_scale":
            assignment[cid] = ModelScaleAttack(
                gamma_range=gamma_range,
                rng_seed=instance_seed,
            )
        else:
            assignment[cid] = GaussianNoiseAttack(sigma=gaussian_sigma)
    return assignment


def family_breakdown(per_client_attacks: Mapping[int, Attack]) -> dict[str, int]:
    """Count how many cohort members got each attack family.

    Useful for the trace metadata ("8-client cohort: 3 flip / 3 scale /
    2 gaussian") so the analysis pipeline can stratify rate_M by family.
    """
    counts = {"label_flip": 0, "model_scale": 0, "gaussian": 0, "other": 0}
    for att in per_client_attacks.values():
        name = getattr(att, "name", "other")
        if name in counts:
            counts[name] += 1
        else:
            counts["other"] += 1
    if counts["other"] == 0:
        del counts["other"]
    return counts
