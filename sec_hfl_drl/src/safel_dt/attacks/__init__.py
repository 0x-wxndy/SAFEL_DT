"""Attacks against the FL pipeline + scheduling policies."""

from safel_dt.attacks.base import Attack, NoAttack
from safel_dt.attacks.gaussian import GaussianNoiseAttack
from safel_dt.attacks.label_flip import LabelFlipAttack, LabelFlippedDataset
from safel_dt.attacks.mixed import build_mixed_attacks, family_breakdown
from safel_dt.attacks.model_scale import ModelScaleAttack
from safel_dt.attacks.schedule import (
    FixedAdversary,
    MaliciousSchedule,
    MixedAdversary,
    NoMalice,
    PeriodicAdversary,
    RampedAdversary,
    StepwiseAdversary,
    WindowedAdversary,
)

ATTACK_REGISTRY: dict[str, type] = {
    "none": NoAttack,
    "label_flip": LabelFlipAttack,
    "model_scale": ModelScaleAttack,
    "gaussian": GaussianNoiseAttack,
}

__all__ = [
    "ATTACK_REGISTRY",
    "Attack",
    "FixedAdversary",
    "GaussianNoiseAttack",
    "LabelFlipAttack",
    "LabelFlippedDataset",
    "MaliciousSchedule",
    "MixedAdversary",
    "ModelScaleAttack",
    "NoAttack",
    "NoMalice",
    "PeriodicAdversary",
    "RampedAdversary",
    "StepwiseAdversary",
    "WindowedAdversary",
    "build_mixed_attacks",
    "family_breakdown",
]
