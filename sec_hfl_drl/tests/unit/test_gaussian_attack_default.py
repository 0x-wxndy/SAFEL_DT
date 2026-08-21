"""Tests for A3: Gaussian attack sigma default bumped 0.5 -> 1.5."""

from __future__ import annotations

import numpy as np

from safel_dt.attacks.gaussian import GaussianNoiseAttack
from safel_dt.runtime.attack_builder import AttackSpec, build_attack


def test_default_sigma_is_1p5() -> None:
    """Bumped from 0.5 because the old value did not dent FedAvg on N-BaIoT."""
    assert GaussianNoiseAttack().sigma == 1.5


def test_attack_spec_default_sigma_is_1p5() -> None:
    assert AttackSpec().gaussian_sigma == 1.5


def test_build_attack_respects_custom_sigma() -> None:
    spec = AttackSpec(name="gaussian", frac=0.3, gaussian_sigma=2.0)
    attack = build_attack(name="gaussian", num_classes=2, spec=spec)
    assert getattr(attack, "sigma") == 2.0


def test_gaussian_noise_actually_perturbs_delta() -> None:
    """Sanity: noise injection changes the delta vector."""
    atk = GaussianNoiseAttack(sigma=1.5)
    rng = np.random.default_rng(0)
    delta = np.ones(100, dtype=np.float64)
    noisy = atk.transform_delta(delta, rng)
    assert np.linalg.norm(noisy - delta) > 0
    assert np.linalg.norm(noisy - delta) > 5.0  # rough sigma=1.5 sqrt(100) magnitude
