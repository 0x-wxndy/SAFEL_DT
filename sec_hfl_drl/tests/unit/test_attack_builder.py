"""Unit tests for PR-9c attack builder + WindowedAdversary."""

from __future__ import annotations

import pytest

from safel_dt.attacks.base import NoAttack
from safel_dt.attacks.gaussian import GaussianNoiseAttack
from safel_dt.attacks.label_flip import LabelFlipAttack
from safel_dt.attacks.model_scale import ModelScaleAttack
from safel_dt.attacks.schedule import NoMalice, WindowedAdversary
from safel_dt.runtime.attack_builder import (
    AttackSpec,
    _pick_malicious,
    build_attack,
    build_schedule,
)


def test_attackspec_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        AttackSpec(name="garbage")  # type: ignore[arg-type]


def test_attackspec_rejects_out_of_range_frac() -> None:
    with pytest.raises(ValueError):
        AttackSpec(name="label_flip", frac=1.5)
    with pytest.raises(ValueError):
        AttackSpec(name="label_flip", frac=-0.1)


def test_pick_malicious_reproducible() -> None:
    a = _pick_malicious(client_ids=list(range(10)), frac=0.4, seed=42)
    b = _pick_malicious(client_ids=list(range(10)), frac=0.4, seed=42)
    c = _pick_malicious(client_ids=list(range(10)), frac=0.4, seed=43)
    assert a == b
    assert a != c
    assert len(a) == 4


def test_pick_malicious_at_least_one_when_frac_positive() -> None:
    """frac=0.001 with 3 clients should still round up to 1 to make attacks visible."""
    out = _pick_malicious(client_ids=[0, 1, 2], frac=0.001, seed=0)
    assert len(out) == 1


def test_pick_malicious_zero_frac() -> None:
    assert _pick_malicious(client_ids=[0, 1, 2], frac=0.0, seed=0) == []


def test_build_attack_label_flip_requires_multi_class() -> None:
    with pytest.raises(ValueError):
        build_attack(name="label_flip", num_classes=1, spec=AttackSpec(name="label_flip"))


def test_build_attack_constructs_correct_classes() -> None:
    assert isinstance(
        build_attack(name="none", num_classes=2, spec=AttackSpec()), NoAttack
    )
    a = build_attack(
        name="label_flip", num_classes=4, spec=AttackSpec(name="label_flip", label_shift=2)
    )
    assert isinstance(a, LabelFlipAttack)
    assert a.shift == 2
    b = build_attack(
        name="model_scale", num_classes=2, spec=AttackSpec(name="model_scale", model_gamma=-1.0)
    )
    assert isinstance(b, ModelScaleAttack)
    assert b.gamma == -1.0
    c = build_attack(
        name="gaussian", num_classes=2, spec=AttackSpec(name="gaussian", gaussian_sigma=0.3)
    )
    assert isinstance(c, GaussianNoiseAttack)
    assert c.sigma == 0.3


def test_build_schedule_no_attack_returns_no_malice() -> None:
    sched, ids = build_schedule(
        spec=AttackSpec(name="none"), num_classes=4, client_ids=[0, 1, 2], seed=0,
    )
    assert isinstance(sched, NoMalice)
    assert ids == []


def test_build_schedule_returns_windowed_adversary_with_window() -> None:
    sched, ids = build_schedule(
        spec=AttackSpec(name="label_flip", frac=0.5, start_round=2, end_round=5),
        num_classes=4,
        client_ids=list(range(8)),
        seed=0,
    )
    assert isinstance(sched, WindowedAdversary)
    assert len(ids) == 4
    assert ids == sorted(ids)
    assert set(sched.malicious_ids()) == set(ids)


def test_windowed_adversary_respects_window() -> None:
    atk = LabelFlipAttack(num_classes=4, shift=1)
    sched = WindowedAdversary(malicious_ids=[0, 1], attack=atk, start_round=3, end_round=6)
    for r in range(3):
        assert sched.attack_for(0, r).name == "none"
    for r in range(3, 6):
        assert sched.attack_for(0, r).name == "label_flip"
    for r in range(6, 10):
        assert sched.attack_for(0, r).name == "none"
    for r in range(0, 10):
        assert sched.attack_for(2, r).name == "none"


def test_windowed_adversary_open_ended_end() -> None:
    atk = LabelFlipAttack(num_classes=4, shift=1)
    sched = WindowedAdversary(malicious_ids=[0], attack=atk, start_round=0, end_round=None)
    for r in range(0, 100):
        assert sched.attack_for(0, r).name == "label_flip"


def test_windowed_adversary_bad_window_raises() -> None:
    atk = LabelFlipAttack(num_classes=4, shift=1)
    with pytest.raises(ValueError):
        WindowedAdversary(malicious_ids=[0], attack=atk, start_round=-1)
    with pytest.raises(ValueError):
        WindowedAdversary(malicious_ids=[0], attack=atk, start_round=3, end_round=3)
