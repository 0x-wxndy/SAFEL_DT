"""Unit tests for the paper's adversarial-schedule pipeline.

Covers:

* :class:`safel_dt.attacks.schedule.StepwiseAdversary` -- piecewise
  malicious-fraction schedule, validation, ``active_malicious_ids``.
* :class:`safel_dt.attacks.schedule.MixedAdversary` wrapper -- correct
  attack object returned per active client, ``NoAttack`` passthrough
  for benign clients, ``per_client_attacks()`` view.
* :func:`safel_dt.attacks.mixed.build_mixed_attacks` -- assignment is
  deterministic given the seed, covers every cohort id, respects
  family weights and gamma range.
* :class:`safel_dt.attacks.label_flip.LabelFlipAttack` with
  ``target_strategy="random"`` -- target is never the true class,
  distribution is roughly uniform over the K-1 alternatives.
* :class:`safel_dt.attacks.model_scale.ModelScaleAttack` with
  ``gamma_range`` -- sampled scalar is in range, fixed for the
  instance's lifetime, distinct seeds give distinct gammas.
* :func:`safel_dt.runtime.attack_builder.parse_stepwise_breakpoints` --
  parser accepts the paper schedule string and rejects malformed
  inputs.
* :func:`safel_dt.runtime.attack_builder.build_schedule` -- the
  ``mixed + stepwise`` combination produces a :class:`MixedAdversary`
  whose inner schedule is a :class:`StepwiseAdversary`, with the
  paper-default cohort sizes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from safel_dt.attacks import (
    GaussianNoiseAttack,
    LabelFlipAttack,
    MixedAdversary,
    ModelScaleAttack,
    NoAttack,
    StepwiseAdversary,
    build_mixed_attacks,
    family_breakdown,
)
from safel_dt.runtime.attack_builder import (
    AttackSpec,
    build_schedule,
    parse_stepwise_breakpoints,
)

PAPER_BREAKPOINTS = ((0, 0.10), (100, 0.15), (200, 0.20), (250, 0.25))


# ----------------- StepwiseAdversary ----------------------------------


def _placeholder_attack() -> LabelFlipAttack:
    return LabelFlipAttack(num_classes=4, shift=1)


def test_stepwise_active_count_jumps_at_breakpoints():
    """Paper schedule at N=30: ceil(f * N) gives 3 / 5 / 6 / 8 clients."""
    sched = StepwiseAdversary(
        cohort=list(range(30)),
        attack=_placeholder_attack(),
        total_clients=30,
        breakpoints=list(PAPER_BREAKPOINTS),
    )
    assert len(sched.active_malicious_ids(0)) == 3
    assert len(sched.active_malicious_ids(99)) == 3
    assert len(sched.active_malicious_ids(100)) == 5
    assert len(sched.active_malicious_ids(199)) == 5
    assert len(sched.active_malicious_ids(200)) == 6
    assert len(sched.active_malicious_ids(249)) == 6
    assert len(sched.active_malicious_ids(250)) == 8
    assert len(sched.active_malicious_ids(299)) == 8


def test_stepwise_returns_attack_only_for_active_prefix():
    cohort = list(range(8))
    sched = StepwiseAdversary(
        cohort=cohort,
        attack=_placeholder_attack(),
        total_clients=30,
        breakpoints=list(PAPER_BREAKPOINTS),
    )
    for cid in cohort[:3]:
        assert sched.attack_for(cid, 0).name == "label_flip"
    for cid in cohort[3:]:
        assert sched.attack_for(cid, 0).name == "none"
    for cid in cohort:
        assert sched.attack_for(cid, 250).name == "label_flip"


def test_stepwise_malicious_ids_is_max_prefix():
    sched = StepwiseAdversary(
        cohort=list(range(30)),
        attack=_placeholder_attack(),
        total_clients=30,
        breakpoints=list(PAPER_BREAKPOINTS),
    )
    assert sched.malicious_ids() == set(range(8))


def test_stepwise_rejects_first_breakpoint_not_zero():
    with pytest.raises(ValueError, match="round 0"):
        StepwiseAdversary(
            cohort=list(range(10)),
            attack=_placeholder_attack(),
            total_clients=10,
            breakpoints=[(50, 0.5)],
        )


def test_stepwise_rejects_non_monotone_rounds():
    with pytest.raises(ValueError, match="strictly increasing"):
        StepwiseAdversary(
            cohort=list(range(10)),
            attack=_placeholder_attack(),
            total_clients=10,
            breakpoints=[(0, 0.1), (100, 0.2), (50, 0.3)],
        )


def test_stepwise_rejects_undersized_cohort():
    with pytest.raises(ValueError, match="cohort too small"):
        StepwiseAdversary(
            cohort=[0, 1],
            attack=_placeholder_attack(),
            total_clients=30,
            breakpoints=list(PAPER_BREAKPOINTS),
        )


# ----------------- MixedAdversary -------------------------------------


def test_mixed_adversary_dispatches_per_client_attack():
    cohort = [0, 1, 2]
    per_client = {
        0: LabelFlipAttack(num_classes=4, shift=1),
        1: ModelScaleAttack(gamma=10.0),
        2: GaussianNoiseAttack(sigma=1.5),
    }
    inner = StepwiseAdversary(
        cohort=cohort,
        attack=per_client[0],
        total_clients=10,
        breakpoints=[(0, 0.3)],
    )
    mixed = MixedAdversary(inner_schedule=inner, per_client_attacks=per_client)
    assert mixed.attack_for(0, 0).name == "label_flip"
    assert mixed.attack_for(1, 0).name == "model_scale"
    assert mixed.attack_for(2, 0).name == "gaussian"
    assert mixed.attack_for(5, 0).name == "none"


def test_mixed_adversary_missing_assignment_raises():
    inner = StepwiseAdversary(
        cohort=[0, 1, 2],
        attack=_placeholder_attack(),
        total_clients=10,
        breakpoints=[(0, 0.3)],
    )
    with pytest.raises(ValueError, match="missing assignments"):
        MixedAdversary(
            inner_schedule=inner,
            per_client_attacks={0: _placeholder_attack()},
        )


def test_mixed_adversary_forwards_active_ids_from_stepwise():
    cohort = list(range(8))
    per_client = {cid: _placeholder_attack() for cid in cohort}
    inner = StepwiseAdversary(
        cohort=cohort,
        attack=_placeholder_attack(),
        total_clients=30,
        breakpoints=list(PAPER_BREAKPOINTS),
    )
    mixed = MixedAdversary(inner_schedule=inner, per_client_attacks=per_client)
    assert len(mixed.active_malicious_ids(0)) == 3
    assert len(mixed.active_malicious_ids(250)) == 8


# ----------------- build_mixed_attacks --------------------------------


def test_build_mixed_attacks_covers_full_cohort():
    cohort = list(range(20))
    assignment = build_mixed_attacks(cohort, num_classes=11, rng_seed=42)
    assert set(assignment.keys()) == set(cohort)


def test_build_mixed_attacks_is_seed_deterministic():
    a = build_mixed_attacks([0, 1, 2, 3], num_classes=5, rng_seed=7)
    b = build_mixed_attacks([0, 1, 2, 3], num_classes=5, rng_seed=7)
    assert [type(a[cid]).__name__ for cid in sorted(a)] == [
        type(b[cid]).__name__ for cid in sorted(b)
    ]


def test_build_mixed_attacks_distribution_is_uniform_under_default_weights():
    """With 600 clients and uniform weights, each family gets ~200."""
    cohort = list(range(600))
    assignment = build_mixed_attacks(cohort, num_classes=11, rng_seed=123)
    counts = family_breakdown(assignment)
    for fam in ("label_flip", "model_scale", "gaussian"):
        assert 150 <= counts[fam] <= 250, counts


def test_build_mixed_attacks_respects_family_weights():
    assignment = build_mixed_attacks(
        cohort=list(range(300)),
        num_classes=11,
        rng_seed=11,
        family_weights={"label_flip": 1.0, "model_scale": 0.0, "gaussian": 0.0},
    )
    counts = family_breakdown(assignment)
    assert counts["label_flip"] == 300
    assert counts["model_scale"] == 0
    assert counts["gaussian"] == 0


def test_build_mixed_attacks_gamma_range_applied():
    assignment = build_mixed_attacks(
        cohort=list(range(20)),
        num_classes=11,
        rng_seed=5,
        family_weights={"model_scale": 1.0},
        gamma_range=(10.0, 50.0),
    )
    for att in assignment.values():
        assert isinstance(att, ModelScaleAttack)
        assert 10.0 <= att.gamma <= 50.0


# ----------------- LabelFlipAttack random target ----------------------


def test_random_label_flip_never_returns_true_class():
    att = LabelFlipAttack(num_classes=5, target_strategy="random", rng_seed=0)
    for true_label in range(5):
        for _ in range(20):
            assert att.transform_label(true_label) != true_label


def test_random_label_flip_covers_alternatives():
    """For K=4 and a fixed true class, all 3 alternatives should appear."""
    att = LabelFlipAttack(num_classes=4, target_strategy="random", rng_seed=42)
    seen = {att.transform_label(0) for _ in range(200)}
    assert seen == {1, 2, 3}


def test_random_label_flip_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="target_strategy"):
        LabelFlipAttack(num_classes=4, target_strategy="bogus")


def test_cyclic_label_flip_unchanged():
    att = LabelFlipAttack(num_classes=4, shift=1)
    assert att.transform_label(0) == 1
    assert att.transform_label(3) == 0


# ----------------- ModelScaleAttack gamma range -----------------------


def test_model_scale_gamma_range_samples_within_interval():
    att = ModelScaleAttack(gamma_range=(10.0, 50.0), rng_seed=1)
    assert 10.0 <= att.gamma <= 50.0


def test_model_scale_gamma_range_distinct_seeds_distinct_gammas():
    a = ModelScaleAttack(gamma_range=(10.0, 50.0), rng_seed=1)
    b = ModelScaleAttack(gamma_range=(10.0, 50.0), rng_seed=2)
    assert a.gamma != b.gamma


def test_model_scale_gamma_range_fixed_after_construction():
    att = ModelScaleAttack(gamma_range=(10.0, 50.0), rng_seed=3)
    g1 = att.gamma
    rng = np.random.default_rng(0)
    att.transform_delta(np.ones(5), rng)
    att.transform_delta(np.ones(5), rng)
    assert att.gamma == g1


def test_model_scale_gamma_range_invalid_raises():
    with pytest.raises(ValueError, match="lo > hi"):
        ModelScaleAttack(gamma_range=(50.0, 10.0))


# ----------------- parse_stepwise_breakpoints -------------------------


def test_parse_stepwise_paper_string():
    bps = parse_stepwise_breakpoints("0:0.10,100:0.15,200:0.20,250:0.25")
    assert bps == ((0, 0.10), (100, 0.15), (200, 0.20), (250, 0.25))


def test_parse_stepwise_tolerates_whitespace():
    bps = parse_stepwise_breakpoints(" 0:0.10 , 100:0.15 ")
    assert bps == ((0, 0.10), (100, 0.15))


def test_parse_stepwise_rejects_malformed():
    with pytest.raises(ValueError, match="round:frac"):
        parse_stepwise_breakpoints("0=0.1,100=0.2")


def test_parse_stepwise_rejects_non_numeric():
    with pytest.raises(ValueError, match="int:float"):
        parse_stepwise_breakpoints("0:0.1,abc:0.2")


# ----------------- build_schedule integration -------------------------


def test_build_schedule_mixed_stepwise_yields_mixed_over_stepwise():
    spec = AttackSpec(
        name="mixed",
        stepwise_breakpoints=PAPER_BREAKPOINTS,
        mixed_gamma_range=(10.0, 50.0),
        gaussian_sigma=1.5,
    )
    schedule, cohort = build_schedule(
        spec=spec,
        num_classes=11,
        client_ids=list(range(30)),
        seed=2026,
        rounds_total=300,
    )
    assert isinstance(schedule, MixedAdversary)
    assert isinstance(schedule._inner, StepwiseAdversary)
    assert len(cohort) == 8
    assert len(schedule.active_malicious_ids(0)) == 3
    assert len(schedule.active_malicious_ids(250)) == 8


def test_build_schedule_mixed_requires_activation_schedule():
    spec = AttackSpec(name="mixed", frac=0.25)
    with pytest.raises(ValueError, match="stepwise_breakpoints or"):
        build_schedule(
            spec=spec,
            num_classes=11,
            client_ids=list(range(30)),
            seed=0,
            rounds_total=300,
        )


def test_build_schedule_stepwise_with_single_family():
    spec = AttackSpec(name="label_flip", stepwise_breakpoints=PAPER_BREAKPOINTS)
    schedule, cohort = build_schedule(
        spec=spec,
        num_classes=11,
        client_ids=list(range(30)),
        seed=7,
        rounds_total=300,
    )
    assert isinstance(schedule, StepwiseAdversary)
    assert len(cohort) == 8
    sample_id = next(iter(schedule.active_malicious_ids(0)))
    assert schedule.attack_for(sample_id, 0).name == "label_flip"
