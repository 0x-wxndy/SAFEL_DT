"""Construct ``(MaliciousSchedule, attack_name)`` from a CLI-style spec.

Single source of truth used by both ``run_simulation.py`` and
``run_sweep.py``. Keeps the simulator's ``malicious_schedule`` field
opaque to the scripts and centralises the "which clients are malicious"
random draw so it stays reproducible across seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from safel_dt.attacks.base import Attack, NoAttack
from safel_dt.attacks.gaussian import GaussianNoiseAttack
from safel_dt.attacks.label_flip import LabelFlipAttack
from safel_dt.attacks.mixed import build_mixed_attacks
from safel_dt.attacks.model_scale import ModelScaleAttack
from safel_dt.attacks.schedule import (
    MaliciousSchedule,
    MixedAdversary,
    NoMalice,
    RampedAdversary,
    StepwiseAdversary,
    WindowedAdversary,
)

AttackName = Literal["none", "label_flip", "model_scale", "gaussian", "mixed"]
ATTACK_NAMES: tuple[AttackName, ...] = (
    "none",
    "label_flip",
    "model_scale",
    "gaussian",
    "mixed",
)


@dataclass(frozen=True)
class AttackSpec:
    """All knobs the sweep / single-run CLIs expose for adversarial behaviour.

    Three schedule shapes are supported:

    * **Windowed (default)** -- ``frac`` of clients are malicious for the
      full window ``[start_round, end_round)``. This is the legacy
      behaviour and matches PR-9c.
    * **Ramped (PR-16)** -- set both ``ramp_early_frac`` and
      ``ramp_late_frac`` to enable a linearly escalating cohort: at
      round 0 ``ramp_early_frac * N`` clients attack, by the final
      round ``ramp_late_frac * N`` do. ``frac`` is ignored when both
      ramp fractions are set; ``start_round``/``end_round`` are also
      ignored (the ramp owns the full ``[0, rounds_total)`` window).
    * **Stepwise** (paper headline) -- set ``stepwise_breakpoints``
      to a list ``[(round, frac), ...]`` with the first round at 0.
      The cohort prefix is piecewise-constant; takes precedence over
      ``frac`` and the ramp pair. Combined with ``name="mixed"``,
      each cohort member is assigned a random attack family at
      construction (paper "mixed attacks" claim).
    """

    name: AttackName = "none"
    frac: float = 0.0
    start_round: int = 0
    end_round: int | None = None
    label_shift: int = 1
    model_gamma: float = 10.0
    gaussian_sigma: float = 1.5
    ramp_early_frac: float | None = None
    ramp_late_frac: float | None = None
    stepwise_breakpoints: tuple[tuple[int, float], ...] | None = None
    mixed_gamma_range: tuple[float, float] = (10.0, 50.0)
    mixed_family_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.name not in ATTACK_NAMES:
            raise ValueError(f"unknown attack name {self.name!r}; expected one of {ATTACK_NAMES}")
        if not 0.0 <= self.frac <= 1.0:
            raise ValueError(f"attack frac must be in [0, 1], got {self.frac}")
        ramp_pair = (self.ramp_early_frac, self.ramp_late_frac)
        if (ramp_pair[0] is None) != (ramp_pair[1] is None):
            raise ValueError(
                "ramp_early_frac and ramp_late_frac must both be set or both None; "
                f"got {ramp_pair}"
            )
        if self.ramp_early_frac is not None:
            for label, v in (
                ("ramp_early_frac", self.ramp_early_frac),
                ("ramp_late_frac", self.ramp_late_frac),
            ):
                if v is None or not 0.0 <= v <= 1.0:
                    raise ValueError(f"{label} must be in [0, 1], got {v}")
        if self.stepwise_breakpoints is not None:
            bps = self.stepwise_breakpoints
            if not bps:
                raise ValueError("stepwise_breakpoints must be non-empty when set")
            if bps[0][0] != 0:
                raise ValueError(
                    f"stepwise_breakpoints must start at round 0, got {bps[0][0]}"
                )
            for i in range(1, len(bps)):
                if bps[i][0] <= bps[i - 1][0]:
                    raise ValueError(
                        f"stepwise_breakpoints rounds must strictly increase, got {bps}"
                    )
            for r, f in bps:
                if not 0.0 <= f <= 1.0:
                    raise ValueError(f"stepwise breakpoint frac must be in [0, 1], got {f}")
        lo, hi = self.mixed_gamma_range
        if lo > hi:
            raise ValueError(f"mixed_gamma_range lo > hi: {(lo, hi)}")

    @property
    def is_ramped(self) -> bool:
        return self.ramp_early_frac is not None and self.ramp_late_frac is not None

    @property
    def is_stepwise(self) -> bool:
        return self.stepwise_breakpoints is not None

    @property
    def is_mixed(self) -> bool:
        return self.name == "mixed"


def parse_stepwise_breakpoints(text: str) -> tuple[tuple[int, float], ...]:
    """Parse the ``--attack-stepwise`` CLI string.

    Accepts comma-separated ``round:fraction`` pairs, e.g.
    ``"0:0.10,100:0.15,200:0.20,250:0.25"``. Whitespace is tolerated.
    Returns a tuple of ``(round, fraction)`` pairs validated against
    :class:`AttackSpec`'s rules (first round must be 0, rounds strictly
    increasing, fractions in [0, 1]).
    """
    raw = text.strip()
    if not raw:
        raise ValueError("--attack-stepwise: empty string")
    bps: list[tuple[int, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--attack-stepwise: each item must be 'round:frac', got {chunk!r}"
            )
        r_str, f_str = chunk.split(":", 1)
        try:
            r = int(r_str.strip())
            f = float(f_str.strip())
        except ValueError as e:
            raise ValueError(
                f"--attack-stepwise: failed to parse {chunk!r} as 'int:float'"
            ) from e
        bps.append((r, f))
    if not bps:
        raise ValueError("--attack-stepwise: no breakpoints parsed")
    return tuple(bps)


def _pick_malicious(*, client_ids: list[int], frac: float, seed: int) -> list[int]:
    """Reproducibly sample ``ceil(frac * |client_ids|)`` adversary ids."""
    if frac <= 0.0 or not client_ids:
        return []
    n = max(1, round(frac * len(client_ids)))
    n = min(n, len(client_ids))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(client_ids), size=n, replace=False)
    return sorted(int(c) for c in chosen.tolist())


def build_attack(*, name: AttackName, num_classes: int, spec: AttackSpec) -> Attack:
    """Instantiate a single-family attack object for the given dataset / spec.

    ``name="mixed"`` is **not** valid here -- the mixed-cohort case
    needs a per-client assignment and is handled by
    :func:`build_schedule` instead. Callers that hit this branch
    have a logic error.
    """
    if name == "none":
        return NoAttack()
    if name == "label_flip":
        if num_classes <= 1:
            raise ValueError(
                f"label_flip needs num_classes > 1 (got {num_classes}); use a "
                "multi-class dataset or pick a different attack."
            )
        return LabelFlipAttack(num_classes=num_classes, shift=spec.label_shift)
    if name == "model_scale":
        return ModelScaleAttack(gamma=spec.model_gamma)
    if name == "gaussian":
        return GaussianNoiseAttack(sigma=spec.gaussian_sigma)
    if name == "mixed":
        raise ValueError(
            "build_attack(name='mixed') is invalid; mixed cohorts use "
            "per-client assignments built in build_schedule()."
        )
    raise ValueError(f"unreachable: {name!r}")


def _ramp_counts(*, n_clients: int, early_frac: float, late_frac: float) -> tuple[int, int]:
    """Convert ramp fractions to integer cohort counts.

    Mirrors :func:`_pick_malicious` rounding: ``ceil`` semantics via
    ``round()``; both endpoints capped to ``n_clients`` and floored at 0.
    Returns ``(early_count, late_count)``.
    """
    def _q(frac: float) -> int:
        if frac <= 0.0:
            return 0
        return max(1, min(n_clients, round(frac * n_clients)))

    return _q(early_frac), _q(late_frac)


def build_schedule(
    *,
    spec: AttackSpec,
    num_classes: int,
    client_ids: list[int],
    seed: int,
    rounds_total: int | None = None,
) -> tuple[MaliciousSchedule, list[int]]:
    """Return ``(schedule, malicious_client_ids)``.

    Branches (in priority order):

    * ``spec.name == "none"`` or zero attack budget -> :class:`NoMalice`.
    * ``spec.is_stepwise`` -> :class:`StepwiseAdversary` as the
      activation layer. When ``spec.is_mixed`` is also set the result
      is wrapped with :class:`MixedAdversary` so each cohort member
      gets its own attack family. ``rounds_total`` may be ``None``
      (stepwise uses the breakpoints to decide activation).
    * ``spec.is_ramped`` -> :class:`RampedAdversary`. ``rounds_total``
      must be supplied by the caller (the simulator knows it).
    * Otherwise -> the legacy :class:`WindowedAdversary` path.

    ``malicious_client_ids`` is the **ever-active** cohort (union over
    rounds) so downstream analysis (rate_M, selection_patterns) keeps
    one consistent definition of "who could ever attack".
    """
    if spec.name == "none" and not spec.is_mixed:
        return NoMalice(), []

    if spec.is_stepwise:
        breakpoints = list(spec.stepwise_breakpoints or ())
        n_clients = len(client_ids)
        if n_clients == 0:
            return NoMalice(), []
        max_frac = max(f for _, f in breakpoints)
        # Ceiling -- matches StepwiseAdversary internals so the cohort
        # pre-allocated here exactly covers the schedule's peak prefix.
        max_count = max(0, min(n_clients, math.ceil(max_frac * n_clients)))
        if max_count == 0:
            return NoMalice(), []
        cohort = _pick_malicious(
            client_ids=client_ids,
            frac=max_count / max(n_clients, 1),
            seed=seed,
        )
        if spec.is_mixed:
            per_client = build_mixed_attacks(
                cohort=cohort,
                num_classes=num_classes,
                rng_seed=seed,
                family_weights=spec.mixed_family_weights,
                gamma_range=spec.mixed_gamma_range,
                gaussian_sigma=spec.gaussian_sigma,
            )
            placeholder = next(iter(per_client.values()))
            inner: MaliciousSchedule = StepwiseAdversary(
                cohort=cohort,
                attack=placeholder,
                total_clients=n_clients,
                breakpoints=breakpoints,
            )
            mixed: MaliciousSchedule = MixedAdversary(
                inner_schedule=inner,
                per_client_attacks=per_client,
            )
            return mixed, sorted(cohort[:max_count])
        attack = build_attack(name=spec.name, num_classes=num_classes, spec=spec)
        stepwise: MaliciousSchedule = StepwiseAdversary(
            cohort=cohort,
            attack=attack,
            total_clients=n_clients,
            breakpoints=breakpoints,
        )
        return stepwise, sorted(cohort[:max_count])

    if spec.is_ramped:
        if rounds_total is None or rounds_total <= 0:
            raise ValueError(
                "build_schedule(rounds_total=...) is required when AttackSpec "
                "uses ramp_early_frac/ramp_late_frac"
            )
        # Always assert non-None because is_ramped guarantees they are set.
        early_frac = float(spec.ramp_early_frac or 0.0)
        late_frac = float(spec.ramp_late_frac or 0.0)
        early_count, late_count = _ramp_counts(
            n_clients=len(client_ids), early_frac=early_frac, late_frac=late_frac,
        )
        max_count = max(early_count, late_count)
        if max_count == 0:
            return NoMalice(), []
        # Reuse _pick_malicious to draw the *maximum* cohort once; the
        # ramp then activates a growing prefix of it deterministically.
        # This means the early-active subset is also reproducibly the
        # "first chosen" subset for any given seed.
        cohort = _pick_malicious(
            client_ids=client_ids,
            frac=max_count / max(len(client_ids), 1),
            seed=seed,
        )
        if spec.is_mixed:
            per_client = build_mixed_attacks(
                cohort=cohort,
                num_classes=num_classes,
                rng_seed=seed,
                family_weights=spec.mixed_family_weights,
                gamma_range=spec.mixed_gamma_range,
                gaussian_sigma=spec.gaussian_sigma,
            )
            placeholder = next(iter(per_client.values()))
            ramped_inner: MaliciousSchedule = RampedAdversary(
                cohort=cohort,
                attack=placeholder,
                rounds_total=rounds_total,
                early_count=early_count,
                late_count=late_count,
            )
            return (
                MixedAdversary(
                    inner_schedule=ramped_inner,
                    per_client_attacks=per_client,
                ),
                sorted(cohort[:max_count]),
            )
        attack = build_attack(name=spec.name, num_classes=num_classes, spec=spec)
        ramped: MaliciousSchedule = RampedAdversary(
            cohort=cohort,
            attack=attack,
            rounds_total=rounds_total,
            early_count=early_count,
            late_count=late_count,
        )
        return ramped, sorted(cohort[:max_count])

    if spec.is_mixed:
        raise ValueError(
            "AttackSpec(name='mixed') requires stepwise_breakpoints or "
            "ramp_early_frac/ramp_late_frac to define the activation "
            "schedule; got neither."
        )

    if spec.frac <= 0.0:
        return NoMalice(), []
    malicious = _pick_malicious(client_ids=client_ids, frac=spec.frac, seed=seed)
    attack = build_attack(name=spec.name, num_classes=num_classes, spec=spec)
    windowed: MaliciousSchedule = WindowedAdversary(
        malicious_ids=malicious,
        attack=attack,
        start_round=spec.start_round,
        end_round=spec.end_round,
    )
    return windowed, malicious
