"""Malicious-client scheduling policies.

Each schedule maps ``(client_id, round_idx) -> Attack``. Most rounds return
`NoAttack` for most clients; a fixed adversary returns the configured
attack for a fixed set of clients.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from safel_dt.attacks.base import Attack, NoAttack


@runtime_checkable
class MaliciousSchedule(Protocol):
    """Structural protocol shared by every schedule implementation."""

    def attack_for(self, client_id: int, round_idx: int) -> Attack: ...

    def malicious_ids(self) -> set[int]: ...


class NoMalice:
    """Schedule that never returns a real attack (benign baseline)."""

    name: str = "none"

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        return NoAttack()

    def malicious_ids(self) -> set[int]:
        return set()


class FixedAdversary:
    """A fixed set of clients always run the same attack."""

    name: str = "fixed"

    def __init__(self, malicious_ids: Iterable[int], attack: Attack) -> None:
        self._ids: set[int] = set(int(i) for i in malicious_ids)
        self._attack = attack

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        return self._attack if client_id in self._ids else NoAttack()

    def malicious_ids(self) -> set[int]:
        return set(self._ids)


class WindowedAdversary:
    """Malicious clients attack only on rounds in ``[start, end)``.

    ``end = None`` means "until the end of the run" (open interval). Used
    by the sweep CLI to express "label-flip from round 0 to 9", "model-
    scale after round 5", etc.
    """

    name: str = "windowed"

    def __init__(
        self,
        malicious_ids: Iterable[int],
        attack: Attack,
        start_round: int = 0,
        end_round: int | None = None,
    ) -> None:
        if start_round < 0:
            raise ValueError(f"start_round must be >= 0, got {start_round}")
        if end_round is not None and end_round <= start_round:
            raise ValueError(
                f"end_round must be > start_round, got start={start_round}, end={end_round}"
            )
        self._ids: set[int] = {int(i) for i in malicious_ids}
        self._attack = attack
        self._start = start_round
        self._end = end_round

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        if client_id not in self._ids:
            return NoAttack()
        if round_idx < self._start:
            return NoAttack()
        if self._end is not None and round_idx >= self._end:
            return NoAttack()
        return self._attack

    def malicious_ids(self) -> set[int]:
        return set(self._ids)


class RampedAdversary:
    """Linearly escalating (or de-escalating) adversary cohort over rounds.

    PR-16 motivation
    ----------------
    The paper claims attacks *escalate* through training (Reviewer-1 #1
    is partly about this -- a flat 30% attack does not match the paper's
    text). This schedule samples one maximum-size cohort at construction
    and then activates a growing prefix of it round by round:

    .. math::

        k(r) = \\mathrm{round}\\!\\Big( early + (late - early)\\cdot\\frac{r}{R-1}\\Big)

    where ``early``/``late`` are integer client counts derived from the
    ``early_frac``/``late_frac`` knobs at construction. At round ``r``
    the first ``k(r)`` clients of the cohort attack; the rest stay
    benign. ``early > late`` is also valid (de-escalation, e.g. for an
    "attacker bots get cleaned up over time" experiment).

    ``malicious_ids()`` returns the **ever-active** cohort (the union of
    all clients that attack on at least one round), which is what the
    `rate_M` analysis script wants for a global "did the policy avoid
    these clients overall?" metric. The per-round active set is exposed
    via :meth:`active_malicious_ids` for round-resolved analysis (e.g.
    "did D3QN switch to Krum once the attack ramped past 20%?").
    """

    name: str = "ramped"

    def __init__(
        self,
        cohort: Iterable[int],
        attack: Attack,
        *,
        rounds_total: int,
        early_count: int,
        late_count: int,
    ) -> None:
        if rounds_total <= 0:
            raise ValueError(f"rounds_total must be > 0, got {rounds_total}")
        cohort_list: list[int] = sorted({int(c) for c in cohort})
        if early_count < 0 or late_count < 0:
            raise ValueError(
                f"early_count/late_count must be >= 0, got "
                f"early={early_count}, late={late_count}"
            )
        max_count = max(early_count, late_count)
        if max_count > len(cohort_list):
            raise ValueError(
                f"cohort too small: need >= max(early, late)={max_count} ids, "
                f"got {len(cohort_list)} ({cohort_list})"
            )
        self._cohort: list[int] = cohort_list
        self._attack = attack
        self._rounds = rounds_total
        self._early = early_count
        self._late = late_count

    def _active_count(self, round_idx: int) -> int:
        """Number of active attackers at ``round_idx``.

        ``round_idx`` is clamped into ``[0, rounds_total - 1]``. Rounds
        beyond the configured horizon stay at the ``late_count`` value
        (matches the open-ended ``end_round=None`` semantics in the
        other schedules).
        """
        if self._rounds == 1:
            return self._late
        r = max(0, min(round_idx, self._rounds - 1))
        t = r / (self._rounds - 1)
        k = self._early + t * (self._late - self._early)
        return round(k)

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        k = self._active_count(round_idx)
        if k <= 0:
            return NoAttack()
        if client_id in self._cohort[:k]:
            return self._attack
        return NoAttack()

    def malicious_ids(self) -> set[int]:
        # Union of clients that attack at any round.
        max_count = max(self._early, self._late)
        return set(self._cohort[:max_count])

    def active_malicious_ids(self, round_idx: int) -> set[int]:
        """Clients that are *actively* attacking at ``round_idx``."""
        return set(self._cohort[: max(0, self._active_count(round_idx))])


class StepwiseAdversary:
    """Stepwise-escalating adversary cohort over rounds (paper schedule).

    The paper's adversarial-schedule paragraph specifies a piecewise-
    constant malicious fraction: ``(round_start, frac)`` breakpoints
    where the active cohort size jumps from one level to the next.
    For the default 300-round headline run this is::

        [(0, 0.10), (100, 0.15), (200, 0.20), (250, 0.25)]

    meaning: from round 0 use 10% of N as the active cohort prefix,
    from round 100 jump to 15%, etc. The breakpoint at index 0 must
    start at round 0; subsequent breakpoints must be strictly
    increasing in round and define the fraction *from that round
    onward*. Fractions are interpreted as fractions of ``N`` (total
    clients), with the cohort prefix size ``round(frac * N)`` clamped
    into ``[0, len(cohort)]``.

    Like :class:`RampedAdversary`, one maximum-size cohort is sampled
    at construction and only a prefix is active at any given round.
    ``malicious_ids()`` returns the union of ever-active clients (used
    by ``rate_M``); :meth:`active_malicious_ids` returns the prefix
    active at a specific round.
    """

    name: str = "stepwise"

    def __init__(
        self,
        cohort: Iterable[int],
        attack: Attack,
        *,
        total_clients: int,
        breakpoints: list[tuple[int, float]],
    ) -> None:
        if total_clients <= 0:
            raise ValueError(f"total_clients must be > 0, got {total_clients}")
        if not breakpoints:
            raise ValueError("breakpoints must be non-empty.")
        if breakpoints[0][0] != 0:
            raise ValueError(
                f"first breakpoint must start at round 0, got {breakpoints[0][0]}"
            )
        for i in range(1, len(breakpoints)):
            if breakpoints[i][0] <= breakpoints[i - 1][0]:
                raise ValueError(
                    f"breakpoints must be strictly increasing in round, "
                    f"got {breakpoints}"
                )
        for r, f in breakpoints:
            if r < 0:
                raise ValueError(f"breakpoint round must be >= 0, got {r}")
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"breakpoint fraction must be in [0, 1], got {f}")
        cohort_list: list[int] = sorted({int(c) for c in cohort})
        max_frac = max(f for _, f in breakpoints)
        # Ceiling so "15% of 30" reads as 5 clients (>= 15%) rather than
        # 4 (banker's round would give 4 here). Keeps the per-round
        # active count an honest *upper* bound on the claimed fraction.
        max_count = math.ceil(max_frac * total_clients)
        if max_count > len(cohort_list):
            raise ValueError(
                f"cohort too small: need >= max prefix={max_count} ids "
                f"(max_frac={max_frac} of N={total_clients}), "
                f"got {len(cohort_list)} ({cohort_list})"
            )
        self._cohort: list[int] = cohort_list
        self._attack = attack
        self._N = total_clients
        self._breakpoints: list[tuple[int, float]] = list(breakpoints)

    def _active_count(self, round_idx: int) -> int:
        """Cohort prefix size at ``round_idx`` from the latest breakpoint.

        Uses ceiling rounding so the active cohort is *at least* the
        claimed fraction (e.g. 15% of 30 -> 5 clients, not 4).
        """
        r = max(0, int(round_idx))
        frac = self._breakpoints[0][1]
        for bp_round, bp_frac in self._breakpoints:
            if r >= bp_round:
                frac = bp_frac
            else:
                break
        k = math.ceil(frac * self._N)
        return max(0, min(k, len(self._cohort)))

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        k = self._active_count(round_idx)
        if k <= 0:
            return NoAttack()
        if client_id in self._cohort[:k]:
            return self._attack
        return NoAttack()

    def malicious_ids(self) -> set[int]:
        max_frac = max(f for _, f in self._breakpoints)
        max_count = round(max_frac * self._N)
        return set(self._cohort[:max_count])

    def active_malicious_ids(self, round_idx: int) -> set[int]:
        return set(self._cohort[: self._active_count(round_idx)])


class MixedAdversary:
    """Wraps an inner activation schedule with per-client attack types.

    The paper's mixed-cohort scenario assigns each malicious client one
    of ``{label_flip, model_scale, gaussian}`` at cohort construction
    and keeps the assignment stable for the whole run. Activation
    timing (when each client is *currently* active) is delegated to an
    inner schedule -- typically :class:`StepwiseAdversary` for the
    paper's stepwise escalation, but :class:`RampedAdversary` /
    :class:`FixedAdversary` also compose. The inner schedule's
    ``attack`` argument is effectively a placeholder; this wrapper
    overrides the returned attack object per client.

    Construction notes
    ------------------
    * ``per_client_attacks`` must cover *every* client in
      ``inner_schedule.malicious_ids()``. A client missing from the
      map is treated as benign (defensive) and a ``ValueError`` is
      raised at construction time to catch the bug early.
    * The helper :func:`safel_dt.attacks.mixed.build_mixed_attacks`
      builds the per-client map from a cohort + RNG seed.
    """

    name: str = "mixed"

    def __init__(
        self,
        inner_schedule: MaliciousSchedule,
        per_client_attacks: dict[int, Attack],
    ) -> None:
        self._inner = inner_schedule
        self._map: dict[int, Attack] = {int(k): v for k, v in per_client_attacks.items()}
        missing = inner_schedule.malicious_ids() - set(self._map.keys())
        if missing:
            raise ValueError(
                f"per_client_attacks is missing assignments for cohort members "
                f"{sorted(missing)}; every malicious client must have a type."
            )

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        base = self._inner.attack_for(client_id, round_idx)
        if getattr(base, "name", "none") == "none":
            return base
        return self._map.get(int(client_id), base)

    def malicious_ids(self) -> set[int]:
        return self._inner.malicious_ids()

    def active_malicious_ids(self, round_idx: int) -> set[int]:
        fast = getattr(self._inner, "active_malicious_ids", None)
        if callable(fast):
            return {int(c) for c in fast(round_idx)}
        return {
            int(c)
            for c in self._inner.malicious_ids()
            if getattr(self._inner.attack_for(int(c), int(round_idx)), "name", "none") != "none"
        }

    def per_client_attacks(self) -> dict[int, Attack]:
        """Read-only view of the cohort-id -> attack assignment.

        Returned dict is a shallow copy; mutating it does not affect
        the schedule. Used by analysis scripts (rate_M-by-type) and by
        unit tests.
        """
        return dict(self._map)


class PeriodicAdversary:
    """Malicious clients attack every ``period`` rounds, starting from ``offset``."""

    name: str = "periodic"

    def __init__(
        self,
        malicious_ids: Iterable[int],
        attack: Attack,
        period: int,
        offset: int = 0,
    ) -> None:
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        self._ids: set[int] = set(int(i) for i in malicious_ids)
        self._attack = attack
        self._period = period
        self._offset = offset

    def attack_for(self, client_id: int, round_idx: int) -> Attack:
        if client_id not in self._ids:
            return NoAttack()
        if (round_idx - self._offset) % self._period == 0 and round_idx >= self._offset:
            return self._attack
        return NoAttack()

    def malicious_ids(self) -> set[int]:
        return set(self._ids)
