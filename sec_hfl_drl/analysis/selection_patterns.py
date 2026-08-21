"""Per-client selection-pattern analysis for a sweep directory.

Reads every ``*.jsonl`` trace under ``run_dir`` and tabulates:

* per-client selection frequency (how often each global client id was
  picked across all rounds / all seeds within a single policy combo),
* the same split into ``malicious`` vs ``benign`` buckets (malicious set
  is taken from the ``malicious_ids`` field logged in the trace),
* an *avoidance ratio* ``benign_rate / malicious_rate`` -- numbers above
  1.0 mean the policy is preferring honest clients, numbers below 1.0
  mean the opposite,
* an *early vs late* selection bias (first half of rounds vs second
  half) so we can see whether RL policies *learn* to prune adversaries.

Trace format assumed (one JSON object per round)::

    {"round_idx": int,
     "selected_per_fog": {fog_id_str: [local_idx, ...]},
     "malicious_ids": [global_id, ...],
     ...}

Local-to-global mapping is reconstructed from the round-robin
``assign_clients_to_fogs`` partition (client ``i`` -> fog ``i %
num_fogs``).

Usage::

    python analysis/selection_patterns.py results/runs/sweep_paper_final \
        --num-clients 10 --num-fogs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Filename template the sweep runner writes:  <fog>__<cloud>__seed<NNNN>.jsonl
# Split on the literal "__" separator (double underscore) since both
# fog and cloud policy names may themselves contain single underscores
# (e.g. ``binary_rl``, ``round_robin``).
_TRACE_NAME_RE = re.compile(r"^(?P<fog>.+?)__(?P<cloud>.+?)__seed(?P<seed>\d+)\.jsonl$")


@dataclass(frozen=True)
class TraceMeta:
    fog_policy: str
    cloud_policy: str
    seed: int
    path: Path


def _round_robin_fogs(num_clients: int, num_fogs: int) -> dict[int, list[int]]:
    """Mirror :func:`safel_dt.data.partition.assign_clients_to_fogs` (round_robin)."""
    return {f: [c for c in range(num_clients) if c % num_fogs == f] for f in range(num_fogs)}


def _parse_trace_name(name: str) -> TraceMeta | None:
    m = _TRACE_NAME_RE.match(name)
    if not m:
        return None
    return TraceMeta(
        fog_policy=m.group("fog"),
        cloud_policy=m.group("cloud"),
        seed=int(m.group("seed")),
        path=Path(name),
    )


def _iter_traces(run_dir: Path) -> Iterable[TraceMeta]:
    for p in sorted(run_dir.glob("*.jsonl")):
        meta = _parse_trace_name(p.name)
        if meta is None:
            continue
        yield TraceMeta(meta.fog_policy, meta.cloud_policy, meta.seed, p)


@dataclass
class _RunCounts:
    """Per-run aggregation buffer."""
    rounds_total: int = 0
    rounds_early: int = 0
    rounds_late: int = 0
    sel_total: dict[int, int] = None  # type: ignore[assignment]
    sel_early: dict[int, int] = None  # type: ignore[assignment]
    sel_late: dict[int, int] = None  # type: ignore[assignment]
    malicious_ids: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.sel_total = defaultdict(int)
        self.sel_early = defaultdict(int)
        self.sel_late = defaultdict(int)
        self.malicious_ids = set()


def _summarise_trace(
    path: Path, *, fog_to_clients: dict[int, list[int]]
) -> _RunCounts | None:
    counts = _RunCounts()
    rounds: list[tuple[int, dict[str, list[int]]]] = []
    last_malicious: set[int] = set()
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if not isinstance(row, dict) or "round_idx" not in row:
                    continue
                rounds.append((int(row["round_idx"]), row.get("selected_per_fog", {})))
                mids = row.get("malicious_ids")
                if isinstance(mids, list):
                    last_malicious = {int(m) for m in mids}
    except OSError:
        return None

    if not rounds:
        return None
    rounds.sort(key=lambda x: x[0])
    counts.malicious_ids = last_malicious
    counts.rounds_total = len(rounds)
    midpoint = counts.rounds_total // 2

    for idx, (_, sel_per_fog) in enumerate(rounds):
        is_late = idx >= midpoint
        if is_late:
            counts.rounds_late += 1
        else:
            counts.rounds_early += 1
        for fog_id_str, local_indices in sel_per_fog.items():
            fog_id = int(fog_id_str)
            globals_for_fog = fog_to_clients.get(fog_id, [])
            for li in local_indices:
                if 0 <= li < len(globals_for_fog):
                    gid = globals_for_fog[li]
                    counts.sel_total[gid] += 1
                    if is_late:
                        counts.sel_late[gid] += 1
                    else:
                        counts.sel_early[gid] += 1
    return counts


@dataclass(frozen=True)
class PolicySummary:
    fog_policy: str
    cloud_policy: str
    n_seeds: int
    # selection rate = picks / rounds
    rate_malicious: float
    rate_benign: float
    rate_early_malicious: float
    rate_late_malicious: float
    rate_early_benign: float
    rate_late_benign: float
    avoidance_ratio: float  # benign / malicious  (higher = better)
    delta_malicious: float  # late - early on malicious (negative = learning to avoid)
    n_malicious: int
    n_benign: int


def _summarise_policy(
    metas: Sequence[TraceMeta], *, fog_to_clients: dict[int, list[int]]
) -> PolicySummary | None:
    runs = [_summarise_trace(m.path, fog_to_clients=fog_to_clients) for m in metas]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None

    # Aggregate across seeds: average per-client selection rate within each
    # bucket (malicious / benign), then combine.
    tot_malicious = tot_benign = 0.0
    early_malicious = early_benign = 0.0
    late_malicious = late_benign = 0.0
    nm = nb = 0
    for r in runs:
        if r.rounds_total == 0:
            continue
        for gid in range(max(max(fog_to_clients.values(), key=len)) + 1):  # noqa: B023
            picks_total = r.sel_total.get(gid, 0)
            picks_early = r.sel_early.get(gid, 0)
            picks_late = r.sel_late.get(gid, 0)
            rate_total = picks_total / r.rounds_total if r.rounds_total else 0.0
            rate_early = picks_early / r.rounds_early if r.rounds_early else 0.0
            rate_late = picks_late / r.rounds_late if r.rounds_late else 0.0
            if gid in r.malicious_ids:
                tot_malicious += rate_total
                early_malicious += rate_early
                late_malicious += rate_late
                nm += 1
            else:
                tot_benign += rate_total
                early_benign += rate_early
                late_benign += rate_late
                nb += 1

    if nm == 0 or nb == 0:
        return None
    rate_malicious = tot_malicious / nm
    rate_benign = tot_benign / nb
    rate_early_m = early_malicious / nm
    rate_late_m = late_malicious / nm
    rate_early_b = early_benign / nb
    rate_late_b = late_benign / nb
    avoidance = (rate_benign / rate_malicious) if rate_malicious > 0 else float("inf")
    delta_m = rate_late_m - rate_early_m

    return PolicySummary(
        fog_policy=metas[0].fog_policy,
        cloud_policy=metas[0].cloud_policy,
        n_seeds=len(runs),
        rate_malicious=rate_malicious,
        rate_benign=rate_benign,
        rate_early_malicious=rate_early_m,
        rate_late_malicious=rate_late_m,
        rate_early_benign=rate_early_b,
        rate_late_benign=rate_late_b,
        avoidance_ratio=avoidance,
        delta_malicious=delta_m,
        n_malicious=nm // len(runs),
        n_benign=nb // len(runs),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--num-fogs", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional CSV path; defaults to <run_dir>/selection_patterns.csv")
    args = p.parse_args(argv)

    if not args.run_dir.is_dir():
        raise SystemExit(f"run_dir {args.run_dir!s} is not a directory.")

    fog_to_clients = _round_robin_fogs(args.num_clients, args.num_fogs)

    by_combo: dict[tuple[str, str], list[TraceMeta]] = defaultdict(list)
    for meta in _iter_traces(args.run_dir):
        by_combo[(meta.fog_policy, meta.cloud_policy)].append(meta)

    if not by_combo:
        print(f"[selection_patterns] no traces found in {args.run_dir}")
        return 1

    summaries: list[PolicySummary] = []
    for (_fp, _cp), metas in sorted(by_combo.items()):
        s = _summarise_policy(metas, fog_to_clients=fog_to_clients)
        if s is not None:
            summaries.append(s)

    out_path = args.out or (args.run_dir / "selection_patterns.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "fog_policy", "cloud_policy", "n_seeds", "n_malicious", "n_benign",
            "rate_malicious", "rate_benign", "avoidance_ratio",
            "rate_early_malicious", "rate_late_malicious", "delta_malicious",
            "rate_early_benign", "rate_late_benign",
        ])
        for s in summaries:
            w.writerow([
                s.fog_policy, s.cloud_policy, s.n_seeds, s.n_malicious, s.n_benign,
                f"{s.rate_malicious:.4f}", f"{s.rate_benign:.4f}",
                f"{s.avoidance_ratio:.3f}",
                f"{s.rate_early_malicious:.4f}", f"{s.rate_late_malicious:.4f}",
                f"{s.delta_malicious:+.4f}",
                f"{s.rate_early_benign:.4f}", f"{s.rate_late_benign:.4f}",
            ])

    print(f"[selection_patterns] wrote {len(summaries)} rows -> {out_path}")
    print()
    header = (
        f"{'fog':<10} {'cloud':<12} {'rate_M':>7} {'rate_B':>7} "
        f"{'avoid':>6} {'early_M':>8} {'late_M':>8} {'delta_M':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        avoid_str = f"{s.avoidance_ratio:>5.2f}x" if s.avoidance_ratio != float("inf") else "    inf"
        print(
            f"{s.fog_policy:<10} {s.cloud_policy:<12} "
            f"{s.rate_malicious:>7.3f} {s.rate_benign:>7.3f} "
            f"{avoid_str:>6} {s.rate_early_malicious:>8.3f} "
            f"{s.rate_late_malicious:>8.3f} {s.delta_malicious:>+8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
