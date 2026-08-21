"""Quick one-off: aggregator pick distribution per cloud policy.

For each ``*.jsonl`` trace under a sweep dir, count how often each
cloud aggregator was used in the first half vs second half of rounds.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

AGGS = ("fedavg", "krum", "multi_krum", "trimmed_mean", "median")
RUN_DIR = Path("results/runs/sweep_paper_final")

combined: dict[tuple[str, str], tuple[Counter, Counter]] = defaultdict(
    lambda: (Counter(), Counter())
)

for trace_path in sorted(RUN_DIR.glob("*.jsonl")):
    parts = trace_path.stem.split("__")
    if len(parts) < 3:
        continue
    fog, cloud, _seed = parts[0], parts[1], parts[2]
    if cloud not in ("round_robin", "d3qn"):
        continue

    # Keep only the first row per round_idx that carries a cloud_aggregator,
    # so we don't double-count fog-level sub-rows.
    by_round: dict[int, str] = {}
    for line in trace_path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (isinstance(row, dict) and "round_idx" in row):
            continue
        rid = int(row["round_idx"])
        agg = row.get("cloud_aggregator")
        if agg and rid not in by_round:
            by_round[rid] = agg

    if not by_round:
        continue
    rounds = sorted(by_round.items())
    midpoint = len(rounds) // 2
    early_c = Counter(a for _, a in rounds[:midpoint])
    late_c = Counter(a for _, a in rounds[midpoint:])
    combined[(fog, cloud)][0].update(early_c)
    combined[(fog, cloud)][1].update(late_c)


def _row(c: Counter) -> str:
    return " ".join(f"{a:<12}{c.get(a, 0):>3}" for a in AGGS)


print()
print("=== Aggregator pick distribution: rounds 0-14 (early) vs 15-29 (late), summed over 3 seeds ===")
print()
header = f'{"fog_policy":<12} {"cloud":<13}  early  ' + " ".join(f"{a:<15}" for a in AGGS) + " | shift"
print(header)
print("-" * len(header))
for (fog, cloud), (early_c, late_c) in sorted(combined.items()):
    early_top = max(AGGS, key=lambda a: early_c.get(a, 0))
    late_top = max(AGGS, key=lambda a: late_c.get(a, 0))
    shift = f"{early_top} -> {late_top}" if early_top != late_top else f"{early_top} (stable)"
    print(f'{fog:<12} {cloud:<13}  early  {_row(early_c)} | {shift}')
    print(f'{"":<12} {"":<13}  late   {_row(late_c)}')
    print()
