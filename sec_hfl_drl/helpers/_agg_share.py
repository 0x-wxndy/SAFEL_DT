"""Compute D3QN aggregator-share table stratified by attack phase."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

TRACE = Path("results/runs/sweep_headline/all__d3qn__seed0000.jsonl")

PHASES = [
    ("Phase 1 (10% mal., r1-100)",   range(0, 100)),
    ("Phase 2 (15% mal., r101-200)", range(100, 200)),
    ("Phase 3 (20% mal., r201-250)", range(200, 250)),
    ("Phase 4 (25% mal., r251-300)", range(250, 300)),
]

AGGREGATORS = ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]

rows = []
with TRACE.open() as f:
    for line in f:
        d = json.loads(line)
        rows.append((d["round_idx"], d.get("cloud_aggregator", "unknown")))

print(f"Total rounds: {len(rows)}\n")
header = f"{'Phase':<28} | " + " | ".join(f"{a:>12s}" for a in AGGREGATORS)
print(header)
print("-" * len(header))

for phase_name, rng in PHASES:
    rng_set = set(rng)
    sel = [agg for r, agg in rows if r in rng_set]
    if not sel:
        continue
    counts = Counter(sel)
    total = len(sel)
    parts = [f"{100.0 * counts.get(a, 0) / total:>11.1f}%" for a in AGGREGATORS]
    print(f"{phase_name:<28} | " + " | ".join(parts))

print(f"\n{'overall (n=1 seed)':<28} | " +
      " | ".join(f"{100.0 * Counter(a for _, a in rows).get(a, 0) / len(rows):>11.1f}%"
                  for a in AGGREGATORS))
