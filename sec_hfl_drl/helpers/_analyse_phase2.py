"""Why did D3QN commit to Multi-Krum in phase 2?

Two hypotheses to check:
  H1: Epsilon decay schedule ends near r=100, so phase 2 starts in
      pure exploitation mode -> whatever has the highest Q-value wins.
  H2: Multi-Krum has the highest average per-round reward during phase 1
      exploration, so its Q-value is the converged greedy choice.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import statistics as st

TRACE = Path("results/runs/sweep_headline/all__d3qn__seed0000.jsonl")

rows = [json.loads(l) for l in TRACE.read_text().splitlines()]

per_agg_reward_phase1: dict[str, list[float]] = defaultdict(list)
per_agg_reward_phase2: dict[str, list[float]] = defaultdict(list)
per_agg_acc_gain_phase1: dict[str, list[float]] = defaultdict(list)

prev_acc = rows[0]["accuracy"]
for r in rows:
    agg = r.get("cloud_aggregator", "?")
    reward = r.get("cloud_reward", 0.0)
    acc = r["accuracy"]
    gain = acc - prev_acc
    if r["round_idx"] < 100:
        per_agg_reward_phase1[agg].append(reward)
        per_agg_acc_gain_phase1[agg].append(gain)
    elif r["round_idx"] < 200:
        per_agg_reward_phase2[agg].append(reward)
    prev_acc = acc

print("=== Phase 1 (rounds 1-100, 10% mal., exploration regime) ===")
print(f"{'aggregator':<14} | {'n_picks':>8} | {'mean cloud_reward':>20} | {'mean d_acc':>11}")
print("-" * 64)
for agg in ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]:
    rs = per_agg_reward_phase1.get(agg, [])
    gs = per_agg_acc_gain_phase1.get(agg, [])
    if not rs:
        print(f"{agg:<14} |    0     | (never picked)")
        continue
    mr = st.mean(rs)
    sr = st.pstdev(rs) if len(rs) > 1 else 0.0
    mg = st.mean(gs) * 100
    print(f"{agg:<14} | {len(rs):>8d} | {mr:>10.4f} +/- {sr:>5.4f} | {mg:>+8.3f}pp")

print()
print("=== Phase 2 (rounds 101-200, 15% mal., greedy regime) ===")
print(f"{'aggregator':<14} | {'n_picks':>8} | {'mean cloud_reward':>20}")
print("-" * 50)
for agg in ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]:
    rs = per_agg_reward_phase2.get(agg, [])
    if not rs:
        print(f"{agg:<14} |    0     | (never picked)")
        continue
    mr = st.mean(rs)
    print(f"{agg:<14} | {len(rs):>8d} | {mr:>10.4f}")

print()
print("=== epsilon decay sanity check (cloud_debug) ===")
for round_idx in [0, 25, 50, 75, 99, 100, 150, 200]:
    if round_idx >= len(rows):
        continue
    dbg = rows[round_idx].get("cloud_debug", {})
    eps = dbg.get("epsilon", "?") if isinstance(dbg, dict) else "?"
    print(f"  r={round_idx:>3d}  epsilon={eps}")
