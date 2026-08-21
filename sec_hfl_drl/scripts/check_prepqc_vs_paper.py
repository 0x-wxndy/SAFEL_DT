"""Check a pre-PQC smoke sweep against paper invariants from experiment.tex.

The full paper used N-BaIoT, T=300, Paillier-1024. A short synthetic smoke
cannot match absolute numbers; this checker validates *structural* and
*ordering* claims that the restored code must still satisfy:

Paper anchors (experiment.tex):
  - N=30, |F|=3, mixed stepwise 10→15→20→25%
  - sac+d3qn cuts comm vs all (~245k vs ~549k ≈ 55% saving)
  - train cost: all ≫ selection (capacity saturation; ~2978 vs ~31)
  - sec cost (Paillier): scales ~linear with cohort (~63 min all vs ~31 sac)
  - tail acc: sac+d3qn ~0.78; all/static FedAvg collapses ~0.09 under attack
  - ablation: SAC owns efficiency; D3QN owns robustness

For short smokes we only assert orderings / plumbing, with soft warnings
when absolute scales are off (expected under synthetic + plain crypto).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Paper Table cost-by-policy (sac+d3qn / all+d3qn rows) — absolute anchors.
PAPER_COMM_ALL = 548_661.0
PAPER_COMM_SAC = 245_495.0
PAPER_TRAIN_ALL = 2_977.0
PAPER_TRAIN_SAC = 31.417
PAPER_SEC_ALL_MIN = 63.2
PAPER_SEC_SAC_MIN = 30.6
PAPER_TAIL_SAC_D3QN = 0.783
PAPER_TAIL_STATIC = 0.093
# Calibrated Paillier-1024 per-device encrypt (helpers/_paillier_recalib_cache.json)
C_ENC_P1024_S = 147.513
C_AUTH_S = 0.004519
C_VERIFY_S = 0.005986


def _load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_cell(name: str) -> tuple[str, str, int] | None:
    # fog__cloud__seed0000.jsonl
    stem = name.replace(".jsonl", "")
    if "__seed" not in stem:
        return None
    left, seed_s = stem.rsplit("__seed", 1)
    if "__" not in left:
        return None
    fog, cloud = left.rsplit("__", 1)
    try:
        seed = int(seed_s)
    except ValueError:
        return None
    return fog, cloud, seed


def _cohort_size(row: dict) -> float:
    sel = row.get("selected_per_fog")
    if sel is None:
        # "all" policy may record null → treat as full federation
        return float(row.get("n_clients_accepted") or 0)
    total = 0
    for v in sel.values():
        if isinstance(v, list):
            total += len(v)
    return float(total)


def _sum_cost(row: dict, key: str) -> float:
    costs = row.get("costs") or {}
    total = 0.0
    for fog_costs in costs.values():
        if isinstance(fog_costs, dict) and key in fog_costs:
            total += float(fog_costs[key])
    return total


def _summarise(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    accs = [float(r["accuracy"]) for r in rows]
    tail_n = max(1, min(10, n // 5 or 1))
    return {
        "rounds": float(n),
        "acc_final": accs[-1],
        "acc_mean": sum(accs) / n,
        "acc_tail": sum(accs[-tail_n:]) / tail_n,
        "acc_max": max(accs),
        "comm_mean": sum(_sum_cost(r, "comm") for r in rows) / n,
        "train_mean": sum(_sum_cost(r, "train") for r in rows) / n,
        "sec_mean": sum(_sum_cost(r, "sec") for r in rows) / n,
        "cohort_mean": sum(_cohort_size(r) for r in rows) / n,
        "n_accepted_mean": sum(float(r.get("n_clients_accepted") or 0) for r in rows) / n,
    }


def _estimate_paillier_sec_min(cohort_mean: float) -> float:
    """Retroactive Paillier-1024 sec cost (minutes), same idea as _paillier_recalib."""
    per_device = C_ENC_P1024_S + C_AUTH_S + C_VERIFY_S
    return (cohort_mean * per_device) / 60.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=0, help="Expected round count (0=auto).")
    args = p.parse_args(argv)

    cells: dict[tuple[str, str], dict[str, float]] = {}
    for path in sorted(args.trace_dir.glob("*.jsonl")):
        parsed = _parse_cell(path.name)
        if parsed is None:
            continue
        fog, cloud, _seed = parsed
        rows = _load_trace(path)
        if args.rounds and len(rows) < args.rounds:
            print(f"FAIL  {path.name}: expected {args.rounds} rounds, got {len(rows)}")
            return 1
        cells[(fog, cloud)] = _summarise(rows)
        print(
            f"cell  {fog:10s} + {cloud:12s}  "
            f"rounds={int(cells[(fog, cloud)]['rounds'])}  "
            f"acc_tail={cells[(fog, cloud)]['acc_tail']:.3f}  "
            f"comm={cells[(fog, cloud)]['comm_mean']:.1f}  "
            f"train={cells[(fog, cloud)]['train_mean']:.2f}  "
            f"cohort={cells[(fog, cloud)]['cohort_mean']:.1f}"
        )

    if not cells:
        print("FAIL  no jsonl cells found")
        return 1

    fails: list[str] = []
    warns: list[str] = []

    # --- plumbing ---
    needed = {("all", "static"), ("sac", "d3qn")}
    missing = needed - set(cells)
    if missing:
        fails.append(f"missing headline cells: {sorted(missing)}")

    all_static = cells.get(("all", "static"))
    sac_d3qn = cells.get(("sac", "d3qn"))
    all_d3qn = cells.get(("all", "d3qn"))
    sac_static = cells.get(("sac", "static"))

    # --- efficiency (SAC owns cost) ---
    if all_static and sac_d3qn:
        if not (sac_d3qn["comm_mean"] < 0.85 * all_static["comm_mean"]):
            fails.append(
                f"comm ordering broken: sac+d3qn={sac_d3qn['comm_mean']:.1f} "
                f"not << all+static={all_static['comm_mean']:.1f} "
                f"(paper ~{PAPER_COMM_SAC:.0f} vs ~{PAPER_COMM_ALL:.0f})"
            )
        else:
            saving = 1.0 - sac_d3qn["comm_mean"] / max(all_static["comm_mean"], 1e-9)
            print(f"OK    comm saving sac vs all = {100*saving:.1f}%  (paper ~55%)")

        if not (sac_d3qn["train_mean"] < 0.5 * all_static["train_mean"]):
            # On short synthetic runs capacity saturation may be milder.
            warns.append(
                f"train gap weaker than paper: sac={sac_d3qn['train_mean']:.2f} "
                f"all={all_static['train_mean']:.2f} (paper ~{PAPER_TRAIN_SAC} vs ~{PAPER_TRAIN_ALL})"
            )
        else:
            print(
                f"OK    train cost all/sac ≈ "
                f"{all_static['train_mean']/max(sac_d3qn['train_mean'],1e-9):.1f}x "
                f"(paper ~95x)"
            )

        if not (sac_d3qn["cohort_mean"] < 0.85 * max(all_static["cohort_mean"], 1.0)):
            fails.append(
                f"cohort not reduced by SAC: sac={sac_d3qn['cohort_mean']:.1f} "
                f"all={all_static['cohort_mean']:.1f} (paper ~13.4 vs 30)"
            )
        else:
            print(
                f"OK    cohort sac={sac_d3qn['cohort_mean']:.1f} "
                f"< all={all_static['cohort_mean']:.1f} (paper ~13.4 vs 30)"
            )

        # Retroactive Paillier ratio (paper sec column).
        sec_all = _estimate_paillier_sec_min(all_static["cohort_mean"])
        sec_sac = _estimate_paillier_sec_min(sac_d3qn["cohort_mean"])
        print(
            f"INFO  estimated Paillier-1024 sec min/round: "
            f"all≈{sec_all:.1f}  sac≈{sec_sac:.1f}  "
            f"(paper {PAPER_SEC_ALL_MIN}/{PAPER_SEC_SAC_MIN})"
        )
        if sec_all > 0 and sec_sac / sec_all > 0.75:
            warns.append(
                f"Paillier sec ratio sac/all={sec_sac/sec_all:.2f} "
                f"(paper ~{PAPER_SEC_SAC_MIN/PAPER_SEC_ALL_MIN:.2f})"
            )

    # --- robustness plumbing (D3QN path exercises aggregators) ---
    if sac_d3qn:
        # At least one non-fedavg choice should appear if d3qn explores.
        aggs: dict[str, int] = defaultdict(int)
        for path in args.trace_dir.glob("sac__d3qn__*.jsonl"):
            for row in _load_trace(path):
                aggs[str(row.get("aggregator", "?"))] += 1
        print(f"INFO  sac+d3qn aggregator histogram: {dict(aggs)}")
        if len(aggs) < 2 and sac_d3qn["rounds"] >= 10:
            warns.append(
                "D3QN never left FedAvg — check epsilon schedule / cloud policy wiring"
            )

    # Soft accuracy note (short smokes are noisy).
    if all_static and sac_d3qn and sac_d3qn["rounds"] >= 15:
        if sac_d3qn["acc_tail"] + 0.05 < all_static["acc_tail"]:
            warns.append(
                f"sac+d3qn tail acc {sac_d3qn['acc_tail']:.3f} < "
                f"all+static {all_static['acc_tail']:.3f} "
                f"(paper {PAPER_TAIL_SAC_D3QN} vs {PAPER_TAIL_STATIC}; "
                f"short smoke may not reproduce collapse)"
            )
        else:
            print(
                f"OK    sac+d3qn tail acc {sac_d3qn['acc_tail']:.3f} "
                f"vs all+static {all_static['acc_tail']:.3f} "
                f"(within soft +0.05 margin; paper {PAPER_TAIL_SAC_D3QN}/{PAPER_TAIL_STATIC})"
            )

    if all_d3qn and sac_static:
        print(
            f"INFO  ablation cells present: all+d3qn tail={all_d3qn['acc_tail']:.3f}, "
            f"sac+static tail={sac_static['acc_tail']:.3f}"
        )

    print()
    for w in warns:
        print(f"WARN  {w}")
    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        print("\nPRE-PQC GATE: FAILED — fix wiring before Phase-1 PQ auth.")
        return 1

    print("PRE-PQC GATE: PASSED — stack matches paper orderings enough to start Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
