"""Aggregate the 3 seeds of the headline SAC+D3QN cell.

Reports tail-window accuracy, per-round comm/train/Paillier-1024 sec costs,
and final multiplier values, as mean ± std across seeds 0/1/2.

Once seeds 1 and 2 finish, run this to produce the numbers we'll splice
into the paper's headline rows.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

SWEEP = Path("results/runs/sweep_headline")
CACHE = Path("_paillier_recalib_cache.json")
TAIL = 10
SEEDS = (0, 1, 2)


def load_seed(seed: int) -> dict | None:
    path = SWEEP / f"sac__d3qn__seed{seed:04d}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 300:
        print(f"[warn] seed {seed}: only {len(rows)} rounds (incomplete)")
        return None
    acc = np.array([float(r["accuracy"]) for r in rows])
    comm = np.zeros(len(rows))
    train = np.zeros(len(rows))
    sec_plain = np.zeros(len(rows))
    for i, r in enumerate(rows):
        c = r.get("costs") or {}
        for cb in c.values():
            comm[i]      += float(cb.get("comm", 0.0))
            train[i]     += float(cb.get("train", 0.0))
            sec_plain[i] += float(cb.get("sec", 0.0))
    final_mult = rows[-1].get("multipliers", {})
    return {
        "tail_acc": float(np.mean(acc[-TAIL:])),
        "comm":     float(np.mean(comm)),
        "train":    float(np.mean(train)),
        "sec_plain": float(np.mean(sec_plain)),
        "nu_lat":  float(final_mult.get("lat",  np.nan)),
        "nu_cap":  float(final_mult.get("cap",  np.nan)),
        "nu_priv": float(final_mult.get("priv", np.nan)),
    }


def paillier_ratio() -> float:
    c = json.loads(CACHE.read_text(encoding="utf-8"))
    return ((float(c["c_enc_p1024"]) + float(c["c_auth"]) + float(c["c_verify"])) /
            (float(c["c_enc_plain"]) + float(c["c_auth"]) + float(c["c_verify"])))


def fmt_mean_std(values: list[float], fmt: str = "{:.3f}") -> str:
    if len(values) < 2:
        return fmt.format(values[0]) + " (n=1)"
    return f"{fmt.format(np.mean(values))} \u00b1 {fmt.format(np.std(values, ddof=1))}"


def main() -> None:
    ratio = paillier_ratio()
    seed_data = {s: load_seed(s) for s in SEEDS}
    completed = [s for s, d in seed_data.items() if d is not None]
    print(f"\n=== sac+d3qn headline cell: {len(completed)}/{len(SEEDS)} seeds complete ===")
    if not completed:
        print("nothing to aggregate yet")
        return

    print(f"completed seeds: {completed}")
    print(f"Paillier-1024/plain ratio: {ratio:.4f}\n")

    for s in completed:
        d = seed_data[s]
        sec_min = d["sec_plain"] * ratio / 60.0
        print(
            f"  seed {s}: tail_acc={d['tail_acc']:.3f}  "
            f"comm={d['comm']:>9,.0f}  train={d['train']:>7.1f}  "
            f"sec={sec_min:>5.1f} min  "
            f"nu=(lat={d['nu_lat']:.2f}, cap={d['nu_cap']:.2f}, priv={d['nu_priv']:.2f})"
        )

    if len(completed) >= 2:
        print("\n--- multi-seed mean \u00b1 std ---")
        accs   = [seed_data[s]["tail_acc"] for s in completed]
        comms  = [seed_data[s]["comm"]     for s in completed]
        trains = [seed_data[s]["train"]    for s in completed]
        secs   = [seed_data[s]["sec_plain"] * ratio / 60.0 for s in completed]
        priv   = [seed_data[s]["nu_priv"]  for s in completed]
        latv   = [seed_data[s]["nu_lat"]   for s in completed]
        print(f"  tail accuracy:           {fmt_mean_std(accs)}")
        print(f"  comm cost (bytes/round): {fmt_mean_std(comms, '{:,.0f}')}")
        print(f"  train cost (units):      {fmt_mean_std(trains, '{:.1f}')}")
        print(f"  Paillier-1024 sec (min): {fmt_mean_std(secs, '{:.1f}')}")
        print(f"  nu_priv (final):         {fmt_mean_std(priv, '{:.2f}')}")
        print(f"  nu_lat  (final):         {fmt_mean_std(latv, '{:.2f}')}")
        print()
        print("--- paper-ready summary line ---")
        print(f"Across {len(completed)} seeds, tail accuracy = "
              f"{np.mean(accs):.3f} \u00b1 {np.std(accs, ddof=1):.3f}, "
              f"comm = {np.mean(comms)/1000:.0f}K bytes/round, "
              f"nu_priv = {np.mean(priv):.2f} \u00b1 {np.std(priv, ddof=1):.2f}.")


if __name__ == "__main__":
    main()
