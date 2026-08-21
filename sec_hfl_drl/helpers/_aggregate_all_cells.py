"""Aggregate per-round costs across whatever seeds exist for every cell."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

SWEEP = Path("results/runs/sweep_headline")
CACHE = Path("_paillier_recalib_cache.json")
TAIL = 10

CELLS = [
    ("all",       "static"),
    ("all",       "round_robin"),
    ("all",       "d3qn"),
    ("random",    "static"),
    ("random",    "round_robin"),
    ("random",    "d3qn"),
    ("heuristic", "static"),
    ("heuristic", "round_robin"),
    ("heuristic", "d3qn"),
    ("sac",       "static"),
    ("sac",       "round_robin"),
    ("sac",       "d3qn"),
]


def load(fog: str, cloud: str, seed: int) -> dict | None:
    p = SWEEP / f"{fog}__{cloud}__seed{seed:04d}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    if len(rows) < 300:
        return None
    comm = np.zeros(len(rows));  train = np.zeros(len(rows));  sec_plain = np.zeros(len(rows))
    acc = np.array([float(r["accuracy"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    for i, r in enumerate(rows):
        for cb in (r.get("costs") or {}).values():
            comm[i]      += float(cb.get("comm", 0.0))
            train[i]     += float(cb.get("train", 0.0))
            sec_plain[i] += float(cb.get("sec", 0.0))
    return {
        "tail_acc":   float(np.mean(acc[-TAIL:])),
        "tail_loss":  float(np.mean(loss[-TAIL:])),
        "comm":       float(np.mean(comm)),
        "train":      float(np.mean(train)),
        "sec_plain":  float(np.mean(sec_plain)),
    }


def main() -> None:
    cache = json.loads(CACHE.read_text())
    ratio = ((cache["c_enc_p1024"] + cache["c_auth"] + cache["c_verify"]) /
             (cache["c_enc_plain"] + cache["c_auth"] + cache["c_verify"]))

    print(f"{'cell':<30} {'n':>2}  {'tail_acc':>22}  {'comm':>22}  {'train':>20}  {'sec_min':>20}")
    print("-" * 130)
    for fog, cloud in CELLS:
        all_seeds = []
        for s in range(10):
            d = load(fog, cloud, s)
            if d is not None:
                all_seeds.append(d)
        n = len(all_seeds)
        if n == 0:
            print(f"{fog+'+'+cloud:<30}  -- no data --")
            continue

        def stat(key, fmt="{:.3f}"):
            vals = [d[key] for d in all_seeds]
            if n >= 2:
                return f"{fmt.format(np.mean(vals))} \u00b1 {fmt.format(np.std(vals, ddof=1))}"
            return fmt.format(vals[0])

        sec_min_seeds = [d["sec_plain"] * ratio / 60.0 for d in all_seeds]
        if n >= 2:
            sec_str = f"{np.mean(sec_min_seeds):.1f} \u00b1 {np.std(sec_min_seeds, ddof=1):.1f}"
        else:
            sec_str = f"{sec_min_seeds[0]:.1f}"

        print(f"{fog+'+'+cloud:<30} {n:>2}  "
              f"{stat('tail_acc'):>22}  "
              f"{stat('comm', '{:,.0f}'):>22}  "
              f"{stat('train', '{:.1f}'):>20}  "
              f"{sec_str:>20}")


if __name__ == "__main__":
    main()
