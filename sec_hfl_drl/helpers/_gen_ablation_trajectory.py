"""Ablation trajectory: 3-panel time-series of the four ablation cells.

For each of:
  - Vanilla FedAvg (all + static)         : no SAC, no D3QN
  - - SAC          (all + d3qn)           : D3QN cloud only
  - - D3QN         (sac + static)         : SAC fog only
  - Full SAFEL-DT  (sac + d3qn)           : both

we plot, over the 300 rounds of the headline mixed-stepwise sweep:

  Top    : per-round test accuracy            (smoothed, w=5)
  Middle : per-round cloud reward             (smoothed, w=10)
  Bottom : cumulative communication cost      (raw running sum)

The four phases (10%, 15%, 20%, 25% adversarial) are shaded in the
background, so the reader can see exactly when the static-cloud cells
collapse (around round 100 when the malicious fraction crosses 10%)
and where the cumulative-cost slope of '- SAC' / Vanilla pulls away
from SAFEL-DT / '- D3QN'.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
T = 300

CELLS = [
    # (label,                         fog,   cloud,    colour,    linestyle)
    ("Vanilla FedAvg",                "all",   "static", "#7f7f7f", ":"),
    ("$-$ SAC (D3QN only)",           "all",   "d3qn",   "#2ca02c", "--"),
    ("$-$ D3QN (SAC only)",           "sac",   "static", "#ff7f0e", "-."),
    ("Full SAFEL-DT (SAC + D3QN)",    "sac",   "d3qn",   "#d62728", "-"),
]

PHASES = [
    (  0, 100, "10% mal"),
    (100, 200, "15% mal"),
    (200, 250, "20% mal"),
    (250, 300, "25% mal"),
]


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def load_cell(fog: str, cloud: str) -> dict | None:
    path = SWEEP / f"{fog}__{cloud}__seed0000.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    n = min(len(rows), T)
    acc = np.array([float(r["accuracy"]) for r in rows[:n]])
    comm_round = np.zeros(n, dtype=np.float64)
    reward = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(rows[:n]):
        c = r.get("costs") or {}
        for cb in c.values():
            comm_round[i] += float(cb.get("comm", 0.0))
        cr = r.get("cloud_reward")
        if cr is None:
            pfr = r.get("per_fog_reward") or {}
            cr = float(sum(float(v) for v in pfr.values())) if pfr else 0.0
        reward[i] = float(cr)
    comm_cum = np.cumsum(comm_round)
    return {"acc": acc, "reward": reward, "comm_cum": comm_cum, "n": n}


def main() -> None:
    cells: list[tuple[str, str, str, dict]] = []
    for label, fog, cloud, colour, ls in CELLS:
        d = load_cell(fog, cloud)
        if d is None:
            print(f"[skip] {fog}+{cloud}")
            continue
        cells.append((label, colour, ls, d))
        print(f"[loaded] {fog}+{cloud}: n={d['n']}, final acc={d['acc'][-1]:.3f}, "
              f"cum comm={d['comm_cum'][-1]:.2e}, mean reward={d['reward'].mean():.3f}")

    fig, (ax_acc, ax_rew, ax_cost) = plt.subplots(
        3, 1, figsize=(11.0, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.10},
    )

    for ax in (ax_acc, ax_rew, ax_cost):
        for lo, hi, _lbl in PHASES:
            shade = "#fff4e6" if lo < 100 else ("#fde0c2" if lo < 200 else
                                                ("#fab784" if lo < 250 else "#e88847"))
            ax.axvspan(lo, hi, color=shade, alpha=0.45, zorder=0)

    for lo, hi, lbl in PHASES:
        ax_acc.text((lo + hi) / 2.0, 1.04, lbl, ha="center", va="bottom",
                    fontsize=9, color="#333", transform=ax_acc.get_xaxis_transform(),
                    fontweight="bold")

    for label, colour, ls, d in cells:
        ax_acc.plot(np.arange(d["n"]), smooth(d["acc"], 5),
                    color=colour, linestyle=ls, linewidth=1.9,
                    label=label, alpha=0.95, zorder=3)
        ax_rew.plot(np.arange(d["n"]), smooth(d["reward"], 10),
                    color=colour, linestyle=ls, linewidth=1.9,
                    label=label, alpha=0.95, zorder=3)
        ax_cost.plot(np.arange(d["n"]), d["comm_cum"] / 1e6,
                     color=colour, linestyle=ls, linewidth=1.9,
                     label=label, alpha=0.95, zorder=3)

    ax_acc.set_ylabel("Test accuracy\n(smoothed, w=5)")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_acc.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9)

    ax_rew.set_ylabel("Cloud reward\n(smoothed, w=10)")
    ax_rew.axhline(0, color="black", linewidth=0.8, alpha=0.6, linestyle="-", zorder=2)
    ax_rew.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax_cost.set_ylabel("Cumulative comm. cost\n($\\times 10^6$ transmission units)")
    ax_cost.set_xlabel("Round")
    ax_cost.set_xlim(0, T)
    ax_cost.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_cost.set_xticks([0, 50, 100, 150, 200, 250, 300])

    sorted_cells = sorted(cells, key=lambda c: c[3]["comm_cum"][-1] / 1e6, reverse=True)
    n = len(sorted_cells)
    y_min = min(c[3]["comm_cum"][-1] / 1e6 for c in cells)
    y_max = max(c[3]["comm_cum"][-1] / 1e6 for c in cells)
    overlap_eps = (y_max - y_min) * 0.06
    placed: list[float] = []
    for i, (label, colour, ls, d) in enumerate(sorted_cells):
        y = d["comm_cum"][-1] / 1e6
        for py in placed:
            if abs(py - y) < overlap_eps:
                y = py - overlap_eps if y < py else py + overlap_eps
        placed.append(y)
        ax_cost.text(
            T + 4, y,
            f"{d['comm_cum'][-1] / 1e6:.0f}M",
            color=colour, fontsize=9, va="center", fontweight="bold",
        )

    fig.suptitle(
        "Component ablation over time (N-BaIoT, $N=30$, $T=300$, mixed-stepwise attack).\n"
        "SAFEL-DT and $-$SAC keep accuracy high; SAFEL-DT and $-$D3QN keep cumulative cost low; "
        "only SAFEL-DT does both.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))

    out_png = OUT_DIR / "ablation_trajectory.png"
    out_pdf = OUT_DIR / "ablation_trajectory.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")


if __name__ == "__main__":
    main()
