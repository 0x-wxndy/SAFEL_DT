"""Lagrangian dual trajectories (primal-dual constraint tracking).

For the headline SAFEL-DT cell (sac+d3qn, seed 0, 300 rounds,
mixed-stepwise attack) we plot the three dual variables computed by
the simulator's dual-ascent loop:

  - nu_lat  : latency  constraint
  - nu_cap  : capacity constraint
  - nu_priv : privacy  constraint

The story per multiplier:
  - nu_priv climbs smoothly 0.008 -> 4.96 within the nu_max=10
    ceiling -- privacy constraint becomes more binding as cohort
    scales and the malicious fraction escalates, but the dual stays
    bounded (so the primal-dual algorithm is converging, not
    saturating).
  - nu_lat  climbs 0.01 -> 2.37 -- latency constraint also becomes
    binding as more clients participate per round.
  - nu_cap  stays flat at 0   -- per-fog cohort cap K_max=5 keeps
    g_cap <= 0 for every feasible action; capacity is not binding
    in this configuration. The flat line is the framework's
    "constraint is automatically respected" signal, not a bug.

The shaded background marks the four-phase attack escalation;
the dotted line at nu_max=10 marks the cap (set by Lagrangian
config). The figure directly supports Proposition 2 of
Sec.~\\ref{sec:framework} by showing all duals stay within the
projected feasible region.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE = SWEEP / "sac__d3qn__seed0000.jsonl"
NU_MAX = 10.0
T = 300

PHASES = [
    (  0, 100, "10% mal"),
    (100, 200, "15% mal"),
    (200, 250, "20% mal"),
    (250, 300, "25% mal"),
]

MULTS = [
    ("lat",  r"$\nu_{\mathrm{lat}}$ (latency)",  "#1f77b4", "-"),
    ("priv", r"$\nu_{\mathrm{priv}}$ (privacy)", "#d62728", "-"),
    ("cap",  r"$\nu_{\mathrm{cap}}$ (capacity)", "#2ca02c", "--"),
]


def main() -> None:
    rows = [json.loads(l) for l in TRACE.read_text(encoding="utf-8").splitlines()]
    n = min(len(rows), T)
    nu = {k: np.zeros(n, dtype=np.float64) for k, _, _, _ in MULTS}
    for i, r in enumerate(rows[:n]):
        m = r.get("multipliers") or {}
        for key, _, _, _ in MULTS:
            nu[key][i] = float(m.get(key, 0.0))

    fig, ax = plt.subplots(figsize=(10.0, 4.6))

    for lo, hi, lbl in PHASES:
        shade = "#fff4e6" if lo < 100 else ("#fde0c2" if lo < 200 else
                                            ("#fab784" if lo < 250 else "#e88847"))
        ax.axvspan(lo, hi, color=shade, alpha=0.55, zorder=0)
        ax.text((lo + hi) / 2.0, 0.88, lbl, ha="center", va="top", fontsize=9,
                color="#333", transform=ax.get_xaxis_transform(), fontweight="bold")

    ax.axhline(NU_MAX, color="black", linestyle=":", linewidth=1.0, alpha=0.5,
               label=rf"$\nu_{{\max}}={int(NU_MAX)}$ (cap)")
    ax.text(T - 6, NU_MAX - 0.25, rf"$\nu_{{\max}}={int(NU_MAX)}$", fontsize=9,
            color="#444", ha="right", va="top")

    for key, label, colour, ls in MULTS:
        ax.plot(np.arange(n), nu[key], color=colour, linestyle=ls,
                linewidth=2.0, label=label, alpha=0.95, zorder=3)

    for key, _, colour, _ in MULTS:
        v_final = nu[key][-1]
        ax.text(n + 4, v_final, f"{v_final:.2f}",
                color=colour, fontsize=10, va="center", fontweight="bold")

    ax.annotate(
        r"$\nu_{\mathrm{cap}}\equiv 0$: capacity not binding under $K_{\max}=5$",
        xy=(150, 0.0), xytext=(150, 1.6),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1, alpha=0.7),
        ha="center", fontsize=9, color="#2ca02c",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel(r"Lagrangian dual multiplier $\nu$")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.2, NU_MAX * 1.05)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    leg_handles = [
        plt.Line2D([0], [0], color=MULTS[0][2], lw=2.0, label=MULTS[0][1]),
        plt.Line2D([0], [0], color=MULTS[1][2], lw=2.0, label=MULTS[1][1]),
        plt.Line2D([0], [0], color=MULTS[2][2], lw=2.0, ls="--", label=MULTS[2][1]),
        plt.Line2D([0], [0], color="black",     lw=1.0, ls=":",  label=rf"$\nu_{{\max}}={int(NU_MAX)}$ (cap)"),
    ]
    ax.legend(handles=leg_handles, loc="center left", fontsize=9,
              framealpha=0.92, ncol=1)
    ax.set_title(
        "Primal--dual constraint tracking (SAFEL-DT, $N=30$, $T=300$, mixed-stepwise schedule). "
        r"$\nu_{\mathrm{priv}}$ and $\nu_{\mathrm{lat}}$ rise monotonically as the cohort and "
        r"attack scale, both bounded well below the cap; $\nu_{\mathrm{cap}}$ stays at zero "
        "because the cohort cap keeps capacity slack non-negative.",
        fontsize=10, wrap=True,
    )
    fig.tight_layout()

    out_png = OUT_DIR / "multiplier_trajectory_new.png"
    out_pdf = OUT_DIR / "multiplier_trajectory_new.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")

    print("\n--- final multiplier values ---")
    for key, label, _, _ in MULTS:
        print(f"  {label:<30s}: {nu[key][-1]:.4f}  (max during run: {nu[key].max():.4f})")
    print(f"  cap utilization (nu_priv / nu_max): {nu['priv'][-1] / NU_MAX * 100:.1f}%")


if __name__ == "__main__":
    main()
