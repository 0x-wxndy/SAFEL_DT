"""Generate aggregator-share-by-phase figure from the headline sweep.

Reads any *__d3qn__seed*.jsonl cell that is complete (>=300 rounds),
pools across them, and renders a stacked bar chart with the stepwise
malicious-fraction phases used in the headline sweep:
    Phase 1: rounds   1-100  (10% malicious)
    Phase 2: rounds 101-200  (15% malicious)
    Phase 3: rounds 201-250  (20% malicious)
    Phase 4: rounds 251-300  (25% malicious)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP   = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ROUNDS = 300

PHASES = [
    (  0, 100, "10%\nadversarial\n(r1-100)"),
    (100, 200, "15%\nadversarial\n(r101-200)"),
    (200, 250, "20%\nadversarial\n(r201-250)"),
    (250, 300, "25%\nadversarial\n(r251-300)"),
]
AGGREGATORS = ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]
AGG_LABELS  = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "median": "Median",
    "krum": "Krum",
    "multi_krum": "Multi-Krum",
}


def load_d3qn_traces() -> list[tuple[str, list[str]]]:
    """Return [(cell_name, per_round_aggregators), ...] for complete d3qn cells."""
    out = []
    for trace in sorted(SWEEP.glob("*__d3qn__seed*.jsonl")):
        rows = trace.read_text(encoding="utf-8").splitlines()
        if len(rows) < TARGET_ROUNDS:
            continue
        aggs = [json.loads(r).get("cloud_aggregator", "unknown") for r in rows[:TARGET_ROUNDS]]
        out.append((trace.stem, aggs))
    return out


def main() -> None:
    cells = load_d3qn_traces()
    if not cells:
        raise SystemExit("no complete d3qn cells found")
    print(f"found {len(cells)} complete d3qn cell(s):")
    for name, _ in cells:
        print(f"  - {name}")

    agg_to_idx = {a: i for i, a in enumerate(AGGREGATORS)}
    counts = np.zeros((len(PHASES), len(AGGREGATORS)), dtype=np.float64)
    totals = np.zeros(len(PHASES), dtype=np.float64)
    for _, aggs in cells:
        for r_idx, a in enumerate(aggs):
            for p_idx, (lo, hi, _lbl) in enumerate(PHASES):
                if lo <= r_idx < hi:
                    counts[p_idx, agg_to_idx.get(a, 0)] += 1.0
                    totals[p_idx] += 1.0
                    break
    shares = counts / np.maximum(totals[:, None], 1.0)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    cmap = plt.get_cmap("tab10", len(AGGREGATORS))
    bottom = np.zeros(len(PHASES))
    for ai, agg in enumerate(AGGREGATORS):
        ax.bar(
            np.arange(len(PHASES)), shares[:, ai],
            bottom=bottom, label=AGG_LABELS[agg], color=cmap(ai),
            edgecolor="white", linewidth=0.8,
        )
        for p_idx in range(len(PHASES)):
            v = shares[p_idx, ai]
            if v >= 0.05:
                ax.text(
                    p_idx, bottom[p_idx] + v / 2.0,
                    f"{v * 100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold",
                )
        bottom += shares[:, ai]

    ax.axhline(1.0 / len(AGGREGATORS), color="black", linestyle=":",
               linewidth=1.0, alpha=0.6, label=f"Uniform ({100 // len(AGGREGATORS)}%)")
    ax.set_xticks(np.arange(len(PHASES)))
    ax.set_xticklabels([p[2] for p in PHASES], fontsize=9)
    ax.set_ylabel("Share of rounds")
    ax.set_ylim(0, 1.0)
    n_cells = len(cells)
    cell_word = "cell" if n_cells == 1 else "cells"
    ax.set_title(
        f"D3QN aggregator share by attack-intensity phase\n"
        f"(mixed-stepwise attack; pooled across {n_cells} D3QN {cell_word}, "
        f"{n_cells * TARGET_ROUNDS} rounds total)",
        fontsize=11,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_png = OUT_DIR / "aggregator_share_phases.png"
    out_pdf = OUT_DIR / "aggregator_share_phases.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")

    print("\n--- numeric shares (%) ---")
    print(f"{'phase':<26} | " + " | ".join(f"{AGG_LABELS[a]:>13s}" for a in AGGREGATORS))
    for p_idx, (_, _, lbl) in enumerate(PHASES):
        clean_lbl = lbl.replace("\n", " ")
        print(f"{clean_lbl:<26} | " +
              " | ".join(f"{shares[p_idx, ai] * 100:>12.1f}%" for ai in range(len(AGGREGATORS))))


if __name__ == "__main__":
    main()
