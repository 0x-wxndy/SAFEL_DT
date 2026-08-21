"""3-panel "participation probability for honest vs malicious clients"
figure, refreshed with the new 300-round mixed-stepwise headline sweep.

Same layout as the original Figure 3 (paper/assets/weight_evolution.png)
but driven from the post-fix sweep (sweep_headline/), so the rate_M /
rate_B numbers reflect the current SAC+D3QN configuration instead of
the broken nu_priv-saturated run.

Each panel plots, per round, the *mean selection probability* of the
honest cohort vs the (then-active) malicious cohort. The horizontal
shading marks the four-phase mixed-stepwise attack schedule
(10% -> 15% -> 20% -> 25% malicious population fraction).

To make trends visible against round-to-round noise, both lines are
smoothed with a 10-round centred moving average; the unsmoothed traces
are also drawn as faint background series.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLIENTS = 30
N_FOGS = 3
CLIENTS_PER_FOG = N_CLIENTS // N_FOGS
T = 300

CELLS = [
    ("all__d3qn__seed0000.jsonl",       "all + D3QN (no fog selection)"),
    ("heuristic__d3qn__seed0000.jsonl", "heuristic + D3QN"),
    ("sac__d3qn__seed0000.jsonl",       "SAFEL-DT (SAC + D3QN)"),
]

PHASES = [
    (  0, 100, "10% mal"),
    (100, 200, "15% mal"),
    (200, 250, "20% mal"),
    (250, 300, "25% mal"),
]


def selection_per_round(trace: Path) -> tuple[np.ndarray, list[int], list[set[int]]]:
    """Return (sel matrix [30, T], malicious_ids, active_malicious_sets per round)."""
    rows = trace.read_text(encoding="utf-8").splitlines()
    sel = np.zeros((N_CLIENTS, T), dtype=np.uint8)
    mal_ids: list[int] = []
    active_sets: list[set[int]] = []
    for r_idx, line in enumerate(rows[:T]):
        d = json.loads(line)
        if r_idx == 0:
            mal_ids = list(int(x) for x in (d.get("malicious_ids") or []))
        active_sets.append(set(int(x) for x in (d.get("active_malicious_ids") or [])))
        spf = d.get("selected_per_fog")
        if spf is None:
            sel[:, r_idx] = 1
            continue
        for fog_key, locals_list in spf.items():
            fog_id = int(fog_key)
            if locals_list is None:
                continue
            for li in locals_list:
                gid = fog_id * CLIENTS_PER_FOG + int(li)
                if 0 <= gid < N_CLIENTS:
                    sel[gid, r_idx] = 1
    return sel, mal_ids, active_sets


def smooth(x: np.ndarray, w: int = 10) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def main() -> None:
    panels: list[tuple[str, np.ndarray, list[int], list[set[int]]]] = []
    for fname, label in CELLS:
        path = SWEEP / fname
        if not path.exists():
            print(f"[skip] missing {fname}")
            continue
        sel, mal_ids, act = selection_per_round(path)
        panels.append((label, sel, mal_ids, act))
        print(f"[loaded] {fname}: sel={sel.shape}, mal={mal_ids}")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(14.0, 4.8),
        sharey=True, gridspec_kw={"wspace": 0.07},
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, sel, mal_ids, act) in zip(axes, panels):
        rate_m = np.zeros(T, dtype=np.float64)
        rate_b = np.zeros(T, dtype=np.float64)
        for r_idx in range(T):
            active_mal = act[r_idx]
            benign = [g for g in range(N_CLIENTS) if g not in active_mal]
            if active_mal:
                rate_m[r_idx] = float(np.mean([sel[g, r_idx] for g in active_mal]))
            else:
                rate_m[r_idx] = np.nan
            rate_b[r_idx] = float(np.mean([sel[g, r_idx] for g in benign])) if benign else np.nan

        for lo, hi, lbl in PHASES:
            shade = "#fff4e6" if lo < 100 else ("#fde0c2" if lo < 200 else
                                                ("#fab784" if lo < 250 else "#e88847"))
            ax.axvspan(lo, hi, color=shade, alpha=0.55, zorder=0)
            ax.text((lo + hi) / 2.0, 0.95, lbl, ha="center", va="top",
                    fontsize=8.0, color="#333",
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1))

        ax.plot(np.arange(T), rate_b, color="#1f77b4", alpha=0.18, linewidth=0.7, zorder=2)
        ax.plot(np.arange(T), rate_m, color="#d62728", alpha=0.18, linewidth=0.7, zorder=2)
        ax.plot(np.arange(T), smooth(rate_b), color="#1f4f8b", linestyle="-",
                linewidth=1.9, label="Honest", zorder=4)
        ax.plot(np.arange(T), smooth(rate_m), color="#9a1e1e", linestyle="--",
                linewidth=1.9, label="Malicious", zorder=4)

        m_mean = float(np.nanmean(rate_m))
        b_mean = float(np.nanmean(rate_b))
        ratio = m_mean / b_mean if b_mean > 0 else float("nan")
        ax.text(
            0.02, 0.04,
            f"mean rate$_M$={m_mean:.2f}\n"
            f"mean rate$_B$={b_mean:.2f}\n"
            f"$\\mathrm{{rate}}_M/\\mathrm{{rate}}_B$={ratio:.2f}",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, pad=3),
        )

        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Round")
        ax.set_xticks([0, 100, 200, 300])
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Mean per-round selection probability")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.85)

    fig.suptitle(
        "Per-round selection probability of honest vs. malicious clients "
        "(N-BaIoT, $N=30$, $T=300$, mixed-stepwise schedule).\n"
        "Lower $\\mathrm{rate}_M/\\mathrm{rate}_B$ means stronger fog-level "
        "discrimination against malicious clients.",
        fontsize=10.5, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))

    out_png = OUT_DIR / "selection_probability.png"
    out_pdf = OUT_DIR / "selection_probability.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")


if __name__ == "__main__":
    main()
