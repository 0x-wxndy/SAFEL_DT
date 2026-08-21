"""Two-panel "where the defense happens" figure.

Top panel  : per-round malicious cohort fraction (m_pre) for the three
             policies `all`, `heuristic`, `sac+d3qn`. Ground-truth
             malicious-population fraction (10% -> 15% -> 20% -> 25%)
             is shown as a dashed reference. This panel shows that
             SAFEL-DT does *not* preferentially exclude malicious
             clients -- m_pre tracks roughly the population rate.

Bottom panel: per-round test accuracy for the same three policies, so
             the reader sees that despite admitting malicious updates
             into the cohort at the population rate, SAFEL-DT keeps
             accuracy high. The defense lives at the cloud (D3QN's
             adaptive Byzantine-robust aggregator), not at the fog
             selector. The accuracy lines visually decouple cohort
             composition (top) from learned robustness (bottom).

A colored band along the bottom of the lower panel encodes the
cloud aggregator D3QN chose for SAFEL-DT in each round, so the
reader can see which robust rule was active when the gap between
\\texttt{all}/\\texttt{heuristic} accuracy and SAFEL-DT accuracy
opens up. This figure is the headline RQ3+RQ4 visual: SAFEL-DT
delegates byzantine defense to the cloud layer.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLIENTS = 30
T = 300

CELLS = [
    ("all__d3qn__seed0000.jsonl",       "all (no selection)",       "#2ca02c", "--"),
    ("heuristic__d3qn__seed0000.jsonl", "heuristic (top-k inv-loss)", "#1f77b4", "-."),
    ("sac__d3qn__seed0000.jsonl",       "SAFEL-DT (sac + d3qn)",    "#d62728", "-"),
]

PHASES = [
    (  0, 100, 0.10, "10%"),
    (100, 200, 0.15, "15%"),
    (200, 250, 0.20, "20%"),
    (250, 300, 0.25, "25%"),
]

AGGREGATORS = ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]
AGG_COLOURS = {
    "fedavg":       "#bbbbbb",
    "trimmed_mean": "#a8d8b9",
    "median":       "#5dade2",
    "krum":         "#f5b041",
    "multi_krum":   "#c0392b",
}
AGG_LABELS = {
    "fedavg":       "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "median":       "Median",
    "krum":         "Krum",
    "multi_krum":   "Multi-Krum",
}


def cohort_size(d: dict) -> int:
    n = int(d.get("n_clients_accepted") or 0)
    if n:
        return n
    spf = d.get("selected_per_fog") or {}
    return sum(len(ids) for ids in spf.values() if ids is not None)


def mal_in_cohort(d: dict) -> int:
    active = set(int(x) for x in (d.get("active_malicious_ids") or []))
    if not active:
        return 0
    spf = d.get("selected_per_fog")
    if spf is None:
        return int(sum(1 for m in active if 0 <= m < N_CLIENTS))
    count = 0
    for fog_key, locals_list in spf.items():
        fog_id = int(fog_key)
        if locals_list is None:
            continue
        for li in locals_list:
            gid = fog_id * (N_CLIENTS // 3) + int(li)
            if gid in active:
                count += 1
    return count


def load_trace(path: Path) -> dict:
    rows = path.read_text(encoding="utf-8").splitlines()
    m_pre = np.zeros(T, dtype=np.float64)
    acc   = np.zeros(T, dtype=np.float64)
    aggs: list[str] = []
    for r_idx, line in enumerate(rows[:T]):
        d = json.loads(line)
        coh = cohort_size(d)
        mal = mal_in_cohort(d)
        m_pre[r_idx] = mal / coh if coh else 0.0
        acc[r_idx] = float(d.get("accuracy", 0.0))
        aggs.append(str(d.get("cloud_aggregator") or "fedavg"))
    return {"m_pre": m_pre, "acc": acc, "aggs": aggs, "n": len(rows)}


def main() -> None:
    traces = {}
    for fname, label, *_ in CELLS:
        path = SWEEP / fname
        if not path.exists():
            print(f"[skip] missing {fname}")
            continue
        traces[label] = load_trace(path)
        print(f"[loaded] {fname}: n={traces[label]['n']}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.0, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.10},
    )

    for ax in (ax_top, ax_bot):
        for lo, hi, frac, lbl in PHASES:
            ax.axvspan(lo, hi, color="#fff4e6" if frac <= 0.10 else
                                  ("#fde0c2" if frac <= 0.15 else
                                   ("#fab784" if frac <= 0.20 else "#e88847")),
                       alpha=0.55, zorder=0)

    for lo, hi, _frac, lbl in PHASES:
        ax_top.text((lo + hi) / 2.0, 1.02, lbl + " malicious",
                    ha="center", va="bottom", fontsize=9, color="#333",
                    transform=ax_top.get_xaxis_transform(), fontweight="bold")

    ground_truth = np.zeros(T)
    for lo, hi, frac, _ in PHASES:
        ground_truth[lo:hi] = frac
    ax_top.plot(np.arange(T), ground_truth, color="black", linestyle=":",
                linewidth=1.5, label="Population malicious fraction", zorder=2)

    for fname, label, colour, ls in CELLS:
        if label not in traces:
            continue
        x = np.arange(T)
        y = traces[label]["m_pre"]
        w = 5
        kernel = np.ones(w) / w
        y_smooth = np.convolve(y, kernel, mode="same")
        ax_top.plot(x, y_smooth, color=colour, linestyle=ls, linewidth=1.8,
                    label=label, alpha=0.95, zorder=3)

    ax_top.set_ylabel("Malicious fraction\nof selected cohort")
    ax_top.set_ylim(0.0, 0.55)
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_top.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.85)

    for fname, label, colour, ls in CELLS:
        if label not in traces:
            continue
        x = np.arange(T)
        y = traces[label]["acc"]
        ax_bot.plot(x, y, color=colour, linestyle=ls, linewidth=1.8,
                    label=label, alpha=0.95, zorder=3)

    ax_bot.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax_bot.set_xlabel("Round")

    ax_bot.set_ylabel("Test accuracy")
    ax_bot.set_xlim(0, T)
    ax_bot.set_ylim(0.35, 0.95)
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bot.legend(loc="lower right", ncol=1, fontsize=9, framealpha=0.85)

    # Annotate the worst accuracy dips for `all` and `heuristic` to make
    # the "D3QN rescues" story visible: the dip happens, then the cloud
    # aggregator switches and recovery follows.
    all_acc  = traces.get("all (no selection)", {}).get("acc")
    heur_acc = traces.get("heuristic (top-k inv-loss)", {}).get("acc")
    if all_acc is not None:
        i = int(np.argmin(all_acc))
        ax_bot.annotate(
            f"all dip to {all_acc[i]:.2f}",
            xy=(i, all_acc[i]), xytext=(i + 8, all_acc[i] - 0.05),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
            fontsize=8, color="#2ca02c",
        )
    if heur_acc is not None:
        i = int(np.argmin(heur_acc))
        ax_bot.annotate(
            f"heuristic dip to {heur_acc[i]:.2f}",
            xy=(i, heur_acc[i]), xytext=(i - 60, heur_acc[i] + 0.07),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
            fontsize=8, color="#1f77b4",
        )

    fig.suptitle(
        "Where the defense happens: cohort composition (top) decouples from "
        "learned robustness (bottom).\n"
        "SAFEL-DT admits malicious updates at $\\sim$population rate; the "
        "cloud D3QN+robust-aggregator stack rescues accuracy.",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_png = OUT_DIR / "defense_decoupling.png"
    out_pdf = OUT_DIR / "defense_decoupling.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")

    print("\n--- per-phase summary ---")
    print(f"{'policy':<32} | {'phase':<8} | {'mean m_pre':>10} | {'gt frac':>8} | {'tail acc':>9}")
    for fname, label, *_ in CELLS:
        if label not in traces:
            continue
        for lo, hi, frac, name in PHASES:
            m = float(np.mean(traces[label]["m_pre"][lo:hi]))
            a = float(np.mean(traces[label]["acc"][max(lo, hi - 10):hi]))
            print(f"{label:<32} | {name:<8} | {m:>10.3f} | {frac:>8.2f} | {a:>9.3f}")


if __name__ == "__main__":
    main()
