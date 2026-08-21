"""Clean 2x2 grouped bar chart for the component ablation.

Four anchor cells of the headline sweep:
  - Vanilla FedAvg (all + static)   : no SAC, no D3QN
  - - SAC          (all + d3qn)     : D3QN cloud only
  - - D3QN         (sac + static)   : SAC fog only
  - Full SAFEL-DT  (sac + d3qn)     : both

Four metrics, one panel each:
  - Tail accuracy (last-10-rounds mean; linear axis)
  - Per-round comm cost                (log axis)
  - Per-round train cost               (log axis -- spans 30 -> 3000)
  - Per-round Paillier-1024 sec cost   (linear, in minutes)

Each bar is annotated with its numeric value, and the SAFEL-DT bar is
red so the reader sees instantly that it wins (or ties for cheapest)
on every cost axis and is on the accuracy plateau with - SAC.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = Path("_paillier_recalib_cache.json")
TAIL = 10

CELLS = [
    # (short label,           long label,                              fog,    cloud,    fill,       edge)
    ("Vanilla\nFedAvg",       "Vanilla FedAvg (no SAC, no D3QN)",      "all",   "static", "#b9bfc6", "#5e6770"),
    ("$-$ SAC\n(D3QN only)",  "D3QN cloud only (no SAC)",              "all",   "d3qn",   "#a8c5dc", "#3a6a8d"),
    ("$-$ D3QN\n(SAC only)",  "SAC fog only (no D3QN)",                "sac",   "static", "#e5be8a", "#8a5a1f"),
    ("Full\nSAFEL-DT",        "Full SAFEL-DT (SAC + D3QN)",            "sac",   "d3qn",   "#3d6b8a", "#1f3d52"),
]


def _load_seed(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 100:
        return None
    acc = np.array([float(r["accuracy"]) for r in rows])
    comm = np.zeros(len(rows))
    train = np.zeros(len(rows))
    sec_plain = np.zeros(len(rows))
    for i, r in enumerate(rows):
        for cb in (r.get("costs") or {}).values():
            comm[i]      += float(cb.get("comm", 0.0))
            train[i]     += float(cb.get("train", 0.0))
            sec_plain[i] += float(cb.get("sec", 0.0))
    return {
        "tail_acc":  float(np.mean(acc[-TAIL:])),
        "comm":      float(np.mean(comm)),
        "train":     float(np.mean(train)),
        "sec_plain": float(np.mean(sec_plain)),
    }


def load_cell(fog: str, cloud: str) -> dict | None:
    """Load every available seed of the headline cell and aggregate.

    Returns mean / std for each metric; std is None when only one seed exists.
    """
    seeds = []
    for s in range(10):
        d = _load_seed(SWEEP / f"{fog}__{cloud}__seed{s:04d}.jsonl")
        if d is not None:
            seeds.append(d)
    if not seeds:
        return None
    def agg(key):
        vals = [d[key] for d in seeds]
        if len(vals) >= 2:
            return float(np.mean(vals)), float(np.std(vals, ddof=1))
        return float(vals[0]), None
    acc_m, acc_s = agg("tail_acc")
    cm_m,  cm_s  = agg("comm")
    tr_m,  tr_s  = agg("train")
    sp_m,  sp_s  = agg("sec_plain")
    return {
        "n_seeds":      len(seeds),
        "tail_acc":     acc_m, "tail_acc_std":  acc_s,
        "comm":         cm_m,  "comm_std":      cm_s,
        "train":        tr_m,  "train_std":     tr_s,
        "sec_plain":    sp_m,  "sec_plain_std": sp_s,
    }


def paillier_ratio() -> float:
    c = json.loads(CACHE.read_text(encoding="utf-8"))
    return ((float(c["c_enc_p1024"]) + float(c["c_auth"]) + float(c["c_verify"])) /
            (float(c["c_enc_plain"]) + float(c["c_auth"]) + float(c["c_verify"])))


def fmt_thousands(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.0f}K"
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def main() -> None:
    ratio = paillier_ratio()
    vals = []
    for _, _, fog, cloud, _, _ in CELLS:
        d = load_cell(fog, cloud)
        if d is None:
            raise SystemExit(f"missing cell: {fog}+{cloud}")
        d["sec_min"]     = d["sec_plain"] * ratio / 60.0
        d["sec_min_std"] = (d["sec_plain_std"] * ratio / 60.0
                            if d["sec_plain_std"] is not None else None)
        vals.append(d)

    labels    = [c[0] for c in CELLS]
    fills     = [c[4] for c in CELLS]
    edges     = [c[5] for c in CELLS]
    text_dark = [c[5] for c in CELLS]
    x = np.arange(len(CELLS))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#3d3d3d",
        "axes.labelcolor": "#2a2a2a",
        "axes.titlecolor": "#1f3d52",
        "xtick.color": "#3d3d3d",
        "ytick.color": "#3d3d3d",
        "axes.linewidth": 0.9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.24})
    fig.patch.set_facecolor("white")

    panels = [
        ("Tail test accuracy", "(higher is better)",
         [v["tail_acc"] for v in vals], [v["tail_acc_std"] for v in vals],
         None, lambda v: f"{v:.3f}", (0, 1.0)),
        ("Per-round comm cost", "(lower is better; log scale)",
         [v["comm"] for v in vals], [v["comm_std"] for v in vals],
         "log", lambda v: fmt_thousands(v), None),
        ("Per-round train cost", "(lower is better; log scale)",
         [v["train"] for v in vals], [v["train_std"] for v in vals],
         "log", lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}", None),
        ("Per-round Paillier-1024 sec cost (min)", "(lower is better)",
         [v["sec_min"] for v in vals], [v["sec_min_std"] for v in vals],
         None, lambda v: f"{v:.1f}", (0, max(v["sec_min"] for v in vals) * 1.22)),
    ]

    for ax, (title, sub, values, stds, yscale, fmt, ylim) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fafbfc")
        bars = ax.bar(
            x, values, color=fills, edgecolor=edges, linewidth=1.0,
            width=0.70, zorder=3,
        )
        err_x = [bars[i].get_x() + bars[i].get_width() / 2.0 for i in range(len(bars))]
        err_y = [v if v > 0 else 1e-9 for v in values]
        err_lo = [s if s is not None else 0.0 for s in stds]
        err_hi = list(err_lo)
        if any(s is not None for s in stds):
            ax.errorbar(
                err_x, err_y, yerr=[err_lo, err_hi],
                fmt="none", ecolor="#1f3d52", elinewidth=1.4,
                capsize=4.5, capthick=1.4, zorder=5,
            )
        if yscale == "log":
            ax.set_yscale("log")

        if ylim is not None:
            ax.set_ylim(*ylim)

        if yscale == "log":
            vmax = max(values)
            for rect, v, c_text in zip(bars, values, text_dark):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v * 1.22,
                    fmt(v),
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=c_text,
                )
            ax.set_ylim(top=vmax * 4.5)
        else:
            vmax = max(values) if max(values) > 0 else 1.0
            for rect, v, c_text in zip(bars, values, text_dark):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v + vmax * 0.03,
                    fmt(v),
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=c_text,
                )

        ax.set_title(title, fontsize=11, fontweight="bold", color="#1f3d52", pad=8)
        ax.text(0.5, -0.34, sub, ha="center", va="top",
                transform=ax.transAxes, fontsize=8.5, color="#666", fontstyle="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color="#2a2a2a")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.9,
                color="#c2cad3", alpha=1.0, zorder=1)
        if yscale == "log":
            ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.6,
                    color="#d6dce2", alpha=0.9, zorder=1)
        ax.set_axisbelow(True)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
        for spine_name in ("left", "bottom"):
            ax.spines[spine_name].set_color("#b5bbc2")

        winning_idx = np.argmax(values) if "accuracy" in title.lower() else np.argmin(values)
        bars[winning_idx].set_linewidth(2.4)
        bars[winning_idx].set_edgecolor("#1f3d52")
        ax.annotate(
            "best",
            xy=(bars[winning_idx].get_x() + bars[winning_idx].get_width() / 2.0,
                ax.get_ylim()[0]),
            xytext=(0, -34), textcoords="offset points",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            color="#1f3d52",
            bbox=dict(boxstyle="round,pad=0.18", fc="#eaf1f6",
                      ec="#1f3d52", lw=0.7),
        )

    fig.suptitle(
        "Component ablation: SAC owns efficiency, D3QN owns robustness, "
        "and only Full SAFEL-DT owns both.",
        fontsize=12, y=0.99, color="#1f3d52", fontweight="bold",
    )
    fig.text(
        0.5, 0.945,
        "Removing D3QN collapses accuracy to near-chance; "
        "removing SAC more than doubles every cost axis. "
        "Error bars are $\\pm$1 std across 3 seeds (Vanilla, Full SAFEL-DT); "
        "the $-$SAC and $-$D3QN bars are single-seed point estimates.",
        ha="center", va="top", fontsize=9.5, color="#555",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))

    out_png = OUT_DIR / "ablation_bars.png"
    out_pdf = OUT_DIR / "ablation_bars.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")

    print("\n--- ablation summary (mean across available seeds) ---")
    print(f"{'variant':<35} {'n':>2} {'acc':>14} {'comm':>16} {'train':>14} {'sec(min)':>14}")
    for (label, long, fog, cloud, _, _), d in zip(CELLS, vals):
        s = label.replace("\n", " ")
        def f(m, s_, fmt):
            return f"{fmt.format(m)}\u00b1{fmt.format(s_)}" if s_ is not None else fmt.format(m)
        print(f"{s:<35} {d['n_seeds']:>2} "
              f"{f(d['tail_acc'], d['tail_acc_std'], '{:.3f}'):>14} "
              f"{f(d['comm'], d['comm_std'], '{:,.0f}'):>16} "
              f"{f(d['train'], d['train_std'], '{:.1f}'):>14} "
              f"{f(d['sec_min'], d['sec_min_std'], '{:.1f}'):>14}")


if __name__ == "__main__":
    main()
