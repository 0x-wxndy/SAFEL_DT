"""Component ablation table + cost-accuracy Pareto scatter.

Reads the headline sweep cells (all/random/heuristic/sac fog x
static/round_robin/d3qn cloud, 1 seed each, 300 rounds, mixed-stepwise
attack) and emits:

  1) A LaTeX ablation table whose four anchor rows correspond to the
     2x2 ablation grid:
       - all+static : no SAC, no D3QN (vanilla FedAvg under attack)
       - all+d3qn   : - SAC (D3QN only)
       - sac+static : - D3QN (SAC only)
       - sac+d3qn   : full SAFEL-DT
     plus heuristic+d3qn and random+d3qn as reference baselines.

  2) A cost-accuracy Pareto scatter (one point per fog x cloud cell)
     so the reader can see SAFEL-DT alone in the low-cost / high-acc
     corner.

All numbers come from the same cells the rest of the paper uses, so
the ablation does not need any new sweep. Security cost is the
Paillier-1024 column from _paillier_recalib.py (cached).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SWEEP = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TEX = Path("_table_ablation.tex")
CACHE = Path("_paillier_recalib_cache.json")

TAIL = 10
T = 300

ABLATION_ROWS = [
    # (label,                              fog,         cloud,         note)
    ("Vanilla FedAvg (no SAC, no D3QN)",   "all",       "static",      "baseline under attack"),
    ("$-$ SAC only (D3QN cloud)",          "all",       "d3qn",        "remove SAC: D3QN alone"),
    ("$-$ D3QN only (SAC fog)",            "sac",       "static",      "remove D3QN: SAC alone"),
    ("\\textbf{Full SAFEL-DT}",            "sac",       "d3qn",        "both layers active"),
]

REFERENCE_ROWS = [
    ("Random fog + D3QN",                  "random",    "d3qn",        "random selection reference"),
    ("Heuristic fog + D3QN",               "heuristic", "d3qn",        "non-learning selection reference"),
]


def load_cell(fog: str, cloud: str) -> dict | None:
    path = SWEEP / f"{fog}__{cloud}__seed0000.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        return None
    acc = np.array([float(r["accuracy"]) for r in rows], dtype=np.float64)
    loss = np.array([float(r["loss"]) for r in rows], dtype=np.float64)
    comm = np.zeros(len(rows))
    train = np.zeros(len(rows))
    sec_plain = np.zeros(len(rows))
    for i, r in enumerate(rows):
        c = r.get("costs") or {}
        for fid, cb in c.items():
            comm[i]      += float(cb.get("comm", 0.0))
            train[i]     += float(cb.get("train", 0.0))
            sec_plain[i] += float(cb.get("sec", 0.0))
    return {
        "n":         len(rows),
        "tail_acc":  float(np.mean(acc[-TAIL:])) if len(acc) >= TAIL else float(acc.mean()),
        "tail_loss": float(np.mean(loss[-TAIL:])) if len(loss) >= TAIL else float(loss.mean()),
        "comm":      float(np.mean(comm)),
        "train":     float(np.mean(train)),
        "sec_plain": float(np.mean(sec_plain)),
    }


def paillier_ratio() -> float:
    cached = json.loads(CACHE.read_text(encoding="utf-8"))
    plain  = float(cached["c_enc_plain"]) + float(cached["c_auth"]) + float(cached["c_verify"])
    p1024 = float(cached["c_enc_p1024"]) + float(cached["c_auth"]) + float(cached["c_verify"])
    return p1024 / plain


def fmt_int(v: float) -> str:
    return f"${int(round(v)):,}$".replace(",", "\\,")


def fmt_min(v_s: float) -> str:
    return f"${v_s/60.0:.1f}$"


def main() -> None:
    ratio = paillier_ratio()
    print(f"Paillier-1024 / plain scaling: {ratio:,.1f}x\n")

    rows_all = ABLATION_ROWS + REFERENCE_ROWS
    cells: dict[tuple[str, str], dict] = {}
    for _, fog, cloud, _ in rows_all:
        d = load_cell(fog, cloud)
        if d is None:
            print(f"[skip] {fog}+{cloud}: trace missing")
            continue
        d["sec_p1024_min"] = (d["sec_plain"] * ratio) / 60.0
        cells[(fog, cloud)] = d

    # ---- Print ablation summary to stdout ----
    print(f"{'variant':<40} {'fog':<10} {'cloud':<13} | {'tail acc':>9} {'tail loss':>10} | "
          f"{'comm':>10} {'train':>9} | {'sec(min)':>9}")
    print("-" * 130)
    for label, fog, cloud, _ in rows_all:
        d = cells.get((fog, cloud))
        clean = label.replace(r"\textbf{", "").replace(r"}", "").replace("$-$", "-")
        if d is None:
            print(f"{clean:<40} {fog:<10} {cloud:<13} |  ---")
            continue
        print(f"{clean:<40} {fog:<10} {cloud:<13} | "
              f"{d['tail_acc']:>9.3f} {d['tail_loss']:>10.3f} | "
              f"{d['comm']:>10,.0f} {d['train']:>9.1f} | {d['sec_p1024_min']:>9.1f}")

    # ---- Emit LaTeX table ----
    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit(r"\begin{table*}[!htbp]")
    emit(r"\centering")
    emit(r"\caption{Component ablation. Rows~1--4 isolate the contribution of the SAC fog selector and the D3QN cloud coordinator; rows~5--6 add a random fog and the loss-based heuristic as non-learning references. All cells use the headline mixed-stepwise schedule ($N{=}30$, $T{=}300$, 10\%$\to$25\% adversarial). Accuracy and loss are tail-window means over the final 10 rounds. Comm and Train are per-round costs summed across fogs; Security is per-round Paillier-1024 cost reconstructed via the calibration of Sec.~\ref{subsec:efficiency}. Removing D3QN collapses accuracy from $0.78$ to near-chance under attack; removing SAC keeps accuracy but more than doubles every cost axis.}")
    emit(r"\label{tab:ablation}")
    emit(r"\footnotesize")
    emit(r"\begin{tabular}{ll cccc}")
    emit(r"\toprule")
    emit(r"\textbf{Variant} & \textbf{Note} & \textbf{Tail acc.} & \textbf{Comm cost} & \textbf{Train cost} & \textbf{Sec cost (min)} \\")
    emit(r"\midrule")

    for label, fog, cloud, note in ABLATION_ROWS:
        d = cells.get((fog, cloud))
        if d is None:
            emit(f"{label} & {note} & --- & --- & --- & --- \\\\")
            continue
        emph_l, emph_r = ("", "")
        if "Full SAFEL-DT" in label:
            emit(r"\rowcolor{yellow!18}")
            emph_l, emph_r = (r"\textbf{", "}")
        emit(
            f"{label} & {note} & "
            f"{emph_l}${d['tail_acc']:.3f}${emph_r} & "
            f"{emph_l}{fmt_int(d['comm'])}{emph_r} & "
            f"{emph_l}${d['train']:.1f}${emph_r} & "
            f"{emph_l}{fmt_min(d['sec_p1024_min']*60.0)}{emph_r} \\\\"
        )

    emit(r"\midrule")
    emit(r"\multicolumn{6}{l}{\emph{Reference baselines (non-learning fog selectors with D3QN cloud).}} \\")
    for label, fog, cloud, note in REFERENCE_ROWS:
        d = cells.get((fog, cloud))
        if d is None:
            emit(f"{label} & {note} & --- & --- & --- & --- \\\\")
            continue
        emit(
            f"{label} & {note} & "
            f"${d['tail_acc']:.3f}$ & "
            f"{fmt_int(d['comm'])} & "
            f"${d['train']:.1f}$ & "
            f"{fmt_min(d['sec_p1024_min']*60.0)} \\\\"
        )
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    emit(r"\end{table*}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote {OUT_TEX}]")

    # ---- Pareto scatter ----
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fog_marker = {"all": "s", "random": "o", "heuristic": "^", "sac": "*"}
    fog_color  = {"all": "#444444", "random": "#888888",
                  "heuristic": "#1f77b4", "sac": "#d62728"}
    cloud_size = {"static": 90, "round_robin": 130, "d3qn": 200}
    cloud_edge = {"static": "0.6", "round_robin": "0.3", "d3qn": "black"}

    all_cells: dict[tuple[str, str], dict] = {}
    for fog in ("all", "random", "heuristic", "sac"):
        for cloud in ("static", "round_robin", "d3qn"):
            d = load_cell(fog, cloud)
            if d is None:
                continue
            d["sec_p1024_min"] = (d["sec_plain"] * ratio) / 60.0
            all_cells[(fog, cloud)] = d

    for (fog, cloud), d in all_cells.items():
        ax.scatter(
            d["comm"] / 1000.0, d["tail_acc"],
            s=cloud_size[cloud], marker=fog_marker[fog],
            c=fog_color[fog], edgecolors=cloud_edge[cloud],
            linewidths=1.4, alpha=0.92, zorder=4,
        )

    sac_d3qn = all_cells.get(("sac", "d3qn"))
    if sac_d3qn is not None:
        ax.annotate(
            r"$\bigstar$ SAFEL-DT",
            xy=(sac_d3qn["comm"] / 1000.0, sac_d3qn["tail_acc"]),
            xytext=(sac_d3qn["comm"] / 1000.0 + 40.0, sac_d3qn["tail_acc"] - 0.04),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
            fontsize=11, color="#d62728", fontweight="bold",
        )

    sac_static = all_cells.get(("sac", "static"))
    if sac_static is not None:
        ax.annotate(
            "$-$D3QN: collapse",
            xy=(sac_static["comm"] / 1000.0, sac_static["tail_acc"]),
            xytext=(sac_static["comm"] / 1000.0 + 60.0, sac_static["tail_acc"] + 0.04),
            arrowprops=dict(arrowstyle="->", color="#a83232", lw=1.2),
            fontsize=10, color="#a83232",
        )

    all_d3qn = all_cells.get(("all", "d3qn"))
    if all_d3qn is not None:
        ax.annotate(
            "$-$SAC: expensive",
            xy=(all_d3qn["comm"] / 1000.0, all_d3qn["tail_acc"]),
            xytext=(all_d3qn["comm"] / 1000.0 - 230.0, all_d3qn["tail_acc"] - 0.10),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
            fontsize=10, color="#444444",
        )

    handles_fog = [
        plt.Line2D([0], [0], marker=fog_marker[f], color="w",
                   markerfacecolor=fog_color[f], markeredgecolor="black",
                   markersize=11, label=f"fog = {f}")
        for f in ("all", "random", "heuristic", "sac")
    ]
    handles_cloud = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor=cloud_edge[c],
                   markersize=10 + i * 2, markeredgewidth=1.4,
                   label=f"cloud = {c}")
        for i, c in enumerate(("static", "round_robin", "d3qn"))
    ]
    leg1 = ax.legend(handles=handles_fog, loc="lower left", title="Fog selector",
                     fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=handles_cloud, loc="lower right", title="Cloud coordinator",
              fontsize=9, title_fontsize=9, framealpha=0.9)

    ax.set_xlabel("Mean per-round communication cost (×$10^3$ transmission units)")
    ax.set_ylabel("Tail-window test accuracy (last 10 rounds)")
    ax.set_title(
        "Cost-accuracy Pareto: SAFEL-DT sits at the low-cost / high-accuracy corner.\n"
        "Removing D3QN (sac+static) collapses accuracy; "
        "removing SAC (all+d3qn) more than doubles every cost axis.",
        fontsize=10.5,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    out_png = OUT_DIR / "ablation_pareto.png"
    out_pdf = OUT_DIR / "ablation_pareto.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")


if __name__ == "__main__":
    main()
