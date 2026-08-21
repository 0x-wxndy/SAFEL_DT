"""3-panel client selection heatmap: all vs heuristic vs SAC+D3QN.

Each panel shows a 30-clients-by-300-rounds binary matrix of
"was this client selected in this round?". Malicious clients are
sorted to the top and colored red (selected) / pink-shaded (not yet
activated); benign clients are at the bottom and colored blue
(selected) / white (not selected). A horizontal separator marks
the malicious / benign split, and a colored phase strip at the
bottom marks the four-phase mixed-stepwise attack schedule
(10\% -> 15\% -> 20\% -> 25\% malicious fraction).

The visual story is that under `all` every cell is lit, under
`heuristic` the malicious top-strip turns on as fast as the benign
block, and under `sac+d3qn` the malicious top-strip is visibly
suppressed -- especially once the Lagrangian dual variables have
converged after ~round 50.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

SWEEP   = Path("results/runs/sweep_headline")
OUT_DIR = Path("results/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLIENTS = 30
N_FOGS = 3
CLIENTS_PER_FOG = N_CLIENTS // N_FOGS  # = 10
T = 300

# Sequential split: global client g lives in fog g // 10, local index g % 10.
def global_id(fog_id: int, local_idx: int) -> int:
    return fog_id * CLIENTS_PER_FOG + local_idx


PHASES = [
    (  0, 100, "10%",  "#fff4e6"),
    (100, 200, "15%",  "#fde0c2"),
    (200, 250, "20%",  "#fab784"),
    (250, 300, "25%",  "#e88847"),
]

CELLS = [
    ("all__d3qn__seed0000.jsonl",       "all (no selection)"),
    ("heuristic__d3qn__seed0000.jsonl", "heuristic (top-k inverse-loss)"),
    ("sac__d3qn__seed0000.jsonl",       "SAFEL-DT (sac + d3qn)"),
]


def build_selection_matrix(trace: Path) -> tuple[np.ndarray, list[int]]:
    """Return (matrix, malicious_global_ids).

    matrix[client, round] in {0, 1}. For the `all` policy
    selected_per_fog is None; in that case every client participates.
    """
    rows = trace.read_text(encoding="utf-8").splitlines()
    rounds = min(len(rows), T)
    sel = np.zeros((N_CLIENTS, rounds), dtype=np.uint8)
    mal_ids_first = None
    for r_idx, line in enumerate(rows[:rounds]):
        d = json.loads(line)
        if mal_ids_first is None:
            mal_ids_first = list(d.get("malicious_ids") or [])
        spf = d.get("selected_per_fog")
        if spf is None:
            sel[:, r_idx] = 1
            continue
        for fog_key, locals_list in spf.items():
            fog_id = int(fog_key)
            if locals_list is None:
                continue
            for li in locals_list:
                gid = global_id(fog_id, int(li))
                if 0 <= gid < N_CLIENTS:
                    sel[gid, r_idx] = 1
    return sel, list(mal_ids_first or [])


def activation_round(trace: Path, mal_ids: list[int]) -> dict[int, int]:
    """Round (0-indexed) in which each malicious client first becomes active."""
    rows = trace.read_text(encoding="utf-8").splitlines()
    out: dict[int, int] = {}
    for r_idx, line in enumerate(rows):
        active = set(int(x) for x in (json.loads(line).get("active_malicious_ids") or []))
        for m in mal_ids:
            if m in active and m not in out:
                out[m] = r_idx
        if len(out) == len(mal_ids):
            break
    return out


def main() -> None:
    cells = []
    for fname, label in CELLS:
        trace = SWEEP / fname
        if not trace.exists():
            print(f"[skip] missing {fname}")
            continue
        sel, mal_ids = build_selection_matrix(trace)
        act = activation_round(trace, mal_ids)
        cells.append((label, sel, mal_ids, act))
        print(f"[loaded] {fname}: sel={sel.shape}, mal={mal_ids}")

    if not cells:
        raise SystemExit("no traces loaded")

    mal_ids = cells[0][2]
    benign_ids = [g for g in range(N_CLIENTS) if g not in mal_ids]
    row_order = mal_ids + benign_ids
    n_mal = len(mal_ids)

    fig_w = 14.0
    fig_h = 0.18 * N_CLIENTS + 1.4  # ~6.8 in for 30 clients + phase strip
    fig, axes = plt.subplots(
        1, len(cells),
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if len(cells) == 1:
        axes = [axes]

    for ax, (label, sel, _mal, act) in zip(axes, cells):
        sel_reord = sel[row_order, :]
        rgb = np.ones((sel_reord.shape[0], sel_reord.shape[1], 3), dtype=np.float32)
        for r in range(sel_reord.shape[0]):
            if r < n_mal:
                colour = np.array([0.78, 0.13, 0.13])  # red
                light  = np.array([1.00, 0.93, 0.93])  # pink-ish
            else:
                colour = np.array([0.10, 0.34, 0.65])  # blue
                light  = np.array([0.94, 0.96, 0.99])  # almost white
            for c in range(sel_reord.shape[1]):
                rgb[r, c] = colour if sel_reord[r, c] else light
        ax.imshow(
            rgb, aspect="auto", interpolation="nearest", origin="lower",
            extent=(-0.5, T - 0.5, -0.5, N_CLIENTS - 0.5),
        )

        ax.axhline(n_mal - 0.5, color="black", linewidth=1.2, linestyle="-", alpha=0.85)

        for m, ar in act.items():
            if m not in mal_ids:
                continue
            row = mal_ids.index(m)
            ax.plot([ar, ar], [row - 0.45, row + 0.45], color="black", linewidth=1.6, alpha=0.55)

        for lo, hi, _name, _col in PHASES:
            ax.axvline(lo, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        ax.set_title(label, fontsize=10)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_xlabel("Round")
        ax.set_xticks([0, 100, 200, 300])

        ax.set_yticks(range(N_CLIENTS))
        ax.set_yticklabels(
            [f"c{cid}" for cid in row_order],
            fontsize=6.5,
        )
        for tick, cid in zip(ax.get_yticklabels(), row_order):
            tick.set_color("#9a1e1e" if cid in mal_ids else "#1e3a6a")
            tick.set_fontweight("bold" if cid in mal_ids else "normal")

        for lo, hi, name, col in PHASES:
            ax.add_patch(Rectangle(
                (lo, -1.8), hi - lo, 0.9,
                facecolor=col, edgecolor="black", linewidth=0.4,
                clip_on=False,
            ))
            ax.text((lo + hi) / 2.0, -1.35, name,
                    ha="center", va="center", fontsize=8, clip_on=False)

    axes[0].set_ylabel("Client (malicious top, benign bottom)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=(0.78, 0.13, 0.13), label="Malicious selected"),
        plt.Rectangle((0, 0), 1, 1, fc=(1.00, 0.93, 0.93), ec="0.6",
                      label="Malicious skipped"),
        plt.Rectangle((0, 0), 1, 1, fc=(0.10, 0.34, 0.65), label="Benign selected"),
        plt.Rectangle((0, 0), 1, 1, fc=(0.94, 0.96, 0.99), ec="0.6",
                      label="Benign skipped"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )

    fig.suptitle(
        "Per-round client selection under escalating mixed attack "
        "(N-BaIoT, $N=30$, 8 malicious; cloud=D3QN, headline seed)",
        fontsize=11, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_png = OUT_DIR / "selection_heatmap.png"
    out_pdf = OUT_DIR / "selection_heatmap.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")

    print("\n--- per-policy, per-phase selection rates (M=mal, B=ben, ratio=M/B; <1 means discriminating) ---")
    header = f"{'policy':<26} | " + " | ".join(
        f"{n+'  M':>7}  {n+'  B':>6}  {n+' M/B':>6}" for _, _, n, _ in PHASES
    )
    print(header)
    for label, sel, _mal, _act in cells:
        clean = label.replace(r"\textbf", "").replace(r"\textsc", "").replace("{", "").replace("}", "").replace("\\texttt", "")
        parts = []
        for lo, hi, _, _ in PHASES:
            # restrict to *active* malicious clients in this phase
            if lo < 100:
                active_mal = [0, 1, 2]
            elif lo < 200:
                active_mal = [0, 1, 2, 7, 8]
            elif lo < 250:
                active_mal = [0, 1, 2, 7, 8, 12]
            else:
                active_mal = mal_ids
            m_mean = sel[active_mal, lo:hi].mean() if active_mal else 0.0
            b_mean = sel[benign_ids, lo:hi].mean()
            ratio = m_mean / b_mean if b_mean > 0 else float("nan")
            parts.append(f"  {m_mean:5.2f}   {b_mean:5.2f}   {ratio:5.2f}")
        print(f"{clean:<26} |" + " |".join(parts))


if __name__ == "__main__":
    main()
