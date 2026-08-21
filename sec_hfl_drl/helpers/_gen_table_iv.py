"""Build the new Table IV (cost by policy) from completed sweep cells.

Reads every *.jsonl trace under results/runs/sweep_headline, computes total
comm/train/sec cost per cell (summed across rounds and fogs), then groups by
(fog_policy, cloud_policy) and reports mean +/- std over the seeds we have.

A "complete" cell has >= TARGET_ROUNDS lines. Cells with fewer lines are
treated as in-progress and reported as '--' so the LaTeX shows empty rows.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import defaultdict

SWEEP = Path("results/runs/sweep_headline")
TARGET_ROUNDS = 300
OUT_TEX = Path("_table_cost.tex")

FOGS = ["all", "random", "heuristic", "sac"]
CLOUDS = ["static", "round_robin", "d3qn"]

# Parse cell name like:  fog__cloud__seed0000.jsonl
CELL_RE = re.compile(r"(?P<fog>[a-z_]+?)__(?P<cloud>[a-z0-9_]+?)__seed(?P<seed>\d{4})$")


def per_cell_totals(trace: Path) -> tuple[int, float, float, float]:
    """Return (n_rounds, mean_per_round_comm, mean_per_round_train, mean_per_round_sec).

    Matches analysis/paper_tables.py methodology: per-round cost = sum across fogs
    for that round; the reported value is the *mean* over rounds (NOT the total)
    so it's comparable to the existing placeholder Table IV units.
    """
    n = 0
    per_round_comm: list[float] = []
    per_round_train: list[float] = []
    per_round_sec: list[float] = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        c = t = s = 0.0
        for fog_costs in d["costs"].values():
            c += float(fog_costs["comm"])
            t += float(fog_costs["train"])
            s += float(fog_costs["sec"])
        per_round_comm.append(c)
        per_round_train.append(t)
        per_round_sec.append(s)
        n += 1
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    return (
        n,
        sum(per_round_comm) / n,
        sum(per_round_train) / n,
        sum(per_round_sec) / n,
    )


def fmt(mean: float, std: float | None, big: bool) -> str:
    r"""Format a (mean, std) pair as '$value \pm std$' or just '$value$'."""
    if big:
        mu = f"{mean:,.0f}".replace(",", "\\,")
        sd = f"{std:,.0f}".replace(",", "\\,") if std is not None else None
    else:
        mu = f"{mean:.3f}"
        sd = f"{std:.3f}" if std is not None else None
    return f"${mu} \\pm {sd}$" if sd is not None else f"${mu}$"


def main() -> None:
    by_combo: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(list)
    incomplete: list[str] = []
    for trace in sorted(SWEEP.glob("*.jsonl")):
        m = CELL_RE.match(trace.stem)
        if m is None:
            continue
        fog, cloud, seed = m.group("fog"), m.group("cloud"), int(m.group("seed"))
        n, c, t, s = per_cell_totals(trace)
        if n < TARGET_ROUNDS:
            incomplete.append(f"{trace.stem}  ({n}/{TARGET_ROUNDS} rounds)")
            continue
        by_combo[(fog, cloud)].append((seed, c, t, s))

    print(f"target_rounds = {TARGET_ROUNDS}")
    print(f"complete cells: {sum(len(v) for v in by_combo.values())}")
    print(f"incomplete cells: {len(incomplete)}")
    for line in incomplete:
        print(f"  - {line}")
    print()

    # Full LaTeX table block
    lines_out: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines_out.append(s)
    emit()
    emit(r"\begin{table*}[!htbp]")
    emit(r"\centering")
    emit(r"\caption{System cost by fog selection policy and cloud coordinator")
    emit(r"  (N-BaIoT, $N=30$ clients, $T=300$ rounds, mixed-stepwise attack")
    emit(r"  schedule). Each value is the mean per-round cost summed across fogs,")
    emit(r"  reported as mean$\,\pm\,$std over the seeds currently available;")
    emit(r"  rows annotated \emph{n=1} have a single seed (variance pending the")
    emit(r"  follow-up multi-seed sweep). Cells marked \texttt{---} are still in")
    emit(r"  flight. Comm cost is in transmission units ($|\mathcal{D}_i|\sigma_i$);")
    emit(r"  training cost in compute-cost units ($\phi_{\mathrm{cpu}} C_{\mathrm{ml}}$);")
    emit(r"  security cost in cryptographic-operation seconds (ECDSA signing /")
    emit(r"  verification; Paillier-1024 encryption overhead reported separately).}")
    emit(r"\label{tab:cost-by-policy}")
    emit(r"\footnotesize")
    emit(r"\begin{tabular}{ll ccc}")
    emit(r"\toprule")
    emit(r"\textbf{Fog policy} & \textbf{Cloud policy} & \textbf{Comm cost} & \textbf{Train cost} & \textbf{Sec cost (s)} \\")
    emit(r"\midrule")
    for fi, fog in enumerate(FOGS):
        if fi > 0:
            emit(r"\midrule")
        for cloud in CLOUDS:
            seeds = by_combo.get((fog, cloud), [])
            label = f"\\texttt{{{cloud.replace('_', r'\_')}}}"
            if not seeds:
                emit(f"\\texttt{{{fog}}} & {label} & --- & --- & --- \\\\")
                continue
            comms = [c for _, c, _, _ in seeds]
            trains = [t for _, _, t, _ in seeds]
            secs = [s for _, _, _, s in seeds]
            n_seeds = len(comms)
            cmu = sum(comms) / n_seeds
            tmu = sum(trains) / n_seeds
            smu = sum(secs) / n_seeds
            if n_seeds > 1:
                csd = math.sqrt(sum((x - cmu) ** 2 for x in comms) / (n_seeds - 1))
                tsd = math.sqrt(sum((x - tmu) ** 2 for x in trains) / (n_seeds - 1))
                ssd = math.sqrt(sum((x - smu) ** 2 for x in secs) / (n_seeds - 1))
            else:
                csd = tsd = ssd = None
            cstr = fmt(cmu, csd, big=True)
            tstr = fmt(tmu, tsd, big=False) if tmu < 1000 else fmt(tmu, tsd, big=True)
            sstr = fmt(smu, ssd, big=False)
            note = "" if n_seeds >= 3 else r"\,{\scriptsize\emph{n=1}}"
            emit(f"\\texttt{{{fog}}} & {label} & {cstr} & {tstr} & {sstr}{note} \\\\")
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    emit(r"\end{table*}")
    OUT_TEX.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"\n[wrote {OUT_TEX} -- {len(lines_out)} lines]")


if __name__ == "__main__":
    main()
