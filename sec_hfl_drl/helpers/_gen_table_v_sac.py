"""Build Table V (cloud-coordinator comparison with SAC fog selection).

Identical to _gen_table_v.py but slices on fog=sac (the paper's headline
cell), so the numbers match the experiment.tex caption text.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import defaultdict

SWEEP = Path("results/runs/sweep_headline")
TARGET_ROUNDS = 300
FOG = "sac"
CLOUDS = ["static", "round_robin", "d3qn"]
OUT_TEX = Path("_table_cloud_sac.tex")

CELL_RE = re.compile(r"(?P<fog>[a-z_]+?)__(?P<cloud>[a-z0-9_]+?)__seed(?P<seed>\d{4})$")


def final_acc_loss(trace: Path, tail: int = 10) -> tuple[int, float, float]:
    """Return (n_rounds, tail-window mean accuracy, tail-window mean loss).

    Using a tail window instead of a single last round is more robust to the
    per-round noise that's visible in the trajectories (the global model
    bounces a few percent around its steady-state under attack).
    """
    rows = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    last = rows[-tail:] if n >= tail else rows
    acc = sum(float(r["accuracy"]) for r in last) / len(last)
    loss = sum(float(r["loss"]) for r in last) / len(last)
    return n, acc, loss


def stat(xs: list[float]) -> tuple[float, float | None]:
    n = len(xs)
    if n == 0:
        return float("nan"), None
    mu = sum(xs) / n
    if n == 1:
        return mu, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, sd


def fmt(mu: float, sd: float | None) -> str:
    if sd is None:
        return f"${mu:.3f}$"
    return f"${mu:.3f} \\pm {sd:.3f}$"


def main() -> None:
    by_cloud: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    incomplete: list[str] = []
    for trace in sorted(SWEEP.glob(f"{FOG}__*.jsonl")):
        m = CELL_RE.match(trace.stem)
        if m is None:
            continue
        fog, cloud, seed = m.group("fog"), m.group("cloud"), int(m.group("seed"))
        if fog != FOG:
            continue
        n, acc, loss = final_acc_loss(trace)
        if n < TARGET_ROUNDS:
            incomplete.append(f"{trace.stem} ({n}/{TARGET_ROUNDS})")
            continue
        by_cloud[cloud].append((seed, acc, loss))

    print(f"FOG = {FOG}")
    print(f"complete cells: {sum(len(v) for v in by_cloud.values())}")
    print(f"incomplete: {incomplete}")
    print()

    static_seeds = by_cloud.get("static", [])
    static_acc_mu = sum(a for _, a, _ in static_seeds) / len(static_seeds) if static_seeds else None

    lines_out: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines_out.append(s)

    emit(r"\begin{table}[!htbp]")
    emit(r"\centering")
    emit(r"\caption{Cloud-coordinator comparison with SAC fog selection")
    emit(r"  (N-BaIoT; $N=30$, $T=300$, mixed-stepwise attack schedule). Accuracy")
    emit(r"  and loss are tail-window means over the final 10 rounds, which")
    emit(r"  smooths the round-to-round bounce visible under sustained")
    emit(r"  attack. Under static FedAvg the global model is unable to")
    emit(r"  defend once the malicious fraction exceeds $10\%$ at round 100;")
    emit(r"  the learned D3QN aggregator selector recovers the bulk of that")
    emit(r"  gap, ending close to its peak.}")
    emit(r"\label{tab:cloud-comparison}")
    emit(r"\footnotesize")
    emit(r"\begin{tabular}{lccc}")
    emit(r"\toprule")
    emit(r"\textbf{Cloud policy} & \textbf{Tail acc.} & \textbf{vs.\ static} & \textbf{Tail loss} \\")
    emit(r"\midrule")
    for cloud in CLOUDS:
        seeds = by_cloud.get(cloud, [])
        cloud_label = cloud.replace("_", r"\_")
        if not seeds:
            emit(f"\\texttt{{{cloud_label}}} & --- & --- & --- \\\\")
            continue
        accs = [a for _, a, _ in seeds]
        losses = [l for _, _, l in seeds]
        a_mu, a_sd = stat(accs)
        l_mu, l_sd = stat(losses)
        delta_str = "baseline" if cloud == "static" else (
            f"${(a_mu - static_acc_mu)*100:+.1f}$\\,pp" if static_acc_mu is not None else "---"
        )
        note = "" if len(seeds) >= 3 else r"\,{\scriptsize\emph{n=1}}"
        bold_l = "\\textbf{" if cloud == "d3qn" else ""
        bold_r = "}" if cloud == "d3qn" else ""
        emit(
            f"{bold_l}\\texttt{{{cloud_label}}}{bold_r} & "
            f"{bold_l}{fmt(a_mu, a_sd)}{bold_r} & "
            f"{bold_l}{delta_str}{bold_r} & "
            f"{bold_l}{fmt(l_mu, l_sd)}{bold_r}{note} \\\\"
        )
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    emit(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"\n[wrote {OUT_TEX}]")


if __name__ == "__main__":
    main()
