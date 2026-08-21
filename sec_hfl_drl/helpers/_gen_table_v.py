"""Build Table V (cloud-coordinator comparison) from completed sweep cells.

Uses the all-fog cells (no fog selection) to isolate the cloud-coordinator
effect. Reports final accuracy + std and final loss + std over seeds.

A "complete" cell has TARGET_ROUNDS lines; partials show '--'.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import defaultdict

SWEEP = Path("results/runs/sweep_headline")
TARGET_ROUNDS = 300
FOG = "all"                                  # which fog policy to slice
CLOUDS = ["static", "round_robin", "d3qn"]
PLACE = "fog=all (no selection)"             # caption note
OUT_TEX = Path("_table_cloud.tex")

CELL_RE = re.compile(r"(?P<fog>[a-z_]+?)__(?P<cloud>[a-z0-9_]+?)__seed(?P<seed>\d{4})$")


def final_acc_loss(trace: Path) -> tuple[int, float, float]:
    """Return (n_rounds, final_acc, final_loss). Reads the last line."""
    txt = trace.read_text(encoding="utf-8").splitlines()
    n = len(txt)
    last = json.loads(txt[-1])
    return n, float(last["accuracy"]), float(last["loss"])


def stat(xs: list[float]) -> tuple[float, float | None]:
    n = len(xs)
    if n == 0:
        return float("nan"), None
    mu = sum(xs) / n
    if n == 1:
        return mu, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, sd


def fmt_acc(mu: float, sd: float | None) -> str:
    if sd is None:
        return f"${mu:.3f}$"
    return f"${mu:.3f} \\pm {sd:.3f}$"


def fmt_loss(mu: float, sd: float | None) -> str:
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

    print(f"complete cells (fog={FOG}): {sum(len(v) for v in by_cloud.values())}")
    print(f"incomplete: {incomplete}")
    print()

    # Static baseline (for vs-static delta column)
    static_seeds = by_cloud.get("static", [])
    static_acc_mu = sum(a for _, a, _ in static_seeds) / len(static_seeds) if static_seeds else None

    lines_out: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines_out.append(s)

    emit(r"\begin{table}[!htbp]")
    emit(r"\centering")
    emit(r"\caption{Cloud-coordinator comparison under the no-selection (fog=\texttt{all})")
    emit(r"  baseline (N-BaIoT; mixed-stepwise attack; final-round accuracy and loss,")
    emit(r"  mean$\,\pm\,$std over seeds where multiple are available). The")
    emit(r"  no-selection baseline isolates the cloud-coordinator effect without the")
    emit(r"  fog-policy layer confounding it; under mixed attacks, both")
    emit(r"  \texttt{round\_robin} and \texttt{d3qn} restore strong accuracy")
    emit(r"  relative to the catastrophic collapse of \texttt{static} FedAvg.}")
    emit(r"\label{tab:cloud-comparison}")
    emit(r"\footnotesize")
    emit(r"\begin{tabular}{lccc}")
    emit(r"\toprule")
    emit(r"\textbf{Cloud policy} & \textbf{Final acc.} & \textbf{vs.\ static} & \textbf{Final loss} \\")
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
            f"{bold_l}{fmt_acc(a_mu, a_sd)}{bold_r} & "
            f"{bold_l}{delta_str}{bold_r} & "
            f"{bold_l}{fmt_loss(l_mu, l_sd)}{bold_r}{note} \\\\"
        )
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    emit(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"\n[wrote {OUT_TEX}]")


if __name__ == "__main__":
    main()
