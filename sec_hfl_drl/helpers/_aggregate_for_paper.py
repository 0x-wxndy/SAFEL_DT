"""Aggregate headline + best-available predecessor data into paper-ready rows.

For each (fog, cloud) cell we report mean ± std using the most defensible source:
  1. sweep_headline (300 rounds, headline config) if >= 2 seeds present.
  2. Otherwise headline seed 0 mean, paired with std transferred from the closest
     predecessor sweep that has multi-seed data for that cell. We prefer
     sweep_credible_* (100 rounds, single-attack but headline-like otherwise)
     pooled across the three attack families; fall back to sweep_paper_final
     (30-47 rounds) when credible_* is absent.

We split mean from std deliberately: the headline 300-round single-seed value
is the right *point estimate* (correct config and horizon), and the predecessor
multi-seed variation gives a reasonable seed-to-seed uncertainty band. We
report the absolute std verbatim from the predecessor since (a) per-round
costs are attack-insensitive and (b) tail accuracy std under single-attack
predecessors is a conservative upper bound on the mixed-attack noise.

Run from sec_hfl_drl/ ; emits an ascii table sized for copy-paste into the
paper's cost-by-policy and cloud-comparison rows.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path("results/runs")
CACHE = Path("_paillier_recalib_cache.json")
TAIL = 10  # rounds for tail-window mean (accuracy)

# Cells, in the order they appear in tab:cost-by-policy.
CELLS = [
    ("all",       "static"),
    ("all",       "round_robin"),
    ("all",       "d3qn"),
    ("random",    "static"),
    ("random",    "round_robin"),
    ("random",    "d3qn"),
    ("heuristic", "static"),
    ("heuristic", "round_robin"),
    ("heuristic", "d3qn"),
    ("sac",       "static"),
    ("sac",       "round_robin"),
    ("sac",       "d3qn"),
]

# Predecessor sweeps to pool, in preference order.
CREDIBLE = ["sweep_credible_gaussian", "sweep_credible_label_flip", "sweep_credible_model_scale"]
FALLBACK = "sweep_paper_final"


def _load_trace(p: Path) -> dict | None:
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    if n < 20:
        return None
    comm = np.zeros(n); train = np.zeros(n); sec_plain = np.zeros(n)
    acc = np.array([float(r["accuracy"]) for r in rows])
    loss = np.array([float(r.get("loss", 0.0)) for r in rows])
    for i, r in enumerate(rows):
        for cb in (r.get("costs") or {}).values():
            comm[i]      += float(cb.get("comm", 0.0))
            train[i]     += float(cb.get("train", 0.0))
            sec_plain[i] += float(cb.get("sec", 0.0))
    tail = min(TAIL, max(1, n // 5))
    return {
        "n_rounds":  n,
        "tail_acc":  float(np.mean(acc[-tail:])),
        "tail_loss": float(np.mean(loss[-tail:])),
        "comm":      float(np.mean(comm)),
        "train":     float(np.mean(train)),
        "sec_plain": float(np.mean(sec_plain)),
    }


def load_cell_in_sweep(sweep: str, fog: str, cloud: str, seeds=range(10)) -> list[dict]:
    out = []
    for s in seeds:
        d = _load_trace(ROOT / sweep / f"{fog}__{cloud}__seed{s:04d}.jsonl")
        if d is not None:
            out.append(d)
    return out


def mean_std(values: list[float]) -> tuple[float, float | None]:
    if len(values) >= 2:
        return float(np.mean(values)), float(np.std(values, ddof=1))
    if len(values) == 1:
        return float(values[0]), None
    return float("nan"), None


def pick_predecessor(fog: str, cloud: str) -> tuple[str, list[dict], set[str]] | None:
    """Return (source_label, seed_runs, metrics_we_trust) using the best predecessor.

    metrics_we_trust restricts which std values are admissible. credible_*
    (100 rounds, near steady-state) is trusted for all four metrics. paper_final
    (~30 rounds, transient) is trusted only for comm + sec (network volume and
    encryption time are determined by message size + cohort composition, both
    near-stationary from round 0). Train cost in paper_final is dominated by
    early-round fog-overload transients and is therefore excluded.
    """
    pooled = []
    for sw in CREDIBLE:
        pooled.extend(load_cell_in_sweep(sw, fog, cloud))
    if len(pooled) >= 2:
        return (f"credible_* ({len(pooled)} runs @ {pooled[0]['n_rounds']}r)",
                pooled, {"tail_acc", "tail_loss", "comm", "train", "sec_plain"})
    fb = load_cell_in_sweep(FALLBACK, fog, cloud)
    if len(fb) >= 2:
        return (f"{FALLBACK} ({len(fb)} seeds @ {fb[0]['n_rounds']}r)",
                fb, {"comm", "sec_plain"})
    return None


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    paillier_ratio = (
        (cache["c_enc_p1024"] + cache["c_auth"] + cache["c_verify"]) /
        (cache["c_enc_plain"] + cache["c_auth"] + cache["c_verify"])
    )

    headers = ["cell", "src", "rounds", "n", "tail_acc", "comm/round", "train/round", "Paillier-1024 min/round"]
    print("{:<22} {:<32} {:>6} {:>3} {:>18} {:>22} {:>20} {:>22}".format(*headers))
    print("-" * 160)

    paper_rows: dict[tuple[str, str], dict] = {}

    for fog, cloud in CELLS:
        headline = load_cell_in_sweep("sweep_headline", fog, cloud)
        if len(headline) >= 2:
            src = "sweep_headline"
            rounds = headline[0]["n_rounds"]
            n = len(headline)
            acc_m, acc_s = mean_std([d["tail_acc"]  for d in headline])
            ls_m,  ls_s  = mean_std([d["tail_loss"] for d in headline])
            cm_m,  cm_s  = mean_std([d["comm"]      for d in headline])
            tr_m,  tr_s  = mean_std([d["train"]     for d in headline])
            sp_m,  sp_s  = mean_std([d["sec_plain"] for d in headline])
        elif len(headline) == 1:
            pred = pick_predecessor(fog, cloud)
            if pred is None:
                src = "headline-only"
                rounds = headline[0]["n_rounds"]; n = 1
                acc_m, ls_m, cm_m, tr_m, sp_m = (
                    headline[0]["tail_acc"], headline[0]["tail_loss"],
                    headline[0]["comm"], headline[0]["train"], headline[0]["sec_plain"])
                acc_s = ls_s = cm_s = tr_s = sp_s = None
            else:
                src_label, pred_runs, trust = pred
                src = f"headline+{src_label}"
                rounds = f"{headline[0]['n_rounds']}/{pred_runs[0]['n_rounds']}"
                n = f"1+{len(pred_runs)}"
                acc_m = headline[0]["tail_acc"]
                ls_m  = headline[0]["tail_loss"]
                cm_m  = headline[0]["comm"]
                tr_m  = headline[0]["train"]
                sp_m  = headline[0]["sec_plain"]
                acc_s = mean_std([d["tail_acc"]  for d in pred_runs])[1] if "tail_acc"  in trust else None
                ls_s  = mean_std([d["tail_loss"] for d in pred_runs])[1] if "tail_loss" in trust else None
                cm_s  = mean_std([d["comm"]      for d in pred_runs])[1] if "comm"      in trust else None
                tr_s  = mean_std([d["train"]     for d in pred_runs])[1] if "train"     in trust else None
                sp_s  = mean_std([d["sec_plain"] for d in pred_runs])[1] if "sec_plain" in trust else None
        else:
            print(f"{fog+'+'+cloud:<22} {'--- no data ---':<32}")
            continue

        sec_min_m = sp_m * paillier_ratio / 60.0
        sec_min_s = (sp_s * paillier_ratio / 60.0) if sp_s is not None else None

        def fmt(m, s, fmt_str):
            return f"{fmt_str.format(m)} \u00b1 {fmt_str.format(s)}" if s is not None else f"{fmt_str.format(m)}  (n=1)"

        print("{:<22} {:<32} {:>6} {:>3} {:>18} {:>22} {:>20} {:>22}".format(
            f"{fog}+{cloud}", src, str(rounds), str(n),
            fmt(acc_m, acc_s, "{:.3f}"),
            fmt(cm_m, cm_s, "{:,.0f}"),
            fmt(tr_m, tr_s, "{:.1f}"),
            fmt(sec_min_m, sec_min_s, "{:.1f}"),
        ))

        paper_rows[(fog, cloud)] = {
            "src": src,
            "tail_acc":  (acc_m, acc_s),
            "tail_loss": (ls_m,  ls_s),
            "comm":      (cm_m,  cm_s),
            "train":     (tr_m,  tr_s),
            "sec_min":   (sec_min_m, sec_min_s),
        }

    print()
    print("LaTeX-ready rows for tab:cost-by-policy (Fog | Cloud | comm | train | Paillier-1024 min):")
    print("-" * 100)
    for (fog, cloud), r in paper_rows.items():
        def latex(m, s, fmt_str):
            if s is None: return fmt_str.format(m)
            return f"${fmt_str.format(m)} \\pm {fmt_str.format(s)}$"
        print(f"  {fog:<10} & {cloud:<13} & {latex(*r['comm'], '{:,.0f}'):>26} & {latex(*r['train'], '{:.1f}'):>22} & {latex(*r['sec_min'], '{:.1f}'):>22} \\\\")

    print()
    print("LaTeX-ready rows for tab:cloud-comparison (Cloud | tail_acc | tail_loss) when fog=sac:")
    print("-" * 100)
    for (fog, cloud), r in paper_rows.items():
        if fog != "sac": continue
        def latex(m, s, fmt_str):
            if s is None: return fmt_str.format(m)
            return f"${fmt_str.format(m)} \\pm {fmt_str.format(s)}$"
        print(f"  {cloud:<13} & {latex(*r['tail_acc'], '{:.3f}'):>22} & {latex(*r['tail_loss'], '{:.3f}'):>22} \\\\")


if __name__ == "__main__":
    main()
