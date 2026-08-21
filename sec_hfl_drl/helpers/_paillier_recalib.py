"""Standalone Paillier crypto-cost calibration + retroactive sec-cost table.

Runs a one-shot calibration of Paillier-1024 and Paillier-2048 on the
current host (matching the 12,299-param N-BaIoT MLP used in the headline
sweep) and reports the per-device c_enc cost. Then, for every cell in
sweep_headline/, recomputes what cost_sec WOULD have been if the sweep
had used Paillier instead of plain encryption -- by substituting the
calibrated c_enc into the existing per-round selection traces.

No re-training is performed. The accuracy / cost_comm / cost_train
numbers stay exactly as observed in the original run; only the
sec-cost column is refreshed.

Outputs LaTeX-ready tables to _table_cost_paillier.tex.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

from safel_dt.crypto.paillier import PaillierContext
from safel_dt.crypto.channel import PlaintextChannel
from safel_dt.crypto.signing import Signer
from safel_dt.crypto.calibration import (
    calibrate_channel_cost,
    calibrate_signing_cost,
)

# Model size used in the headline sweep (from calibration JSONs).
VEC_SIZE = 12_299
SWEEP = Path("results/runs/sweep_headline")
OUT_TEX = Path("_table_cost_paillier.tex")
CACHE = Path("_paillier_recalib_cache.json")
CELL_RE = re.compile(r"(?P<fog>[a-z_]+?)__(?P<cloud>[a-z0-9_]+?)__seed(?P<seed>\d{4})$")


def main() -> None:
    # ---- 1. Calibrate the three crypto costs at multiple key sizes ----
    print(f"=== calibration (vec_size={VEC_SIZE:,} params) ===\n", flush=True)

    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        c_auth     = float(cached["c_auth"])
        c_verify   = float(cached["c_verify"])
        c_enc_plain = float(cached["c_enc_plain"])
        c_enc_p1024 = float(cached["c_enc_p1024"])
        c_enc_p2048 = float(cached["c_enc_p2048"])
        print(f"  (loaded from cache: {CACHE.name})", flush=True)
        print(f"  c_auth (ECDSA sign):     {c_auth*1000:8.3f} ms / device")
        print(f"  c_verify (ECDSA verify): {c_verify*1000:8.3f} ms / device")
        print(f"  c_enc (plain):           {c_enc_plain*1000:8.3f} ms / device")
        print(f"  c_enc (Paillier-1024):   {c_enc_p1024:8.3f} s / device")
        print(f"  c_enc (Paillier-2048):   {c_enc_p2048:8.3f} s / device")
    else:
        signer = Signer.generate()
        c_auth, c_verify = calibrate_signing_cost(signer, n_warmup=2, n_trials=10)
        print(f"  c_auth (ECDSA sign):     {c_auth*1000:8.3f} ms / device", flush=True)
        print(f"  c_verify (ECDSA verify): {c_verify*1000:8.3f} ms / device", flush=True)

        plain = PlaintextChannel()
        c_enc_plain = calibrate_channel_cost(plain, vec_size=VEC_SIZE, n_warmup=1, n_trials=3)
        print(f"  c_enc (plain):           {c_enc_plain*1000:8.3f} ms / device", flush=True)

        print("\n  generating Paillier-1024 key pair (slow)...", flush=True)
        paillier_1024 = PaillierContext.generate(n_length=1024)
        print("  ...key ready, encrypting (1 warmup + 2 trials)...", flush=True)
        c_enc_p1024 = calibrate_channel_cost(paillier_1024, vec_size=VEC_SIZE, n_warmup=1, n_trials=2)
        print(f"  c_enc (Paillier-1024):   {c_enc_p1024:8.3f} s / device", flush=True)

        print("\n  generating Paillier-2048 key pair (slower)...", flush=True)
        paillier_2048 = PaillierContext.generate(n_length=2048)
        print("  ...key ready, encrypting (0 warmup + 1 trial, the slow part)...", flush=True)
        c_enc_p2048 = calibrate_channel_cost(paillier_2048, vec_size=VEC_SIZE, n_warmup=0, n_trials=1)
        print(f"  c_enc (Paillier-2048):   {c_enc_p2048:8.3f} s / device", flush=True)

        CACHE.write_text(json.dumps({
            "vec_size":    VEC_SIZE,
            "c_auth":      c_auth,
            "c_verify":    c_verify,
            "c_enc_plain": c_enc_plain,
            "c_enc_p1024": c_enc_p1024,
            "c_enc_p2048": c_enc_p2048,
        }, indent=2), encoding="utf-8")
        print(f"\n  [cached to {CACHE.name}]", flush=True)

    # ---- 2. Recompute mean per-round cost_sec for each cell ----
    print("\n=== retroactive cost_sec by cell (mean per-round, summed across fogs) ===\n")

    # The simulator records ``costs[<fog>].sec`` for every fog in every round,
    # computed from (c_enc + c_auth + c_verify) per device, scaled by
    # ``record_size_kb_i / mean_record_size_kb`` so that fast / slow devices
    # contribute proportionally. Because both ``c_enc`` and the ECDSA pair
    # share the *same* per-device record-size scaling, switching from plain
    # to Paillier is a uniform multiplicative factor on each row -- which
    # preserves the per-row heterogeneity already encoded in the trace
    # (different selected cohorts -> slightly different totals) without
    # re-running anything.
    per_device_plain = c_enc_plain + c_auth + c_verify
    per_device_p1024 = c_enc_p1024 + c_auth + c_verify
    per_device_p2048 = c_enc_p2048 + c_auth + c_verify
    ratio_p1024 = per_device_p1024 / per_device_plain
    ratio_p2048 = per_device_p2048 / per_device_plain
    print(f"  per-device totals: plain={per_device_plain*1000:.3f} ms, "
          f"P-1024={per_device_p1024:.3f} s, P-2048={per_device_p2048:.3f} s")
    print(f"  scaling ratios   : P-1024 / plain = {ratio_p1024:,.1f}x, "
          f"P-2048 / plain = {ratio_p2048:,.1f}x\n")

    results: dict[tuple[str, str], dict[str, float]] = {}
    for trace in sorted(SWEEP.glob("*.jsonl")):
        m = CELL_RE.match(trace.stem)
        if m is None:
            continue
        fog, cloud = m.group("fog"), m.group("cloud")
        rows = trace.read_text(encoding="utf-8").splitlines()
        cohort_per_round: list[int] = []
        sec_plain_per_round: list[float] = []
        for line in rows:
            d = json.loads(line)
            total = int(d.get("n_clients_accepted") or 0)
            if total == 0:
                spf = d.get("selected_per_fog") or {}
                total = sum(len(ids) for ids in spf.values() if ids is not None)
            cohort_per_round.append(total)
            costs = d.get("costs") or {}
            sec_round = 0.0
            for _fog_id, fog_costs in costs.items():
                sec_round += float(fog_costs.get("sec", 0.0))
            sec_plain_per_round.append(sec_round)
        if not cohort_per_round:
            continue
        mean_cohort = float(np.mean(cohort_per_round))
        mean_sec_plain = float(np.mean(sec_plain_per_round))
        results[(fog, cloud)] = {
            "mean_cohort": mean_cohort,
            "sec_plain":   mean_sec_plain,
            "sec_p1024":   mean_sec_plain * ratio_p1024,
            "sec_p2048":   mean_sec_plain * ratio_p2048,
        }
        print(f"  {fog:>10}+{cloud:<12}  cohort={mean_cohort:5.2f}  "
              f"plain={results[(fog, cloud)]['sec_plain']:6.3f}s  "
              f"P-1024={results[(fog, cloud)]['sec_p1024']:8.2f}s  "
              f"P-2048={results[(fog, cloud)]['sec_p2048']:8.2f}s")

    # ---- 3. Emit a LaTeX table ----
    fogs = ["all", "random", "heuristic", "sac"]
    clouds = ["static", "round_robin", "d3qn"]

    def fmt_s(v: float) -> str:
        """Format in seconds, with thousands sep for large values."""
        if v < 1.0:
            return f"${v:.3f}$"
        if v < 100.0:
            return f"${v:.1f}$"
        return f"${v:,.0f}$".replace(",", "\\,")

    def fmt_min(v_s: float) -> str:
        """Format a value originally in seconds as minutes (one decimal)."""
        return f"${v_s/60.0:.1f}$"

    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit()
    emit(r"% Generated by _paillier_recalib.py -- retroactive Paillier sec-cost")
    emit(r"% Per-device c_enc values (this host):")
    emit(rf"%   plain:        {c_enc_plain*1000:.3f} ms")
    emit(rf"%   Paillier-1024: {c_enc_p1024:.3f} s")
    emit(rf"%   Paillier-2048: {c_enc_p2048:.3f} s")
    emit(rf"% c_auth + c_verify (ECDSA, channel-independent): {(c_auth+c_verify)*1000:.3f} ms / device")
    emit(rf"% Model size: {VEC_SIZE:,} parameters")
    emit()
    emit(r"\begin{table}[!htbp]")
    emit(r"\centering")
    emit(r"\caption{Per-round security cost under three encryption configurations,")
    emit(r"  reconstructed by combining the cohort traces from the headline sweep")
    emit(r"  with crypto-cost calibration on the same host. The plain column matches")
    emit(r"  the run that produced the accuracy and comm-cost tables; the Paillier")
    emit(rf"  columns scale the measured per-device $c_{{\mathrm{{enc}}}}$ values")
    emit(rf"  (1024-bit: $\sim${c_enc_p1024:.1f}\,s, 2048-bit: $\sim${c_enc_p2048:.1f}\,s")
    emit(rf"  for the {VEC_SIZE:,}-parameter N-BaIoT MLP) by the mean cohort size")
    emit(r"  selected over the 300 rounds. $c_{\mathrm{auth}}+c_{\mathrm{verify}}$")
    emit(rf"  contribute a host-dependent constant $\sim${(c_auth+c_verify)*1000:.1f}\,ms per device")
    emit(r"  across all three columns.}")
    emit(r"\label{tab:sec-by-encryption}")
    emit(r"\footnotesize")
    emit(r"\begin{tabular}{ll ccc}")
    emit(r"\toprule")
    emit(r"\textbf{Fog policy} & \textbf{Cloud policy} & "
         r"\textbf{plain (s)} & \textbf{Paillier-1024 (s)} & \textbf{Paillier-2048 (s)} \\")
    emit(r"\midrule")
    for fi, fog in enumerate(fogs):
        if fi > 0:
            emit(r"\midrule")
        for cloud in clouds:
            r = results.get((fog, cloud))
            label_c = cloud.replace("_", r"\_")
            if r is None:
                emit(f"\\texttt{{{fog}}} & \\texttt{{{label_c}}} & --- & --- & --- \\\\")
                continue
            emit(f"\\texttt{{{fog}}} & \\texttt{{{label_c}}} & "
                 f"{fmt_s(r['sec_plain'])} & {fmt_s(r['sec_p1024'])} & {fmt_s(r['sec_p2048'])} \\\\")
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    emit(r"\end{table}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote] {OUT_TEX}")


if __name__ == "__main__":
    main()
