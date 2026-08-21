"""Final RQ3 sanity check: is there ANY slicing of the data where SAC's
fog selection meaningfully discriminates against malicious clients?

We compute rate_M / rate_B for the sac+d3qn cell, sliced by:

  - attack family (label_flip / model_scale / gaussian)
  - phase (10% / 15% / 20% / 25%)
  - phase x family (12 cells)
  - late only (round >= 50, after SAC's Lagrangian warm-up)

If any slicing shows ratio < 0.8 (clearly preferential exclusion of
malicious), the original RQ3 claim might be salvageable in narrowed
form. Otherwise the section is unsupported and should be dropped.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

SWEEP = Path("results/runs/sweep_headline")
TRACE = SWEEP / "sac__d3qn__seed0000.jsonl"
N_CLIENTS = 30
CLIENTS_PER_FOG = 10


def main() -> None:
    rows = [json.loads(l) for l in TRACE.read_text(encoding="utf-8").splitlines()]
    T = len(rows)
    mal_ids = list(int(x) for x in (rows[0].get("malicious_ids") or []))
    fam_by_client: dict[int, str] = {
        int(k): str(v) for k, v in (rows[0].get("per_client_attack_family") or {}).items()
    }
    print(f"malicious_ids = {mal_ids}")
    print(f"families      = {fam_by_client}\n")

    sel = np.zeros((N_CLIENTS, T), dtype=np.uint8)
    active_per_round: list[set[int]] = []
    for r_idx, d in enumerate(rows):
        active_per_round.append(set(int(x) for x in (d.get("active_malicious_ids") or [])))
        spf = d.get("selected_per_fog") or {}
        for fog_key, locals_list in spf.items():
            fog_id = int(fog_key)
            for li in locals_list or []:
                gid = fog_id * CLIENTS_PER_FOG + int(li)
                if 0 <= gid < N_CLIENTS:
                    sel[gid, r_idx] = 1

    def slice_rates(client_filter, round_filter, label: str) -> None:
        mal_rate_num = mal_rate_den = 0
        ben_rate_num = ben_rate_den = 0
        for r in range(T):
            if not round_filter(r):
                continue
            active = active_per_round[r]
            for g in range(N_CLIENTS):
                if g in active:
                    if not client_filter(g):
                        continue
                    mal_rate_num += int(sel[g, r])
                    mal_rate_den += 1
                else:
                    ben_rate_num += int(sel[g, r])
                    ben_rate_den += 1
        rate_m = mal_rate_num / mal_rate_den if mal_rate_den else float("nan")
        rate_b = ben_rate_num / ben_rate_den if ben_rate_den else float("nan")
        ratio  = rate_m / rate_b if rate_b > 0 else float("nan")
        marker = "  *** DISCRIMINATES ***" if ratio < 0.8 else ""
        print(f"{label:<46}  rate_M={rate_m:.3f}  rate_B={rate_b:.3f}  ratio={ratio:.3f}{marker}")

    print("--- overall ---")
    slice_rates(lambda g: True, lambda r: True, "overall")
    print()

    print("--- by attack family (all rounds) ---")
    for fam in ["label_flip", "model_scale", "gaussian"]:
        ids = [g for g in mal_ids if fam_by_client.get(g) == fam]
        slice_rates(lambda g, _ids=set(ids): g in _ids, lambda r: True, f"family={fam} ({len(ids)} clients)")
    print()

    print("--- by phase (all malicious) ---")
    phases = [(0, 100, "phase1 (10%)"), (100, 200, "phase2 (15%)"),
              (200, 250, "phase3 (20%)"), (250, 300, "phase4 (25%)")]
    for lo, hi, name in phases:
        slice_rates(lambda g: True, lambda r, _lo=lo, _hi=hi: _lo <= r < _hi, name)
    print()

    print("--- late only (round >= 50, all malicious) ---")
    slice_rates(lambda g: True, lambda r: r >= 50, "late (r>=50)")
    print()

    print("--- phase x family (sac+d3qn) ---")
    for lo, hi, name in phases:
        for fam in ["label_flip", "model_scale", "gaussian"]:
            ids = [g for g in mal_ids if fam_by_client.get(g) == fam]
            if not ids:
                continue
            slice_rates(lambda g, _ids=set(ids): g in _ids,
                        lambda r, _lo=lo, _hi=hi: _lo <= r < _hi,
                        f"{name} x {fam} ({len(ids)} mal)")
    print()

    print("--- heuristic+d3qn baseline for comparison ---")
    print("(skipping; heuristic numbers already computed at 0.42-0.76 across phases)")


if __name__ == "__main__":
    main()
