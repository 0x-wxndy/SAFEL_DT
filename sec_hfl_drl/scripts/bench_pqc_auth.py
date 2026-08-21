#!/usr/bin/env python3
"""Microbench: T_sig / T_verify (ECDSA vs ML-DSA) and T_kem (ECDH vs ML-KEM).

Writes a markdown latency table (stdout + optional --out) for the Phase-1
report and the Fmsa-DT P4 bridge (T_cycle ≤ 2000 ms).

Usage (venv active, from sec_hfl_drl/):
  pip install -e '.[pqc]'
  python -m scripts.bench_pqc_auth --trials 200 --out analysis/pqc_phase1_latency.md
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from safel_dt.crypto.calibration import calibrate_signing_cost
from safel_dt.crypto.signing import Signer, signature_nbytes


def _ms(seconds: float) -> float:
    return 1000.0 * seconds


def _bench_kem(alg: str, *, n_warmup: int, n_trials: int) -> dict[str, float]:
    from safel_dt.transport.kem import KemEndpoint, kem_nbytes

    server = KemEndpoint.generate(alg)
    client = KemEndpoint.generate(alg)
    for _ in range(n_warmup):
        r = client.encapsulate(server.public_key)
        server.decapsulate(r.ciphertext)

    encap_t: list[float] = []
    decap_t: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        r = client.encapsulate(server.public_key)
        encap_t.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        server.decapsulate(r.ciphertext)
        decap_t.append(time.perf_counter() - t1)

    sizes = kem_nbytes(alg)
    return {
        "T_encap_ms": _ms(statistics.mean(encap_t)),
        "T_decap_ms": _ms(statistics.mean(decap_t)),
        "pk_B": float(sizes["public_key"]),
        "ct_B": float(sizes["ciphertext"]),
    }


def _bench_sig(alg: str, *, n_warmup: int, n_trials: int, body_size: int) -> dict[str, float]:
    signer = Signer.generate(alg)
    c_auth, c_verify = calibrate_signing_cost(
        signer, n_warmup=n_warmup, n_trials=n_trials, body_size=body_size
    )
    signed = signer.sign(b"\x00" * body_size)
    return {
        "T_sig_ms": _ms(c_auth),
        "T_verify_ms": _ms(c_verify),
        "sig_B": float(len(signed.signature)),
        "sig_B_nominal": float(signature_nbytes(alg)),
    }


def _fmsa_bridge(rows: dict[str, dict[str, float]]) -> str:
    """Compare new PQ auth+KEM costs against Fmsa-DT P4 classical ECDSA budget."""
    # Fmsa-DT P4 medians (Thesis-plan / SoSyM): T_sig=6.7, T_verify=6.0, δ=2000 ms
    fmsa_sig = 6.7
    fmsa_ver = 6.0
    delta = 2000.0
    ecdsa = rows.get("ecdsa", {})
    mldsa = rows.get("mldsa", {})
    ecdh = rows.get("ecdh_kem", {})
    mlkem = rows.get("mlkem", {})

    lines = [
        "## Fmsa-DT P4 bridge (`T_cycle ≤ 2000 ms`)",
        "",
        "Fmsa-DT discharged P4 with classical ECDSA on the sensing→actuation path:",
        "",
        "| Stage (Fmsa-DT) | Median (ms) |",
        "|---|---:|",
        "| `T_sig` ECDSA | 6.7 |",
        "| `T_verify` ECDSA | 6.0 |",
        "| `T_cycle` budget `δ` | **2000** |",
        "",
        "Substituting Phase-1 PQ primitives measured on this host:",
        "",
    ]
    if mldsa and ecdsa:
        pq_auth = mldsa["T_sig_ms"] + mldsa["T_verify_ms"]
        cl_auth = ecdsa["T_sig_ms"] + ecdsa["T_verify_ms"]
        lines.append(
            f"- Auth wall time: ECDSA `{cl_auth:.3f}` ms → ML-DSA-65 "
            f"`{pq_auth:.3f}` ms "
            f"(Δ `{pq_auth - cl_auth:+.3f}` ms vs Fmsa report "
            f"`{fmsa_sig + fmsa_ver:.1f}` ms)."
        )
    if mlkem and ecdh:
        pq_kem = mlkem["T_encap_ms"] + mlkem["T_decap_ms"]
        cl_kem = ecdh["T_encap_ms"] + ecdh["T_decap_ms"]
        lines.append(
            f"- KEM handshake: ECDH `{cl_kem:.3f}` ms → ML-KEM-768 "
            f"`{pq_kem:.3f}` ms (Δ `{pq_kem - cl_kem:+.3f}` ms)."
        )
        budget_left = delta - (mldsa.get("T_sig_ms", 0) + mldsa.get("T_verify_ms", 0) + pq_kem)
        lines.append(
            f"- Residual headroom vs `δ=2000` ms if only auth+KEM change "
            f"(ignoring Paillier/ML): **`{budget_left:.1f}` ms** "
            f"{'(OK)' if budget_left > 0 else '(TIGHT / FAIL)'}."
        )
    lines.append("")
    lines.append(
        "**Note:** SAFEL-DT FL rounds are dominated by local train + (Phase 2) HE; "
        "Phase 1 only asks whether Fmsa-DT's actuation cycle still clears `δ` when "
        "ECDSA/ECDH are swapped for ML-DSA/ML-KEM. Full FL `cost_sec` recalibration "
        "is separate (calibration sidecar)."
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--body-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows: dict[str, dict[str, float]] = {}
    md: list[str] = [
        "# Phase 1 — PQ authentication latency",
        "",
        f"Host microbench: `{args.trials}` trials, warmup `{args.warmup}`, "
        f"sign body `{args.body_size}` B.",
        "",
        "## Signatures (`T_sig`, `T_verify`)",
        "",
        "| Alg | `T_sig` (ms) | `T_verify` (ms) | sig bytes |",
        "|---|---:|---:|---:|",
    ]

    for alg in ("hmac", "ecdsa", "mldsa"):
        try:
            r = _bench_sig(alg, n_warmup=args.warmup, n_trials=args.trials, body_size=args.body_size)
        except ImportError as exc:
            md.append(f"| `{alg}` | — | — | skipped ({exc}) |")
            continue
        rows[alg] = r
        md.append(
            f"| `{alg}` | {r['T_sig_ms']:.4f} | {r['T_verify_ms']:.4f} | "
            f"{int(r['sig_B'])} |"
        )

    md.extend(
        [
            "",
            "## KEM / TLS–MQTT bridge (`T_encap`, `T_decap`)",
            "",
            "| Alg | `T_encap` (ms) | `T_decap` (ms) | pk B | ct B |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for alg, key in (("ecdh", "ecdh_kem"), ("mlkem", "mlkem")):
        try:
            r = _bench_kem(alg, n_warmup=args.warmup, n_trials=args.trials)
        except ImportError as exc:
            md.append(f"| `{alg}` | — | — | — | skipped ({exc}) |")
            continue
        rows[key] = r
        md.append(
            f"| `{alg}` | {r['T_encap_ms']:.4f} | {r['T_decap_ms']:.4f} | "
            f"{int(r['pk_B'])} | {int(r['ct_B'])} |"
        )

    md.append("")
    md.append(_fmsa_bridge(rows))
    md.append("")
    text = "\n".join(md)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[bench_pqc_auth] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
