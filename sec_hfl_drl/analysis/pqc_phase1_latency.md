# Phase 1 — PQ authentication latency

Host microbench: `50` trials, warmup `3`, sign body `256` B.

## Signatures (`T_sig`, `T_verify`)

| Alg | `T_sig` (ms) | `T_verify` (ms) | sig bytes |
|---|---:|---:|---:|
| `hmac` | 0.0103 | 0.0071 | 32 |
| `ecdsa` | 0.1502 | 0.1790 | 71 |
| `mldsa` | 0.2886 | 0.1203 | 3309 |

## KEM / TLS–MQTT bridge (`T_encap`, `T_decap`)

| Alg | `T_encap` (ms) | `T_decap` (ms) | pk B | ct B |
|---|---:|---:|---:|---:|
| `ecdh` | 0.2369 | 0.2140 | 91 | 91 |
| `mlkem` | 0.0472 | 0.0470 | 1184 | 1088 |

## Fmsa-DT P4 bridge (`T_cycle ≤ 2000 ms`)

Fmsa-DT discharged P4 with classical ECDSA on the sensing→actuation path:

| Stage (Fmsa-DT) | Median (ms) |
|---|---:|
| `T_sig` ECDSA | 6.7 |
| `T_verify` ECDSA | 6.0 |
| `T_cycle` budget `δ` | **2000** |

Substituting Phase-1 PQ primitives measured on this host:

- Auth wall time: ECDSA `0.329` ms → ML-DSA-65 `0.409` ms (Δ `+0.080` ms vs Fmsa report `12.7` ms).
- KEM handshake: ECDH `0.451` ms → ML-KEM-768 `0.094` ms (Δ `-0.357` ms).
- Residual headroom vs `δ=2000` ms if only auth+KEM change (ignoring Paillier/ML): **`1999.5` ms** (OK).

**Note:** SAFEL-DT FL rounds are dominated by local train + (Phase 2) HE; Phase 1 only asks whether Fmsa-DT's actuation cycle still clears `δ` when ECDSA/ECDH are swapped for ML-DSA/ML-KEM. Full FL `cost_sec` recalibration is separate (calibration sidecar).
