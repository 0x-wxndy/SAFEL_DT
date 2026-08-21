"""One-shot crypto cost calibration for device profile substitution."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from safel_dt.crypto.channel import SecureChannel
from safel_dt.crypto.signing import Signer


@dataclass(frozen=True)
class CryptoCosts:
    c_enc: float
    c_auth: float
    c_verify: float


@dataclass(frozen=True)
class CalibrationReport:
    costs: CryptoCosts
    encryption_mode: str
    paillier_keybits: int | None
    vec_size: int
    n_warmup: int
    n_trials: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_channel_cost(
    channel: SecureChannel,
    *,
    vec_size: int,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> float:
    """Mean wall time (seconds) to encrypt a ``vec_size`` vector."""
    vec = np.zeros(vec_size, dtype=np.float64)
    for _ in range(max(0, n_warmup)):
        channel.encrypt_vector(vec)
    times: list[float] = []
    for _ in range(max(1, n_trials)):
        t0 = time.perf_counter()
        channel.encrypt_vector(vec)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def calibrate_signing_cost(
    signer: Signer,
    *,
    n_warmup: int = 1,
    n_trials: int = 10,
    body_size: int = 64,
) -> tuple[float, float]:
    """Return ``(c_auth, c_verify)`` mean wall times in seconds."""
    body = b"\x00" * body_size
    for _ in range(max(0, n_warmup)):
        signed = signer.sign(body)
        signer.verify(signed)
    auth_times: list[float] = []
    verify_times: list[float] = []
    for _ in range(max(1, n_trials)):
        t0 = time.perf_counter()
        signed = signer.sign(body)
        auth_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        ok = signer.verify(signed)
        verify_times.append(time.perf_counter() - t1)
        if not ok:
            raise RuntimeError("signer failed self-verify during calibration")
    return float(np.mean(auth_times)), float(np.mean(verify_times))


def run_calibration(
    *,
    channel: SecureChannel,
    signer: Signer,
    vec_size: int,
    encryption_mode: str,
    paillier_keybits: int | None,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> CalibrationReport:
    """Calibrate encrypt + sign/verify costs for the live algorithms."""
    c_enc = calibrate_channel_cost(
        channel, vec_size=vec_size, n_warmup=n_warmup, n_trials=n_trials
    )
    c_auth, c_verify = calibrate_signing_cost(
        signer, n_warmup=n_warmup, n_trials=max(n_trials, 5)
    )
    return CalibrationReport(
        costs=CryptoCosts(c_enc=c_enc, c_auth=c_auth, c_verify=c_verify),
        encryption_mode=encryption_mode,
        paillier_keybits=paillier_keybits,
        vec_size=vec_size,
        n_warmup=n_warmup,
        n_trials=n_trials,
    )
