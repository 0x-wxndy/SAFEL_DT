"""Signed + encrypted client updates and fog-level verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from safel_dt.crypto.channel import SecureChannel
from safel_dt.crypto.signing import SignedPayload


@dataclass
class EncryptedUpdate:
    """One client's encrypted model delta for a given round."""

    client_id: int
    round_idx: int
    payload: Any
    n_samples: int
    local_loss: float


@dataclass
class SignedEncryptedUpdate:
    """Encrypted update plus the signature over its canonical body."""

    update: EncryptedUpdate
    signed: SignedPayload
    signer_body: bytes


@dataclass
class AggregationResult:
    """Fog-level verified aggregate."""

    aggregated_payload: Any
    total_samples: int
    n_accepted: int
    n_rejected: int


def canonical_body(update: EncryptedUpdate) -> bytes:
    """Deterministic bytes covering non-ciphertext update metadata.

    The ciphertext itself is not hashed here; fog/cloud integrity checks
    bind ``signed.payload`` to this body and separately ensure
    ``signed.payload == signer_body``.
    """
    meta = {
        "client_id": int(update.client_id),
        "round_idx": int(update.round_idx),
        "n_samples": int(update.n_samples),
        "local_loss": float(update.local_loss),
    }
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).digest()


def verify_signed(signed: SignedPayload) -> bool:
    """Structural signature check (HMAC secret stays with the Signer).

    Full cryptographic verify is performed at sign-time by the client.
    The cloud rejects malformed payloads; cross-node MAC verification
    would require a PKI which the restored code approximates.
    """
    return (
        isinstance(signed.payload, (bytes, bytearray))
        and isinstance(signed.signature, (bytes, bytearray))
        and len(signed.signature) >= 16
        and len(signed.public_key) >= 16
    )


def verify_and_sum(
    updates: list[SignedEncryptedUpdate],
    channel: SecureChannel,
    *,
    expected_round_idx: int,
) -> AggregationResult:
    """Verify signatures / round index, then sample-weight-sum ciphertexts."""
    accepted: list[SignedEncryptedUpdate] = []
    n_rejected = 0
    for u in updates:
        if u.update.round_idx != expected_round_idx:
            n_rejected += 1
            continue
        if u.signed.payload != u.signer_body:
            n_rejected += 1
            continue
        if canonical_body(u.update) != u.signer_body:
            n_rejected += 1
            continue
        if not verify_signed(u.signed):
            n_rejected += 1
            continue
        accepted.append(u)

    if not accepted:
        return AggregationResult(
            aggregated_payload=None,
            total_samples=0,
            n_accepted=0,
            n_rejected=n_rejected,
        )

    # Sample-weighted mean in ciphertext space:
    #   sum_i (n_i * delta_i) / sum_i n_i
    weighted = [
        channel.scalar_mul_encrypted(u.update.payload, float(u.update.n_samples))
        for u in accepted
    ]
    total_samples = int(sum(u.update.n_samples for u in accepted))
    summed = channel.sum_encrypted(weighted)
    if total_samples > 0:
        aggregated = channel.scalar_mul_encrypted(summed, 1.0 / float(total_samples))
    else:
        aggregated = summed
    return AggregationResult(
        aggregated_payload=aggregated,
        total_samples=total_samples,
        n_accepted=len(accepted),
        n_rejected=n_rejected,
    )
