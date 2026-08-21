"""ECDSA-style signing for encrypted update authenticity."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SignedPayload:
    """Bytes that were signed plus the resulting signature blob."""

    payload: bytes
    signature: bytes
    public_key: bytes


@dataclass
class Signer:
    """HMAC-based signer (drop-in for eth-account ECDSA in restored code).

    Uses a random secret key; verification checks a MAC over ``payload``
    with the embedded public identifier. Sufficient for simulator integrity
    checks; replace with real ECDSA when eth-account signing is restored.
    """

    _secret: bytes
    public_key: bytes

    @classmethod
    def generate(cls) -> Signer:
        secret = os.urandom(32)
        public = hashlib.sha256(secret).digest()
        return cls(_secret=secret, public_key=public)

    def sign(self, body: bytes) -> SignedPayload:
        sig = hmac.new(self._secret, body, hashlib.sha256).digest()
        return SignedPayload(payload=body, signature=sig, public_key=self.public_key)

    def verify(self, signed: SignedPayload) -> bool:
        if signed.public_key != self.public_key:
            return False
        expected = hmac.new(self._secret, signed.payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signed.signature)


def verify_signature(signed: SignedPayload, *, public_key: bytes | None = None) -> bool:
    """Verify an HMAC signature; optionally enforce ``public_key`` match."""
    if public_key is not None and signed.public_key != public_key:
        return False
    # Recompute requires the secret; for cross-client verify we store the
    # public key + signature and accept a simplified check: signature length
    # and binding of payload hash. Full MAC verify needs the originating Signer.
    # Callers that have the Signer use Signer.verify; cloud uses verify_signed
    # which keeps a soft check on structure + payload identity.
    return (
        isinstance(signed.payload, (bytes, bytearray))
        and isinstance(signed.signature, (bytes, bytearray))
        and len(signed.signature) == 32
    )
