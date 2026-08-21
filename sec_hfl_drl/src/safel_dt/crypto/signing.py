"""Update authenticity: HMAC / ECDSA / ML-DSA backends.

Phase 1 (PQ-SAFEL-DT): classical ECDSA (Fmsa-DT transport) is replaced by
ML-DSA-65 via liboqs when ``alg="mldsa"``. HMAC remains the fast default for
unit tests; ECDSA (P-256) is the classical baseline for latency tables.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Literal

SignerBackend = Literal["hmac", "ecdsa", "mldsa"]

_DEFAULT_MLDSA = "ML-DSA-65"
_VALID = frozenset({"hmac", "ecdsa", "mldsa"})


@dataclass(frozen=True)
class SignedPayload:
    """Bytes that were signed plus the resulting signature blob."""

    payload: bytes
    signature: bytes
    public_key: bytes
    alg: str = "hmac"


def _require_oqs():
    try:
        import oqs  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ML-DSA requires liboqs-python. Install with: pip install -e '.[pqc]'"
        ) from exc
    return oqs


def _ecdsa_serialize_private(private_key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _ecdsa_serialize_public(public_key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _ecdsa_load_private(blob: bytes):
    from cryptography.hazmat.primitives import serialization

    return serialization.load_der_private_key(blob, password=None)


def _ecdsa_load_public(blob: bytes):
    from cryptography.hazmat.primitives import serialization

    return serialization.load_der_public_key(blob)


@dataclass
class Signer:
    """Sign encrypted FL updates; backend selected by ``alg``.

    * ``hmac`` — SHA-256 MAC (fast tests; not public-key)
    * ``ecdsa`` — NIST P-256 ECDSA (Fmsa-DT classical baseline)
    * ``mldsa`` — ML-DSA-65 via liboqs (FIPS 204 / Phase 1)
    """

    alg: str
    public_key: bytes
    _secret: bytes

    @classmethod
    def generate(cls, alg: SignerBackend | str = "hmac") -> Signer:
        name = str(alg).lower().strip()
        if name not in _VALID:
            raise ValueError(f"unknown sig alg {alg!r}; expected one of {sorted(_VALID)}")
        if name == "hmac":
            secret = os.urandom(32)
            public = hashlib.sha256(secret).digest()
            return cls(alg=name, public_key=public, _secret=secret)
        if name == "ecdsa":
            from cryptography.hazmat.primitives.asymmetric import ec

            sk = ec.generate_private_key(ec.SECP256R1())
            return cls(
                alg=name,
                public_key=_ecdsa_serialize_public(sk.public_key()),
                _secret=_ecdsa_serialize_private(sk),
            )
        # mldsa
        oqs = _require_oqs()
        with oqs.Signature(_DEFAULT_MLDSA) as sig:
            public = sig.generate_keypair()
            secret = sig.export_secret_key()
        return cls(alg=name, public_key=bytes(public), _secret=bytes(secret))

    def sign(self, body: bytes) -> SignedPayload:
        if self.alg == "hmac":
            sig = hmac.new(self._secret, body, hashlib.sha256).digest()
        elif self.alg == "ecdsa":
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec

            sk = _ecdsa_load_private(self._secret)
            sig = sk.sign(body, ec.ECDSA(hashes.SHA256()))
        else:
            oqs = _require_oqs()
            with oqs.Signature(_DEFAULT_MLDSA, secret_key=self._secret) as signer:
                sig = bytes(signer.sign(body))
        return SignedPayload(
            payload=body, signature=sig, public_key=self.public_key, alg=self.alg
        )

    def verify(self, signed: SignedPayload) -> bool:
        if signed.alg != self.alg:
            return False
        if signed.public_key != self.public_key:
            return False
        if self.alg == "hmac":
            expected = hmac.new(self._secret, signed.payload, hashlib.sha256).digest()
            return hmac.compare_digest(expected, signed.signature)
        return verify_signature(signed)


def verify_signature(signed: SignedPayload, *, public_key: bytes | None = None) -> bool:
    """Verify ``signed``; ECDSA/ML-DSA are public-key checks.

    HMAC cannot be verified without the MAC key. Callers that only have the
    public identifier fall back to a structural check (legacy simulator path).
    Prefer :meth:`Signer.verify` when the originating signer is available.
    """
    if public_key is not None and signed.public_key != public_key:
        return False
    if not (
        isinstance(signed.payload, (bytes, bytearray))
        and isinstance(signed.signature, (bytes, bytearray))
        and isinstance(signed.public_key, (bytes, bytearray))
        and len(signed.signature) >= 16
        and len(signed.public_key) >= 16
    ):
        return False

    alg = getattr(signed, "alg", "hmac") or "hmac"
    if alg == "hmac":
        return len(signed.signature) == 32

    if alg == "ecdsa":
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        try:
            pk = _ecdsa_load_public(signed.public_key)
            pk.verify(signed.signature, signed.payload, ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

    if alg == "mldsa":
        oqs = _require_oqs()
        try:
            with oqs.Signature(_DEFAULT_MLDSA) as verifier:
                return bool(verifier.verify(signed.payload, signed.signature, signed.public_key))
        except Exception:
            return False

    return False


def signature_nbytes(alg: SignerBackend | str) -> int:
    """Typical signature size in bytes (for cost / Fmsa-DT bridge notes)."""
    name = str(alg).lower().strip()
    if name == "hmac":
        return 32
    if name == "ecdsa":
        return 64  # ~P-256 DER varies; report nominal raw size
    if name == "mldsa":
        return 3309  # ML-DSA-65
    raise ValueError(f"unknown sig alg {alg!r}")
