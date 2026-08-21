"""Key encapsulation: classical ECDH vs ML-KEM (TLS/MQTT transport bridge).

Fmsa-DT uses MQTT over TLS 1.3 with classical ECDH. Phase 1 of PQ-SAFEL-DT
replaces that handshake with ML-KEM-768 (FIPS 203) via liboqs. This module
is a **simulator / microbench adapter** — it does not replace OpenSSL; it
measures encapsulate/decapsulate costs and shared-secret agreement so we
can fold ``T_kem`` into the Fmsa-DT ``T_cycle ≤ 2000 ms`` budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

KemBackend = Literal["ecdh", "mlkem"]

_DEFAULT_MLKEM = "ML-KEM-768"
_VALID = frozenset({"ecdh", "mlkem"})


def _require_oqs():
    try:
        import oqs  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ML-KEM requires liboqs-python. Install with: pip install -e '.[pqc]'"
        ) from exc
    return oqs


@dataclass(frozen=True)
class KemHandshakeResult:
    """One client→server KEM handshake outcome."""

    shared_secret: bytes
    ciphertext: bytes
    public_key: bytes
    alg: str


@dataclass
class KemEndpoint:
    """Local KEM key pair (fog/cloud or TLS terminator stand-in)."""

    alg: str
    public_key: bytes
    _secret: bytes

    @classmethod
    def generate(cls, alg: KemBackend | str = "ecdh") -> KemEndpoint:
        name = str(alg).lower().strip()
        if name not in _VALID:
            raise ValueError(f"unknown kem alg {alg!r}; expected one of {sorted(_VALID)}")
        if name == "ecdh":
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ec

            sk = ec.generate_private_key(ec.SECP256R1())
            public = sk.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            secret = sk.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            return cls(alg=name, public_key=public, _secret=secret)

        oqs = _require_oqs()
        with oqs.KeyEncapsulation(_DEFAULT_MLKEM) as kem:
            public = kem.generate_keypair()
            secret = kem.export_secret_key()
        return cls(alg=name, public_key=bytes(public), _secret=bytes(secret))

    def encapsulate(self, peer_public_key: bytes) -> KemHandshakeResult:
        """Client side: encapsulate to the peer's public key."""
        if self.alg == "ecdh":
            # For ECDH the "encapsulator" also holds an ephemeral key; the
            # peer public is the server static/ephemeral key.
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF

            peer = serialization.load_der_public_key(peer_public_key)
            eph = ec.generate_private_key(ec.SECP256R1())
            shared = eph.exchange(ec.ECDH(), peer)  # type: ignore[arg-type]
            secret = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"safel-dt/ecdh",
            ).derive(shared)
            ct = eph.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            return KemHandshakeResult(
                shared_secret=secret,
                ciphertext=ct,
                public_key=peer_public_key,
                alg=self.alg,
            )

        oqs = _require_oqs()
        with oqs.KeyEncapsulation(_DEFAULT_MLKEM) as kem:
            ct, ss = kem.encap_secret(peer_public_key)
        return KemHandshakeResult(
            shared_secret=bytes(ss),
            ciphertext=bytes(ct),
            public_key=peer_public_key,
            alg=self.alg,
        )

    def decapsulate(self, ciphertext: bytes) -> bytes:
        """Server side: recover the shared secret from the client CT."""
        if self.alg == "ecdh":
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF

            sk = serialization.load_der_private_key(self._secret, password=None)
            peer_eph = serialization.load_der_public_key(ciphertext)
            shared = sk.exchange(ec.ECDH(), peer_eph)  # type: ignore[arg-type]
            return HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"safel-dt/ecdh",
            ).derive(shared)

        oqs = _require_oqs()
        with oqs.KeyEncapsulation(_DEFAULT_MLKEM, secret_key=self._secret) as kem:
            return bytes(kem.decap_secret(ciphertext))


def mqtt_session_key(shared_secret: bytes, *, topic: str = "safel/dt") -> bytes:
    """Derive a 32-byte MQTT/TLS application key from the KEM shared secret."""
    return hashlib.sha256(shared_secret + b"|" + topic.encode("utf-8")).digest()


def handshake_roundtrip(
    alg: KemBackend | str = "mlkem",
) -> tuple[KemHandshakeResult, bytes]:
    """Server keygen → client encapsulate → server decapsulate (agreement check)."""
    server = KemEndpoint.generate(alg)
    # Client only needs the alg label for ECDH ephemeral; for ML-KEM encapsulate
    # uses the server public key via a throwaway endpoint of the same alg.
    client = KemEndpoint.generate(alg)
    result = client.encapsulate(server.public_key)
    recovered = server.decapsulate(result.ciphertext)
    if not hmac_compare(result.shared_secret, recovered):
        raise RuntimeError(f"{alg} KEM shared-secret mismatch")
    return result, mqtt_session_key(recovered)


def hmac_compare(a: bytes, b: bytes) -> bool:
    import hmac as _hmac

    return _hmac.compare_digest(a, b)


def kem_nbytes(alg: KemBackend | str) -> dict[str, int]:
    """Nominal public-key / ciphertext sizes (bytes)."""
    name = str(alg).lower().strip()
    if name == "ecdh":
        return {"public_key": 91, "ciphertext": 91, "shared_secret": 32}  # DER P-256
    if name == "mlkem":
        return {"public_key": 1184, "ciphertext": 1088, "shared_secret": 32}  # ML-KEM-768
    raise ValueError(f"unknown kem alg {alg!r}")
