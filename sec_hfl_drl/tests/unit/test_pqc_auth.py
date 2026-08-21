"""Unit tests for Phase-1 PQ authentication (HMAC / ECDSA / ML-DSA + KEM)."""

from __future__ import annotations

import pytest

from safel_dt.crypto.signing import Signer, signature_nbytes, verify_signature
from safel_dt.fl.secure_aggregation import verify_signed


def test_hmac_roundtrip() -> None:
    signer = Signer.generate("hmac")
    body = b"round-7-update"
    signed = signer.sign(body)
    assert signed.alg == "hmac"
    assert signer.verify(signed)
    assert verify_signature(signed)
    assert verify_signed(signed)
    assert not signer.verify(
        signed.__class__(
            payload=b"tampered",
            signature=signed.signature,
            public_key=signed.public_key,
            alg=signed.alg,
        )
    )


def test_ecdsa_roundtrip_and_public_verify() -> None:
    signer = Signer.generate("ecdsa")
    body = b"fog-aggregate-body"
    signed = signer.sign(body)
    assert signed.alg == "ecdsa"
    assert signer.verify(signed)
    assert verify_signature(signed)
    assert verify_signed(signed)
    bad = signed.__class__(
        payload=b"evil",
        signature=signed.signature,
        public_key=signed.public_key,
        alg=signed.alg,
    )
    assert not verify_signature(bad)


@pytest.mark.requires_pqc
def test_mldsa_roundtrip_and_public_verify() -> None:
    pytest.importorskip("oqs")
    signer = Signer.generate("mldsa")
    body = b"ml-dsa-client-delta"
    signed = signer.sign(body)
    assert signed.alg == "mldsa"
    assert len(signed.signature) == signature_nbytes("mldsa")
    assert signer.verify(signed)
    assert verify_signature(signed)
    assert verify_signed(signed)
    other = Signer.generate("mldsa")
    assert not other.verify(signed)


@pytest.mark.requires_pqc
def test_mldsa_rejects_wrong_public_key() -> None:
    pytest.importorskip("oqs")
    a = Signer.generate("mldsa")
    b = Signer.generate("mldsa")
    signed = a.sign(b"payload")
    forged = signed.__class__(
        payload=signed.payload,
        signature=signed.signature,
        public_key=b.public_key,
        alg=signed.alg,
    )
    assert not verify_signature(forged)


def test_unknown_sig_alg_raises() -> None:
    with pytest.raises(ValueError, match="unknown sig alg"):
        Signer.generate("rsa")  # type: ignore[arg-type]


@pytest.mark.requires_pqc
def test_mlkem_handshake_agrees() -> None:
    pytest.importorskip("oqs")
    from safel_dt.transport.kem import handshake_roundtrip, kem_nbytes

    result, session = handshake_roundtrip("mlkem")
    assert result.alg == "mlkem"
    assert len(session) == 32
    sizes = kem_nbytes("mlkem")
    assert len(result.ciphertext) == sizes["ciphertext"]


def test_ecdh_handshake_agrees() -> None:
    from safel_dt.transport.kem import handshake_roundtrip

    result, session = handshake_roundtrip("ecdh")
    assert result.alg == "ecdh"
    assert len(session) == 32
