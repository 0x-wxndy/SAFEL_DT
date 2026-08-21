"""Paillier additive HE channel (via ``phe``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from phe import paillier


@dataclass
class PaillierContext:
    """Shared Paillier keypair used as a :class:`SecureChannel`.

    Floats are encoded with a fixed-point scale before encryption so the
    FedAvg path (sum + scalar mul) stays exact under additive HE.
    """

    public_key: paillier.PaillierPublicKey
    private_key: paillier.PaillierPrivateKey | None = None
    scale: float = 1e6

    @classmethod
    def generate(cls, n_length: int = 1024, *, scale: float = 1e6) -> PaillierContext:
        pub, priv = paillier.generate_paillier_keypair(n_length=n_length)
        return cls(public_key=pub, private_key=priv, scale=float(scale))

    @property
    def has_private_key(self) -> bool:
        return self.private_key is not None

    def encrypt_vector(self, vec: np.ndarray) -> list[Any]:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        return [self.public_key.encrypt(int(round(float(v) * self.scale))) for v in arr]

    def decrypt_vector(self, payload: list[Any]) -> np.ndarray:
        if self.private_key is None:
            raise RuntimeError("cannot decrypt without private key")
        vals = [self.private_key.decrypt(c) / self.scale for c in payload]
        return np.asarray(vals, dtype=np.float64)

    def sum_encrypted(self, payloads: list[list[Any]]) -> list[Any]:
        if not payloads:
            raise ValueError("sum_encrypted requires at least one payload")
        acc = list(payloads[0])
        for other in payloads[1:]:
            if len(other) != len(acc):
                raise ValueError("encrypted vector length mismatch")
            acc = [a + b for a, b in zip(acc, other, strict=True)]
        return acc

    def scalar_mul_encrypted(self, payload: list[Any], scalar: float) -> list[Any]:
        # Fixed-point: multiplying plaintext by s is multiplying ciphertext
        # by s. For non-integer scalars we keep float mul (phe supports it
        # via EncryptedNumber.__mul__).
        return [c * float(scalar) for c in payload]
