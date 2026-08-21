"""Secure channel abstractions (plaintext + Paillier HE)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SecureChannel(Protocol):
    """Minimal HE / plaintext channel used by fog + cloud aggregation."""

    @property
    def has_private_key(self) -> bool: ...

    def encrypt_vector(self, vec: np.ndarray) -> Any: ...

    def decrypt_vector(self, payload: Any) -> np.ndarray: ...

    def sum_encrypted(self, payloads: list[Any]) -> Any: ...

    def scalar_mul_encrypted(self, payload: Any, scalar: float) -> Any: ...


class PlaintextChannel:
    """Identity channel: payloads are plain ``np.ndarray`` vectors."""

    @property
    def has_private_key(self) -> bool:
        return True

    def encrypt_vector(self, vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64).copy()

    def decrypt_vector(self, payload: np.ndarray) -> np.ndarray:
        return np.asarray(payload, dtype=np.float64).copy()

    def sum_encrypted(self, payloads: list[np.ndarray]) -> np.ndarray:
        if not payloads:
            raise ValueError("sum_encrypted requires at least one payload")
        acc = np.asarray(payloads[0], dtype=np.float64).copy()
        for p in payloads[1:]:
            acc = acc + np.asarray(p, dtype=np.float64)
        return acc

    def scalar_mul_encrypted(self, payload: np.ndarray, scalar: float) -> np.ndarray:
        return np.asarray(payload, dtype=np.float64) * float(scalar)
