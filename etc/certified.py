"""
etc.certified
=============
Certified: a typed wrapper that pairs a mathematical value with its proof.

Every discrete output in ETC (sign, winding_number, ...) returns a
Certified object so the caller always has access to the underlying proof
certificate alongside the computed value.

Usage
-----
    result = sign(x)
    print(result.value)   # -1, 0, or +1
    print(result.proof)   # Certificate object
    print(result)         # Certified(value=-1, claim='sign(x) = -1', VALID)
"""

from __future__ import annotations
from typing import Any, Generic, TypeVar

V = TypeVar("V")


class Certified(Generic[V]):
    """
    A value paired with a proof certificate.

    Attributes
    ----------
    value : V
        The computed result (e.g. -1, 0, 1 for sign; int for winding_number).
    proof : Certificate
        The machine-checkable evidence that `value` is correct.

    The certificate is always a ``etc.verify.certificate.Certificate``
    instance (or subclass).  Use ``result.proof.summary()`` to inspect
    the evidence, or ``result.proof.to_json()`` to serialise it.
    """

    __slots__ = ("value", "proof")

    def __init__(self, value: V, proof: Any) -> None:
        self.value = value
        self.proof = proof

    def is_valid(self) -> bool:
        """Re-run the proof check."""
        return self.proof.is_valid()

    def __repr__(self) -> str:
        status = "VALID" if self.proof.valid else "INVALID"
        return (
            f"Certified(value={self.value!r}, "
            f"claim='{self.proof.claim}', {status})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare the *value* only (not the proof)."""
        if isinstance(other, Certified):
            return self.value == other.value
        return self.value == other

    def __int__(self) -> int:
        return int(self.value)  # type: ignore[arg-type]

    def __index__(self) -> int:
        return int(self.value)  # type: ignore[arg-type]
