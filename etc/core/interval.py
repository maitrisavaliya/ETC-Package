"""
etc.core.interval
=================
Interval: a closed interval [lo, hi] of ExactReal values.

Interval arithmetic is the fast inner layer for space predicates and
path checking.  Every operation produces an interval that is guaranteed
to contain the true result — no false negatives, possible false positives
(conservatism).

An interval [lo, hi] represents the set { x ∈ ℝ | lo ≤ x ≤ hi }.
Width w = hi − lo measures over-approximation.

All internal bounds are kept as Fraction (exact), evaluated lazily.
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import Tuple
from etc.core.real import ExactReal


class Interval:
    """
    A closed real interval [lo, hi] with exact rational endpoints.

    Construction
    ------------
    Interval(lo, hi)                 – from two ExactReal bounds
    Interval.from_real(x, prec)      – [x−ε, x+ε] where ε = 2^{-prec}
    Interval.from_rationals(a, b)    – from two Fraction/int endpoints
    Interval.point(x)                – degenerate interval [x, x]

    Arithmetic
    ----------
    add, sub, mul, div, neg, abs_val, pow_int, sqrt, recip
    All return Interval (over-approximate).

    Predicates
    ----------
    contains_zero()  – True if 0 ∈ [lo, hi]
    is_positive()    – True if lo > 0
    is_negative()    – True if hi < 0
    width()          – hi − lo as Fraction
    certainly_lt(other)  – hi < other.lo
    certainly_gt(other)  – lo > other.hi
    possibly_eq(other)   – intervals overlap
    """

    def __init__(self, lo: Fraction, hi: Fraction):
        if lo > hi:
            raise ValueError(
                f"Interval: lo={lo} > hi={hi}; endpoints must satisfy lo ≤ hi."
            )
        self.lo: Fraction = lo
        self.hi: Fraction = hi

    # --------------------------------------------------------- constructors

    @staticmethod
    def from_real(x: ExactReal, prec: int = 40) -> "Interval":
        """
        Build [x−ε, x+ε] where ε = 2^{-prec}.
        The true value lies inside this interval by the Cauchy guarantee.
        """
        mid = x.eval(prec)
        eps = Fraction(1, 2 ** prec)
        return Interval(mid - eps, mid + eps)

    @staticmethod
    def from_rationals(a: Fraction, b: Fraction) -> "Interval":
        lo, hi = (a, b) if a <= b else (b, a)
        return Interval(lo, hi)

    @staticmethod
    def point(x: Fraction) -> "Interval":
        return Interval(x, x)

    # --------------------------------------------------------- properties

    def width(self) -> Fraction:
        return self.hi - self.lo

    def midpoint(self) -> Fraction:
        return (self.lo + self.hi) / 2

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def is_positive(self) -> bool:
        """Certainly positive: lo > 0."""
        return self.lo > 0

    def is_negative(self) -> bool:
        """Certainly negative: hi < 0."""
        return self.hi < 0

    def contains(self, v: Fraction) -> bool:
        return self.lo <= v <= self.hi

    # --------------------------------------------------------- arithmetic

    def neg(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def mul(self, other: "Interval") -> "Interval":
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def recip(self) -> "Interval":
        """1/self.  Raises if self contains 0."""
        if self.contains_zero():
            raise ZeroDivisionError(
                f"Interval.recip: interval {self} contains 0."
            )
        return Interval(Fraction(1) / self.hi, Fraction(1) / self.lo)

    def div(self, other: "Interval") -> "Interval":
        return self.mul(other.recip())

    def abs_val(self) -> "Interval":
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        if self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        return Interval(Fraction(0), max(-self.lo, self.hi))

    def pow_int(self, k: int) -> "Interval":
        """
        Return self^k for k ≥ 0.
        Handles the sign correctly for even/odd k.
        """
        if k < 0:
            return self.recip().pow_int(-k)
        if k == 0:
            return Interval.point(Fraction(1))
        if k % 2 == 0:
            # Even power: minimum at 0 if interval straddles 0
            a = self.lo ** k
            b = self.hi ** k
            if self.contains_zero():
                return Interval(Fraction(0), max(a, b))
            return Interval(min(a, b), max(a, b))
        else:
            # Odd power: monotone
            return Interval(self.lo ** k, self.hi ** k)

    def sqrt(self) -> "Interval":
        """Return √self.  Requires self.lo ≥ 0."""
        if self.lo < 0:
            raise ValueError("Interval.sqrt: interval has negative lower bound")
        # Use rational sqrt approximation for bounds
        def _isqrt_frac(f: Fraction) -> Fraction:
            if f == 0:
                return Fraction(0)
            # Newton's method over Fraction
            x = Fraction(math.isqrt(f.numerator * f.denominator),
                         f.denominator)
            for _ in range(10):
                x = (x + f / x) / 2
            return x

        lo_sqrt = _isqrt_frac(self.lo)
        hi_sqrt = _isqrt_frac(self.hi)
        # Ensure the interval is valid (Newton may over/undershoot).
        while lo_sqrt * lo_sqrt > self.lo:
            lo_sqrt -= Fraction(1, 2 ** 60)
        while hi_sqrt * hi_sqrt < self.hi:
            hi_sqrt += Fraction(1, 2 ** 60)
        return Interval(lo_sqrt, hi_sqrt)

    # --------------------------------------------------------- comparison

    def certainly_lt(self, other: "Interval") -> bool:
        """Return True if every element of self is < every element of other."""
        return self.hi < other.lo

    def certainly_gt(self, other: "Interval") -> bool:
        return other.certainly_lt(self)

    def certainly_le(self, other: "Interval") -> bool:
        return self.hi <= other.lo

    def certainly_eq(self, other: "Interval") -> bool:
        """Can only be certain if both are point intervals with same value."""
        return self.lo == other.lo == self.hi == other.hi

    def possibly_eq(self, other: "Interval") -> bool:
        """True if the intervals overlap."""
        return not (self.certainly_lt(other) or self.certainly_gt(other))

    def certainly_contains_value(self, v: Fraction) -> bool:
        return self.lo <= v <= self.hi

    # --------------------------------------------------------- Python ops

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Interval.point(Fraction(other))
        return self.add(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Interval.point(Fraction(other))
        return self.sub(other)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Interval.point(Fraction(other))
        return self.mul(other)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Interval.point(Fraction(other))
        return self.div(other)

    def __neg__(self):
        return self.neg()

    def __repr__(self) -> str:
        return f"[{float(self.lo):.8g}, {float(self.hi):.8g}]"

    def __contains__(self, v) -> bool:
        return self.contains(Fraction(v))


def interval_eval(x: ExactReal, prec: int = 40) -> Interval:
    """Convenience: build an Interval from an ExactReal."""
    return Interval.from_real(x, prec)
