"""
etc.analysis.series
===================
PowerSeries and TaylorSeries with exact coefficient arithmetic.
"""

from __future__ import annotations
from fractions import Fraction
from typing import List, Callable
from etc.core.real import ExactReal


class PowerSeries:
    """
    A formal power series Σ_{k=0}^{N} aₖ·xᵏ with ExactReal coefficients.

    Parameters
    ----------
    coeffs : List[ExactReal]   – [a₀, a₁, ..., a_N]

    Methods
    -------
    eval(x, prec)    – evaluate at ExactReal x to precision prec
    add(other)       – term-wise addition
    mul(other)       – Cauchy product
    differentiate()  – formal derivative
    integrate()      – formal antiderivative (constant term = 0)
    truncate(N)      – keep first N+1 terms
    """

    def __init__(self, coeffs: List[ExactReal]):
        self.coeffs: List[ExactReal] = coeffs

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def eval(self, x: ExactReal, prec: int = 40) -> ExactReal:
        """Evaluate using Horner's method for numerical stability."""
        if not self.coeffs:
            return ExactReal.zero()
        result = self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            result = result.mul(x).add(c)
        return result

    def add(self, other: "PowerSeries") -> "PowerSeries":
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs  + [ExactReal.zero()] * (n - len(self.coeffs))
        b = other.coeffs + [ExactReal.zero()] * (n - len(other.coeffs))
        return PowerSeries([ai.add(bi) for ai, bi in zip(a, b)])

    def mul(self, other: "PowerSeries") -> "PowerSeries":
        """Cauchy (convolution) product."""
        n = len(self.coeffs) + len(other.coeffs) - 1
        result = [ExactReal.zero()] * n
        for i, ai in enumerate(self.coeffs):
            for j, bj in enumerate(other.coeffs):
                result[i + j] = result[i + j].add(ai.mul(bj))
        return PowerSeries(result)

    def scale(self, c: ExactReal) -> "PowerSeries":
        return PowerSeries([c.mul(a) for a in self.coeffs])

    def differentiate(self) -> "PowerSeries":
        """Formal derivative: d/dx Σ aₖxᵏ = Σ k·aₖ·xᵏ⁻¹."""
        if len(self.coeffs) <= 1:
            return PowerSeries([ExactReal.zero()])
        return PowerSeries([
            ExactReal.from_rational(k).mul(self.coeffs[k])
            for k in range(1, len(self.coeffs))
        ])

    def integrate(self) -> "PowerSeries":
        """Formal antiderivative with zero constant: ∫ Σ aₖxᵏ dx = Σ aₖ/(k+1)·xᵏ⁺¹."""
        result = [ExactReal.zero()]
        for k, ak in enumerate(self.coeffs):
            result.append(ak.div(ExactReal.from_rational(k + 1)))
        return PowerSeries(result)

    def truncate(self, N: int) -> "PowerSeries":
        return PowerSeries(self.coeffs[:N + 1])

    def __repr__(self) -> str:
        terms = []
        for k, c in enumerate(self.coeffs[:6]):
            cf = float(c.eval(20))
            if abs(cf) > 1e-15:
                terms.append(f"{cf:.4g}·x^{k}" if k > 0 else f"{cf:.4g}")
        return "PowerSeries(" + " + ".join(terms) + ("..." if len(self.coeffs) > 6 else "") + ")"


class TaylorSeries:
    """
    Taylor series utilities for constructing PowerSeries approximations
    of standard functions around a point a.

    All coefficients are computed exactly as fractions.
    """

    @staticmethod
    def exp(N: int) -> PowerSeries:
        """e^x = Σ x^k/k!  up to degree N."""
        coeffs = [ExactReal.from_rational(1)]
        for k in range(1, N + 1):
            prev = coeffs[-1]
            coeffs.append(prev.div(ExactReal.from_rational(k)))
        return PowerSeries(coeffs)

    @staticmethod
    def sin(N: int) -> PowerSeries:
        """sin(x) = Σ (-1)^k x^{2k+1} / (2k+1)!  up to degree N."""
        coeffs = [ExactReal.zero()] * (N + 1)
        sign   = Fraction(1)
        fact   = Fraction(1)
        for k in range(N + 1):
            if k % 2 == 1:
                coeffs[k] = ExactReal.from_rational(sign / fact)
                if k % 4 == 3:
                    sign = -sign
            if k > 0:
                fact *= k
        return PowerSeries(coeffs)

    @staticmethod
    def cos(N: int) -> PowerSeries:
        """cos(x) = Σ (-1)^k x^{2k} / (2k)!  up to degree N."""
        coeffs = [ExactReal.zero()] * (N + 1)
        sign   = Fraction(1)
        fact   = Fraction(1)
        for k in range(N + 1):
            if k % 2 == 0:
                coeffs[k] = ExactReal.from_rational(sign / fact)
                sign = -sign
            if k > 0:
                fact *= k
        return PowerSeries(coeffs)

    @staticmethod
    def log1p(N: int) -> PowerSeries:
        """ln(1+x) = Σ_{k=1}^N (-1)^{k+1} x^k / k."""
        coeffs = [ExactReal.zero()]
        for k in range(1, N + 1):
            sign = Fraction(1) if k % 2 == 1 else Fraction(-1)
            coeffs.append(ExactReal.from_rational(sign, k))
        return PowerSeries(coeffs)

    @staticmethod
    def geometric(N: int) -> PowerSeries:
        """1/(1-x) = Σ x^k up to degree N."""
        return PowerSeries([ExactReal.one()] * (N + 1))
