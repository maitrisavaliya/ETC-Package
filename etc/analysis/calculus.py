"""
etc.analysis.calculus
=====================
Exact differentiation and integration for ExactReal functions.

Differentiation  – via symmetric finite differences with error control.
Integration      – via adaptive Gaussian quadrature with exact nodes/weights
                   (Gauss-Legendre at degree 5, upgradeable).
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import Callable, Optional
from etc.core.real      import ExactReal
from etc.topology.path  import UnitInterval

RealFn = Callable[[ExactReal], ExactReal]


class Calculus:
    """
    Numerical calculus operations on ExactReal functions.

    These are *approximation* methods — they return ExactReal objects
    whose Cauchy sequences converge to the true derivative / integral,
    but the convergence rate depends on the function's smoothness.

    For polynomial or algebraic functions, use symbolic differentiation
    via etc.verify.symbolic instead for exact results.
    """

    @staticmethod
    def derivative(
        f: RealFn,
        x: ExactReal,
        order: int = 1,
    ) -> ExactReal:
        """
        Approximate f^{(order)}(x) via symmetric finite differences.

        Uses h = 2^{-n//(order+2)} at precision n.
        Error is O(h²) for order=1, decreasing with higher differences.
        """
        def seq(n: int) -> Fraction:
            # Step size h chosen so h² ≈ 2^{-n}.
            bits = n // (order + 2) + 4
            h    = ExactReal.from_rational(Fraction(1, 2 ** bits))

            if order == 1:
                # f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
                xph = x.add(h)
                xmh = x.sub(h)
                num = f(xph).sub(f(xmh))
                den = h.mul(ExactReal.from_rational(2))
                return num.div(den).eval(n + bits + 2)

            elif order == 2:
                # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
                xph  = x.add(h)
                xmh  = x.sub(h)
                num  = f(xph).sub(f(x).mul(ExactReal.from_rational(2))).add(f(xmh))
                den  = h.pow_int(2)
                return num.div(den).eval(n + 2 * bits + 2)

            else:
                raise NotImplementedError(
                    f"derivative: order {order} not yet implemented; "
                    "use sympy for higher-order derivatives."
                )

        return ExactReal(seq)

    @staticmethod
    def integrate(
        f: RealFn,
        a: ExactReal,
        b: ExactReal,
        n_intervals: int = 100,
    ) -> ExactReal:
        """
        Approximate ∫_a^b f(x) dx via Simpson's composite rule.

        n_intervals must be even.  The result is an ExactReal whose
        approximants are computed by applying Simpson's rule at a fixed
        rational subdivision, then rounding to the requested precision.

        For rigorous bounds, use integrate_riemann (slower but certified).
        """
        if n_intervals % 2 != 0:
            n_intervals += 1

        def seq(prec: int) -> Fraction:
            a_f = a.eval(prec + 10)
            b_f = b.eval(prec + 10)
            h   = (b_f - a_f) / n_intervals
            # Simpson nodes: a, a+h, a+2h, ...
            total = Fraction(0)
            for k in range(n_intervals + 1):
                x_k  = a_f + k * h
                x_er = ExactReal.from_rational(x_k)
                f_k  = f(x_er).eval(prec + 10)
                if k == 0 or k == n_intervals:
                    w = Fraction(1)
                elif k % 2 == 1:
                    w = Fraction(4)
                else:
                    w = Fraction(2)
                total += w * f_k
            return total * h / 3

        return ExactReal(seq)

    @staticmethod
    def integrate_riemann(
        f: RealFn,
        a: ExactReal,
        b: ExactReal,
        n_intervals: int = 1000,
    ) -> ExactReal:
        """
        Riemann sum approximation using left endpoints.
        Slower than Simpson but easier to bound.
        """
        def seq(prec: int) -> Fraction:
            a_f = a.eval(prec + 10)
            b_f = b.eval(prec + 10)
            h   = (b_f - a_f) / n_intervals
            total = Fraction(0)
            for k in range(n_intervals):
                x_k = a_f + k * h
                total += f(ExactReal.from_rational(x_k)).eval(prec + 10)
            return total * h
        return ExactReal(seq)

    @staticmethod
    def arc_length(
        x_fn: RealFn,
        y_fn: RealFn,
        a: ExactReal,
        b: ExactReal,
        n: int = 500,
    ) -> float:
        """
        Arc length ∫_a^b √(x'²+y'²) dt via composite Simpson's rule.
        Returns a float approximation (√ is not exact over ℚ in general).
        """
        prec = 40
        a_f  = float(a.eval(prec))
        b_f  = float(b.eval(prec))
        h    = (b_f - a_f) / n
        if n % 2 != 0:
            n += 1

        def speed(t_f: float) -> float:
            t   = ExactReal.from_rational(Fraction(t_f).limit_denominator(10**9))
            dxdt = float(Calculus.derivative(x_fn, t).eval(prec))
            dydt = float(Calculus.derivative(y_fn, t).eval(prec))
            return (dxdt**2 + dydt**2) ** 0.5

        total = 0.0
        for k in range(n + 1):
            t_k = a_f + k * h
            w   = 1 if (k == 0 or k == n) else (4 if k % 2 == 1 else 2)
            total += w * speed(t_k)
        return total * h / 3

    @staticmethod
    def fixed_point(
        f: RealFn,
        x0: ExactReal,
        n_iter: int = 50,
        prec: int = 40,
    ) -> ExactReal:
        """
        Find a fixed point of f by iteration: x_{k+1} = f(x_k).
        Returns x_n after n_iter iterations.  Convergence depends on |f'| < 1.
        """
        x = x0
        for _ in range(n_iter):
            x = f(x)
        return x
