"""
etc.geometry.curves
===================
AlgebraicCurve   – { (x,y) | f(x,y) = 0 } defined by a sympy polynomial.
ParametricCurve  – (x(t), y(t)) for ExactReal functions of t.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Tuple, Optional
from etc.core.real      import ExactReal
from etc.topology.space import Space
from etc.topology.path  import UnitInterval, Path


class AlgebraicCurve:
    """
    A plane algebraic curve defined by f(x, y) = 0 for a polynomial f.

    Parameters
    ----------
    sympy_poly : sympy.Expr
        The defining polynomial in x and y.
    x_sym, y_sym : sympy.Symbol
    name : str

    This is a thin wrapper: it builds a Space from the polynomial and
    exposes curve-specific methods.

    Methods
    -------
    as_space(check_prec)  – return a Space object
    point(x, y)           – construct and certify a point on the curve
    tangent_vector(pt, prec) – approximate tangent direction at a point
    """

    def __init__(self, sympy_poly, x_sym, y_sym, name: str = "curve"):
        import sympy as sp
        self.poly   = sympy_poly
        self.x_sym  = x_sym
        self.y_sym  = y_sym
        self.name   = name

    def as_space(self, check_prec: int = 40) -> Space:
        """Build a Space whose predicate evaluates f(x,y) ≈ 0."""
        tol  = Fraction(1, 2 ** (check_prec // 2))
        poly = self.poly
        xs   = self.x_sym
        ys   = self.y_sym

        import sympy as sp

        def predicate(coords):
            x, y = coords
            xf   = x.eval(check_prec)
            yf   = y.eval(check_prec)
            val  = poly.subs(xs, sp.Rational(xf.numerator, xf.denominator))
            val  = val.subs(ys, sp.Rational(yf.numerator, yf.denominator))
            val  = float(val.evalf(20))
            return abs(val) < float(tol)

        return Space(
            predicate    = predicate,
            dim          = 2,
            sympy_pred   = poly,
            sympy_coords = (xs, ys),
            name         = self.name,
            description  = str(poly) + " = 0",
        )

    def point(self, x: ExactReal, y: ExactReal) -> "etc.topology.space.Point":
        from etc.topology.space import Point
        return Point((x, y), self.as_space())

    def tangent_direction(
        self, pt: Tuple[ExactReal, ExactReal], prec: int = 30
    ) -> Tuple[float, float]:
        """
        Return approximate unit tangent vector at pt by computing
        the gradient of f and rotating 90°: tangent ∝ (−∂f/∂y, ∂f/∂x).
        """
        import sympy as sp
        x, y  = pt
        xf    = x.eval(prec)
        yf    = y.eval(prec)
        xr    = sp.Rational(xf.numerator, xf.denominator)
        yr    = sp.Rational(yf.numerator, yf.denominator)
        dfdx  = float(self.poly.diff(self.x_sym).subs(self.x_sym, xr).subs(self.y_sym, yr).evalf())
        dfdy  = float(self.poly.diff(self.y_sym).subs(self.x_sym, xr).subs(self.y_sym, yr).evalf())
        # Tangent direction: perpendicular to gradient = (−dfdy, dfdx)
        norm = (dfdx**2 + dfdy**2) ** 0.5
        if norm < 1e-30:
            return (1.0, 0.0)
        return (-dfdy / norm, dfdx / norm)

    def __repr__(self) -> str:
        return f"AlgebraicCurve('{self.name}', f={self.poly})"


class ParametricCurve:
    """
    A parametric curve (x(t), y(t)) for t ∈ [0,1].

    Parameters
    ----------
    x_fn, y_fn : UnitInterval → ExactReal
    space : Space  (optional — if given, curve is certified to lie in the space)
    name  : str

    Methods
    -------
    as_path()          – return a Path (requires space)
    arc_length(n)      – numerical arc length approximation with n subintervals
    curvature_at(t)    – signed curvature κ(t) (approximate)
    """

    def __init__(
        self,
        x_fn: Callable[[UnitInterval], ExactReal],
        y_fn: Callable[[UnitInterval], ExactReal],
        space: Optional[Space] = None,
        name:  str = "curve",
    ):
        self.x_fn  = x_fn
        self.y_fn  = y_fn
        self.space = space
        self.name  = name

    def __call__(self, t: UnitInterval) -> Tuple[ExactReal, ExactReal]:
        return (self.x_fn(t), self.y_fn(t))

    def as_path(self, sympy_exprs=None, sympy_param=None) -> Path:
        """Convert to a certified Path if a space is attached."""
        if self.space is None:
            raise ValueError("ParametricCurve.as_path: no space attached.")

        def path_fn(t: UnitInterval):
            return (self.x_fn(t), self.y_fn(t))

        return Path(
            path_fn,
            self.space,
            sympy_path  = sympy_exprs,
            sympy_param = sympy_param,
            name        = self.name,
        )

    def arc_length(self, n: int = 100) -> float:
        """
        Approximate arc length ∫₀¹ √(x'²+y'²) dt via the trapezoidal rule
        on n subintervals.  Uses finite-difference derivative approximation.
        """
        h      = Fraction(1, n)
        prec   = 40
        length = 0.0
        prev_x = float(self.x_fn(UnitInterval.from_rational(0)).eval(prec))
        prev_y = float(self.y_fn(UnitInterval.from_rational(0)).eval(prec))
        for k in range(1, n + 1):
            t_k   = UnitInterval.from_rational(Fraction(k, n))
            cur_x = float(self.x_fn(t_k).eval(prec))
            cur_y = float(self.y_fn(t_k).eval(prec))
            dx    = cur_x - prev_x
            dy    = cur_y - prev_y
            length += (dx**2 + dy**2) ** 0.5
            prev_x, prev_y = cur_x, cur_y
        return length

    def __repr__(self) -> str:
        return f"ParametricCurve('{self.name}')"
