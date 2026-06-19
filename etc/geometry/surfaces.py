"""
etc.geometry.surfaces
=====================
AlgebraicSurface – { (x,y,z) | f(x,y,z) = 0 } in ℝ³.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Tuple
from etc.core.real      import ExactReal
from etc.topology.space import Space


class AlgebraicSurface:
    """
    A surface in ℝ³ defined by f(x, y, z) = 0.

    Parameters
    ----------
    sympy_poly : sympy.Expr
    x_sym, y_sym, z_sym : sympy.Symbol
    name : str

    Methods
    -------
    as_space(check_prec)  – return a Space
    point(x, y, z)        – certify a point
    normal_vector(pt)     – unit normal ∇f/|∇f| at a point
    """

    def __init__(self, sympy_poly, x_sym, y_sym, z_sym, name: str = "surface"):
        self.poly  = sympy_poly
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.z_sym = z_sym
        self.name  = name

    def as_space(self, check_prec: int = 40) -> Space:
        tol  = Fraction(1, 2 ** (check_prec // 2))
        poly = self.poly
        xs, ys, zs = self.x_sym, self.y_sym, self.z_sym

        import sympy as sp

        def predicate(coords):
            x, y, z = coords
            xf = x.eval(check_prec)
            yf = y.eval(check_prec)
            zf = z.eval(check_prec)
            val = poly
            for sym, f in [(xs, xf), (ys, yf), (zs, zf)]:
                val = val.subs(sym, sp.Rational(f.numerator, f.denominator))
            return abs(float(val.evalf(20))) < float(tol)

        return Space(
            predicate    = predicate,
            dim          = 3,
            sympy_pred   = poly,
            sympy_coords = (xs, ys, zs),
            name         = self.name,
            description  = str(poly) + " = 0",
        )

    def point(
        self, x: ExactReal, y: ExactReal, z: ExactReal
    ) -> "etc.topology.space.Point":
        from etc.topology.space import Point
        return Point((x, y, z), self.as_space())

    def normal_vector(
        self, pt: Tuple[ExactReal, ExactReal, ExactReal], prec: int = 30
    ) -> Tuple[float, float, float]:
        """Return approximate unit normal ∇f/|∇f| at pt."""
        import sympy as sp
        x, y, z = pt
        xr = sp.Rational(*x.eval(prec).as_integer_ratio())
        yr = sp.Rational(*y.eval(prec).as_integer_ratio())
        zr = sp.Rational(*z.eval(prec).as_integer_ratio())

        def diff_eval(sym):
            d = self.poly.diff(sym)
            d = d.subs(self.x_sym, xr).subs(self.y_sym, yr).subs(self.z_sym, zr)
            return float(d.evalf())

        nx, ny, nz = diff_eval(self.x_sym), diff_eval(self.y_sym), diff_eval(self.z_sym)
        norm = (nx**2 + ny**2 + nz**2) ** 0.5
        if norm < 1e-30:
            return (1.0, 0.0, 0.0)
        return (nx / norm, ny / norm, nz / norm)

    def __repr__(self) -> str:
        return f"AlgebraicSurface('{self.name}', f={self.poly})"
