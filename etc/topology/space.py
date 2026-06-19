"""
etc.topology.space
==================
Space: a constructive topological space defined by an exact predicate.
Point: a certified element of a Space.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Tuple, Optional
from etc.core.real     import ExactReal
from etc.core.interval import Interval


Coords = Tuple[ExactReal, ...]


class Space:
    """
    A constructive topological space defined by:

      predicate   : Coords -> bool
          Exact membership check (interval arithmetic).

      sympy_pred  : sympy.Expr, optional
          Symbolic expression that equals 0 on the space.
          Used for symbolic path / homotopy verification.

      sympy_coords: Tuple[sympy.Symbol, ...], optional
          Coordinate symbols matching predicate arity.

      dim         : int
          Ambient dimension (length of coordinate tuples).

      name        : str
          Human-readable label.

    The predicate is required; sympy fields are optional (needed only for
    symbolic verification).
    """

    def __init__(
        self,
        predicate: Callable[[Coords], bool],
        dim: int,
        sympy_pred=None,
        sympy_coords=None,
        name: str = "Space",
        description: str = "",
    ):
        self.predicate     = predicate
        self.dim           = dim
        self.sympy_pred    = sympy_pred
        self.sympy_coords  = sympy_coords
        self.name          = name
        self.description   = description

    # -------------------------------------------------------------- membership

    def contains(self, coords: Coords) -> bool:
        """Check membership via exact predicate."""
        if len(coords) != self.dim:
            raise ValueError(
                f"Space '{self.name}' has dim={self.dim}, "
                f"got {len(coords)} coordinates."
            )
        return self.predicate(coords)

    def __contains__(self, coords) -> bool:
        return self.contains(tuple(coords))

    # -------------------------------------------------------------- symbolic

    def verify_path_symbolic(
        self,
        path_exprs: Tuple,        # tuple of sympy expressions
        param_sym,                # sympy Symbol
    ) -> bool:
        """
        Symbolically prove path ⊆ Space for all values of param_sym.
        Substitutes path_exprs into sympy_pred and simplifies.
        Returns True iff the result is identically 0.
        """
        import sympy as sp
        if self.sympy_pred is None or self.sympy_coords is None:
            raise ValueError(
                f"Space '{self.name}' has no symbolic predicate."
            )
        expr = self.sympy_pred
        for sym, coord in zip(self.sympy_coords, path_exprs):
            expr = expr.subs(sym, coord)
        return sp.simplify(sp.expand(expr)) == 0

    def verify_homotopy_symbolic(
        self,
        homotopy_exprs: Tuple,   # tuple of sympy expressions in (s, t)
        s_sym, t_sym,
    ) -> bool:
        """
        Symbolically prove homotopy H(s,t) ⊆ Space for all (s,t) ∈ [0,1]².
        """
        import sympy as sp
        if self.sympy_pred is None or self.sympy_coords is None:
            raise ValueError(
                f"Space '{self.name}' has no symbolic predicate."
            )
        expr = self.sympy_pred
        for sym, coord in zip(self.sympy_coords, homotopy_exprs):
            expr = expr.subs(sym, coord)
        return sp.simplify(sp.expand(expr)) == 0

    # -------------------------------------------------------------- display

    def __repr__(self) -> str:
        return f"Space('{self.name}', dim={self.dim})"


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------

class Point:
    """
    A certified element of a Space.

    Certification is performed at construction time via the space predicate.
    Raises ValueError if the coordinates do not satisfy the predicate.
    """

    def __init__(self, coords: Coords, space: Space):
        if not space.contains(coords):
            approxs = [float(c.eval(20)) for c in coords]
            raise ValueError(
                f"Point({approxs}) does not satisfy '{space.name}'."
            )
        self.coords: Coords = coords
        self.space:  Space  = space

    def coord(self, i: int) -> ExactReal:
        return self.coords[i]

    def eval(self, prec: int) -> Tuple[Fraction, ...]:
        """Return rational approximants at precision prec."""
        return tuple(c.eval(prec) for c in self.coords)

    def __repr__(self) -> str:
        approxs = [float(c.eval(53)) for c in self.coords]
        return f"Point({approxs!r}, space='{self.space.name}')"


# ---------------------------------------------------------------------------
# Common space factories
# ---------------------------------------------------------------------------

def make_Lp_circle(p: int) -> Space:
    """
    L_p unit circle: { (x,y) ∈ ℝ² | |x|^p + |y|^p = 1 }.
    Predicate uses interval arithmetic at CHECK_PREC = 40 bits.
    """
    CHECK_PREC = 40
    tol        = Fraction(1, 2 ** (CHECK_PREC // 2))

    def predicate(coords: Coords) -> bool:
        x, y = coords
        xp = x.abs_val().pow_int(p).eval(CHECK_PREC)
        yp = y.abs_val().pow_int(p).eval(CHECK_PREC)
        return abs(xp + yp - 1) < tol

    import sympy as sp
    sx, sy   = sp.symbols("x y", real=True)
    sym_pred = sx**p + sy**p - 1

    return Space(
        predicate    = predicate,
        dim          = 2,
        sympy_pred   = sym_pred,
        sympy_coords = (sx, sy),
        name         = f"L{p}_circle",
        description  = f"|x|^{p} + |y|^{p} = 1",
    )


def make_sphere(n: int = 2) -> Space:
    """
    Standard (n-1)-sphere in ℝ^n: { x ∈ ℝ^n | Σ xᵢ² = 1 }.
    Currently supports n ∈ {2, 3}.
    """
    CHECK_PREC = 40
    tol        = Fraction(1, 2 ** (CHECK_PREC // 2))

    def predicate(coords: Coords) -> bool:
        s = sum(c.pow_int(2).eval(CHECK_PREC) for c in coords)
        return abs(s - 1) < tol

    import sympy as sp
    syms     = sp.symbols(f"x0:{n}", real=True)
    sym_pred = sum(s**2 for s in syms) - 1

    return Space(
        predicate    = predicate,
        dim          = n,
        sympy_pred   = sym_pred,
        sympy_coords = tuple(syms),
        name         = f"S{n-1}",
        description  = f"Unit sphere in ℝ^{n}",
    )


def make_torus(R: ExactReal, r: ExactReal) -> Space:
    """
    Torus in ℝ³ with major radius R, minor radius r:
    (√(x²+y²) − R)² + z² = r²
    """
    CHECK_PREC = 40
    tol        = Fraction(1, 2 ** (CHECK_PREC // 2))
    R_f        = R.eval(CHECK_PREC)
    r_f        = r.eval(CHECK_PREC)

    def predicate(coords: Coords) -> bool:
        x, y, z = coords
        x_f = x.eval(CHECK_PREC)
        y_f = y.eval(CHECK_PREC)
        z_f = z.eval(CHECK_PREC)
        from fractions import Fraction
        import math
        rxy   = Fraction(math.isqrt((x_f**2 + y_f**2).numerator *
                                     (x_f**2 + y_f**2).denominator),
                         (x_f**2 + y_f**2).denominator)
        # Approximate √(x²+y²) via Newton
        rxy_f = x_f*x_f + y_f*y_f
        val   = (rxy_f**Fraction(1,2) - R_f)**2 + z_f**2
        return abs(val - r_f**2) < tol

    import sympy as sp
    sx, sy, sz = sp.symbols("x y z", real=True)
    sym_pred   = (sp.sqrt(sx**2 + sy**2) - sp.Symbol("R"))**2 + sz**2 - sp.Symbol("r")**2

    return Space(
        predicate  = predicate,
        dim        = 3,
        name       = "Torus",
        description= f"Torus R={float(R):.4g}, r={float(r):.4g}",
    )
