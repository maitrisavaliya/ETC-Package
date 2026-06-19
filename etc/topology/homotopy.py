"""
etc.topology.homotopy
=====================
Homotopy: a certified continuous map H : [0,1]² → Space connecting
two paths γ₀ and γ₁.

H satisfies:
    H(0, t) = γ₀(t)   for all t ∈ [0,1]
    H(1, t) = γ₁(t)   for all t ∈ [0,1]
    H(s, t) ∈ Space    for all (s,t) ∈ [0,1]²

Verification is analogous to Path: spot-check on a grid, then symbolic.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Tuple, Optional
from etc.core.real      import ExactReal
from etc.topology.space import Space, Coords
from etc.topology.path  import UnitInterval, Path


Homotopy_fn = Callable[[UnitInterval, UnitInterval], Coords]


class Homotopy:
    """
    A certified path homotopy H : [0,1]² → Space.

    Parameters
    ----------
    homotopy_fn : (s: UnitInterval, t: UnitInterval) → Coords
    space  : Space
    path0  : Path   – H(0, ·) should equal path0
    path1  : Path   – H(1, ·) should equal path1
    sympy_homotopy : Tuple[sympy.Expr, ...] in (s_sym, t_sym)
    s_sym, t_sym   : sympy.Symbol

    Verification
    ------------
    verify_endpoints(prec)  – check H(0,·)=γ₀ and H(1,·)=γ₁ at spot points
    spot_check(n)            – check membership on n×n grid
    symbolic_verify()        – full algebraic proof
    verify()                 – all three
    """

    def __init__(
        self,
        homotopy_fn:    Homotopy_fn,
        space:          Space,
        path0:          Path,
        path1:          Path,
        sympy_homotopy: Optional[Tuple]  = None,
        s_sym  = None,
        t_sym  = None,
        name:           str = "H",
    ):
        self.homotopy_fn    = homotopy_fn
        self.space          = space
        self.path0          = path0
        self.path1          = path1
        self.sympy_homotopy = sympy_homotopy
        self.s_sym          = s_sym
        self.t_sym          = t_sym
        self.name           = name

    def __call__(self, s: UnitInterval, t: UnitInterval) -> Coords:
        coords = self.homotopy_fn(s, t)
        if not self.space.contains(coords):
            raise RuntimeError(
                f"Homotopy '{self.name}' left space '{self.space.name}'."
            )
        return coords

    # ---------------------------------------------------------------- verify

    def verify_endpoints(self, n_points: int = 5) -> bool:
        """
        Check H(0,t) ≈ γ₀(t) and H(1,t) ≈ γ₁(t) at spot t values.
        """
        prec = 30
        eps  = 2 ** -20
        params = [Fraction(k, n_points - 1) for k in range(n_points)]

        for t_frac in params:
            t    = UnitInterval.from_rational(t_frac)
            h0   = self.homotopy_fn(UnitInterval.zero(), t)
            p0   = self.path0.path_fn(t)
            h1   = self.homotopy_fn(UnitInterval.one(), t)
            p1   = self.path1.path_fn(t)
            for a, b in zip(h0, p0):
                if abs(float(a.eval(prec)) - float(b.eval(prec))) > eps:
                    return False
            for a, b in zip(h1, p1):
                if abs(float(a.eval(prec)) - float(b.eval(prec))) > eps:
                    return False
        return True

    def spot_check(self, n: int = 5) -> bool:
        """Check H(s,t) ∈ Space on an n×n grid."""
        for i in range(n):
            for j in range(n):
                s = UnitInterval.from_rational(Fraction(i, n - 1) if n > 1 else Fraction(0))
                t = UnitInterval.from_rational(Fraction(j, n - 1) if n > 1 else Fraction(0))
                coords = self.homotopy_fn(s, t)
                if not self.space.contains(coords):
                    return False
        return True

    def symbolic_verify(self) -> bool:
        """Full algebraic proof via space predicate."""
        if self.sympy_homotopy is None:
            raise ValueError(
                f"Homotopy '{self.name}' has no symbolic description."
            )
        return self.space.verify_homotopy_symbolic(
            self.sympy_homotopy, self.s_sym, self.t_sym
        )

    def verify(self) -> bool:
        """Full verification: endpoints, grid, symbolic."""
        return (
            self.verify_endpoints()
            and self.spot_check()
            and (self.symbolic_verify() if self.sympy_homotopy is not None else True)
        )

    # ---------------------------------------------------------------- slices

    def path_at(self, s: UnitInterval) -> Path:
        """Return the path H(s, ·) : [0,1] → Space."""
        def fn(t: UnitInterval) -> Coords:
            return self.homotopy_fn(s, t)
        return Path(fn, self.space, name=f"{self.name}(s={float(s.value.eval(20)):.4g})")

    def __repr__(self) -> str:
        return (
            f"Homotopy('{self.name}', "
            f"path0='{self.path0.name}', path1='{self.path1.name}', "
            f"space='{self.space.name}')"
        )
