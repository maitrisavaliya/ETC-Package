"""
etc.topology.path
=================
Path: a certified continuous map γ : [0,1] → Space.

Verification is three-layer:
  1. Type check   — parameter must be a UnitInterval.
  2. Spot checks  — evaluate at a grid of rational t values.
  3. Symbolic proof — substitute sympy path expressions into the space
                      predicate and confirm the result is identically 0.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, Tuple, Optional
from etc.core.real    import ExactReal
from etc.topology.space import Space, Coords


class UnitInterval:
    """
    An exact real number constrained to [0, 1].

    Construction raises ValueError if the value is not in [0, 1],
    verified via interval arithmetic at two precision levels.
    """

    def __init__(self, x: ExactReal, _skip_check: bool = False):
        if not _skip_check:
            for prec in (4, 16):
                approx = x.eval(prec)
                eps    = Fraction(1, 2 ** prec)
                if approx + eps < 0 or approx - eps > 1:
                    raise ValueError(
                        f"UnitInterval: {float(approx):.6f} is not in [0, 1]"
                    )
        self.value: ExactReal = x

    @staticmethod
    def from_rational(p, q: int = 1) -> "UnitInterval":
        return UnitInterval(ExactReal.from_rational(p, q))

    @staticmethod
    def zero() -> "UnitInterval":
        return UnitInterval(ExactReal.zero(), _skip_check=True)

    @staticmethod
    def one() -> "UnitInterval":
        return UnitInterval(ExactReal.one(), _skip_check=True)

    @staticmethod
    def half() -> "UnitInterval":
        return UnitInterval(ExactReal.from_rational(Fraction(1, 2)), _skip_check=True)

    def complement(self) -> "UnitInterval":
        """Return 1 − self, also in [0,1]."""
        one_minus = ExactReal.one().sub(self.value)
        return UnitInterval(one_minus, _skip_check=True)

    def __repr__(self) -> str:
        return f"UnitInterval(≈{float(self.value.eval(53)):.10g})"


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class Path:
    """
    A certified continuous path γ : [0,1] → Space.

    Parameters
    ----------
    path_fn : UnitInterval → Coords
        The path function.  Must return coordinates in the space.
    space : Space
        The target space.
    sympy_path : Tuple[sympy.Expr, ...], optional
        Symbolic parametric coordinates in sympy_param.
    sympy_param : sympy.Symbol, optional
        The parameter symbol.
    name : str

    Verification
    ------------
    verify()       – run all checks, return True/False
    spot_check(n)  – check at n equally-spaced rational t values
    symbolic_verify() – full algebraic proof via space.verify_path_symbolic

    Evaluation
    ----------
    __call__(t : UnitInterval) → Coords  (with runtime space check)
    start()   – γ(0)
    end()     – γ(1)
    reverse() – Path running from γ(1) to γ(0)
    concat(other) – concatenate two paths (requires end == other.start)
    """

    def __init__(
        self,
        path_fn: Callable[[UnitInterval], Coords],
        space: Space,
        sympy_path: Optional[Tuple] = None,
        sympy_param=None,
        name: str = "γ",
    ):
        self.path_fn     = path_fn
        self.space       = space
        self.sympy_path  = sympy_path
        self.sympy_param = sympy_param
        self.name        = name

    # ---------------------------------------------------------------- eval

    def __call__(self, t: UnitInterval) -> Coords:
        coords = self.path_fn(t)
        if not self.space.contains(coords):
            raise RuntimeError(
                f"Path '{self.name}' left space '{self.space.name}' "
                f"at t ≈ {float(t.value.eval(30)):.8g}"
            )
        return coords

    def start(self) -> Coords:
        return self(UnitInterval.zero())

    def end(self) -> Coords:
        return self(UnitInterval.one())

    # ---------------------------------------------------------------- verify

    def spot_check(self, n_points: int = 9) -> bool:
        """
        Evaluate at n_points equally-spaced t values in [0,1] and check
        each returned point lies in the space.
        Default: t = 0, 1/(n-1), ..., 1.
        """
        for k in range(n_points):
            t_frac = Fraction(k, n_points - 1) if n_points > 1 else Fraction(0)
            t      = UnitInterval.from_rational(t_frac)
            coords = self.path_fn(t)
            if not self.space.contains(coords):
                return False
        return True

    def symbolic_verify(self) -> bool:
        """Full symbolic proof via space predicate simplification."""
        if self.sympy_path is None or self.sympy_param is None:
            raise ValueError(
                f"Path '{self.name}' has no symbolic description; "
                "cannot perform symbolic verification."
            )
        return self.space.verify_path_symbolic(self.sympy_path, self.sympy_param)

    def verify(self) -> bool:
        """
        Full verification:
          1. Spot-check at 9 equally-spaced t values.
          2. Symbolic proof (if available).
        Returns True only if both pass.
        """
        if not self.spot_check():
            return False
        if self.sympy_path is not None:
            return self.symbolic_verify()
        return True

    # ---------------------------------------------------------------- combinators

    def reverse(self) -> "Path":
        """Return the reversed path γ̄(t) = γ(1−t)."""
        def rev_fn(t: UnitInterval) -> Coords:
            return self.path_fn(t.complement())

        sympy_path = None
        if self.sympy_path is not None and self.sympy_param is not None:
            import sympy as sp
            t  = self.sympy_param
            one = sp.Integer(1)
            sympy_path = tuple(
                e.subs(t, one - t) for e in self.sympy_path
            )
        return Path(
            rev_fn, self.space,
            sympy_path=sympy_path,
            sympy_param=self.sympy_param,
            name=f"reverse({self.name})",
        )

    def concat(self, other: "Path") -> "Path":
        """
        Concatenate self and other: run self on [0, 1/2], other on [1/2, 1].
        Requires self.end ≈ other.start (checked at prec 30).
        """
        # Check endpoints match
        end_coords   = self.end()
        start_coords = other.start()
        for e, s in zip(end_coords, start_coords):
            diff = abs(float(e.eval(30)) - float(s.eval(30)))
            if diff > 2 ** -20:
                raise ValueError(
                    f"concat: endpoints do not match (diff={diff:.2e})"
                )

        def concat_fn(t: UnitInterval) -> Coords:
            t_val = t.value.eval(4)
            if t_val <= Fraction(1, 2):
                # Reparametrize [0, 1/2] → [0, 1]
                s = UnitInterval(
                    t.value.mul(ExactReal.from_rational(2)), _skip_check=True
                )
                return self.path_fn(s)
            else:
                # Reparametrize [1/2, 1] → [0, 1]
                two   = ExactReal.from_rational(2)
                one   = ExactReal.one()
                s_val = t.value.mul(two).sub(one)
                s     = UnitInterval(s_val, _skip_check=True)
                return other.path_fn(s)

        return Path(
            concat_fn, self.space,
            name=f"({self.name} * {other.name})",
        )

    def __repr__(self) -> str:
        return f"Path('{self.name}', space='{self.space.name}')"
