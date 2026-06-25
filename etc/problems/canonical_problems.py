"""
etc.problems.canonical_problems
================================
Seven canonical floating-point failure problems used in the ETC paper.

These problems are designed to showcase cases where IEEE 754 double precision
fails catastrophically or silently, while ETC computes exact answers with
machine-checkable proof certificates.


"""

from __future__ import annotations
import math
import time
from fractions import Fraction
from typing import Any

from etc.core.real             import ExactReal
from etc.certified             import Certified
from etc.verify.certificate    import Certificate
from etc.problems.base         import Problem
from etc.topology.space        import make_Lp_circle
from etc.topology.path         import Path, UnitInterval
from etc.topology.invariants   import winding_number as _winding_number


class RumpPolynomial(Problem):
    """
    Rump's polynomial — canonical catastrophic cancellation failure.

    The polynomial evaluates to -54767/66192 ≈ -0.8274 at (a, b) = (77617, 33096).
    IEEE 754 double precision returns -1.18 × 10²¹, an error factor of 1.43 × 10²¹.

    The failure occurs because intermediate terms reach 10³⁶ while their sum
    is O(1), losing all significant digits in the cancellation.

    ETC computes the exact rational answer by propagating Fraction objects
    through all arithmetic operations, with no loss of precision.
    """

    @property
    def name(self) -> str:
        return "rump_polynomial"

    @property
    def description(self) -> str:
        return (
            "Rump's polynomial: f(77617, 33096) where IEEE 754 fails with "
            "catastrophic cancellation (error × 10²¹)"
        )

    def solve(self) -> Certified[Fraction]:
        """
        Compute f(77617, 33096) exactly and return Certified[Fraction].

        Algorithm: evaluate via exact rational arithmetic.
        """
        a = ExactReal.from_rational(77617)
        b = ExactReal.from_rational(33096)

        # Constants
        c_333_75 = ExactReal.from_rational(Fraction(33375, 100))
        c_11     = ExactReal.from_rational(11)
        c_121    = ExactReal.from_rational(121)
        c_2      = ExactReal.from_rational(2)
        c_55     = ExactReal.from_rational(Fraction(11, 2))

        # Precompute powers
        b2 = b.pow_int(2)
        b4 = b.pow_int(4)
        b6 = b.pow_int(6)
        b8 = b.pow_int(8)
        a2 = a.pow_int(2)

        # Compute terms
        t1 = c_333_75.mul(b6)
        inner = c_11.mul(a2).mul(b2).sub(b6).sub(c_121.mul(b4)).sub(c_2)
        t2 = a2.mul(inner)
        t3 = c_55.mul(b8)
        t4 = a.div(c_2.mul(b))

        # Sum all terms
        result_exact = t1.add(t2).add(t3).add(t4).eval(80)

        # The exact answer (verified independently)
        exact_frac = Fraction(-54767, 66192)

        # Build certificate
        cert = Certificate(
            claim   = "Rump's polynomial f(77617, 33096) = -54767/66192",
            method  = "exact_rational_arithmetic",
            valid   = (result_exact == exact_frac),
            evidence = {
                "a":           77617,
                "b":           33096,
                "result":      str(result_exact),
                "exact":       str(exact_frac),
                "float_error": "1.43e21 (IEEE 754 returns -1.18e21)",
                "precision_bits": 80,
            },
        )

        return Certified(result_exact, cert)

    def verify(self, answer: Any) -> bool:
        """
        Verify that the claimed answer is correct.

        Parameters
        ----------
        answer : Fraction or int  – the claimed value

        Returns
        -------
        bool  – True iff answer == -54767/66192
        """
        if isinstance(answer, Fraction):
            return answer == Fraction(-54767, 66192)
        # Try to convert to Fraction
        try:
            frac = Fraction(answer)
            return frac == Fraction(-54767, 66192)
        except (TypeError, ValueError):
            return False


class L5WindingNumber(Problem):
    """
    Winding number of a closed circular path — exact topological integer.

    A circular path traversed `winds` times around the origin has exact
    winding number `winds`.  Floating-point arctan2 angle accumulation
    introduces O(n_steps * eps_m) rounding error in the running angle
    total, so float can only report a *rounded* integer with no way to
    certify it.  ETC computes the exact integer via angle-lifting
    (etc.topology.invariants.winding_number) and reports a certificate
    carrying the precision, the path-closure gap, and the rounding
    error at which the integer was certified.

    Parameters
    ----------
    winds   : int  – number of times the path winds around the origin (default 1)
    n_steps : int  – number of angle-sample points used for lifting (default 400)
    """

    def __init__(self, winds: int = 1, n_steps: int = 400) -> None:
        self.winds   = winds
        self.n_steps = n_steps

    @property
    def name(self) -> str:
        return "l5_winding_number"

    @property
    def description(self) -> str:
        return (
            f"Winding number of a circle traversed {self.winds}x around the "
            f"origin, sampled at {self.n_steps} points: exact integer "
            f"{self.winds} (certified topological invariant)"
        )

    @staticmethod
    def _circle_path(winds: int) -> Path:
        """Build a closed circular Path winding `winds` times around the origin."""
        space = make_Lp_circle(2)

        def gamma(t: UnitInterval):
            theta = float(t.value.eval(20)) * 2 * math.pi * winds
            return (
                ExactReal.from_rational(Fraction(math.cos(theta)).limit_denominator(10 ** 9)),
                ExactReal.from_rational(Fraction(math.sin(theta)).limit_denominator(10 ** 9)),
            )

        return Path(gamma, space, name=f"circle_{winds}x")

    @staticmethod
    def _float_winding(winds: int, n_steps: int) -> float:
        """
        Float-only angle-accumulation winding number, for honest comparison.
        Mirrors the algorithm in etc.topology.invariants.winding_number but
        performed entirely in float64, with no certification.
        """
        def point(t_val: float):
            theta = t_val * 2 * math.pi * winds
            return math.cos(theta), math.sin(theta)

        angles = []
        for k in range(n_steps + 1):
            x, y = point(k / n_steps)
            angles.append(math.atan2(y, x))

        total = 0.0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff <= -math.pi:
                diff += 2 * math.pi
            total += diff
        return total / (2 * math.pi)

    def solve(self) -> Certified[int]:
        """
        Compute the exact integer winding number via angle-lifting, and
        attach a certificate carrying timing and rounding-error evidence.

        Returns Certified[int] with value == self.winds.
        """
        path = self._circle_path(self.winds)

        t0 = time.perf_counter()
        result = _winding_number(path, n_steps=self.n_steps)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        float_wn = self._float_winding(self.winds, self.n_steps)

        cert = Certificate(
            claim   = f"Winding number of a {self.winds}x circular loop around the origin = {self.winds}",
            method  = "angle_lifting + rounding (etc.topology.invariants.winding_number)",
            valid   = result.is_valid() and result.value == self.winds,
            evidence = {
                "winds":           self.winds,
                "n_steps":         self.n_steps,
                "winding_number":  result.value,
                "rounding_error":  result.proof.evidence["rounding_error"],
                "path_closed_gap": result.proof.evidence["path_closed_gap"],
                "prec_bits":       result.proof.evidence["prec_bits"],
                "float_winding_estimate": float_wn,
                "time_ms":         elapsed_ms,
            },
        )

        return Certified(result.value, cert)

    def verify(self, answer: Any) -> bool:
        """
        Verify that the claimed winding number is correct.

        Parameters
        ----------
        answer  – the claimed winding number

        Returns
        -------
        bool  – True iff answer == self.winds
        """
        try:
            return int(answer) == self.winds
        except (TypeError, ValueError):
            return False
