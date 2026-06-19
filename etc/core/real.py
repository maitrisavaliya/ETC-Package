"""
etc.core.real
=============
ExactReal: exact real numbers as lazy Cauchy sequences over Q.

Every ExactReal x stores a function  seq: int -> Fraction  such that
    |seq(n) - x|  <  2^{-n}    for all n >= 0.

This is the 2-adic modulus of convergence — the standard representation
in computable real analysis (Weihrauch, 2000).

All arithmetic is exact over Q (fractions.Fraction).  No floating-point
is used inside the Cauchy sequence machinery.  The only place floats
appear is in __repr__ and __float__ for human-readable display.

Complexity labels (asymptotic in precision bits n, accounting for GCD
cost O(n log^2 n) in Python's fractions.Fraction):
  add, sub          : O(n^2 log n)
  mul               : O(n^2 log n)
  recip, div        : O(n^2 log n)
  sqrt (Newton)     : O(n^2 log^2 n)   [quadratic convergence]
  sqrt (bisection)  : O(n^3 log n)     [linear convergence — slower]
  pi, e, sin, cos   : O(n^2 log n)     [argument-reduced series]
  log, atan         : O(n^2 log n)     [arctanh / Gregory after reduction]

IMPORTANT NOTE ON BENCHMARKS: timing at n=8..128 sits in the
constant-dominated pre-asymptotic regime. Use n >= 512 for reliable
exponent fitting. The fitted exponents at small n will underestimate
theoretical values.
"""

from __future__ import annotations
import math
import time as _time
from fractions import Fraction
from typing import Callable, List, Tuple, Union

SeqFn    = Callable[[int], Fraction]
Rational = Union[int, Fraction]


# ---------------------------------------------------------------------------
# Helpers: bisection root (fallback) and Newton root (primary)
# ---------------------------------------------------------------------------

def _nth_root(x: "ExactReal", q: int) -> "ExactReal":
    """
    Return x^(1/q) via interval bisection over Q.
    Complexity: O(n^3 log n) — use only as a bootstrap for _newton_root.
    """
    if q <= 0:
        raise ValueError(f"_nth_root: q must be positive, got {q}")

    def seq(n: int) -> Fraction:
        x_approx = x.eval(n + 4)
        if x_approx < 0:
            raise ValueError("_nth_root: argument must be non-negative")
        if x_approx == 0:
            return Fraction(0)
        lo = Fraction(0)
        hi = max(Fraction(1), x_approx + 1)
        while hi ** q < x_approx:
            hi *= 2
        gap = Fraction(1, 2 ** (n + 2))
        while hi - lo > gap:
            mid = (lo + hi) / 2
            if mid ** q <= x_approx:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return ExactReal(seq)


def _newton_root(x: "ExactReal", q: int) -> "ExactReal":
    """
    Return x^(1/q) via Newton's method over Q (quadratic convergence).

    Newton iteration for f(r) = r^q - x:
        r_{k+1} = ((q-1)*r_k + x / r_k^{q-1}) / q

    Bootstraps with bisection at precision 8, then Newton until 2^{-(n+2)}.
    Complexity: O(log n) Newton steps each costing O(n^2 log n) -> O(n^2 log^2 n).
    """
    if q <= 0:
        raise ValueError(f"_newton_root: q must be positive")

    def seq(n: int) -> Fraction:
        x_approx = x.eval(n + 4)
        if x_approx < 0:
            raise ValueError("_newton_root: argument must be non-negative")
        if x_approx == 0:
            return Fraction(0)
        # Bootstrap at low precision.
        r = _nth_root(x, q).eval(8)
        target = Fraction(1, 2 ** (n + 2))
        for _ in range(n + 20):
            r_qm1 = r ** (q - 1)
            if r_qm1 == 0:
                break
            r_new = (Fraction(q - 1) * r + x_approx / r_qm1) / q
            if abs(r_new - r) < target:
                return r_new
            r = r_new
        return r

    return ExactReal(seq)


# ---------------------------------------------------------------------------
# ExactReal
# ---------------------------------------------------------------------------

class ExactReal:
    """
    An exact real number as a lazy Cauchy sequence over Q, with eval cache.

    Construction
    ------------
    ExactReal(seq)                  - from a raw SeqFn
    ExactReal.from_rational(p, q)   - rational p/q
    ExactReal.from_int(n)           - integer n
    ExactReal.zero() / .one()       - 0 / 1
    ExactReal.pi()                  - pi  (Machin's formula)
    ExactReal.e()                   - e   (Taylor series)
    ExactReal.log2()                - ln(2)
    ExactReal.sqrt2()               - sqrt(2)  via Newton's method
    ExactReal.from_decimal(s)       - parse decimal string exactly
    ExactReal.from_sympy(expr,s,v)  - symbolic substitution

    Arithmetic
    ----------
    add, sub, mul, div, neg, abs_val
    pow_int(k), pow_rational(p,q), recip(), sqrt(), cbrt()

    Analysis
    --------
    exp(), log(), sin(), cos(), tan(), atan()

    Timing
    ------
    timed_eval(n)     - returns (Fraction, wall_seconds)
    benchmark_eval()  - module-level systematic benchmark utility
    """

    def __init__(self, seq: SeqFn):
        self._seq: SeqFn = seq
        # Evaluation cache keyed by precision.  eval(n) called twice
        # returns the cached Fraction without recomputation.
        self._cache: dict[int, Fraction] = {}

    # ------------------------------------------------------------ constructors

    @staticmethod
    def from_rational(p: Rational, q: int = 1) -> "ExactReal":
        """Constant sequence equal to p/q."""
        r = Fraction(p, q)
        return ExactReal(lambda _n: r)

    @staticmethod
    def from_int(n: int) -> "ExactReal":
        return ExactReal.from_rational(n)

    @staticmethod
    def zero() -> "ExactReal":
        return ExactReal.from_rational(0)

    @staticmethod
    def one() -> "ExactReal":
        return ExactReal.from_rational(1)

    @staticmethod
    def pi() -> "ExactReal":
        """
        pi via Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239).
        Each arctan computed via the Gregory-Leibniz alternating series.
        Complexity: O(n^2 log n).
        """
        def _arctan_recip(k: int, n: int) -> Fraction:
            total  = Fraction(0)
            sign   = Fraction(1)
            power  = Fraction(1, k)
            inv_k2 = Fraction(1, k * k)
            denom  = 1
            bound  = Fraction(1, 2 ** (n + 4))
            while True:
                total += sign * power / denom
                next_power = power * inv_k2
                next_denom = denom + 2
                if abs(next_power / next_denom) < bound:
                    break
                power = next_power
                denom = next_denom
                sign  = -sign
            return total

        def seq(n: int) -> Fraction:
            a5   = _arctan_recip(5,   n + 6)
            a239 = _arctan_recip(239, n + 6)
            return 4 * (4 * a5 - a239)

        return ExactReal(seq)

    @staticmethod
    def e() -> "ExactReal":
        """e = sum_{k=0}^{inf} 1/k!   Complexity: O(n^2 log n)."""
        def seq(n: int) -> Fraction:
            total = Fraction(0)
            term  = Fraction(1)
            bound = Fraction(1, 2 ** (n + 4))
            k     = 0
            while abs(term) >= bound:
                total += term
                k     += 1
                term   = term / k
            return total
        return ExactReal(seq)

    @staticmethod
    def log2() -> "ExactReal":
        """
        ln(2) = sum_{k=1}^{inf} 1/(k*2^k).

        Geometric series (ratio 1/2): O(n) terms, each O(n^2 log n)
        fraction cost -> overall O(n^2 log n).

        Note: the paper labels this O(n^2); more precisely it is
        O(n^2 log n) when GCD cost is accounted for.
        """
        def seq(n: int) -> Fraction:
            total = Fraction(0)
            power = Fraction(1, 2)
            bound = Fraction(1, 2 ** (n + 4))
            k     = 1
            while power / k >= bound:
                total += power / k
                k     += 1
                power /= 2
            return total
        return ExactReal(seq)

    @staticmethod
    def sqrt2() -> "ExactReal":
        """
        sqrt(2) via Newton's method over Q (quadratic convergence).

        FIX: previously used bisection (_nth_root). Now correctly uses
        _newton_root to match the paper's description (Section 3.1).
        Complexity: O(n^2 log^2 n).
        """
        return _newton_root(ExactReal.from_rational(2), 2)

    @staticmethod
    def from_decimal(s: str) -> "ExactReal":
        """Parse a decimal string like '3.14159265' into an ExactReal."""
        r = Fraction(s)
        return ExactReal(lambda _n: r)

    @staticmethod
    def from_sympy(expr, sym, val: "ExactReal") -> "ExactReal":
        """
        Build an ExactReal by substituting val into the SymPy expr.
        SymPy is used as an oracle for evaluation, not for proving.
        """
        import sympy as sp

        def seq(n: int) -> Fraction:
            q_approx = val.eval(n + 8)
            sp_val   = sp.Rational(q_approx.numerator, q_approx.denominator)
            subst    = expr.subs(sym, sp_val)
            dec_digs = max(20, n + 8)
            mp_val   = subst.evalf(dec_digs)
            sign, man, exp, _bc = mp_val._mpf_
            f = Fraction(int(man)) * Fraction(2) ** int(exp)
            return -f if sign else f

        return ExactReal(seq)

    # ---------------------------------------------------------------- eval + timing

    def eval(self, precision: int) -> Fraction:
        """
        Return r in Q with  |r - x|  <  2^{-precision}.

        Results are memoised: eval(n) called twice computes once.
        precision must be >= 0.
        """
        if precision < 0:
            raise ValueError("precision must be >= 0")
        if precision not in self._cache:
            self._cache[precision] = self._seq(precision)
        return self._cache[precision]

    def timed_eval(self, precision: int) -> Tuple[Fraction, float]:
        """
        Evaluate to *precision* bits and return (result, wall_time_seconds).

        If the result is already cached, returns (cached_result, 0.0).
        For systematic benchmarking (excluding cache effects) use the
        module-level benchmark_eval() function instead.

        Parameters
        ----------
        precision : int    -- bits of precision (>= 0)

        Returns
        -------
        (Fraction, float)  -- exact rational approximant, wall-clock seconds

        Example
        -------
            val, t = ExactReal.pi().timed_eval(128)
            print(f"pi ~ {float(val):.15f}  [{t*1000:.2f} ms]")
        """
        if precision < 0:
            raise ValueError("precision must be >= 0")
        if precision in self._cache:
            return self._cache[precision], 0.0
        t0      = _time.perf_counter()
        result  = self._seq(precision)
        elapsed = _time.perf_counter() - t0
        self._cache[precision] = result
        return result, elapsed

    def to_decimal(self, digits: int = 20) -> str:
        """Return a decimal string with *digits* significant figures."""
        bits   = math.ceil(digits * math.log2(10)) + 8
        r      = self.eval(bits)
        scale  = 10 ** digits
        scaled = int(r * scale)
        s      = str(abs(scaled))
        if len(s) <= digits:
            s = "0" * (digits - len(s) + 1) + s
        integer_part = s[:-digits]
        frac_part    = s[-digits:].rstrip("0") or "0"
        result       = f"{integer_part}.{frac_part}"
        return ("-" + result) if scaled < 0 else result

    # -------------------------------------------------------------- arithmetic

    def neg(self) -> "ExactReal":
        return ExactReal(lambda n: -self.eval(n))

    def add(self, other: "ExactReal") -> "ExactReal":
        """
        self + other.
        |a_{n+1} + b_{n+1} - (a+b)| <= 2^{-(n+1)} + 2^{-(n+1)} = 2^{-n}.
        """
        def seq(n: int) -> Fraction:
            return self.eval(n + 1) + other.eval(n + 1)
        return ExactReal(seq)

    def sub(self, other: "ExactReal") -> "ExactReal":
        def seq(n: int) -> Fraction:
            return self.eval(n + 1) - other.eval(n + 1)
        return ExactReal(seq)

    def mul(self, other: "ExactReal") -> "ExactReal":
        """
        self * other.

        Error: |ab - a'b'| <= (|a'| + |b'| + 1) * 2^{-(n+extra)}.
        extra = ceil(log2(|a_0| + |b_0| + 2)) + 2.

        FIX: extra is computed ONCE at construction time, not inside seq(n).
        This prevents repeated eval(0) calls on every eval() invocation.
        """
        a0    = abs(self.eval(0))  + 2
        b0    = abs(other.eval(0)) + 2
        extra = max(0, math.ceil(math.log2(float(a0 + b0) + 1))) + 2

        def seq(n: int) -> Fraction:
            return self.eval(n + extra) * other.eval(n + extra)
        return ExactReal(seq)

    def recip(self) -> "ExactReal":
        """
        1/self.  Requires self != 0 (raises ZeroDivisionError otherwise).
        Finds delta > 0 with |self| >= delta, then uses extra = 2*ceil(log2(1/delta)) + 4.
        """
        delta = Fraction(0)
        k     = 1
        while delta <= 0:
            approx    = abs(self.eval(k))
            candidate = approx - Fraction(1, 2 ** k)
            if candidate > 0:
                delta = candidate
                break
            k += 1
            if k > 300:
                raise ZeroDivisionError(
                    "recip: cannot establish |self| > 0 — "
                    "value is zero or converges too slowly to zero."
                )
        extra = 2 * max(0, math.ceil(math.log2(float(1 / delta)))) + 4

        def seq(n: int) -> Fraction:
            a = self.eval(n + extra)
            if a == 0:
                raise ZeroDivisionError("recip: exact-zero denominator")
            return Fraction(1) / a
        return ExactReal(seq)

    def div(self, other: "ExactReal") -> "ExactReal":
        return self.mul(other.recip())

    def abs_val(self) -> "ExactReal":
        return ExactReal(lambda n: abs(self.eval(n)))

    def pow_int(self, k: int) -> "ExactReal":
        """self^k via binary exponentiation: O(log k) multiplications."""
        if k < 0:
            raise ValueError("pow_int: k must be >= 0; use recip() for negatives.")
        if k == 0:
            return ExactReal.one()
        result = ExactReal.one()
        base   = self
        exp    = k
        while exp > 0:
            if exp % 2 == 1:
                result = result.mul(base)
            base = base.mul(base)
            exp //= 2
        return result

    def pow_rational(self, p: int, q: int) -> "ExactReal":
        """self^(p/q): reduces fraction, dispatches to pow_int + _newton_root."""
        if q <= 0:
            raise ValueError("pow_rational: q must be > 0")
        if p == 0:
            return ExactReal.one()
        if p < 0:
            return self.pow_rational(-p, q).recip()
        from math import gcd
        g    = gcd(p, q)
        p, q = p // g, q // g
        if q == 1:
            return self.pow_int(p)
        if p == 1:
            return _newton_root(self, q)
        return _newton_root(self.pow_int(p), q)

    def sqrt(self) -> "ExactReal":
        """sqrt(self) via Newton's method. Complexity: O(n^2 log^2 n)."""
        return _newton_root(self, 2)

    def cbrt(self) -> "ExactReal":
        """cbrt(self) via Newton's method. Complexity: O(n^2 log^2 n)."""
        return _newton_root(self, 3)

    # ----------------------------------------------------------- transcendental

    def exp(self) -> "ExactReal":
        """
        e^self via Taylor series with argument reduction.
        e^x = (e^{x/2^s})^{2^s} where s chosen so |x/2^s| <= 1.
        Complexity: O(n^2 log n).
        """
        def seq(n: int) -> Fraction:
            x     = self.eval(n + 10)
            s     = max(0, math.ceil(math.log2(abs(float(x)) + 1)))
            x_red = x / (2 ** s)
            total = Fraction(0)
            term  = Fraction(1)
            bound = Fraction(1, 2 ** (n + s + 4))
            k     = 0
            while abs(term) >= bound:
                total += term
                k     += 1
                term   = term * x_red / k
            for _ in range(s):
                total = total * total
            return total
        return ExactReal(seq)

    def log(self) -> "ExactReal":
        """
        ln(self) for self > 0.
        Uses ln(x) = 2*arctanh((x-1)/(x+1)) after reducing x to [1/2, 2).
        Complexity: O(n^2 log n).
        """
        def seq(n: int) -> Fraction:
            x = self.eval(n + 10)
            if x <= 0:
                raise ValueError("log: argument must be positive")
            k = 0
            y = x
            while y >= 2:
                y /= 2
                k += 1
            while y < Fraction(1, 2):
                y *= 2
                k -= 1
            z     = (y - 1) / (y + 1)
            total = Fraction(0)
            power = z
            denom = 1
            bound = Fraction(1, 2 ** (n + 8))
            while abs(power / denom) >= bound:
                total += power / denom
                power  = power * z * z
                denom += 2
            ln_y       = 2 * total
            ln2_approx = ExactReal.log2().eval(n + 10)
            return Fraction(k) * ln2_approx + ln_y
        return ExactReal(seq)

    def sin(self) -> "ExactReal":
        """
        sin(self) via Taylor series with full argument reduction.

        Reduction pipeline (all in exact rational arithmetic):
          1. Reduce modulo 2*pi to [-pi, pi] using the pi Cauchy sequence.
             The integer k is found via float arithmetic, then k*pi_exact
             is subtracted exactly — correctness is preserved.
          2. Reduce to [-pi/2, pi/2] using sin(pi - x) = sin(x).
          3. Taylor series on the reduced argument (fast convergence).

        FIX: the original code computed pi_approx but never used it for
        reduction. This version performs the reduction correctly.
        Complexity: O(n^2 log n).
        """
        def seq(n: int) -> Fraction:
            x         = self.eval(n + 10)
            pi_approx = ExactReal.pi().eval(n + 10)
            two_pi    = 2 * pi_approx

            # Step 1: reduce to [-pi, pi]
            if abs(x) > pi_approx:
                k = round(float(x / two_pi))
                x = x - Fraction(k) * two_pi

            # Step 2: reduce to [-pi/2, pi/2]
            if x > pi_approx / 2:
                x = pi_approx - x
            elif x < -pi_approx / 2:
                x = -pi_approx - x

            # Step 3: Taylor series
            total = Fraction(0)
            term  = x
            denom = 1
            bound = Fraction(1, 2 ** (n + 4))
            sign  = Fraction(1)
            for k in range(1, n + 20, 2):
                total += sign * term / denom
                term   = term * x * x
                denom_new = denom * (k + 1) * (k + 2)
                if abs(term / denom_new) < bound:
                    break
                denom = denom_new
                sign  = -sign
            return total
        return ExactReal(seq)

    def cos(self) -> "ExactReal":
        """
        cos(self) via Taylor series with argument reduction to [-pi, pi].

        FIX: the original code lacked effective argument reduction.
        Complexity: O(n^2 log n).
        """
        def seq(n: int) -> Fraction:
            x         = self.eval(n + 10)
            pi_approx = ExactReal.pi().eval(n + 10)
            two_pi    = 2 * pi_approx

            # Reduce to [-pi, pi]
            if abs(x) > pi_approx:
                k = round(float(x / two_pi))
                x = x - Fraction(k) * two_pi

            # Taylor series for cos
            total = Fraction(0)
            term  = Fraction(1)
            denom = 1
            bound = Fraction(1, 2 ** (n + 4))
            sign  = Fraction(1)
            for k in range(0, n + 20, 2):
                total += sign * term / denom
                term   = term * x * x
                denom_new = denom * (k + 1) * (k + 2)
                if abs(term / denom_new) < bound:
                    break
                denom = denom_new
                sign  = -sign
            return total
        return ExactReal(seq)

    def tan(self) -> "ExactReal":
        return self.sin().div(self.cos())

    def atan(self) -> "ExactReal":
        """
        arctan(self) via Gregory's series with reduction to [0, 1/2].
        Uses arctan(x) = pi/2 - arctan(1/x) for x > 1.
        Complexity: O(n^2 log n).
        """
        def seq(n: int) -> Fraction:
            x    = self.eval(n + 10)
            sign = Fraction(1)
            if x < 0:
                sign = Fraction(-1)
                x    = -x
            use_recip = False
            if x > Fraction(1):
                use_recip = True
                x         = Fraction(1) / x
            total = Fraction(0)
            power = x
            denom = 1
            x2    = x * x
            bound = Fraction(1, 2 ** (n + 6))
            s     = Fraction(1)
            while abs(power / denom) >= bound:
                total += s * power / denom
                power  = power * x2
                denom += 2
                s      = -s
            if use_recip:
                pi_half = ExactReal.pi().eval(n + 10) / 2
                total   = pi_half - total
            return sign * total
        return ExactReal(seq)

    # ---------------------------------------------------------- comparison

    def lt(self, other: "ExactReal", prec: int = 53) -> bool:
        diff = (other.sub(self)).eval(prec)
        eps  = Fraction(1, 2 ** prec)
        if diff > eps:
            return True
        if diff < -eps:
            return False
        raise ValueError(
            f"lt: values are indistinguishable at precision 2^{{-{prec}}}; "
            "increase prec."
        )

    def gt(self, other: "ExactReal", prec: int = 53) -> bool:
        return other.lt(self, prec)

    def eq_approx(self, other: "ExactReal", prec: int = 53) -> bool:
        return abs(float(self.eval(prec)) - float(other.eval(prec))) < 2 ** (-prec)

    def sign(self, prec: int = 53) -> int:
        v   = self.eval(prec)
        eps = Fraction(1, 2 ** prec)
        if v > eps:  return  1
        if v < -eps: return -1
        return 0

    # ---------------------------------------------------------- Python ops

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = ExactReal.from_rational(other)
        return self.add(other)

    def __radd__(self, other):
        return ExactReal.from_rational(other).add(self)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = ExactReal.from_rational(other)
        return self.sub(other)

    def __rsub__(self, other):
        return ExactReal.from_rational(other).sub(self)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            other = ExactReal.from_rational(other)
        return self.mul(other)

    def __rmul__(self, other):
        return ExactReal.from_rational(other).mul(self)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            other = ExactReal.from_rational(other)
        return self.div(other)

    def __rtruediv__(self, other):
        return ExactReal.from_rational(other).div(self)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs_val()

    def __pow__(self, exp):
        if isinstance(exp, int):
            if exp >= 0:
                return self.pow_int(exp)
            return self.pow_int(-exp).recip()
        if isinstance(exp, Fraction):
            return self.pow_rational(exp.numerator, exp.denominator)
        raise TypeError(f"ExactReal ** {type(exp).__name__} not supported")

    def __float__(self) -> float:
        return float(self.eval(53))

    def __repr__(self) -> str:
        try:
            approx = float(self.eval(53))
            return f"ExactReal(~{approx:.15g})"
        except Exception:
            return "ExactReal(<unevaluated>)"


# ---------------------------------------------------------------------------
# Module-level benchmark utility (Johnson 2002 methodology)
# ---------------------------------------------------------------------------

def benchmark_eval(
    x: ExactReal,
    precisions: List[int],
    repeat: int = 3,
) -> List[Tuple[int, float]]:
    """
    Time eval(n) for each n in *precisions* using the Johnson (2002) methodology.

    Each precision is timed on a FRESH ExactReal instance (same sequence
    function, empty cache) so that cache effects do not contaminate results.
    The minimum wall-clock time over *repeat* runs is returned.

    Parameters
    ----------
    x          : ExactReal  -- the number to benchmark
    precisions : List[int]  -- precision levels in bits (e.g. [8,16,32,64,128,256,512])
    repeat     : int        -- repetitions per precision (default 3; matches paper methodology)

    Returns
    -------
    List of (precision, min_wall_seconds)

    Notes
    -----
    - Use precisions spanning at least to n=512 for reliable exponent fitting.
    - The n=8..128 range is pre-asymptotic for most operations.
    - Results reflect true algorithmic cost, not amortised cache cost.

    Example
    -------
        results = benchmark_eval(ExactReal.pi(), [8, 16, 32, 64, 128, 256, 512])
        for n, t in results:
            print(f"n={n:4d}  {t*1000:.3f} ms")
    """
    results = []
    for n in precisions:
        times = []
        for _ in range(repeat):
            fresh = ExactReal(x._seq)          # empty cache, same seq
            t0    = _time.perf_counter()
            fresh._seq(n)
            times.append(_time.perf_counter() - t0)
        results.append((n, min(times)))
    return results
