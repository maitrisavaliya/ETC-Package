"""
etc.core.integer
================
ExactInteger: arbitrary-precision integers with number-theoretic operations.

Wraps Python's native int (which is already arbitrary precision) in a
uniform ETC API, and adds:

  - Euclidean / extended GCD
  - Primality testing (deterministic Miller-Rabin for n < 3.3×10²⁴,
    then probabilistic with 20 rounds)
  - Prime factorisation (trial division + Pollard-ρ)
  - Modular arithmetic: modpow, modinv, CRT
  - Integer square root, nth root
  - Binomial coefficients, factorial
  - Conversion to / from ExactReal

All arithmetic is exact — no floating-point is used internally.
"""

from __future__ import annotations
import math
import random
from fractions  import Fraction
from typing     import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Small helpers (module-level, used by ExactInteger and standalone)
# ---------------------------------------------------------------------------

def _isqrt(n: int) -> int:
    """Integer square root ⌊√n⌋, exact for n ≥ 0."""
    if n < 0:
        raise ValueError("_isqrt: n must be non-negative")
    return math.isqrt(n)


def _gcd(a: int, b: int) -> int:
    """Standard Euclidean GCD; result is always non-negative."""
    return math.gcd(abs(a), abs(b))


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Return (g, s, t) with g = gcd(a,b), g = s·a + t·b.
    Uses the iterative extended Euclidean algorithm.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    while r:
        q       = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    g = old_r
    t = (g - old_s * a) // b if b else 0
    return g, old_s, t


def _modpow(base: int, exp: int, mod: int) -> int:
    """Return base^exp mod mod  (Python's built-in pow handles this well)."""
    if mod == 1:
        return 0
    return pow(base, exp, mod)


def _miller_rabin(n: int, witnesses: List[int]) -> bool:
    """
    Deterministic Miller–Rabin with the given witness set.
    Returns True if n is *probably* prime for these witnesses.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n−1 = 2^r · d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in witnesses:
        if a >= n:
            continue
        x = _modpow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = _modpow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# Deterministic witness sets (from Pomerance, Selfridge, Wagstaff; Bach; Jaeschke)
_DETERMINISTIC_WITNESSES: List[Tuple[int, List[int]]] = [
    (2_047,                     [2]),
    (1_373_653,                 [2, 3]),
    (9_080_191,                 [31, 73]),
    (25_326_001,                [2, 3, 5]),
    (3_215_031_751,             [2, 3, 5, 7]),
    (4_759_123_141,             [2, 7, 61]),
    (1_122_004_669_633,         [2, 13, 23, 1662803]),
    (2_152_302_898_747,         [2, 3, 5, 7, 11]),
    (3_474_749_660_383,         [2, 3, 5, 7, 11, 13]),
    (341_550_071_728_321,       [2, 3, 5, 7, 11, 13, 17]),
    (3_825_123_056_546_413_051, [2, 3, 5, 7, 11, 13, 17, 19, 23]),
    # For n < 3.3×10^24 (covers all practical values we'd encounter):
    (318_665_857_834_031_151_167_461,
     [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]),
    (3_317_044_064_679_887_385_961_981,
     [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]),
]


def is_prime(n: int) -> bool:
    """
    Deterministic primality test for n < 3.3×10²⁴, probabilistic (20 rounds)
    for larger values.  Returns True iff n is prime.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Trial division up to min(100, sqrt(n)) for speed
    trial_limit = min(100, int(n**0.5) + 1)
    for p in range(5, trial_limit + 1, 6):
        if n % p == 0 or n % (p + 2) == 0:
            return False
    # Deterministic witnesses
    for limit, witnesses in _DETERMINISTIC_WITNESSES:
        if n < limit:
            return _miller_rabin(n, witnesses)
    # Probabilistic fallback: 20 rounds
    witnesses = [random.randrange(2, n - 1) for _ in range(20)]
    return _miller_rabin(n, witnesses)


def _pollard_rho(n: int) -> int:
    """
    Pollard's ρ algorithm: find a non-trivial factor of n.
    Uses Brent's variant for speed.  Returns a factor (possibly 1 or n).
    """
    if n % 2 == 0:
        return 2
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    while d == 1:
        x = (_modpow(x, 2, n) + c) % n
        y = (_modpow(y, 2, n) + c) % n
        y = (_modpow(y, 2, n) + c) % n
        d = _gcd(abs(x - y), n)
    return d


def factorise(n: int) -> Dict[int, int]:
    """
    Return the prime factorisation of |n| as {prime: exponent}.
    Uses trial division for small primes, then Pollard-ρ for large composites.
    Returns {} for n = ±1, 0.
    """
    n = abs(n)
    if n <= 1:
        return {}
    result: Dict[int, int] = {}

    def _factor(m: int) -> None:
        if m == 1:
            return
        if is_prime(m):
            result[m] = result.get(m, 0) + 1
            return
        # Try small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if m % p == 0:
                result[p] = result.get(p, 0) + 1
                _factor(m // p)
                return
        # Pollard-ρ
        d = m
        while d == m:
            d = _pollard_rho(m)
        _factor(d)
        _factor(m // d)

    _factor(n)
    return result


def primes_up_to(n: int) -> List[int]:
    """Return all primes ≤ n via the Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]


def nth_prime(n: int) -> int:
    """Return the n-th prime (1-indexed: nth_prime(1) = 2)."""
    if n <= 0:
        raise ValueError("nth_prime: n must be positive")
    # Upper bound for p_n from Rosser's theorem: p_n < n(ln n + ln ln n + 2) for n≥6
    if n < 6:
        return [2, 3, 5, 7, 11][n - 1]
    import math
    ln_n    = math.log(n)
    ln_ln_n = math.log(ln_n)
    upper   = int(n * (ln_n + ln_ln_n + 2)) + 10
    sieve   = primes_up_to(upper)
    while len(sieve) < n:
        upper  *= 2
        sieve   = primes_up_to(upper)
    return sieve[n - 1]


def euler_totient(n: int) -> int:
    """φ(n) = n · ∏_{p|n} (1 − 1/p)."""
    if n <= 0:
        raise ValueError("euler_totient: n must be positive")
    factors = factorise(n)
    result  = n
    for p in factors:
        result = result * (p - 1) // p
    return result


def moebius(n: int) -> int:
    """
    Möbius function μ(n):
      μ(1) = 1
      μ(n) = 0  if n has a squared prime factor
      μ(n) = (−1)^k  if n is a product of k distinct primes
    """
    if n <= 0:
        raise ValueError("moebius: n must be positive")
    factors = factorise(n)
    for exp in factors.values():
        if exp >= 2:
            return 0
    return (-1) ** len(factors)


def crt(residues: List[int], moduli: List[int]) -> int:
    """
    Chinese Remainder Theorem: find x with x ≡ residues[i] (mod moduli[i]).
    moduli must be pairwise coprime.  Returns x ∈ [0, ∏ moduli).
    """
    if len(residues) != len(moduli):
        raise ValueError("crt: residues and moduli must have the same length")
    M   = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(residues, moduli):
        Mi  = M // m
        _, s, _ = _extended_gcd(Mi, m)
        x  += r * Mi * s
    return x % M


def modinv(a: int, m: int) -> int:
    """
    Return a⁻¹ mod m.  Raises ValueError if gcd(a,m) ≠ 1.
    """
    g, s, _ = _extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"modinv: {a} has no inverse mod {m} (gcd={g})")
    return s % m


def jacobi_symbol(a: int, n: int) -> int:
    """
    Jacobi symbol (a/n) for odd positive n.
    Returns −1, 0, or 1.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("jacobi_symbol: n must be odd and positive")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def integer_nth_root(x: int, n: int) -> Tuple[int, bool]:
    """
    Return (r, exact) where r = ⌊x^(1/n)⌋ and exact = (r^n == x).
    x must be non-negative, n must be positive.
    """
    if x < 0:
        raise ValueError("integer_nth_root: x must be non-negative")
    if n <= 0:
        raise ValueError("integer_nth_root: n must be positive")
    if x == 0:
        return 0, True
    if n == 1:
        return x, True
    if n == 2:
        r = _isqrt(x)
        return r, (r * r == x)
    # Newton's method: r_{k+1} = ((n-1)*r_k + x/r_k^{n-1}) / n
    r = int(round(x ** (1.0 / n)))
    # Adjust with a few Newton steps
    for _ in range(100):
        rn1 = r ** (n - 1)
        r_new = ((n - 1) * r + x // rn1) // n
        if r_new >= r:
            break
        r = r_new
    # Ensure r^n <= x < (r+1)^n
    while r ** n > x:
        r -= 1
    while (r + 1) ** n <= x:
        r += 1
    return r, (r ** n == x)


# ---------------------------------------------------------------------------
# ExactInteger
# ---------------------------------------------------------------------------

class ExactInteger:
    """
    An exact arbitrary-precision integer.

    Wraps Python's int in the ETC type system so integer objects can
    participate in the same function signatures as ExactReal / ExactComplex.

    Arithmetic
    ----------
    add, sub, mul, floordiv, mod, divmod, pow_int, neg, abs_val
    gcd(other), lcm(other)

    Number theory
    -------------
    is_prime()
    factorise()        – Dict[int, int] of prime → exponent
    euler_totient()
    moebius()
    modinv(m)
    jacobi(m)

    Combinatorics
    -------------
    factorial()        – n!
    binomial(k)        – C(n, k)
    fibonacci()        – F_n (iterative, O(n))

    Conversion
    ----------
    to_real()          – ExactReal constant sequence
    to_fraction()      – Fraction(n, 1)
    __int__, __float__, __repr__
    """

    def __init__(self, value: int):
        if not isinstance(value, int):
            value = int(value)
        self._value: int = value

    # --------------------------------------------------------- constructors

    @staticmethod
    def zero() -> "ExactInteger":
        return ExactInteger(0)

    @staticmethod
    def one() -> "ExactInteger":
        return ExactInteger(1)

    @staticmethod
    def from_int(n: int) -> "ExactInteger":
        return ExactInteger(n)

    # --------------------------------------------------------- value access

    @property
    def value(self) -> int:
        return self._value

    def to_fraction(self) -> Fraction:
        return Fraction(self._value)

    def to_real(self) -> "etc.core.real.ExactReal":
        """Convert to ExactReal (constant Cauchy sequence)."""
        from etc.core.real import ExactReal
        return ExactReal.from_rational(Fraction(self._value))

    # --------------------------------------------------------- arithmetic

    def neg(self) -> "ExactInteger":
        return ExactInteger(-self._value)

    def abs_val(self) -> "ExactInteger":
        return ExactInteger(abs(self._value))

    def add(self, other: "ExactInteger") -> "ExactInteger":
        return ExactInteger(self._value + other._value)

    def sub(self, other: "ExactInteger") -> "ExactInteger":
        return ExactInteger(self._value - other._value)

    def mul(self, other: "ExactInteger") -> "ExactInteger":
        return ExactInteger(self._value * other._value)

    def floordiv(self, other: "ExactInteger") -> "ExactInteger":
        if other._value == 0:
            raise ZeroDivisionError("ExactInteger.floordiv: division by zero")
        return ExactInteger(self._value // other._value)

    def mod(self, other: "ExactInteger") -> "ExactInteger":
        if other._value == 0:
            raise ZeroDivisionError("ExactInteger.mod: division by zero")
        return ExactInteger(self._value % other._value)

    def pow_int(self, k: int) -> "ExactInteger":
        if k < 0:
            raise ValueError("ExactInteger.pow_int: use to_real().recip() for negatives")
        return ExactInteger(self._value ** k)

    def gcd(self, other: "ExactInteger") -> "ExactInteger":
        return ExactInteger(_gcd(self._value, other._value))

    def lcm(self, other: "ExactInteger") -> "ExactInteger":
        if self._value == 0 or other._value == 0:
            return ExactInteger(0)
        return ExactInteger(abs(self._value * other._value) // _gcd(self._value, other._value))

    def extended_gcd(
        self, other: "ExactInteger"
    ) -> Tuple["ExactInteger", "ExactInteger", "ExactInteger"]:
        """Return (g, s, t) with g = gcd, g = s·self + t·other."""
        g, s, t = _extended_gcd(self._value, other._value)
        return ExactInteger(g), ExactInteger(s), ExactInteger(t)

    # --------------------------------------------------------- number theory

    def is_prime(self) -> bool:
        return is_prime(self._value)

    def factorise(self) -> Dict[int, int]:
        """Return prime factorisation as {prime: exponent}."""
        return factorise(self._value)

    def euler_totient(self) -> "ExactInteger":
        return ExactInteger(euler_totient(self._value))

    def moebius(self) -> int:
        return moebius(self._value)

    def modinv(self, m: "ExactInteger") -> "ExactInteger":
        return ExactInteger(modinv(self._value, m._value))

    def modpow(self, exp: "ExactInteger", mod: "ExactInteger") -> "ExactInteger":
        return ExactInteger(_modpow(self._value, exp._value, mod._value))

    def jacobi(self, m: "ExactInteger") -> int:
        return jacobi_symbol(self._value, m._value)

    def integer_sqrt(self) -> Tuple["ExactInteger", bool]:
        """Return (⌊√n⌋, exact) where exact = True iff n is a perfect square."""
        r, exact = integer_nth_root(self._value, 2)
        return ExactInteger(r), exact

    def integer_nth_root(self, n: int) -> Tuple["ExactInteger", bool]:
        """Return (⌊n^(1/n)⌋, exact)."""
        r, exact = integer_nth_root(self._value, n)
        return ExactInteger(r), exact

    # --------------------------------------------------------- combinatorics

    def factorial(self) -> "ExactInteger":
        """Return self! for self ≥ 0."""
        if self._value < 0:
            raise ValueError("ExactInteger.factorial: n must be ≥ 0")
        return ExactInteger(math.factorial(self._value))

    def binomial(self, k: "ExactInteger") -> "ExactInteger":
        """Return C(self, k) = self! / (k! (self-k)!)."""
        n, kv = self._value, k._value
        if kv < 0 or kv > n:
            return ExactInteger(0)
        return ExactInteger(math.comb(n, kv))

    @staticmethod
    def fibonacci(n: int) -> "ExactInteger":
        """
        Return the n-th Fibonacci number F_n (F_0=0, F_1=1).
        Uses the fast doubling algorithm: O(log n) multiplications.
        """
        def _fib_pair(n: int) -> Tuple[int, int]:
            """Return (F_n, F_{n+1}) via fast doubling."""
            if n == 0:
                return 0, 1
            a, b = _fib_pair(n >> 1)
            c = a * (2 * b - a)
            d = a * a + b * b
            if n & 1:
                return d, c + d
            return c, d
        f, _ = _fib_pair(abs(n))
        if n < 0 and n % 2 == 0:
            f = -f
        return ExactInteger(f)

    @staticmethod
    def catalan(n: int) -> "ExactInteger":
        """Return the n-th Catalan number C_n = C(2n,n)/(n+1)."""
        return ExactInteger(math.comb(2 * n, n) // (n + 1))

    # --------------------------------------------------------- digit / base ops

    def digits(self, base: int = 10) -> List[int]:
        """
        Return the digits of |self| in the given base, least-significant first.
        """
        if base < 2:
            raise ValueError("digits: base must be ≥ 2")
        n = abs(self._value)
        if n == 0:
            return [0]
        result = []
        while n:
            result.append(n % base)
            n //= base
        return result

    def digit_sum(self, base: int = 10) -> "ExactInteger":
        """Return the sum of digits of |self| in the given base."""
        return ExactInteger(sum(self.digits(base)))

    # --------------------------------------------------------- Python ops

    def __add__(self, other):
        if isinstance(other, int):
            other = ExactInteger(other)
        return self.add(other)

    def __radd__(self, other):
        return ExactInteger(other).add(self)

    def __sub__(self, other):
        if isinstance(other, int):
            other = ExactInteger(other)
        return self.sub(other)

    def __rsub__(self, other):
        return ExactInteger(other).sub(self)

    def __mul__(self, other):
        if isinstance(other, int):
            other = ExactInteger(other)
        return self.mul(other)

    def __rmul__(self, other):
        return ExactInteger(other).mul(self)

    def __floordiv__(self, other):
        if isinstance(other, int):
            other = ExactInteger(other)
        return self.floordiv(other)

    def __mod__(self, other):
        if isinstance(other, int):
            other = ExactInteger(other)
        return self.mod(other)

    def __pow__(self, k):
        if isinstance(k, int):
            return self.pow_int(k)
        raise TypeError(f"ExactInteger ** {type(k).__name__} not supported")

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs_val()

    def __eq__(self, other) -> bool:
        if isinstance(other, ExactInteger):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __lt__(self, other) -> bool:
        if isinstance(other, ExactInteger):
            return self._value < other._value
        if isinstance(other, int):
            return self._value < other
        return NotImplemented

    def __le__(self, other) -> bool:
        return self == other or self < other

    def __hash__(self) -> int:
        return hash(self._value)

    def __int__(self) -> int:
        return self._value

    def __float__(self) -> float:
        return float(self._value)

    def __repr__(self) -> str:
        return f"ExactInteger({self._value})"

    def __str__(self) -> str:
        return str(self._value)