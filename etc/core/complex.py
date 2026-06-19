"""
etc.core.complex
================
ExactComplex: exact complex numbers z = re + im·i where re, im are ExactReal.

Supports full arithmetic and transcendental operations, useful for
Riemann surfaces, conformal maps, algebraic curves over ℂ.
"""

from __future__ import annotations
from fractions import Fraction
from etc.core.real import ExactReal


class ExactComplex:
    """
    An exact complex number z = re + im·i.

    Both re (real part) and im (imaginary part) are ExactReal instances,
    so z is represented by two lazy Cauchy sequences over ℚ.

    Construction
    ------------
    ExactComplex(re, im)
    ExactComplex.from_real(x)        – x + 0i
    ExactComplex.from_rationals(a,b) – a + bi  for rational a, b
    ExactComplex.i()                 – 0 + 1i
    ExactComplex.zero()              – 0 + 0i
    ExactComplex.one()               – 1 + 0i
    ExactComplex.from_polar(r, θ)    – r·e^{iθ}

    Arithmetic
    ----------
    add, sub, mul, div, neg, conj, abs_val (modulus), arg (argument)
    pow_int(k), pow_rational(p,q)
    recip()

    Transcendental
    --------------
    exp()  – e^z = e^re · (cos(im) + i·sin(im))
    log()  – principal value ln|z| + i·arg(z)
    sqrt() – principal square root
    sin(), cos(), tan()

    Display
    -------
    eval(prec) → (Fraction, Fraction)  (real, imag parts)
    __repr__
    """

    def __init__(self, re: ExactReal, im: ExactReal):
        self.re: ExactReal = re
        self.im: ExactReal = im

    # --------------------------------------------------------- constructors

    @staticmethod
    def from_real(x: ExactReal) -> "ExactComplex":
        return ExactComplex(x, ExactReal.zero())

    @staticmethod
    def from_rationals(a: Fraction, b: Fraction) -> "ExactComplex":
        return ExactComplex(
            ExactReal.from_rational(a),
            ExactReal.from_rational(b),
        )

    @staticmethod
    def zero() -> "ExactComplex":
        return ExactComplex(ExactReal.zero(), ExactReal.zero())

    @staticmethod
    def one() -> "ExactComplex":
        return ExactComplex(ExactReal.one(), ExactReal.zero())

    @staticmethod
    def i() -> "ExactComplex":
        return ExactComplex(ExactReal.zero(), ExactReal.one())

    @staticmethod
    def from_polar(r: ExactReal, theta: ExactReal) -> "ExactComplex":
        """Return r·e^{iθ} = r·cos(θ) + r·sin(θ)·i."""
        return ExactComplex(r.mul(theta.cos()), r.mul(theta.sin()))

    # --------------------------------------------------------- evaluation

    def eval(self, prec: int) -> tuple[Fraction, Fraction]:
        """Return (re_approx, im_approx) each within 2^{-prec} of true value."""
        return self.re.eval(prec), self.im.eval(prec)

    # --------------------------------------------------------- arithmetic

    def neg(self) -> "ExactComplex":
        return ExactComplex(self.re.neg(), self.im.neg())

    def conj(self) -> "ExactComplex":
        """Return the complex conjugate z̄ = re − im·i."""
        return ExactComplex(self.re, self.im.neg())

    def add(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.re.add(other.re), self.im.add(other.im))

    def sub(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.re.sub(other.re), self.im.sub(other.im))

    def mul(self, other: "ExactComplex") -> "ExactComplex":
        """(a+bi)(c+di) = (ac−bd) + (ad+bc)i."""
        ac = self.re.mul(other.re)
        bd = self.im.mul(other.im)
        ad = self.re.mul(other.im)
        bc = self.im.mul(other.re)
        return ExactComplex(ac.sub(bd), ad.add(bc))

    def abs_sq(self) -> ExactReal:
        """Return |z|² = re² + im²."""
        return self.re.pow_int(2).add(self.im.pow_int(2))

    def abs_val(self) -> ExactReal:
        """Return |z| = √(re² + im²)."""
        return self.abs_sq().sqrt()

    def recip(self) -> "ExactComplex":
        """Return 1/z = z̄ / |z|²."""
        mod_sq = self.abs_sq()
        return ExactComplex(
            self.re.div(mod_sq),
            self.im.neg().div(mod_sq),
        )

    def div(self, other: "ExactComplex") -> "ExactComplex":
        """Return self / other."""
        return self.mul(other.recip())

    def pow_int(self, k: int) -> "ExactComplex":
        """Return self^k for k ≥ 0."""
        if k < 0:
            return self.pow_int(-k).recip()
        if k == 0:
            return ExactComplex.one()
        result = ExactComplex.one()
        base   = self
        exp    = k
        while exp > 0:
            if exp % 2 == 1:
                result = result.mul(base)
            base = base.mul(base)
            exp //= 2
        return result

    def arg(self) -> ExactReal:
        """
        Return arg(z) ∈ (−π, π], the principal argument.
        Uses atan2 via:  arg(z) = 2·arctan(im / (|z| + re))  (Wnuk formula)
        """
        mod   = self.abs_val()
        denom = mod.add(self.re)
        return self.im.div(denom).atan().mul(ExactReal.from_rational(2))

    # --------------------------------------------------------- transcendental

    def exp(self) -> "ExactComplex":
        """e^z = e^re · (cos(im) + i·sin(im))."""
        e_re = self.re.exp()
        return ExactComplex(
            e_re.mul(self.im.cos()),
            e_re.mul(self.im.sin()),
        )

    def log(self) -> "ExactComplex":
        """Principal value: log(z) = ln|z| + i·arg(z)."""
        return ExactComplex(self.abs_val().log(), self.arg())

    def sqrt(self) -> "ExactComplex":
        """
        Principal square root of z.
        √z = √(|z|) · e^{i·arg(z)/2}
        """
        r     = self.abs_val()
        theta = self.arg()
        half  = ExactReal.from_rational(Fraction(1, 2))
        return ExactComplex.from_polar(r.sqrt(), theta.mul(half))

    def sin(self) -> "ExactComplex":
        """sin(z) = sin(re)·cosh(im) + i·cos(re)·sinh(im)."""
        # cosh(x) = (e^x + e^{-x})/2,  sinh(x) = (e^x - e^{-x})/2
        ep  = self.im.exp()
        em  = self.im.neg().exp()
        two = ExactReal.from_rational(2)
        cosh_im = ep.add(em).div(two)
        sinh_im = ep.sub(em).div(two)
        return ExactComplex(
            self.re.sin().mul(cosh_im),
            self.re.cos().mul(sinh_im),
        )

    def cos(self) -> "ExactComplex":
        """cos(z) = cos(re)·cosh(im) − i·sin(re)·sinh(im)."""
        ep  = self.im.exp()
        em  = self.im.neg().exp()
        two = ExactReal.from_rational(2)
        cosh_im = ep.add(em).div(two)
        sinh_im = ep.sub(em).div(two)
        return ExactComplex(
            self.re.cos().mul(cosh_im),
            self.re.sin().neg().mul(sinh_im),
        )

    # --------------------------------------------------------- Python ops

    def __add__(self, other):
        if isinstance(other, ExactReal):
            other = ExactComplex.from_real(other)
        return self.add(other)

    def __sub__(self, other):
        if isinstance(other, ExactReal):
            other = ExactComplex.from_real(other)
        return self.sub(other)

    def __mul__(self, other):
        if isinstance(other, ExactReal):
            other = ExactComplex.from_real(other)
        return self.mul(other)

    def __truediv__(self, other):
        if isinstance(other, ExactReal):
            other = ExactComplex.from_real(other)
        return self.div(other)

    def __neg__(self):
        return self.neg()

    def __repr__(self) -> str:
        re_f = float(self.re.eval(53))
        im_f = float(self.im.eval(53))
        sign = "+" if im_f >= 0 else "-"
        return f"ExactComplex({re_f:.10g} {sign} {abs(im_f):.10g}i)"
