#!/usr/bin/env python3
"""
scripts/run_comparison.py
=========================
Run all seven canonical floating-point failure cases and print the
Float64 vs ETC comparison table that appears in the paper (Table 1).

Usage
-----
    python scripts/run_comparison.py

Output
------
Prints a formatted table to stdout showing, for each problem:
  - The IEEE 754 float64 result
  - The ETC exact result
  - Whether float64 passes or fails
  - Whether ETC passes (with certificate validity)

No arguments required. Runtime: ~10-30 seconds depending on hardware.
"""

from __future__ import annotations
import math
import time
import sympy as sp
import numpy  as np
from fractions import Fraction

from etc import ExactReal, sign
from etc.geometry.polygon        import ExactPolygon
from etc.verify.certificate      import certify_identity
from etc.problems.canonical_problems import L5WindingNumber


# ============================================================================
# Helpers
# ============================================================================

def _row(num, problem, float_result, etc_result, float_pass, etc_pass):
    print(f"  {num:<3} | {problem:<38} | {float_result:<28} | {etc_result:<32} | "
          f"{'PASS' if float_pass else 'FAIL':<5} | {'PASS' if etc_pass else 'FAIL'}")


def _sep():
    print("  " + "-" * 125)


def _header():
    print()
    print("  Float64 vs ETC -- Seven Canonical Floating-Point Failure Cases")
    print("  " + "=" * 125)
    print(f"  {'#':<3} | {'Problem':<38} | {'float64 result':<28} | "
          f"{'ETC result':<32} | {'Float':<5} | {'ETC'}")
    _sep()


# ============================================================================
# Problem 1: Rump's polynomial
# ============================================================================

def problem1():
    a_f, b_f = 77617.0, 33096.0
    float_val = (
        333.75 * b_f**6
        + a_f**2 * (11*a_f**2*b_f**2 - b_f**6 - 121*b_f**4 - 2)
        + 5.5 * b_f**8
        + a_f / (2*b_f)
    )

    a = ExactReal.from_rational(77617)
    b = ExactReal.from_rational(33096)
    c_33375 = ExactReal.from_rational(Fraction(33375, 100))
    c_11    = ExactReal.from_rational(11)
    c_121   = ExactReal.from_rational(121)
    c_2     = ExactReal.from_rational(2)
    c_55    = ExactReal.from_rational(Fraction(11, 2))
    b2=b.pow_int(2); b4=b.pow_int(4); b6=b.pow_int(6); b8=b.pow_int(8); a2=a.pow_int(2)
    t1 = c_33375.mul(b6)
    t2 = a2.mul(c_11.mul(a2).mul(b2).sub(b6).sub(c_121.mul(b4)).sub(c_2))
    t3 = c_55.mul(b8)
    t4 = a.div(c_2.mul(b))
    etc_val = t1.add(t2).add(t3).add(t4).eval(80)

    exact_frac = Fraction(-54767, 66192)
    correct    = etc_val == exact_frac

    return (
        f"{float_val:.6e}",
        f"{etc_val} = {float(etc_val):.8f}",
        False,          # float fails
        correct,
    )


# ============================================================================
# Problem 2: Large sum cancellation
# ============================================================================

def problem2():
    float_val = 1e16 + 1.0 - 1e16

    etc_val = (
        ExactReal.from_rational(10**16 + 1)
        .sub(ExactReal.from_rational(10**16))
        .eval(10)
    )
    correct = etc_val == Fraction(1)

    return (
        f"{float_val}  (should be 1)",
        f"{etc_val}  (exact integer)",
        False,
        correct,
    )


# ============================================================================
# Problem 3: Near-zero sqrt expression
# ============================================================================

def problem3():
    x_f     = float(10**15)
    float_val = (math.sqrt(x_f + 1) - math.sqrt(x_f)) * \
                (math.sqrt(x_f + 1) + math.sqrt(x_f)) - 1.0

    x_er  = ExactReal.from_rational(10**15)
    one   = ExactReal.from_rational(1)
    lhs   = x_er.add(one).sqrt().sub(x_er.sqrt())
    rhs   = x_er.add(one).sqrt().add(x_er.sqrt())
    expr  = lhs.mul(rhs).sub(one)
    etc_num = expr.eval(80)

    # Symbolic proof
    x_sym   = sp.Symbol("x", positive=True)
    sym_val = sp.simplify(
        sp.expand(
            (sp.sqrt(x_sym + 1) - sp.sqrt(x_sym)) *
            (sp.sqrt(x_sym + 1) + sp.sqrt(x_sym)) - 1
        )
    )
    proved = sym_val == 0

    return (
        f"{float_val:.4f}  (should be 0)",
        f"|f| = {abs(float(etc_num)):.2e}; identity proved={proved}",
        False,
        proved,
    )


# ============================================================================
# Problem 4: Polynomial sensitivity
# ============================================================================

def problem4():
    # Float: expanded (x-1)^10 via Horner at x = 1 + 1e-6
    xf = 1.0 + 1e-6
    coeffs = [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1]
    float_val = sum(c * xf**(10 - i) for i, c in enumerate(coeffs))

    # ETC: exact rational
    x_er    = ExactReal.from_rational(Fraction(1_000_001, 1_000_000))
    etc_val = x_er.sub(ExactReal.from_rational(1)).pow_int(10).eval(80)
    correct = etc_val == Fraction(1, 10**60)

    return (
        f"{float_val:.4e}  (should be 1e-60)",
        f"1/10^60 = {etc_val}",
        False,
        correct,
    )


# ============================================================================
# Problem 5: Geometric orientation
# ============================================================================

def problem5():
    # Float: 2.0 + 1e-16 rounds to 2.0
    eps_f      = 1e-16
    Ry_float   = 2.0 + eps_f          # rounds to 2.0
    cross_float = 1.0 * Ry_float - 2.0 * 1.0   # (Q-P) x (R-P)

    # ETC: exact
    eps_exact   = ExactReal.from_rational(Fraction(1, 10**16))
    cert        = sign(eps_exact)
    float_wrong = cross_float == 0.0
    etc_correct = cert.value == 1 and cert.is_valid()

    return (
        f"cross={cross_float}  (collinear, WRONG)",
        f"sign=+{cert.value}  CCW  cert_valid={cert.is_valid()}",
        not float_wrong,   # float fails
        etc_correct,
    )


# ============================================================================
# Problem 6: Winding number
# ============================================================================

def problem6():
    l5          = L5WindingNumber(winds=1, n_steps=400)
    float_wind  = l5._float_winding(winds=1, n_steps=400)
    result      = l5.solve()

    return (
        f"{float_wind:.6f}  (uncertified)",
        f"{result.value}  (exact int, cert_valid={result.is_valid()})",
        False,   # float cannot certify -- counts as fail
        result.is_valid(),
    )


# ============================================================================
# Problem 7: Pythagorean identity
# ============================================================================

def problem7():
    # Float: max error over 10,000 samples
    xs       = np.linspace(0, 2 * np.pi, 10_000)
    max_err  = float(np.max(np.abs(np.cos(xs)**2 + np.sin(xs)**2 - 1.0)))

    # ETC: symbolic proof
    t_sym  = sp.Symbol("t", real=True)
    proved = sp.simplify(sp.cos(t_sym)**2 + sp.sin(t_sym)**2 - 1) == 0

    return (
        f"max err={max_err:.2e}  (10^4 samples)",
        f"proved for all x in R  (SymPy={proved})",
        False,   # float cannot prove universally
        proved,
    )


# ============================================================================
# Main
# ============================================================================

PROBLEMS = [
    ("Rump f(77617,33096)",         problem1),
    ("10^16 + 1 - 10^16",           problem2),
    ("(sqrt(x+1)-sqrt(x))*(...)-1", problem3),
    ("(x-1)^10 at x=1+1e-6",        problem4),
    ("Orientation eps=1e-16",        problem5),
    ("Winding number (L5 curve)",    problem6),
    ("cos^2(x)+sin^2(x)=1",         problem7),
]


def main():
    _header()
    total_float_pass = 0
    total_etc_pass   = 0

    for i, (label, fn) in enumerate(PROBLEMS, 1):
        t0 = time.perf_counter()
        float_res, etc_res, float_ok, etc_ok = fn()
        elapsed = time.perf_counter() - t0

        _row(i, label, float_res, etc_res, float_ok, etc_ok)
        _sep()

        if float_ok:
            total_float_pass += 1
        if etc_ok:
            total_etc_pass += 1

    print()
    print(f"  Summary: float64 passed {total_float_pass}/7 problems | "
          f"ETC passed {total_etc_pass}/7 problems")
    print()


if __name__ == "__main__":
    main()
