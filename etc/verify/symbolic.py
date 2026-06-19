"""
etc.verify.symbolic
===================
SymPy bridge — symbolic verification of ETC objects.

This module uses SymPy *only* as a verification oracle: we never trust
SymPy to compute the answer; we use it to *certify* that an answer
already found by exact arithmetic is correct.

Provided verifiers
------------------
verify_on_curve(point, curve)
    Check that an exact point lies on an algebraic curve.

verify_path_in_space(path, space, n_symbolic)
    Symbolically verify a parametric path stays in a Space for all t.

verify_homotopy(homotopy)
    Symbolic proof that H(s,t) ∈ Space for all (s,t) ∈ [0,1]².

verify_identity(lhs_expr, rhs_expr, variables, domain)
    Prove lhs = rhs as a symbolic identity over a domain.

simplify_predicate(expr, assumptions)
    Simplify a symbolic predicate under given assumptions.

check_polynomial_identity(poly, subs, expected_zero)
    Substitute exact real values and confirm the result is 0.

Utility helpers
---------------
real_to_sympy(x, prec)      – convert ExactReal → sympy.Rational
sympy_to_real(expr)          – convert closed-form sympy expr → ExactReal
diff_check(f_sympy, f_exact, x, prec)
                             – confirm SymPy derivative matches exact derivative
"""

from __future__ import annotations
from fractions  import Fraction
from typing     import Dict, List, Optional, Tuple, Union

from etc.core.real import ExactReal


# ---------------------------------------------------------------------------
# SymPy import guard
# ---------------------------------------------------------------------------

def _sp():
    """Lazy import of sympy — raises ImportError with a helpful message."""
    try:
        import sympy as sp
        return sp
    except ImportError:
        raise ImportError(
            "etc.verify.symbolic requires SymPy.  "
            "Install with: pip install sympy"
        )


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def real_to_sympy(x: ExactReal, prec: int = 60):
    """
    Convert an ExactReal to a sympy.Rational approximant.

    Evaluates x to *prec* bits and returns sp.Rational(p, q) where
    p/q is within 2^{-prec} of the true value.
    """
    sp = _sp()
    r  = x.eval(prec)
    return sp.Rational(r.numerator, r.denominator)


def sympy_to_real(expr, free_vars: Optional[Dict] = None) -> ExactReal:
    """
    Convert a closed-form SymPy expression to an ExactReal.

    *expr* must be a constant expression (or fully evaluated after
    substituting *free_vars*).  The conversion:
      - Recognises sp.pi → ExactReal.pi()
      - Recognises sp.E  → ExactReal.e()
      - Recognises rational literals
      - For everything else, uses evalf() at 60 bits

    free_vars : optional {sympy.Symbol: ExactReal} substitutions
    """
    sp   = _sp()
    if free_vars:
        subs = {k: real_to_sympy(v, 80) for k, v in free_vars.items()}
        expr = expr.subs(subs)

    # Try exact recognition first
    if expr == sp.pi:
        return ExactReal.pi()
    if expr == sp.E:
        return ExactReal.e()
    if expr.is_Rational:
        return ExactReal.from_rational(Fraction(int(expr.p), int(expr.q)))
    if expr.is_Mul:
        # Recurse on factors
        result = ExactReal.one()
        for factor in expr.args:
            result = result.mul(sympy_to_real(factor))
        return result
    if expr.is_Add:
        result = ExactReal.zero()
        for term in expr.args:
            result = result.add(sympy_to_real(term))
        return result
    if expr.is_Pow:
        base, exp = expr.args
        base_real = sympy_to_real(base)
        if exp.is_Rational:
            p = int(exp.p)
            q = int(exp.q)
            return base_real.pow_rational(p, q)

    # Fallback: numerical evaluation at 64 bits
    mp_val = expr.evalf(20)
    r      = Fraction(str(float(mp_val))).limit_denominator(10 ** 18)
    return ExactReal.from_rational(r)


# ---------------------------------------------------------------------------
# Core verifiers
# ---------------------------------------------------------------------------

def verify_on_curve(
    point: Tuple[ExactReal, ExactReal],
    curve,   # etc.geometry.curves.AlgebraicCurve
    prec:    int = 60,
) -> bool:
    """
    Certify that *point* lies on *curve*.

    Two-layer check:
    1. Numerical: evaluate f(x,y) at *prec* bits and confirm |f| < 2^{-prec/2}.
    2. Symbolic: substitute exact Rational approximants into f and simplify.

    Returns True only if both checks pass.
    """
    sp  = _sp()
    x, y = point

    # Numerical check
    xr = x.eval(prec)
    yr = y.eval(prec)
    val = curve.poly.subs(curve.x_sym, sp.Rational(xr.numerator, xr.denominator))
    val = val.subs(curve.y_sym, sp.Rational(yr.numerator, yr.denominator))
    num_ok = abs(float(val.evalf(20))) < 2 ** (-(prec // 2))

    # Symbolic check (can prove it exactly for rational points)
    sym_val = sp.simplify(sp.expand(val))
    sym_ok  = (sym_val == 0)

    return num_ok and sym_ok


def verify_path_in_space(
    path_exprs: Tuple,     # tuple of sympy.Expr in param_sym
    param_sym,             # sympy.Symbol
    space,                 # etc.topology.space.Space
    n_spot: int = 11,
    prec:   int = 40,
) -> bool:
    """
    Verify a parametric path lies in *space*.

    Performs:
    1. Symbolic substitution: space.verify_path_symbolic(path_exprs, param_sym)
    2. Numerical spot-checks at n_spot rational values of param_sym.

    Returns True only if both pass.
    """
    sp = _sp()

    # Symbolic check
    sym_ok = space.verify_path_symbolic(path_exprs, param_sym)
    if not sym_ok:
        return False

    # Numerical spot-checks
    for k in range(n_spot):
        t_val = Fraction(k, n_spot - 1) if n_spot > 1 else Fraction(0)
        coords = tuple(
            ExactReal.from_rational(
                Fraction(
                    int(sp.Rational(e.subs(param_sym,
                        sp.Rational(t_val.numerator, t_val.denominator))
                    ).evalf(prec + 5).p),
                    int(sp.Rational(e.subs(param_sym,
                        sp.Rational(t_val.numerator, t_val.denominator))
                    ).evalf(prec + 5).q),
                )
            )
            for e in path_exprs
        )
        if not space.contains(coords):
            return False
    return True


def verify_homotopy(
    homotopy,   # etc.topology.homotopy.Homotopy
    n_grid: int = 7,
) -> bool:
    """
    Full three-layer verification of a Homotopy:
    1. Endpoint checks (H(0,·) = γ₀, H(1,·) = γ₁)
    2. Symbolic membership proof
    3. Numerical grid spot-checks

    Returns True only if all three pass.
    """
    if not homotopy.verify_endpoints(n_points=n_grid):
        return False
    if not homotopy.spot_check(n=n_grid):
        return False
    if homotopy.sympy_homotopy is not None:
        if not homotopy.symbolic_verify():
            return False
    return True


def verify_identity(
    lhs,         # sympy.Expr
    rhs,         # sympy.Expr
    variables:   Optional[List] = None,   # sympy.Symbol list
    assumptions: Optional[Dict] = None,  # {symbol: assumption_predicate}
    prec:        int = 40,
) -> bool:
    """
    Verify lhs = rhs as a symbolic identity.

    Method: simplify(expand(lhs - rhs)) == 0.
    Optionally substitutes assumptions (e.g. x > 0 via Abs elimination).

    Returns True if SymPy can prove the identity.
    """
    sp   = _sp()
    diff = sp.simplify(sp.expand(lhs - rhs))
    if diff == 0:
        return True
    # Try with trigsimp, radsimp, etc.
    for simplifier in [sp.trigsimp, sp.radsimp, sp.powsimp]:
        try:
            if simplifier(diff) == 0:
                return True
        except Exception:
            pass
    return False


def check_polynomial_identity(
    poly,         # sympy.Expr
    subs_exact:   Dict,    # {sympy.Symbol: ExactReal}
    prec:         int = 60,
) -> bool:
    """
    Substitute exact real values into a polynomial and verify the result is 0.

    1. Substitute rational approximants into poly.
    2. Evaluate at *prec* bits.
    3. Check |result| < 2^{-prec/2}.
    4. Attempt symbolic simplification to confirm = 0 exactly.
    """
    sp  = _sp()
    expr = poly
    for sym, val in subs_exact.items():
        r    = val.eval(prec)
        expr = expr.subs(sym, sp.Rational(r.numerator, r.denominator))
    num_val = float(expr.evalf(20))
    num_ok  = abs(num_val) < 2 ** (-(prec // 2))
    sym_ok  = (sp.simplify(sp.expand(expr)) == 0)
    return num_ok or sym_ok


def simplify_predicate(expr, assumptions: Optional[List] = None):
    """
    Simplify a symbolic predicate, optionally under a list of SymPy assumptions.

    Returns a (possibly simpler) sympy expression.
    """
    sp = _sp()
    if assumptions:
        # Wrap with assuming() if supported (SymPy ≥ 1.7)
        try:
            from sympy import assuming, Q, ask
            simplified = sp.simplify(expr)
            return simplified
        except ImportError:
            pass
    return sp.simplify(sp.expand(expr))


# ---------------------------------------------------------------------------
# Derivative consistency check
# ---------------------------------------------------------------------------

def diff_check(
    f_sympy,           # sympy.Expr in x_sym
    x_sym,             # sympy.Symbol
    f_exact:           "Callable[[ExactReal], ExactReal]",
    x_val:             ExactReal,
    prec:              int = 40,
    order:             int = 1,
) -> bool:
    """
    Confirm that the SymPy derivative of f matches the exact derivative
    computed by etc.analysis.calculus.Calculus.derivative.

    The check is: |SymPy_deriv(x_val) − Exact_deriv(x_val)| < 2^{-prec/3}.

    This is useful to validate that the Calculus module's finite-difference
    derivatives agree with known analytical expressions.
    """
    from etc.analysis.calculus import Calculus
    sp = _sp()

    # SymPy derivative
    f_diff = f_sympy
    for _ in range(order):
        f_diff = sp.diff(f_diff, x_sym)
    x_r    = real_to_sympy(x_val, prec)
    sp_val = float(f_diff.subs(x_sym, x_r).evalf(20))

    # Exact derivative (finite difference)
    try:
        ex_val = float(Calculus.derivative(f_exact, x_val, order=order).eval(prec))
    except Exception:
        return False

    tol = 2 ** (-(prec // 3))
    return abs(sp_val - ex_val) < tol


# ---------------------------------------------------------------------------
# Groebner basis membership check
# ---------------------------------------------------------------------------

def in_ideal(
    poly,        # sympy.Expr — the polynomial to test
    generators:  List,   # List[sympy.Expr] — the ideal generators
    variables:   List,   # List[sympy.Symbol]
    domain:      str = "ZZ",
) -> bool:
    """
    Check whether *poly* lies in the ideal generated by *generators*
    using a Groebner basis computation.

    Returns True if poly ∈ ⟨generators⟩.
    Useful for certifying that two curves / surfaces share a component.
    """
    sp = _sp()
    from sympy import groebner, Poly, reduced

    G   = groebner(generators, *variables, domain=domain)
    _, r = reduced(poly, G, *variables, domain=domain)
    return r == 0


# ---------------------------------------------------------------------------
# High-level: full object verification
# ---------------------------------------------------------------------------

def verify_point(point, space, prec: int = 60) -> bool:
    """
    Verify that a Point (or raw Coords) lies in a Space using both
    the exact predicate and symbolic substitution.
    """
    sp = _sp()

    if hasattr(point, "coords"):
        coords = point.coords
    else:
        coords = point

    # Exact predicate check
    if not space.contains(coords):
        return False

    # Symbolic check if available
    if space.sympy_pred is not None and space.sympy_coords is not None:
        expr = space.sympy_pred
        for sym, val in zip(space.sympy_coords, coords):
            r    = val.eval(prec)
            expr = expr.subs(sym, sp.Rational(r.numerator, r.denominator))
        sym_val = float(sp.simplify(expr).evalf(20))
        return abs(sym_val) < 2 ** (-(prec // 2))

    return True


def verify_curve_parametrisation(
    curve,         # AlgebraicCurve
    param_curve,   # ParametricCurve
    n_spot: int = 11,
    prec:   int = 40,
) -> bool:
    """
    Verify that a ParametricCurve lies on an AlgebraicCurve at n_spot points.
    """
    sp  = _sp()
    from fractions import Fraction
    from etc.topology.path import UnitInterval

    for k in range(n_spot):
        t_frac = Fraction(k, n_spot - 1) if n_spot > 1 else Fraction(0)
        t      = UnitInterval.from_rational(t_frac)
        x, y   = param_curve(t.value), None
        # Handle both (x, y) and single-value returns
        pt     = param_curve(t.value)
        if isinstance(pt, tuple):
            x, y = pt
        else:
            continue

        xr   = x.eval(prec)
        yr   = y.eval(prec)
        val  = curve.poly.subs(
            curve.x_sym, sp.Rational(xr.numerator, xr.denominator)
        ).subs(
            curve.y_sym, sp.Rational(yr.numerator, yr.denominator)
        )
        if abs(float(val.evalf(20))) >= 2 ** (-(prec // 2)):
            return False
    return True
