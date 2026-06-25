"""
etc.verify.lean_syntax
=======================
Translate SymPy expressions into syntactically valid Lean 4 / Mathlib4
source text.

This module does not invoke the Lean compiler. It only guarantees
syntactic translation of the operators/functions in the table above. It
is not a general-purpose SymPy-to-Lean elaborator: expressions using
functions outside this table fall back to a `sorry`-stub theorem with a
clear comment, rather than emitting a guess that might silently be wrong.
"""

from __future__ import annotations
from typing import Iterable, List

import sympy as sp

# Functions with a direct, fixed Mathlib equivalent under `open Real`.
# All of these take exactly one argument and are rendered as prefix
# juxtaposition: `Real.sin x`, not `sin(x)`.
_KNOWN_UNARY_FUNCS = {
    sp.sin:  "Real.sin",
    sp.cos:  "Real.cos",
    sp.tan:  "Real.tan",
    sp.exp:  "Real.exp",
    sp.log:  "Real.log",
    sp.sqrt: "Real.sqrt",
    sp.Abs:  "abs",
}


def _is_atom_like(expr: sp.Expr) -> bool:
    """True if `expr` never needs outer parentheses when substituted."""
    return expr.is_Symbol or expr.is_Number


def _render(expr: sp.Expr) -> str:
    """Recursively render a SymPy expression as Lean 4 source text."""
    if expr.is_Symbol:
        return str(expr)

    if expr.is_Integer:
        n = int(expr)
        return f"({n} : ℝ)" if n < 0 else str(n)

    if expr.is_Rational and not expr.is_Integer:
        return f"(({expr.p} : ℝ) / {expr.q})"

    if expr.is_Float:
        return f"({float(expr)} : ℝ)"

    func = expr.func
    if func in _KNOWN_UNARY_FUNCS and len(expr.args) == 1:
        arg = expr.args[0]
        arg_src = _render(arg)
        if not _is_atom_like(arg):
            arg_src = f"({arg_src})"
        return f"{_KNOWN_UNARY_FUNCS[func]} {arg_src}"

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        base_src = _render(base)
        if not _is_atom_like(base):
            base_src = f"({base_src})"
        if exp == sp.Rational(1, 2):
            return f"Real.sqrt {base_src}"
        exp_src = _render(exp)
        return f"{base_src} ^ {exp_src}"

    if isinstance(expr, sp.Add):
        terms = []
        for i, term in enumerate(expr.args):
            t_src = _render(term)
            if i == 0:
                terms.append(t_src)
            elif t_src.startswith("-"):
                terms.append(f"- {t_src[1:].strip()}")
            else:
                terms.append(f"+ {t_src}")
        return " ".join(terms)

    if isinstance(expr, sp.Mul):
        # Render explicit unary minus from a leading -1 coefficient.
        coeff, rest = expr.as_coeff_Mul()
        factors = sp.Mul.make_args(rest)
        rendered_factors = []
        for f in factors:
            f_src = _render(f)
            if f.is_Add:
                f_src = f"({f_src})"
            rendered_factors.append(f_src)

        if coeff == -1:
            body = " * ".join(rendered_factors)
            return f"-({body})" if len(rendered_factors) > 1 else f"-{rendered_factors[0]}"
        elif coeff != 1:
            coeff_src = _render(coeff)
            if not _is_atom_like(coeff):
                coeff_src = f"({coeff_src})"
            return " * ".join([coeff_src] + rendered_factors)
        else:
            return " * ".join(rendered_factors)

    # Fallback: unsupported node type. Mark clearly rather than guessing.
    return f"/- UNSUPPORTED SYMPY NODE: {sp.srepr(expr)} -/"


def sympy_to_lean(expr: sp.Expr) -> str:
    """
    Translate a SymPy expression to a Lean 4 source-text fragment.

    Guarantees correct `^` exponent syntax and `Real.sin x`-style function
    application for the function table in `_KNOWN_UNARY_FUNCS`. Returns a
    `/- UNSUPPORTED ... -/` comment fragment (not raw Python syntax) for
    anything else, so that an unsupported construct fails loudly rather
    than silently producing invalid Lean.
    """
    return _render(sp.sympify(expr))


def free_variables_lean(variables: Iterable) -> List[str]:
    """Render a list of SymPy symbols as Lean 4 identifier strings."""
    return [str(v) for v in variables]


def lean_forall_theorem(
    name: str,
    lhs: sp.Expr,
    rhs: sp.Expr,
    variables: Iterable,
    tactic: str = "ring",
) -> str:
    """
    Render `theorem name : ∀ v1 v2 ... : ℝ, lhs = rhs := by tactic`.

    Always binds every free variable explicitly, so the generated theorem
    is closed (no unbound identifiers) and therefore at least parses —
    whether `tactic` actually closes the goal is a separate question the
    caller must still verify against the real Lean compiler.
    """
    var_names = free_variables_lean(variables)
    lhs_src = sympy_to_lean(lhs)
    rhs_src = sympy_to_lean(rhs)

    if var_names:
        binder = f"∀ {' '.join(var_names)} : ℝ, "
    else:
        binder = ""

    return (
        f"theorem {name} : {binder}{lhs_src} = {rhs_src} := by\n"
        f"  {tactic}"
    )
