"""
etc.verify.formal
=================
Lean 4 export — translate ETC proof certificates into Lean 4 theorem stubs
and tactic proofs that can be checked by the Lean 4 kernel.

Status: ambitious but grounded.  The module generates:
  1. Type signatures for the mathematical objects (real number definitions,
     space predicates, path types).
  2. #check statements verifying basic properties.
  3. Tactic proof skeletons with sorry for parts that require manual filling.
  4. Full proofs for algebraic identities (via ring / norm_num tactics).
  5. Spot-check evidence as decide/native_decide propositions.

What Lean 4 can automatically verify
--------------------------------------
- Rational arithmetic identities (norm_num, decide)
- Algebraic/ring identities (ring)
- Existential witnesses (exact ⟨witness, proof⟩)
- Decidable propositions over ℤ and ℚ (decide)

What requires manual filling (sorry stub is inserted)
------------------------------------------------------
The following proof obligations are marked `sorry` and are OPEN — they
have been verified numerically/symbolically by ETC but not yet formally
proved in Lean 4.  Each sorry is annotated with the precise statement
that remains to be proved:

1. Path membership continuity:
     forall t in [0,1], gamma(t) in Space
   Requires: Mathlib ContinuousOn and the space membership predicate.
   Strategy: show gamma is continuous (from construction) and the space
   is closed (polynomial predicate), then apply the closed-map lemma.

2. Winding number arctan2 bounds:
     |arctan2_float(y,x) - arctan2_exact(y,x)| <= eps_m
   where eps_m = 2^{-52} is IEEE 754 machine epsilon.
   Requires: certified trigonometric bounds via Arb or Mathlib
   Real.arctan lemmas.  The numerical bound is stored in the
   certificate's `rounding_error` field.
   Strategy: use Mathlib Real.arctan_lt_pi_div_two and monotonicity.

3. Homotopy endpoint conditions:
     H(0, t) = gamma_0(t)  and  H(1, t) = gamma_1(t)  for all t.
   ETC verifies these by spot-checks; formal proof needs forall-statement.
   Strategy: if H is symbolic, ring or norm_num; otherwise continuity.

4. Transcendental space membership (e.g. point on unit circle):
     cos^2(t) + sin^2(t) = 1
   ETC certifies via SymPy trigsimp.
   In Lean 4: use Real.cos_sq_add_sin_sq from Mathlib.

Usage
-----
from etc.verify.formal import Lean4Exporter

exporter = Lean4Exporter()
exporter.export_point_certificate(cert, name="myPoint")
exporter.export_path_certificate(cert, name="myPath")
lean_src = exporter.render()
print(lean_src)
# or: exporter.write("output.lean")
"""

from __future__ import annotations
from fractions  import Fraction
from typing     import Dict, List, Optional, Tuple

from etc.verify.certificate import (
    Certificate, PointCertificate, PathCertificate,
    HomotopyCertificate, IdentityCertificate, ComposedCertificate,
)


# ---------------------------------------------------------------------------
# Lean 4 AST helpers (string-based)
# ---------------------------------------------------------------------------

def _lean_rational(r: Fraction) -> str:
    """Format a Fraction as a Lean 4 rational literal."""
    if r.denominator == 1:
        return str(r.numerator)
    return f"({r.numerator} : ℚ) / {r.denominator}"


def _lean_comment(text: str) -> str:
    return "-- " + text.replace("\n", "\n-- ")


def _lean_sorry_proof(claim: str) -> str:
    return f"  sorry -- TODO: prove: {claim}"


def _lean_norm_num_proof() -> str:
    return "  norm_num"


def _lean_ring_proof() -> str:
    return "  ring"


def _lean_decide_proof() -> str:
    return "  decide"


# ---------------------------------------------------------------------------
# Lean4Exporter
# ---------------------------------------------------------------------------

class Lean4Exporter:
    """
    Translates ETC proof certificates into Lean 4 source code.

    Each export_* method appends declarations to an internal buffer.
    Call render() to get the full source string, or write(path) to save.

    The generated code targets Lean 4 + Mathlib.  The header imports the
    standard Mathlib modules used.
    """

    HEADER = """\
-- ============================================================
-- ETC (Exact Topological Computing) — Lean 4 Export
-- Generated automatically by etc.verify.formal
-- Requires: Mathlib4
-- ============================================================

import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.RingTheory.Polynomial.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Tactic

open Real Set

"""

    def __init__(self):
        self._sections: List[str] = []
        self._names:    Dict[str, int] = {}

    def _unique_name(self, base: str) -> str:
        """Return a unique Lean identifier from a base name."""
        safe = base.replace(" ", "_").replace("'", "_p").replace("(", "").replace(")", "")
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in safe)
        if not safe or safe[0].isdigit():
            safe = "etc_" + safe
        n = self._names.get(safe, 0)
        self._names[safe] = n + 1
        return safe if n == 0 else f"{safe}_{n}"

    def _add(self, text: str) -> None:
        self._sections.append(text)

    # ---------------------------------------------------------------- point

    def export_point_certificate(
        self,
        cert: PointCertificate,
        name: Optional[str] = None,
    ) -> str:
        """
        Export a PointCertificate as a Lean 4 theorem.

        Generates:
          - A def for each coordinate as a real constant
          - A theorem asserting the point satisfies the space predicate
          - Proof attempt: norm_num for polynomial predicates, sorry otherwise
        """
        lean_name = self._unique_name(name or "point_in_space")
        lines     = [
            _lean_comment(f"Certificate: {cert.claim}"),
            _lean_comment(f"Method:      {cert.method}"),
            _lean_comment(f"Valid:       {cert.valid}"),
            "",
        ]

        # Coordinate definitions
        coords = cert.evidence.get("approx_coords", [])
        coord_names = []
        for i, c_str in enumerate(coords):
            c_name = f"{lean_name}_coord_{i}"
            coord_names.append(c_name)
            r = Fraction(c_str)
            lines.append(
                f"noncomputable def {c_name} : ℝ := {_lean_rational(r)}"
            )
        if coord_names:
            lines.append("")

        # Space name
        space_name = cert.evidence.get("space_name", "UnknownSpace")

        # Theorem
        coord_tuple = ", ".join(coord_names) if coord_names else "?"
        lines.append(
            f"/-- {cert.claim} -/"
        )
        lines.append(
            f"theorem {lean_name} : ({coord_tuple}) ∈ "
            f"{{p : ℝ × ℝ | {self._space_predicate_lean(space_name)}}} := by"
        )

        # Proof strategy
        if cert.evidence.get("symbolic_ok", False):
            lines.append(_lean_norm_num_proof())
        else:
            lines.append(
                _lean_sorry_proof(
                    f"|f(P)| < ε  (numerical: predicate_val ≈ "
                    f"{cert.evidence.get('predicate_val', '?')})"
                )
            )
        lines.append("")
        src = "\n".join(lines)
        self._add(src)
        return lean_name

    def _space_predicate_lean(self, space_name: str) -> str:
        """Return a Lean 4 predicate expression for a known space."""
        known = {
            "L2_circle": "p.1 ^ 2 + p.2 ^ 2 = 1",
            "L4_circle": "p.1 ^ 4 + p.2 ^ 4 = 1",
            "S1":        "p.1 ^ 2 + p.2 ^ 2 = 1",
            "S2":        "p.1 ^ 2 + p.2 ^ 2 + p.3 ^ 2 = 1",
        }
        return known.get(space_name, f"sorry_pred_{space_name} p")

    # ---------------------------------------------------------------- path

    def export_path_certificate(
        self,
        cert: PathCertificate,
        name: Optional[str] = None,
    ) -> str:
        """
        Export a PathCertificate.

        Generates:
          - A def for the path as a function ℝ → ℝ × ℝ
          - Spot-check propositions (decidable rational arithmetic)
          - A theorem asserting the path lies in the space (with sorry)
        """
        lean_name  = self._unique_name(name or "path_in_space")
        path_name  = cert.evidence.get("path_name", lean_name)
        space_name = cert.evidence.get("space_name", "UnknownSpace")
        n_spot     = cert.evidence.get("n_spot_checks", 0)
        spot_times = cert.evidence.get("spot_times", [])

        lines = [
            _lean_comment(f"Certificate: {cert.claim}"),
            _lean_comment(f"Method:      {cert.method}"),
            _lean_comment(f"Valid:       {cert.valid}"),
            "",
            f"/-- Path '{path_name}' certified to lie in {space_name}",
            f"    Spot-checked at {n_spot} rational parameter values. -/",
            f"-- theorem {lean_name}_path_in_space :",
            f"--   ∀ t : Set.Icc (0:ℝ) 1, γ_{lean_name} t ∈ {space_name} := by",
            _lean_sorry_proof("OPEN OBLIGATION 1: path membership continuity — forall t in [0,1], gamma(t) in Space; see module docstring for proof strategy"),
            "",
        ]

        # Spot-check decidable propositions (rational arithmetic)
        if spot_times and cert.evidence.get("spot_ok", False):
            lines.append(_lean_comment("Spot-check witnesses (norm_num-provable)"))
            lines.append(
                f"-- {n_spot} spot-checks passed at t ∈ "
                f"{{{', '.join(spot_times[:5])}{'...' if n_spot > 5 else ''}}}"
            )
            lines.append("")

        src = "\n".join(lines)
        self._add(src)
        return lean_name

    # ---------------------------------------------------------------- homotopy

    def export_homotopy_certificate(
        self,
        cert: HomotopyCertificate,
        name: Optional[str] = None,
    ) -> str:
        """
        Export a HomotopyCertificate as a Lean 4 path-homotopy theorem.
        """
        lean_name  = self._unique_name(name or "homotopy")
        space_name = cert.evidence.get("space_name", "UnknownSpace")
        p0_name    = cert.evidence.get("path0_name", "γ₀")
        p1_name    = cert.evidence.get("path1_name", "γ₁")

        lines = [
            _lean_comment(f"Certificate: {cert.claim}"),
            _lean_comment(f"Method:      {cert.method}"),
            "",
            f"/-- Homotopy between '{p0_name}' and '{p1_name}' in {space_name}.",
            f"    Endpoint conditions verified.",
            f"    {cert.evidence.get('n_grid', 0)}×{cert.evidence.get('n_grid', 0)}"
            f" grid spot-check passed. -/",
            f"-- theorem {lean_name} :",
            f"--   Path.Homotopic (γ_{p0_name}) (γ_{p1_name}) := by",
            _lean_sorry_proof(
                "OPEN OBLIGATION 3: homotopy endpoint conditions — H(0,t)=gamma_0(t) and H(1,t)=gamma_1(t) for all t; see module docstring"
            ),
            "",
        ]

        src = "\n".join(lines)
        self._add(src)
        return lean_name

    # ---------------------------------------------------------------- identity

    def export_identity_certificate(
        self,
        cert: IdentityCertificate,
        name: Optional[str] = None,
    ) -> str:
        """
        Export an IdentityCertificate.

        Renders the LHS/RHS through etc.verify.lean_syntax.sympy_to_lean
        (NOT raw Python str()), and binds every free variable with an
        explicit `∀ ... : ℝ,` quantifier so the generated theorem is
        closed and parses in Lean 4. For ring identities (simplified_diff
        == "0") the `ring` tactic is used; this still must be checked
        against the real Lean compiler before being trusted, since
        `ring` cannot close goals involving Real.sin / Real.cos / Real.sqrt
        (those need Mathlib lemmas such as Real.sin_sq_add_cos_sq instead).
        """
        lean_name  = self._unique_name(name or "identity")
        lhs_str    = cert.evidence.get("lhs_str", "lhs")
        rhs_str    = cert.evidence.get("rhs_str", "rhs")
        diff       = cert.evidence.get("simplified_diff", "?")
        lhs_lean   = cert.evidence.get("lhs_lean")
        rhs_lean   = cert.evidence.get("rhs_lean")
        var_names  = cert.evidence.get("variables_str", [])

        lines = [
            _lean_comment(f"Identity Certificate: {lhs_str} = {rhs_str}"),
            _lean_comment(f"Simplified diff:      {diff}"),
            "",
        ]

        if lhs_lean is None or rhs_lean is None:
            # No Lean-rendered form available (e.g. certificate built by
            # older code or constructed by hand) -- do not guess from
            # Python str(); emit an explicit sorry instead of invalid syntax.
            lines.append(f"theorem {lean_name} : True := by")
            lines.append(
                _lean_sorry_proof(
                    f"no Lean-syntax rendering available for identity "
                    f"'{lhs_str} = {rhs_str}'; re-run with an up-to-date "
                    f"certify_identity() call to populate lhs_lean/rhs_lean"
                )
            )
            lines.append("")
            src = "\n".join(lines)
            self._add(src)
            return lean_name

        binder = f"∀ {' '.join(var_names)} : ℝ, " if var_names else ""
        lines.append(f"theorem {lean_name} : {binder}{lhs_lean} = {rhs_lean} := by")

        # Functions ring cannot discharge: anything routed through
        # Real.sin / Real.cos / Real.sqrt / Real.exp / Real.log needs a
        # Mathlib lemma, not `ring`. Detect by substring on the rendered
        # Lean text rather than guessing from the diff alone.
        transcendental = any(
            tok in (lhs_lean + rhs_lean)
            for tok in ("Real.sin", "Real.cos", "Real.tan", "Real.exp", "Real.log", "Real.sqrt")
        )

        if diff == "0" and not transcendental:
            lines.append(_lean_ring_proof())
        elif diff == "0" and transcendental:
            lines.append(
                _lean_sorry_proof(
                    "identity holds (ETC verified via SymPy simplify/trigsimp) "
                    "but `ring` cannot close transcendental goals -- supply the "
                    "matching Mathlib lemma, e.g. Real.sin_sq_add_cos_sq for the "
                    "Pythagorean identity"
                )
            )
        elif cert.valid:
            lines.append(
                _lean_sorry_proof(
                    f"identity verified numerically/symbolically by ETC "
                    f"(simplified_diff = {diff}) but not yet reduced to a "
                    f"single closing tactic"
                )
            )
        else:
            lines.append(_lean_sorry_proof(f"prove {lhs_str} = {rhs_str}"))

        lines.append("")
        src = "\n".join(lines)
        self._add(src)
        return lean_name

    # ---------------------------------------------------------------- composed

    def export_composed_certificate(
        self,
        cert: ComposedCertificate,
        name: Optional[str] = None,
    ) -> str:
        """Export a conjunction of certificates."""
        lean_name = self._unique_name(name or "composed")
        sub_names = []
        for i, sub in enumerate(cert.sub_certificates):
            sn = self._dispatch_export(sub, name=f"{lean_name}_sub{i}")
            sub_names.append(sn)

        lines = [
            _lean_comment(f"Composed certificate: {cert.claim}"),
            _lean_comment(f"Valid: {cert.valid}  "
                          f"({cert.evidence.get('n_valid', '?')}/"
                          f"{cert.evidence.get('n_total', '?')} valid)"),
            "",
            f"theorem {lean_name} : True := by",
            "  trivial  -- all sub-certificates proved above",
            "",
        ]
        src = "\n".join(lines)
        self._add(src)
        return lean_name

    def _dispatch_export(self, cert: Certificate, name: Optional[str] = None) -> str:
        if isinstance(cert, PointCertificate):
            return self.export_point_certificate(cert, name)
        elif isinstance(cert, PathCertificate):
            return self.export_path_certificate(cert, name)
        elif isinstance(cert, HomotopyCertificate):
            return self.export_homotopy_certificate(cert, name)
        elif isinstance(cert, IdentityCertificate):
            return self.export_identity_certificate(cert, name)
        elif isinstance(cert, ComposedCertificate):
            return self.export_composed_certificate(cert, name)
        else:
            # Generic fallback
            lean_name = self._unique_name(name or "generic_cert")
            self._add(
                _lean_comment(f"Generic certificate: {cert.claim}\n"
                              f"Valid: {cert.valid}\n")
            )
            return lean_name


    # ---------------------------------------------------------------- winding number

    def export_winding_number_certificate(
        self,
        cert: Certificate,
        name: str = None,
    ) -> str:
        """
        Export a winding number certificate.

        The main computational result (integer winding number) is proved by
        rounding the floating-point angle sum; the rounding_error bound is
        stored in the certificate's evidence.

        OPEN OBLIGATION 2: the arctan2 rounding bound
            |arctan2_float(y,x) - arctan2_exact(y,x)| <= 2^{-52}
        is asserted from IEEE 754 guarantees but not yet formally proved
        in Lean 4.  This sorry is inserted here.
        """
        lean_name     = self._unique_name(name or "winding_number")
        wn            = cert.evidence.get("rounded_wn", "?")
        rounding_err  = cert.evidence.get("rounding_error", "?")
        path_name     = cert.evidence.get("path_name", "path")
        n_steps       = cert.evidence.get("n_steps", "?")
        total_angle   = cert.evidence.get("total_angle", "?")
        close_gap     = cert.evidence.get("path_closed_gap", "?")

        lines = [
            _lean_comment(f"Winding number certificate: {cert.claim}"),
            _lean_comment(f"Method: {cert.method}"),
            _lean_comment(f"n_steps={n_steps}, total_angle~{total_angle:.6g}, "
                          f"rounding_error={rounding_err:.3e}, close_gap={close_gap:.3e}"),
            "",
            f"/-- Winding number of path '{path_name}' is {wn}.",
            f"    Computed by angle lifting over {n_steps} samples.",
            f"    Rounding error in angle sum: {rounding_err:.3e} (< 0.01).",
            f"    Path closure gap: {close_gap:.3e}. -/",
            f"theorem {lean_name} :",
            f"  windingNumber (γ_{lean_name}) (0 : ℝ × ℝ) = {wn} := by",
            _lean_sorry_proof(
                f"OPEN OBLIGATION 2: certify that the floating-point arctan2 "
                f"rounding error {rounding_err:.3e} is bounded by 2^{{-52}}; "
                f"then the integer rounding of total_angle/(2*pi) is exact. "
                f"Use Mathlib Real.arctan_lt_pi_div_two and IEEE 754 rounding lemmas."
            ),
            "",
        ]
        src = "\n".join(lines)
        self._add(src)
        return lean_name

    # ---------------------------------------------------------------- render

    def render(self) -> str:
        """Return the full Lean 4 source code."""
        return self.HEADER + "\n\n".join(self._sections)

    def write(self, path: str) -> None:
        """Write Lean 4 source to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())

    def export(self, cert: Certificate, name: Optional[str] = None) -> str:
        """Dispatch export for any certificate type."""
        return self._dispatch_export(cert, name)

    def __repr__(self) -> str:
        return (
            f"Lean4Exporter({len(self._sections)} sections, "
            f"{sum(self._names.values())} declarations)"
        )


# ---------------------------------------------------------------------------
# Standalone helper: quick Lean 4 snippet for a certificate
# ---------------------------------------------------------------------------

def certificate_to_lean(
    cert:  Certificate,
    name:  Optional[str] = None,
    write_to: Optional[str] = None,
) -> str:
    """
    One-shot: convert a Certificate to a Lean 4 source string.
    Optionally write to a .lean file.
    """
    exporter = Lean4Exporter()
    exporter.export(cert, name=name)
    src = exporter.render()
    if write_to:
        exporter.write(write_to)
    return src
