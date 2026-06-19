"""
etc.core.sign
=============
sign(x) — returns Certified[int] with value in {-1, 0, +1}.

The sign of an ExactReal is computed by evaluating its Cauchy sequence
to increasing precision until the value is unambiguously non-zero, or
until the caller's requested precision is reached (in which case 0 is
returned with a certificate noting the indeterminacy).

The certificate records:
  - The approximant used
  - The precision at which the sign was determined
  - Whether the result is exact (non-zero) or approximate (zero)
"""

from __future__ import annotations
from fractions import Fraction

from etc.core.real       import ExactReal
from etc.certified       import Certified
from etc.verify.certificate import Certificate


def sign(x: ExactReal, max_prec: int = 128) -> Certified[int]:
    """
    Return Certified(-1 | 0 | +1) — the sign of x.

    Algorithm
    ---------
    Evaluate x.eval(n) for n = 8, 16, 32, 64, 128, ... up to max_prec.
    At each precision n, if |approx| > 2^{-(n-1)}, the sign is certain.
    If no precision settles it, return 0 (certifiably within 2^{-max_prec} of 0).

    Parameters
    ----------
    x        : ExactReal
    max_prec : int   – maximum precision bits to try (default 128)

    Returns
    -------
    Certified[int]  with .value ∈ {-1, 0, +1}
    """
    prec = 8
    while prec <= max_prec:
        approx = x.eval(prec)
        eps    = Fraction(1, 2 ** (prec - 1))
        if approx > eps:
            v = 1
            cert = Certificate(
                claim   = f"sign(x) = +1",
                method  = f"cauchy_evaluation(prec={prec})",
                valid   = True,
                evidence = {
                    "approx":    str(approx),
                    "prec_bits": prec,
                    "certain":   True,
                    "eps":       str(eps),
                },
            )
            return Certified(v, cert)
        if approx < -eps:
            v = -1
            cert = Certificate(
                claim   = f"sign(x) = -1",
                method  = f"cauchy_evaluation(prec={prec})",
                valid   = True,
                evidence = {
                    "approx":    str(approx),
                    "prec_bits": prec,
                    "certain":   True,
                    "eps":       str(eps),
                },
            )
            return Certified(v, cert)
        prec *= 2

    # Could not distinguish from zero at max_prec
    approx = x.eval(max_prec)
    cert = Certificate(
        claim   = f"sign(x) = 0  (|x| < 2^{{-{max_prec}}})",
        method  = f"cauchy_evaluation(prec={max_prec}, indeterminate)",
        valid   = True,
        evidence = {
            "approx":        str(approx),
            "prec_bits":     max_prec,
            "certain":       False,
            "note":          (
                f"Value is within 2^{{-{max_prec}}} of 0; "
                "increase max_prec for a stronger statement."
            ),
        },
    )
    return Certified(0, cert)
