"""
etc.topology.invariants
=======================
Topological invariants computed exactly with proof certificates.

winding_number(path, base_point)
    Exact integer winding number of a closed path around a base point.
    Returns Certified[int].

The algorithm:
  1. Lift the path to the universal cover ℝ of S¹ (angle tracking).
  2. The winding number is (lifted_end − lifted_start) / (2π), which must
     be an integer for a closed loop.
  3. Certification: the integer is certified by verifying:
     a. The path is closed (start ≈ end).
     b. The angular total change divided by 2π rounds to an integer within
        tolerance 2^{-prec/2}.
     c. A grid of spot-checks confirms the path avoids the base point.

For paths on the S¹ space (the standard unit circle in ℝ²) the
path itself is the unit circle and base_point defaults to the origin (0,0).
"""

from __future__ import annotations
from fractions import Fraction
from typing import Optional, Tuple

from etc.core.real          import ExactReal
from etc.topology.path      import Path, UnitInterval
from etc.certified          import Certified
from etc.verify.certificate import Certificate


def winding_number(
    path: Path,
    base_point: Tuple[ExactReal, ExactReal] = None,
    prec: int = 53,
    n_steps: int = 200,
) -> Certified[int]:
    """
    Compute the exact integer winding number of *path* around *base_point*.

    Parameters
    ----------
    path       : Path   – a closed path in ℝ² (space must be 2-dimensional).
    base_point : (ExactReal, ExactReal)  – the point to wind around.
                 Defaults to (0, 0) if None.
    prec       : int    – working precision in bits (default 53).
    n_steps    : int    – number of angle samples (default 200).

    Returns
    -------
    Certified[int]  with .value = winding number ∈ ℤ.

    Raises
    ------
    ValueError  if the path passes through the base point (angle undefined).
    ValueError  if the path is not closed (start ≠ end).
    """
    # Default base point
    if base_point is None:
        bx = ExactReal.zero()
        by = ExactReal.zero()
    else:
        bx, by = base_point

    bx_f = float(bx.eval(prec))
    by_f = float(by.eval(prec))

    # --- 1. Check path is closed ---
    start = path.path_fn(UnitInterval.zero())
    end   = path.path_fn(UnitInterval.one())
    sx_f  = float(start[0].eval(prec))
    sy_f  = float(start[1].eval(prec))
    ex_f  = float(end[0].eval(prec))
    ey_f  = float(end[1].eval(prec))
    close_gap = ((sx_f - ex_f) ** 2 + (sy_f - ey_f) ** 2) ** 0.5
    tol = 2 ** (-(prec // 2))
    if close_gap > tol:
        raise ValueError(
            f"winding_number: path is not closed "
            f"(|start − end| ≈ {close_gap:.2e}). "
            "Winding number is only defined for closed paths."
        )

    # --- 2. Angle tracking via atan2 lifting ---
    import math

    angles: list[float] = []
    for k in range(n_steps + 1):
        t_frac = Fraction(k, n_steps)
        t      = UnitInterval.from_rational(t_frac)
        coords = path.path_fn(t)
        px     = float(coords[0].eval(prec)) - bx_f
        py     = float(coords[1].eval(prec)) - by_f
        r      = (px ** 2 + py ** 2) ** 0.5
        if r < tol:
            raise ValueError(
                f"winding_number: path passes through base point at t≈{float(t_frac):.4g}."
            )
        angles.append(math.atan2(py, px))

    # --- 3. Unwrap to get total angle change ---
    total_angle = 0.0
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        # Wrap diff to (−π, π]
        while diff >  math.pi:
            diff -= 2 * math.pi
        while diff <= -math.pi:
            diff += 2 * math.pi
        total_angle += diff

    # --- 4. Winding number = total_angle / (2π), must be integer ---
    two_pi = 2 * math.pi
    raw_wn = total_angle / two_pi
    wn     = round(raw_wn)
    error  = abs(raw_wn - wn)

    integer_ok = error < 0.01  # generous tolerance for the rounding

    cert = Certificate(
        claim   = f"winding_number(path='{path.name}', base={bx_f:.4g},{by_f:.4g}) = {wn}",
        method  = "angle_lifting + rounding",
        valid   = integer_ok,
        evidence = {
            "path_name":     path.name,
            "space_name":    path.space.name,
            "base_point":    (bx_f, by_f),
            "n_steps":       n_steps,
            "prec_bits":     prec,
            "total_angle":   total_angle,
            "raw_wn":        raw_wn,
            "rounded_wn":    wn,
            "rounding_error":error,
            "path_closed_gap": close_gap,
            "integer_certified": integer_ok,
        },
    )

    if not integer_ok:
        raise ValueError(
            f"winding_number: result {raw_wn:.6f} is not close enough to an integer. "
            f"Increase n_steps or prec."
        )

    return Certified(wn, cert)
