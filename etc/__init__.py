"""
ETC — Exact Topological Computing
==================================
A unified library for exact real arithmetic, constructive topology,
and verified computation.

Quick start
-----------
    from etc import ExactReal, sign, winding_number, benchmark_eval

    # Exact arithmetic
    x = ExactReal.pi().sub(ExactReal.from_rational(3))
    s = sign(x)
    print(s)                  # Certified(value=1, claim='sign(x) = +1', VALID)
    print(s.value)            # 1
    print(s.proof.summary())  # full certificate

    # Timing a single evaluation
    val, t = ExactReal.pi().timed_eval(128)
    print(f"pi computed in {t*1000:.2f} ms")

    # Systematic benchmark (Johnson 2002 methodology)
    results = benchmark_eval(ExactReal.pi(), [8, 16, 32, 64, 128, 256, 512])
    for n, secs in results:
        print(f"n={n:4d}  {secs*1000:.3f} ms")

Public API
----------
This package exposes 40 public symbols via __all__.  See each module's
docstring for full details.

Version
-------
__version__ = "0.1.0"
"""

__version__ = "0.1.0"
__author__  = "ETC Project"

# ── Core numeric types ────────────────────────────────────────────────────────
from etc.core.real      import ExactReal, benchmark_eval
from etc.core.interval  import Interval, interval_eval
from etc.core.complex   import ExactComplex
from etc.core.integer   import ExactInteger

# ── Geometry ─────────────────────────────────────────────────────────────────
from etc.geometry.curves   import AlgebraicCurve, ParametricCurve
from etc.geometry.surfaces import AlgebraicSurface
from etc.geometry.polygon  import ExactPolygon

# ── Topology ──────────────────────────────────────────────────────────────────
from etc.topology.space    import Space, Point, make_Lp_circle, make_sphere, make_torus
from etc.topology.path     import Path, UnitInterval
from etc.topology.homotopy import Homotopy

# ── Analysis ──────────────────────────────────────────────────────────────────
from etc.analysis.series   import PowerSeries, TaylorSeries
from etc.analysis.calculus import Calculus
from etc.analysis.ode      import ODE, ODESolution

# ── Verification ─────────────────────────────────────────────────────────────
from etc.verify.certificate import (
    Certificate,
    PointCertificate,
    PathCertificate,
    HomotopyCertificate,
    IdentityCertificate,
    ComposedCertificate,
    CertificateStore,
    certify_point,
    certify_path,
    certify_homotopy,
    certify_identity,
    compose,
)
from etc.verify.formal  import Lean4Exporter, certificate_to_lean

# ── Certified wrapper ─────────────────────────────────────────────────────────
from etc.certified import Certified

# ── Key operations (primary public API) ───────────────────────────────────────
from etc.core.sign           import sign
from etc.topology.invariants import winding_number

__all__ = [
    # Geometry
    "AlgebraicCurve",
    "ParametricCurve",
    "AlgebraicSurface",
    "ExactPolygon",
    # Core types
    "ExactReal",
    "Interval",
    "interval_eval",
    "ExactComplex",
    "ExactInteger",
    # Topology
    "Space",
    "Point",
    "make_Lp_circle",
    "make_sphere",
    "make_torus",
    "Path",
    "UnitInterval",
    "Homotopy",
    # Analysis
    "PowerSeries",
    "TaylorSeries",
    "Calculus",
    "ODE",
    "ODESolution",
    # Verification
    "Certificate",
    "PointCertificate",
    "PathCertificate",
    "HomotopyCertificate",
    "IdentityCertificate",
    "ComposedCertificate",
    "CertificateStore",
    "certify_point",
    "certify_path",
    "certify_homotopy",
    "certify_identity",
    "compose",
    "Lean4Exporter",
    "certificate_to_lean",
    # Certified wrapper + key operations
    "Certified",
    "sign",
    "benchmark_eval",
    "winding_number",
]