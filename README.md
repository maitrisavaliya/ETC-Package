# ETC: Exact Topological Computing

A Python library for **exact real arithmetic**, **constructive topology**,
and **verified computation** with machine-checkable proof certificates.

ETC resolves floating-point failure through three mechanisms:
- **ExactReal** -- lazy Cauchy sequences over Q; no floating-point inside the machinery
- **Certified[V]** -- every discrete output (sign, winding number, orientation) paired with a Certificate
- **Lean 4 Export** -- certificates translate to Lean 4 theorem stubs checkable by the Lean kernel

## Installation

    pip install .
    pip install ".[dev]"   # with pytest, black, mypy

See INSTALL for full instructions including Lean 4 setup.

## Running the tests

    pytest tests/ -v


## Running the benchmark

    python scripts/benchmark.py

Reproduces the paper's timing tables (pi scaling, transcendental constants,
arithmetic operations, sign certification).

## Package structure

| Module | Contents |
|---|---|
| `etc.core` | ExactReal, Interval, ExactComplex, ExactInteger |
| `etc.geometry` | AlgebraicCurve, AlgebraicSurface, ExactPolygon |
| `etc.topology` | Space, Path, Homotopy, winding_number |
| `etc.analysis` | Calculus, PowerSeries, TaylorSeries, ODE |
| `etc.verify` | Certificate, CertificateStore, Lean4Exporter |
| `etc.problems` | Problem, ProblemRegistry, canonical benchmark problems |
| `etc.certified` | Certified[V] generic wrapper |

Full API reference: `doc/manual.pdf`

## Author

Maitri Aniruddhbhai Savaliya  
GSFC University, Vadodara, Gujarat, India  
maitrisavaliya05@gmail.com  
ORCID: 0009-0003-8107-3817
