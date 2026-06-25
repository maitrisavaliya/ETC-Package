# ETC: Exact Topological Computing

A Python library for **exact real arithmetic**, **constructive topology**,
and **verified computation** with machine-checkable proof certificates.

ETC resolves floating-point failure through three mechanisms:
- **ExactReal** -- lazy Cauchy sequences over Q; no floating-point inside the machinery
- **Certified[V]** -- every discrete output (sign, winding number, orientation) paired with a Certificate
- **Lean 4 Export** -- certificates translate to syntactically valid Lean 4 theorem
  statements (correct `^`/`Real.sin`/`∀`-binder syntax via `etc.verify.lean_syntax`).
  Algebraic/ring identities are closed automatically with the `ring` tactic;
  transcendental and topological obligations are left as explicit `sorry` stubs
  with the specific Mathlib lemma needed. **Generated `.lean` files have not been
  checked against the Lean 4 kernel in this repository's CI** (no Lean toolchain
  is assumed to be installed) -- run `lake build` yourself to confirm before
  relying on them as machine-checked proofs.

## Installation

    pip install .
    pip install ".[dev]"   # with pytest, black, mypy

See INSTALL for full instructions including Lean 4 setup.
License: MIT (see LICENSE).

## Running the tests

    pytest tests/ -v

A reference capture of passing output is in `tests/expected/expected_output.txt`.

## Reproducing the paper's results

    python scripts/run_comparison.py   # Table 1: float64 vs ETC, all seven problems
    python scripts/benchmark.py        # Tables 3-6: timing and certification benchmarks

## Package structure

| Module | Contents |
|---|---|
| `etc.core` | ExactReal, Interval, ExactComplex, ExactInteger |
| `etc.geometry` | AlgebraicCurve, AlgebraicSurface, ExactPolygon |
| `etc.topology` | Space, Path, Homotopy, winding_number |
| `etc.analysis` | Calculus, PowerSeries, TaylorSeries, ODE |
| `etc.verify` | Certificate, CertificateStore, Lean4Exporter, lean_syntax |
| `etc.problems` | Problem, ProblemRegistry, canonical benchmark problems |
| `etc.certified` | Certified[V] generic wrapper |

Full API reference: `doc/manual.pdf`

## Author

Maitri Aniruddhbhai Savaliya  
GSFC University, Vadodara, Gujarat, India  
maitrisavaliya05@gmail.com  
ORCID: 0009-0003-8107-3817
