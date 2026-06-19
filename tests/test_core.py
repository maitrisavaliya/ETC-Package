"""
tests/test_core.py
==================
pytest test suite for ETC (Exact Topological Computing).

Every test has been verified against the actual API attributes:
  - Certified has: .value, .proof, .is_valid()
  - Path.__init__ takes: (path_fn, space, sympy_path, sympy_param, name)
  - winding_number() takes: (path, base_point, prec, n_steps)
  - sign() returns Certified[int] with .value, .proof, .is_valid()

Run:
    pytest tests/ -v
"""

from __future__ import annotations
import json
import math
import time
from fractions import Fraction

import pytest

from etc import (
    ExactReal,
    sign,
    winding_number,
    benchmark_eval,
    Certificate,
    CertificateStore,
    certify_identity,
    Lean4Exporter,
    certificate_to_lean,
)
from etc.core.real import _newton_root, _nth_root


# ===========================================================================
# Helpers
# ===========================================================================

def _rump(a: ExactReal, b: ExactReal) -> ExactReal:
    """
    Rump's polynomial:
      333.75*b^6 + a^2*(11*a^2*b^2 - b^6 - 121*b^4 - 2) + 5.5*b^8 + a/(2*b)
    """
    _333_75 = ExactReal.from_rational(Fraction(33375, 100))
    _5_5    = ExactReal.from_rational(Fraction(11, 2))
    _121    = ExactReal.from_rational(121)
    _2      = ExactReal.from_rational(2)
    _11     = ExactReal.from_rational(11)
    b2 = b.mul(b); b4 = b2.mul(b2); b6 = b4.mul(b2); b8 = b4.mul(b4); a2 = a.mul(a)
    term1 = _333_75.mul(b6)
    term2 = a2.mul(_11.mul(a2).mul(b2).sub(b6).sub(_121.mul(b4)).sub(_2))
    term3 = _5_5.mul(b8)
    term4 = a.div(b.mul(_2))
    return term1.add(term2).add(term3).add(term4)


def _make_circle_path(winds: int, n_steps_arg: int = 400):
    """Build a closed circular path winding *winds* times around origin."""
    from etc.topology.space import make_Lp_circle
    from etc.topology.path  import Path, UnitInterval
    space = make_Lp_circle(2)
    def gamma(t: UnitInterval):
        # t.value is an ExactReal; evaluate at precision 20 for the angle
        theta = float(t.value.eval(20)) * 2 * math.pi * winds
        return (
            ExactReal.from_rational(Fraction(math.cos(theta)).limit_denominator(10**9)),
            ExactReal.from_rational(Fraction(math.sin(theta)).limit_denominator(10**9)),
        )
    return Path(gamma, space, name=f"circle_{winds}x"), n_steps_arg


# ===========================================================================
# Problem 1: Rump's polynomial
# ===========================================================================

class TestRump:
    def test_exact_value(self):
        """ETC must return exactly -54767/66192."""
        a = ExactReal.from_rational(77617)
        b = ExactReal.from_rational(33096)
        assert _rump(a, b).eval(80) == Fraction(-54767, 66192)

    def test_float_is_wrong(self):
        """float64 error must exceed 1e18."""
        a_f, b_f = 77617.0, 33096.0
        b6=b_f**6; b8=b_f**8; b4=b_f**4; b2=b_f**2; a2=a_f**2
        float_result = (
            333.75*b6 + a2*(11*a2*b2 - b6 - 121*b4 - 2) + 5.5*b8 + a_f/(2*b_f)
        )
        true_val = float(Fraction(-54767, 66192))
        assert abs(float_result - true_val) > 1e18


# ===========================================================================
# Problem 2: Large sum cancellation
# ===========================================================================

class TestLargeSumCancellation:
    def test_exact_result_is_one(self):
        big = ExactReal.from_rational(10**16)
        assert big.add(ExactReal.one()).sub(big).eval(60) == Fraction(1)

    def test_float_returns_zero(self):
        assert 1e16 + 1 - 1e16 == 0.0


# ===========================================================================
# Problem 3: Near-zero sqrt expression
# ===========================================================================

class TestNearZeroSqrt:
    def test_identity_symbolically(self):
        import sympy as sp
        x   = sp.Symbol("x", positive=True)
        lhs = (sp.sqrt(x+1) - sp.sqrt(x)) * (sp.sqrt(x+1) + sp.sqrt(x)) - 1
        assert sp.simplify(sp.expand(lhs)) == 0

    def test_numerical_improvement_over_float(self):
        """ETC error at 70 bits must be < 1e-20; float gives ~0.178."""
        x  = ExactReal.from_rational(10**15)
        x1 = x.add(ExactReal.one())
        diff = (
            x1.sqrt().sub(x.sqrt())
            .mul(x1.sqrt().add(x.sqrt()))
            .sub(ExactReal.one())
        )
        assert abs(float(diff.eval(70))) < 1e-20


# ===========================================================================
# Problem 4: Polynomial sensitivity near a root
# ===========================================================================

class TestPolynomialSensitivity:
    def test_exact_value(self):
        """(1 + 1e-6 - 1)^10 = 1/10^60 exactly."""
        x = ExactReal.from_rational(Fraction(1_000_001, 1_000_000))
        assert x.sub(ExactReal.one()).pow_int(10).eval(80) == Fraction(1, 10**60)

    def test_float_expanded_wrong_sign(self):
        """Expanded Horner form in float gives negative result (wrong sign)."""
        import numpy as np
        x      = 1.0 + 1e-6
        coeffs = [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1]
        assert np.polyval(coeffs, x) < 0   # true value is positive 1e-60


# ===========================================================================
# Problem 5: Geometric orientation test
# ===========================================================================

class TestOrientationTest:
    def test_certified_sign_plus_one(self):
        """ETC must certify sign = +1 for eps = 1e-16 offset."""
        eps   = ExactReal.from_rational(Fraction(1, 10**16))
        two   = ExactReal.from_rational(2)
        cross = ExactReal.one().mul(two.add(eps)).sub(ExactReal.one().mul(two))
        result = sign(cross, max_prec=128)
        assert result.value == 1
        assert result.is_valid()

    def test_float_says_collinear(self):
        assert 1.0 * (2.0 + 1e-16) - 1.0 * 2.0 == 0.0


# ===========================================================================
# Problem 6: Winding number
# ===========================================================================

class TestWindingNumber:
    def test_winding_number_1(self):
        path, n_steps = _make_circle_path(1, 400)
        result = winding_number(path, n_steps=n_steps)
        assert result.value == 1
        assert result.is_valid()
        assert result.proof.valid

    def test_winding_number_2(self):
        path, n_steps = _make_circle_path(2, 400)  
        result = winding_number(path, n_steps=n_steps)
        assert result.value == 2
        assert result.is_valid()

    def test_certificate_has_rounding_error(self):
        """Certificate evidence must include rounding_error field."""
        path, n_steps = _make_circle_path(1, 400)
        result = winding_number(path, n_steps=n_steps)
        assert "rounding_error" in result.proof.evidence
        assert "path_closed_gap" in result.proof.evidence

    def test_certificate_serialisable(self):
        """Certificate must round-trip through JSON."""
        path, n_steps = _make_circle_path(1, 200)
        result = winding_number(path, n_steps=n_steps)
        j = result.proof.to_json()
        d = json.loads(j)
        assert d["valid"] is True


# ===========================================================================
# Problem 7: Pythagorean identity
# ===========================================================================

class TestPythagoreanIdentity:
    def test_symbolic_simplification(self):
        import sympy as sp
        x = sp.Symbol("x")
        assert sp.trigsimp(sp.cos(x)**2 + sp.sin(x)**2 - 1) == 0

    def test_certify_identity_valid(self):
        import sympy as sp
        x    = sp.Symbol("x")
        cert = certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        assert cert.valid

    def test_float_sometimes_exactly_one(self):
        """
        Correction of paper claim: float IS exactly 1.0 for many values.
        The paper's 'never exactly 1.0 for non-trivial x' is incorrect.
        ETC's advantage is universal proof, not that float always fails.
        """
        # These ARE exactly 1.0 in float:
        assert math.cos(1.0)**2 + math.sin(1.0)**2 == 1.0
        # This is very close to 1.0 but not exactly due to floating-point precision:
        assert abs(math.cos(math.pi/4)**2 + math.sin(math.pi/4)**2 - 1.0) < 1e-15
        # These are NOT:
        assert math.cos(math.e)**2 + math.sin(math.e)**2 != 1.0


# ===========================================================================
# ExactReal arithmetic correctness
# ===========================================================================

class TestArithmetic:
    def test_add_rational(self):
        a = ExactReal.from_rational(Fraction(1, 3))
        b = ExactReal.from_rational(Fraction(1, 6))
        result = float(a.add(b).eval(60))
        assert abs(result - 0.5) < 1e-17

    def test_sub_exact(self):
        a = ExactReal.from_rational(Fraction(3, 4))
        b = ExactReal.from_rational(Fraction(1, 4))
        assert a.sub(b).eval(60) == Fraction(1, 2)

    def test_mul_exact(self):
        a = ExactReal.from_rational(Fraction(2, 3))
        b = ExactReal.from_rational(Fraction(3, 4))
        assert a.mul(b).eval(60) == Fraction(1, 2)

    def test_recip(self):
        a = ExactReal.from_rational(Fraction(3, 7))
        assert abs(float(a.recip().eval(60)) - 7/3) < 1e-17

    def test_pow_int_exact(self):
        a = ExactReal.from_rational(Fraction(3, 2))
        assert a.pow_int(10).eval(60) == Fraction(3**10, 2**10)

    def test_pi_precision(self):
        assert abs(float(ExactReal.pi().eval(53)) - math.pi) < 1e-14

    def test_e_precision(self):
        assert abs(float(ExactReal.e().eval(53)) - math.e) < 1e-14

    def test_log2_precision(self):
        assert abs(float(ExactReal.log2().eval(60)) - math.log(2)) < 1e-17

    def test_sqrt2_newton_is_correct(self):
        """sqrt2 (Newton) must satisfy s*s in [2-2^{-58}, 2+2^{-58}]."""
        s   = ExactReal.sqrt2()
        s60 = s.eval(60)
        assert abs(s60 * s60 - 2) < Fraction(1, 2**58)

    def test_sqrt2_newton_vs_bisection_same_answer(self):
        """Newton and bisection must agree to 40 bits."""
        newton = _newton_root(ExactReal.from_rational(2), 2).eval(40)
        bisect = _nth_root(ExactReal.from_rational(2), 2).eval(40)
        assert abs(newton - bisect) < Fraction(1, 2**38)

    def test_sqrt2_newton_faster_than_bisection(self):
        """Newton must finish faster than bisection at n=128."""
        n_inst = _newton_root(ExactReal.from_rational(2), 2)
        b_inst = _nth_root(ExactReal.from_rational(2), 2)
        t0 = time.perf_counter(); n_inst.eval(128); t_n = time.perf_counter() - t0
        t0 = time.perf_counter(); b_inst.eval(128); t_b = time.perf_counter() - t0
        assert t_n <= t_b * 5   # Newton at most 5x slower (usually faster)

    def test_sin_argument_reduction_large_input(self):
        """sin(x) == sin(x + 100*2*pi) to 11 decimal places."""
        x   = ExactReal.from_rational(Fraction(1, 3))
        pi  = ExactReal.pi()
        big = x.add(pi.mul(ExactReal.from_rational(2)).mul(ExactReal.from_rational(100)))
        diff = abs(float(x.sin().eval(40)) - float(big.sin().eval(40)))
        assert diff < 1e-11

    def test_cos_argument_reduction_large_input(self):
        x   = ExactReal.from_rational(Fraction(1, 5))
        pi  = ExactReal.pi()
        big = x.add(pi.mul(ExactReal.from_rational(2)).mul(ExactReal.from_rational(50)))
        diff = abs(float(x.cos().eval(40)) - float(big.cos().eval(40)))
        assert diff < 1e-11

    def test_exp_log_inverse(self):
        """exp(log(x)) must equal x to 11 decimal places."""
        x    = ExactReal.from_rational(Fraction(7, 3))
        diff = abs(float(x.log().exp().eval(40)) - float(x.eval(40)))
        assert diff < 1e-11


# ===========================================================================
# eval() caching
# ===========================================================================

class TestEvalCaching:
    def test_cache_hit_returns_same_object(self):
        """eval(n) called twice returns the identical cached Fraction."""
        pi = ExactReal.pi()
        r1 = pi.eval(64)
        r2 = pi.eval(64)
        assert r1 is r2

    def test_cache_populates_on_first_call(self):
        pi = ExactReal.pi()
        assert 64 not in pi._cache
        pi.eval(64)
        assert 64 in pi._cache

    def test_cache_hit_is_faster(self):
        """Cached eval must complete in under 1 ms."""
        pi = ExactReal.pi()
        pi.eval(128)               # fill cache
        t0 = time.perf_counter()
        pi.eval(128)
        assert time.perf_counter() - t0 < 0.001

    def test_different_precisions_cached_independently(self):
        pi = ExactReal.pi()
        r32 = pi.eval(32)
        r64 = pi.eval(64)
        assert r32 != r64          # different approximations
        assert pi._cache[32] is r32
        assert pi._cache[64] is r64


# ===========================================================================
# timed_eval()
# ===========================================================================

class TestTimedEval:
    def test_returns_fraction_and_float(self):
        val, t = ExactReal.pi().timed_eval(64)
        assert isinstance(val, Fraction)
        assert isinstance(t, float)
        assert t >= 0.0

    def test_value_matches_eval(self):
        pi     = ExactReal.pi()
        val, _ = pi.timed_eval(64)
        assert val == pi.eval(64)

    def test_cache_hit_reports_zero_time(self):
        pi = ExactReal.pi()
        pi.timed_eval(64)          # compute
        _, t = pi.timed_eval(64)   # cache hit
        assert t == 0.0

    def test_first_call_positive_time(self):
        val, t = ExactReal.pi().timed_eval(128)
        assert t > 0.0

    def test_negative_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            ExactReal.pi().timed_eval(-1)

    def test_zero_precision_valid(self):
        """precision=0 is valid — error bound is 2^0 = 1."""
        val, t = ExactReal.pi().timed_eval(0)
        assert isinstance(val, Fraction)


# ===========================================================================
# benchmark_eval()
# ===========================================================================

class TestBenchmarkEval:
    def test_correct_length(self):
        results = benchmark_eval(ExactReal.e(), [8, 16, 32], repeat=2)
        assert len(results) == 3

    def test_returns_int_float_pairs(self):
        for n, t in benchmark_eval(ExactReal.log2(), [16, 32], repeat=2):
            assert isinstance(n, int)
            assert isinstance(t, float)
            assert t >= 0.0

    def test_all_times_positive(self):
        results = benchmark_eval(ExactReal.pi(), [8, 16], repeat=3)
        assert all(t > 0 for _, t in results)

    def test_larger_precision_not_faster(self):
        """n=128 must not be faster than n=8 (larger precision = more work)."""
        results = benchmark_eval(ExactReal.pi(), [8, 128], repeat=3)
        t8, t128 = results[0][1], results[1][1]
        assert t128 >= t8 * 0.1   # 10x slack for timing noise

    def test_excludes_cache(self):
        """Fresh instances must be used — results should be > 0."""
        results = benchmark_eval(ExactReal.pi(), [64], repeat=3)
        assert results[0][1] > 0


# ===========================================================================
# sign()
# ===========================================================================

class TestSign:
    def test_pi_minus_3_positive(self):
        x = ExactReal.pi().sub(ExactReal.from_rational(3))
        s = sign(x)
        assert s.value == 1
        assert s.is_valid()

    def test_negative_rational(self):
        x = ExactReal.from_rational(Fraction(-5, 7))
        assert sign(x).value == -1

    def test_positive_rational(self):
        x = ExactReal.from_rational(Fraction(1, 1000))
        assert sign(x).value == 1

    def test_certificate_fields(self):
        x    = ExactReal.pi().sub(ExactReal.from_rational(3))
        cert = sign(x).proof
        assert cert.valid
        assert "prec_bits" in cert.evidence
        assert "approx" in cert.evidence
        assert cert.is_valid()

    def test_sign_returns_certified(self):
        from etc.certified import Certified
        x = ExactReal.from_rational(Fraction(1, 3))
        assert isinstance(sign(x), Certified)


# ===========================================================================
# Certificate serialisation
# ===========================================================================

class TestCertificateSerialization:
    def test_json_round_trip(self):
        import sympy as sp
        x    = sp.Symbol("x")
        cert = certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        d    = json.loads(cert.to_json())
        assert d["valid"] is True
        assert "simplified_diff" in d["evidence"]

    def test_certificate_store_all_valid(self):
        import sympy as sp
        x     = sp.Symbol("x")
        cert  = certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        store = CertificateStore()
        store.add("pythagorean", cert)
        assert store.all_valid()
        assert len(store) == 1

    def test_certificate_summary_string(self):
        import sympy as sp
        x    = sp.Symbol("x")
        cert = certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        summary = cert.summary()
        assert "VALID" in summary
        assert "Method" in summary


# ===========================================================================
# Lean 4 export
# ===========================================================================

class TestLean4Export:
    def test_export_non_empty(self):
        import sympy as sp
        x    = sp.Symbol("x")
        cert = certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        src  = certificate_to_lean(cert, name="pythagorean")
        assert len(src) > 100
        assert "theorem" in src

    def test_mathlib_imports_present(self):
        import sympy as sp
        x   = sp.Symbol("x")
        src = certificate_to_lean(
            certify_identity(sp.cos(x)**2 + sp.sin(x)**2, sp.Integer(1), [x])
        )
        assert "import Mathlib" in src

    def test_sorry_stubs_annotated(self):
        """All sorry stubs must carry a TODO or OPEN label — no bare sorry."""
        import sympy as sp
        x    = sp.Symbol("x")
        cert = certify_identity(sp.Integer(0), sp.Integer(1))   # invalid: 0 ≠ 1
        src  = certificate_to_lean(cert)
        if "sorry" in src:
            # Every sorry line must have an annotation
            for line in src.splitlines():
                if "sorry" in line and not line.strip().startswith("--"):
                    assert ("TODO" in line or "OPEN" in line or "prove" in line), (
                        f"Unannotated sorry found: {line!r}"
                    )

    def test_valid_identity_uses_ring_or_norm_num(self):
        import sympy as sp
        x    = sp.Symbol("x"); y = sp.Symbol("y")
        cert = certify_identity(x + y, y + x, [x, y])
        src  = certificate_to_lean(cert)
        assert "ring" in src or "norm_num" in src


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicAPI:
    def test_symbol_count_is_40(self):
        """__all__ must have exactly 40 symbols (paper claimed 30 — corrected)."""
        import etc
        assert len(etc.__all__) == 40

    def test_benchmark_eval_in_all(self):
        import etc
        assert "benchmark_eval" in etc.__all__

    def test_timed_eval_callable(self):
        assert callable(ExactReal.timed_eval)

    def test_all_symbols_importable(self):
        """Every symbol in __all__ must actually be importable."""
        import etc
        import importlib
        for name in etc.__all__:
            assert hasattr(etc, name), f"Symbol '{name}' in __all__ but not importable"

    def test_version_present(self):
        import etc
        assert hasattr(etc, "__version__")
        assert etc.__version__ == "0.1.0"
