"""
tests/test_problems.py
======================
Tests for the etc.problems module.
Covers: Problem base class, ProblemRegistry, RumpPolynomial, L5WindingNumber.
"""
from __future__ import annotations
import pytest
from fractions import Fraction

from etc.problems.base               import Problem
from etc.problems.registry           import ProblemRegistry, registry
from etc.problems.canonical_problems import RumpPolynomial, L5WindingNumber
from etc.certified                   import Certified


class TestProblemBase:
    def test_rump_is_problem_instance(self):
        assert isinstance(RumpPolynomial(), Problem)
    def test_l5_is_problem_instance(self):
        assert isinstance(L5WindingNumber(), Problem)
    def test_problem_has_name(self):
        p = RumpPolynomial()
        assert isinstance(p.name, str) and len(p.name) > 0
    def test_problem_has_description(self):
        p = RumpPolynomial()
        assert isinstance(p.description, str) and len(p.description) > 0
    def test_problem_repr(self):
        assert "Problem" in repr(RumpPolynomial())


class TestProblemRegistry:
    def test_register_and_get(self):
        reg = ProblemRegistry()
        p = RumpPolynomial()
        reg.register(p)
        assert reg.get(p.name) is p
    def test_get_unknown_returns_none(self):
        assert ProblemRegistry().get("no_such_problem") is None
    def test_len(self):
        reg = ProblemRegistry()
        assert len(reg) == 0
        reg.register(RumpPolynomial())
        assert len(reg) == 1
    def test_list_returns_names(self):
        reg = ProblemRegistry()
        reg.register(RumpPolynomial())
        reg.register(L5WindingNumber())
        assert len(reg.list()) == 2
    def test_global_registry_is_registry_instance(self):
        assert isinstance(registry, ProblemRegistry)


class TestRumpPolynomial:
    def test_solve_returns_certified(self):
        assert isinstance(RumpPolynomial().solve(), Certified)
    def test_solve_exact_value(self):
        result = RumpPolynomial().solve()
        assert result.value == Fraction(-54767, 66192)
    def test_solve_certificate_valid(self):
        assert RumpPolynomial().solve().is_valid()
    def test_verify_exact_answer(self):
        assert RumpPolynomial().verify(Fraction(-54767, 66192)) is True
    def test_verify_wrong_answer(self):
        assert RumpPolynomial().verify(Fraction(0)) is False
    def test_float_gives_wrong_answer(self):
        a, b = 77617.0, 33096.0
        fp = (333.75*b**6 + a**2*(11*a**2*b**2 - b**6 - 121*b**4 - 2)
              + 5.5*b**8 + a/(2*b))
        assert abs(fp - float(Fraction(-54767, 66192))) > 1e10


class TestL5WindingNumber:
    def test_solve_returns_certified(self):
        assert isinstance(L5WindingNumber().solve(), Certified)
    def test_solve_winding_is_one(self):
        assert L5WindingNumber().solve().value == 1
    def test_solve_certificate_valid(self):
        assert L5WindingNumber().solve().is_valid()
    def test_verify_correct_answer(self):
        assert L5WindingNumber().verify(1) is True
    def test_verify_wrong_answer(self):
        assert L5WindingNumber().verify(0) is False
