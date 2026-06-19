"""
etc.problems
============
Problem interface and registry.
"""
from etc.problems.base                 import Problem
from etc.problems.registry             import ProblemRegistry, registry
from etc.problems.canonical_problems   import RumpPolynomial, L5WindingNumber

__all__ = ["Problem", "ProblemRegistry", "registry", "RumpPolynomial", "L5WindingNumber"]
