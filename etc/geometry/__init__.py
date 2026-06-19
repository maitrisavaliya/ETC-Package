"""
etc.geometry
============
Exact algebraic geometry: curves, surfaces, and polygon arithmetic.
"""

from etc.geometry.curves   import AlgebraicCurve, ParametricCurve
from etc.geometry.surfaces import AlgebraicSurface
from etc.geometry.polygon  import ExactPolygon

__all__ = ["AlgebraicCurve", "ParametricCurve", "AlgebraicSurface", "ExactPolygon"]
