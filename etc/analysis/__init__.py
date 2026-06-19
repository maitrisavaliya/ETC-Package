"""
etc.analysis
============
Exact analysis: convergent series, differentiation, integration, ODEs.
"""
from etc.analysis.series   import PowerSeries, TaylorSeries
from etc.analysis.calculus import Calculus
from etc.analysis.ode      import ODE, ODESolution

__all__ = ["PowerSeries", "TaylorSeries", "Calculus", "ODE", "ODESolution"]
