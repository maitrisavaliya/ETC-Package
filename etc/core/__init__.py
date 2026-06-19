"""
etc.core
========
Exact numeric primitives: real, interval, complex, integer.
"""
from etc.core.real      import ExactReal
from etc.core.interval  import Interval, interval_eval
from etc.core.complex   import ExactComplex
from etc.core.integer   import ExactInteger
from etc.core.sign      import sign

__all__ = ["ExactReal", "Interval", "interval_eval", "ExactComplex", "ExactInteger", "sign"]