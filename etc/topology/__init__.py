"""
etc.topology
============
Constructive topology: Spaces, Points, Paths, Homotopies, Manifolds,
and topological invariants.
"""
from etc.topology.space      import Space, Point, make_Lp_circle, make_sphere, make_torus
from etc.topology.path       import Path, UnitInterval
from etc.topology.homotopy   import Homotopy
from etc.topology.invariants import winding_number

__all__ = [
    "Space", "Point", "make_Lp_circle", "make_sphere", "make_torus",
    "Path", "UnitInterval",
    "Homotopy",
    "winding_number",
]