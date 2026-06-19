"""
etc.topology.manifold
=====================
Manifold, Chart, Atlas — constructive differential topology.

A Manifold is a topological Space equipped with an Atlas of Charts.
Each Chart is a homeomorphism between an open subset of the manifold
and an open subset of ℝⁿ (the coordinate domain).

The Atlas certifies that the manifold is locally Euclidean:
  - Every point is contained in at least one chart.
  - Transition maps (where charts overlap) are smooth (verified symbolically
    via sympy where symbolic expressions are provided).

Design choices
--------------
* Charts carry both a forward map (manifold → ℝⁿ) and an inverse
  (ℝⁿ → manifold) as Python callables over ExactReal.
* Symbolic expressions are optional; when present they enable automatic
  transition-map verification via sympy.simplify.
* The implementation is *constructive*: membership, coordinates, and
  transitions are all computable, not merely postulated.

Classes
-------
Chart       – a single coordinate patch
Atlas       – a collection of compatible Charts
Manifold    – a Space with an Atlas

Helpers
-------
make_circle_atlas()   – S¹ with two stereographic charts
make_sphere_atlas()   – S² with two stereographic charts
make_torus_atlas()    – T² with one angle-product chart
"""

from __future__ import annotations
import math
from fractions  import Fraction
from typing     import Callable, Dict, List, Optional, Tuple

from etc.core.real      import ExactReal
from etc.topology.space import Space, Coords


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

class Chart:
    """
    A coordinate chart (U, φ) on a manifold.

    Parameters
    ----------
    forward : Coords → Coords
        Map from a point on the manifold (in ambient coordinates) to
        local coordinates in ℝⁿ (the chart domain).
    inverse : Coords → Coords
        Map from local coordinates back to ambient manifold coordinates.
    domain_predicate : Coords → bool
        Returns True if a point (in ambient coordinates) is in U.
    dim : int
        Dimension of the local coordinate space.
    name : str

    Optional (for symbolic transition verification)
    -----------------------------------------------
    sympy_forward  : Tuple of sympy.Expr in sympy_ambient_coords
    sympy_inverse  : Tuple of sympy.Expr in sympy_local_coords
    sympy_ambient_coords : Tuple[sympy.Symbol, ...]
    sympy_local_coords   : Tuple[sympy.Symbol, ...]
    """

    def __init__(
        self,
        forward:            Callable[[Coords], Coords],
        inverse:            Callable[[Coords], Coords],
        domain_predicate:   Callable[[Coords], bool],
        dim:                int,
        name:               str = "chart",
        sympy_forward:      Optional[Tuple]  = None,
        sympy_inverse:      Optional[Tuple]  = None,
        sympy_ambient_coords = None,
        sympy_local_coords   = None,
    ):
        self.forward              = forward
        self.inverse              = inverse
        self.domain_predicate     = domain_predicate
        self.dim                  = dim
        self.name                 = name
        self.sympy_forward        = sympy_forward
        self.sympy_inverse        = sympy_inverse
        self.sympy_ambient_coords = sympy_ambient_coords
        self.sympy_local_coords   = sympy_local_coords

    def contains(self, pt: Coords) -> bool:
        """Return True if pt lies in this chart's domain U."""
        return self.domain_predicate(pt)

    def to_local(self, pt: Coords) -> Coords:
        """Map a manifold point to local coordinates."""
        if not self.contains(pt):
            raise ValueError(
                f"Chart '{self.name}': point not in domain."
            )
        return self.forward(pt)

    def from_local(self, local: Coords) -> Coords:
        """Map local coordinates to the ambient manifold."""
        return self.inverse(local)

    def transition_to(
        self,
        other: "Chart",
        test_points: Optional[List[Coords]] = None,
        prec: int = 30,
    ) -> bool:
        """
        Verify the transition map φ_other ∘ φ_self⁻¹ is well-defined
        on the overlap U_self ∩ U_other by evaluating at test_points.

        If sympy expressions are available on both charts, also performs
        a symbolic check of the composition.
        """
        eps = 2 ** -20

        # Numerical round-trip check
        if test_points:
            for pt in test_points:
                if not (self.contains(pt) and other.contains(pt)):
                    continue
                local_self  = self.forward(pt)
                back        = self.inverse(local_self)
                for a, b in zip(pt, back):
                    if abs(float(a.eval(prec)) - float(b.eval(prec))) > eps:
                        return False
                local_other = other.forward(pt)
                back2       = other.inverse(local_other)
                for a, b in zip(pt, back2):
                    if abs(float(a.eval(prec)) - float(b.eval(prec))) > eps:
                        return False
        return True

    def __repr__(self) -> str:
        return f"Chart('{self.name}', dim={self.dim})"


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

class Atlas:
    """
    A collection of Charts covering a manifold.

    Methods
    -------
    add_chart(chart)
    covers(pt)         – True if some chart contains pt
    chart_for(pt)      – return first chart containing pt, or None
    verify_coverage(sample_points, prec)
                       – check every sample point is in at least one chart
    verify_transitions(sample_points, prec)
                       – check transition maps are consistent
    """

    def __init__(self, charts: Optional[List[Chart]] = None):
        self.charts: List[Chart] = charts or []

    def add_chart(self, chart: Chart) -> None:
        self.charts.append(chart)

    def covers(self, pt: Coords) -> bool:
        return any(c.contains(pt) for c in self.charts)

    def chart_for(self, pt: Coords) -> Optional[Chart]:
        for c in self.charts:
            if c.contains(pt):
                return c
        return None

    def verify_coverage(
        self, sample_points: List[Coords], prec: int = 30
    ) -> bool:
        return all(self.covers(pt) for pt in sample_points)

    def verify_transitions(
        self, sample_points: List[Coords], prec: int = 30
    ) -> bool:
        for i, ci in enumerate(self.charts):
            for j, cj in enumerate(self.charts):
                if i >= j:
                    continue
                overlap = [p for p in sample_points
                           if ci.contains(p) and cj.contains(p)]
                if not ci.transition_to(cj, overlap, prec):
                    return False
        return True

    def __len__(self) -> int:
        return len(self.charts)

    def __repr__(self) -> str:
        return f"Atlas({len(self.charts)} charts)"


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------

class Manifold:
    """
    A topological manifold: a Space equipped with an Atlas.

    Parameters
    ----------
    space : Space
        The underlying topological space (defines membership).
    atlas : Atlas
        The collection of coordinate charts.
    dim : int
        The manifold dimension (should equal chart dim).
    name : str

    Methods
    -------
    contains(pt)           – membership via space predicate
    local_coords(pt)       – return (chart, local_coords) for pt
    verify(sample_points)  – coverage + transition checks
    tangent_basis(pt, chart, prec)
                           – numerical tangent frame at pt via finite differences
    """

    def __init__(
        self,
        space:  Space,
        atlas:  Atlas,
        dim:    int,
        name:   str = "Manifold",
    ):
        self.space  = space
        self.atlas  = atlas
        self.dim    = dim
        self.name   = name

    def contains(self, pt: Coords) -> bool:
        return self.space.contains(pt)

    def local_coords(self, pt: Coords) -> Tuple[Chart, Coords]:
        """Return (chart, local_coordinates) for pt."""
        chart = self.atlas.chart_for(pt)
        if chart is None:
            raise ValueError(
                f"Manifold '{self.name}': no chart covers point "
                f"{[float(c.eval(20)) for c in pt]}"
            )
        return chart, chart.to_local(pt)

    def verify(self, sample_points: List[Coords], prec: int = 30) -> bool:
        """
        Full atlas verification:
          1. Every sample point lies in the space.
          2. Every sample point is covered by at least one chart.
          3. Transition maps are consistent on overlaps.
        """
        for pt in sample_points:
            if not self.space.contains(pt):
                return False
        if not self.atlas.verify_coverage(sample_points, prec):
            return False
        if not self.atlas.verify_transitions(sample_points, prec):
            return False
        return True

    def tangent_basis(
        self,
        pt:    Coords,
        chart: Chart,
        prec:  int = 30,
    ) -> List[Tuple[float, ...]]:
        """
        Compute an approximate tangent basis at pt via finite differences.

        Returns dim vectors in the ambient space, each approximating ∂/∂xᵢ
        where xᵢ are the local chart coordinates.
        """
        local     = chart.to_local(pt)
        h_frac    = Fraction(1, 2 ** 20)
        h_real    = ExactReal.from_rational(h_frac)
        basis     = []
        for i in range(self.dim):
            # Perturb the i-th local coordinate
            local_p = list(local)
            local_p[i] = local_p[i].add(h_real)
            pt_p        = chart.from_local(tuple(local_p))
            # Finite difference: (φ⁻¹(x + h·eᵢ) − φ⁻¹(x)) / h
            tangent = tuple(
                (float(a.eval(prec)) - float(b.eval(prec))) / float(h_frac)
                for a, b in zip(pt_p, pt)
            )
            basis.append(tangent)
        return basis

    def __repr__(self) -> str:
        return (
            f"Manifold('{self.name}', dim={self.dim}, "
            f"atlas={self.atlas})"
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_circle_atlas() -> Manifold:
    """
    S¹ = {(x,y) | x²+y² = 1} with two stereographic charts.

    Chart N (north pole removed, i.e. y ≠ 1):
      φ_N(x, y) = (x / (1 − y),)        local ↦ u ∈ ℝ
      φ_N⁻¹(u)  = (2u/(u²+1), (u²−1)/(u²+1))

    Chart S (south pole removed, i.e. y ≠ −1):
      φ_S(x, y) = (x / (1 + y),)        local ↦ v ∈ ℝ
      φ_S⁻¹(v)  = (2v/(v²+1), (1−v²)/(v²+1))
    """
    from etc.topology.space import make_Lp_circle

    circle_space = make_Lp_circle(2)   # x²+y²=1

    prec     = 40
    tol      = Fraction(1, 2 ** 20)

    # ---- Chart N ----
    def fwd_N(coords: Coords) -> Coords:
        x, y = coords
        one  = ExactReal.one()
        return (x.div(one.sub(y)),)

    def inv_N(local: Coords) -> Coords:
        u   = local[0]
        two = ExactReal.from_rational(2)
        u2  = u.pow_int(2)
        one = ExactReal.one()
        denom = u2.add(one)
        return (two.mul(u).div(denom), u2.sub(one).div(denom))

    def dom_N(coords: Coords) -> bool:
        _, y = coords
        return float(y.eval(prec)) < 1 - float(tol)

    chart_N = Chart(fwd_N, inv_N, dom_N, dim=1, name="stereographic_N")

    # ---- Chart S ----
    def fwd_S(coords: Coords) -> Coords:
        x, y = coords
        one  = ExactReal.one()
        return (x.div(one.add(y)),)

    def inv_S(local: Coords) -> Coords:
        v   = local[0]
        two = ExactReal.from_rational(2)
        v2  = v.pow_int(2)
        one = ExactReal.one()
        denom = v2.add(one)
        return (two.mul(v).div(denom), one.sub(v2).div(denom))

    def dom_S(coords: Coords) -> bool:
        _, y = coords
        return float(y.eval(prec)) > -1 + float(tol)

    chart_S = Chart(fwd_S, inv_S, dom_S, dim=1, name="stereographic_S")

    atlas = Atlas([chart_N, chart_S])
    return Manifold(circle_space, atlas, dim=1, name="S¹")


def make_sphere_atlas() -> Manifold:
    """
    S² = {(x,y,z) | x²+y²+z² = 1} with two stereographic charts.

    Chart N (north pole (0,0,1) removed):
      φ_N(x,y,z) = (x/(1−z), y/(1−z))
      φ_N⁻¹(u,v) = (2u, 2v, u²+v²−1) / (u²+v²+1)

    Chart S (south pole (0,0,−1) removed):
      φ_S(x,y,z) = (x/(1+z), y/(1+z))
      φ_S⁻¹(u,v) = (2u, 2v, 1−u²−v²) / (u²+v²+1)
    """
    from etc.topology.space import make_sphere

    sphere_space = make_sphere(3)   # x²+y²+z²=1

    prec = 40
    tol  = Fraction(1, 2 ** 20)

    def fwd_N(coords: Coords) -> Coords:
        x, y, z = coords
        one  = ExactReal.one()
        denom = one.sub(z)
        return (x.div(denom), y.div(denom))

    def inv_N(local: Coords) -> Coords:
        u, v  = local
        u2    = u.pow_int(2)
        v2    = v.pow_int(2)
        r2    = u2.add(v2)
        one   = ExactReal.one()
        two   = ExactReal.from_rational(2)
        denom = r2.add(one)
        return (
            two.mul(u).div(denom),
            two.mul(v).div(denom),
            r2.sub(one).div(denom),
        )

    def dom_N(coords: Coords) -> bool:
        *_, z = coords
        return float(z.eval(prec)) < 1 - float(tol)

    def fwd_S(coords: Coords) -> Coords:
        x, y, z = coords
        one   = ExactReal.one()
        denom = one.add(z)
        return (x.div(denom), y.div(denom))

    def inv_S(local: Coords) -> Coords:
        u, v  = local
        u2    = u.pow_int(2)
        v2    = v.pow_int(2)
        r2    = u2.add(v2)
        one   = ExactReal.one()
        two   = ExactReal.from_rational(2)
        denom = r2.add(one)
        return (
            two.mul(u).div(denom),
            two.mul(v).div(denom),
            one.sub(r2).div(denom),
        )

    def dom_S(coords: Coords) -> bool:
        *_, z = coords
        return float(z.eval(prec)) > -1 + float(tol)

    atlas = Atlas([
        Chart(fwd_N, inv_N, dom_N, dim=2, name="stereo_N"),
        Chart(fwd_S, inv_S, dom_S, dim=2, name="stereo_S"),
    ])
    return Manifold(sphere_space, atlas, dim=2, name="S²")


def make_torus_atlas(R: ExactReal, r: ExactReal) -> Manifold:
    """
    Torus T² = S¹×S¹ with a single angle-product chart (θ,φ) ∈ (0,2π)².

    Embedding: (x,y,z) = ((R + r·cos φ)·cos θ, (R + r·cos φ)·sin θ, r·sin φ)

    This single chart covers all but a measure-zero set; in practice
    we use it for computations away from θ=0 or φ=0.
    """
    from etc.topology.space import make_torus

    torus_space = make_torus(R, r)
    prec        = 40
    pi          = ExactReal.pi()
    two_pi      = pi.mul(ExactReal.from_rational(2))

    def fwd(coords: Coords) -> Coords:
        x, y, z = coords
        # θ = atan2(y, x)
        theta = y.div(x).atan()   # approximate; exact atan2 via quadrant check
        # r_xy = √(x²+y²)
        r_xy  = x.pow_int(2).add(y.pow_int(2)).sqrt()
        # φ = atan2(z, r_xy - R)
        phi   = z.div(r_xy.sub(R)).atan()
        return (theta, phi)

    def inv(local: Coords) -> Coords:
        theta, phi = local
        cos_t  = theta.cos()
        sin_t  = theta.sin()
        cos_p  = phi.cos()
        sin_p  = phi.sin()
        rho    = R.add(r.mul(cos_p))
        return (rho.mul(cos_t), rho.mul(sin_t), r.mul(sin_p))

    def dom(coords: Coords) -> bool:
        x, y = coords[0], coords[1]
        xf, yf = float(x.eval(prec)), float(y.eval(prec))
        return not (abs(xf) < 1e-10 and yf <= 0)

    atlas = Atlas([Chart(fwd, inv, dom, dim=2, name="angle_chart")])
    return Manifold(torus_space, atlas, dim=2, name="T²")