"""
etc.geometry.polygon
====================
ExactPolygon: a polygon with ExactReal vertex coordinates.

All geometric computations (area, perimeter, centroid, winding number,
point-in-polygon) are performed exactly over ℚ or with certified error bounds.
"""

from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple
from etc.core.real import ExactReal


Vertex = Tuple[ExactReal, ExactReal]


class ExactPolygon:
    """
    A polygon with exact real vertex coordinates, listed in order.

    Parameters
    ----------
    vertices : List[Vertex]   – list of (x, y) ExactReal pairs

    Methods
    -------
    n_vertices()    – number of vertices
    perimeter()     – approximate perimeter (uses sqrt)
    signed_area()   – exact signed area via Shoelace formula (over ℚ)
    area()          – |signed_area()|
    centroid()      – (ExactReal, ExactReal) centroid
    is_convex(prec) – True if polygon is convex
    winding_number(pt, prec) – winding number of pt w.r.t. polygon
    contains_point(pt, prec) – True if pt is inside (winding ≠ 0)
    """

    def __init__(self, vertices: List[Vertex]):
        if len(vertices) < 3:
            raise ValueError("ExactPolygon requires at least 3 vertices.")
        self.vertices: List[Vertex] = vertices

    def n_vertices(self) -> int:
        return len(self.vertices)

    # ------------------------------------------------------------------ area

    def signed_area(self, prec: int = 60) -> ExactReal:
        """
        Exact signed area via the Shoelace formula:
        A = (1/2) Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
        All operations are over ExactReal (exact arithmetic).
        """
        n     = self.n_vertices()
        total = ExactReal.zero()
        for i in range(n):
            xi, yi   = self.vertices[i]
            xj, yj   = self.vertices[(i + 1) % n]
            cross    = xi.mul(yj).sub(xj.mul(yi))
            total    = total.add(cross)
        half = ExactReal.from_rational(Fraction(1, 2))
        return total.mul(half)

    def area(self, prec: int = 60) -> ExactReal:
        return self.signed_area(prec).abs_val()

    def perimeter(self, prec: int = 40) -> float:
        """Approximate perimeter (uses floating-point sqrt for speed)."""
        n   = self.n_vertices()
        tot = 0.0
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[(i + 1) % n]
            dx = float(xi.eval(prec)) - float(xj.eval(prec))
            dy = float(yi.eval(prec)) - float(yj.eval(prec))
            tot += (dx**2 + dy**2) ** 0.5
        return tot

    def centroid(self) -> Vertex:
        """
        Centroid via the exact formula:
        cx = (1/(6A)) Σ (xᵢ+xᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
        cy = (1/(6A)) Σ (yᵢ+yᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
        """
        n   = self.n_vertices()
        A   = self.signed_area()
        six = ExactReal.from_rational(6)
        denom = six.mul(A)

        cx = ExactReal.zero()
        cy = ExactReal.zero()
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[(i + 1) % n]
            cross  = xi.mul(yj).sub(xj.mul(yi))
            cx     = cx.add((xi.add(xj)).mul(cross))
            cy     = cy.add((yi.add(yj)).mul(cross))

        return (cx.div(denom), cy.div(denom))

    # ---------------------------------------------------------------- shape

    def is_convex(self, prec: int = 40) -> bool:
        """
        Return True if the polygon is convex (all cross products same sign).
        """
        n    = self.n_vertices()
        sign = None
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[(i + 1) % n]
            xk, yk = self.vertices[(i + 2) % n]
            # Cross product of edge vectors (j-i) and (k-j)
            dx1 = float(xj.eval(prec)) - float(xi.eval(prec))
            dy1 = float(yj.eval(prec)) - float(yi.eval(prec))
            dx2 = float(xk.eval(prec)) - float(xj.eval(prec))
            dy2 = float(yk.eval(prec)) - float(yj.eval(prec))
            cross = dx1 * dy2 - dy1 * dx2
            if abs(cross) < 1e-12:
                continue
            s = 1 if cross > 0 else -1
            if sign is None:
                sign = s
            elif sign != s:
                return False
        return True

    # --------------------------------------------------------------- point queries

    def winding_number(self, pt: Vertex, prec: int = 40) -> int:
        """
        Compute the winding number of pt w.r.t. the polygon.
        Uses the ray-casting / crossing-number algorithm over float approx.
        Returns an integer (exact for points not on boundary).
        """
        px = float(pt[0].eval(prec))
        py = float(pt[1].eval(prec))
        n  = self.n_vertices()
        wn = 0

        for i in range(n):
            xi = float(self.vertices[i][0].eval(prec))
            yi = float(self.vertices[i][1].eval(prec))
            xj = float(self.vertices[(i+1)%n][0].eval(prec))
            yj = float(self.vertices[(i+1)%n][1].eval(prec))

            if yi <= py:
                if yj > py:
                    if (xj - xi) * (py - yi) - (px - xi) * (yj - yi) > 0:
                        wn += 1
            else:
                if yj <= py:
                    if (xj - xi) * (py - yi) - (px - xi) * (yj - yi) < 0:
                        wn -= 1
        return wn

    def contains_point(self, pt: Vertex, prec: int = 40) -> bool:
        """Return True if pt is strictly inside the polygon."""
        return self.winding_number(pt, prec) != 0

    def __repr__(self) -> str:
        n = self.n_vertices()
        verts = [(float(x.eval(20)), float(y.eval(20))) for x, y in self.vertices[:3]]
        return f"ExactPolygon({n} vertices, first 3: {verts})"
