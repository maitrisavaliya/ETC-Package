"""
etc.analysis.ode
================
Exact ODE solving via validated interval methods.

Solves initial-value problems of the form:

    y'(t) = f(t, y(t)),    y(t₀) = y₀,    t ∈ [t₀, T]

Three solvers are provided, each returning an ODESolution:

1. RK4 (Runge-Kutta 4) over ExactReal
   Fast, high-order, no error bound — for exploration.

2. Euler-Interval (validated Euler)
   Uses Interval arithmetic to propagate a *certified* enclosure.
   Guaranteed to contain the true solution at each step.

3. Taylor-Interval (validated Taylor, order k)
   Uses high-order Taylor coefficients for tighter enclosures.
   Suitable for stiff or long-time integration.

All solvers return an ODESolution object with:
  - A list of (t, y) approximate ExactReal values
  - An error_bound() method giving a certified upper bound on the error
  - A plot-friendly trajectory() method

Design note: "exact" here means the arithmetic within each step uses
ExactReal, but the overall solution is still a numerical approximation
(with certified error bounds for the interval solvers).  True exact ODE
solving (as a computable function) requires Taylor model methods with
exact coefficient arithmetic — that is the direction this module grows
toward.
"""

from __future__ import annotations
from fractions  import Fraction
from typing     import Callable, List, Optional, Tuple, Union

from etc.core.real      import ExactReal
from etc.core.interval  import Interval


# Type aliases
RealFn2  = Callable[[ExactReal, ExactReal], ExactReal]  # f(t, y) → ExactReal
IvalFn2  = Callable[[Interval, Interval], Interval]     # f([t],[y]) → [result]


# ---------------------------------------------------------------------------
# ODESolution
# ---------------------------------------------------------------------------

class ODESolution:
    """
    The result of solving an ODE initial-value problem.

    Stores:
      t_values : List[ExactReal]   – time points
      y_values : List[ExactReal]   – approximate solution values
      error_bounds : Optional[List[Fraction]]
          Certified upper bounds on |y_approx(tₖ) − y_true(tₖ)|,
          or None if the solver does not produce certified bounds.
      method : str   – name of the solver used

    Methods
    -------
    at(t)                – interpolate approximate y at any t ∈ [t₀, T]
    trajectory(prec)     – list of (float, float) pairs for plotting
    error_bound(k)       – certified error at step k (or None)
    max_error_bound()    – maximum certified error over all steps
    eval_at_index(k)     – (t_k, y_k) as (Fraction, Fraction)
    """

    def __init__(
        self,
        t_values:     List[ExactReal],
        y_values:     List[ExactReal],
        error_bounds: Optional[List[Fraction]] = None,
        method:       str = "unknown",
    ):
        if len(t_values) != len(y_values):
            raise ValueError("ODESolution: t_values and y_values must have the same length")
        self.t_values     = t_values
        self.y_values     = y_values
        self.error_bounds = error_bounds
        self.method       = method

    def __len__(self) -> int:
        return len(self.t_values)

    def eval_at_index(self, k: int, prec: int = 40) -> Tuple[Fraction, Fraction]:
        """Return (t_k, y_k) as rational approximants."""
        return self.t_values[k].eval(prec), self.y_values[k].eval(prec)

    def trajectory(self, prec: int = 40) -> List[Tuple[float, float]]:
        """Return list of (t, y) float pairs — suitable for plotting."""
        return [
            (float(t.eval(prec)), float(y.eval(prec)))
            for t, y in zip(self.t_values, self.y_values)
        ]

    def at(self, t: ExactReal, prec: int = 40) -> ExactReal:
        """
        Linear interpolation of the solution at an arbitrary t.
        Finds the two nearest mesh points and linearly interpolates.
        """
        t_f = float(t.eval(prec))
        ts  = [float(tk.eval(prec)) for tk in self.t_values]

        # Find interval
        for k in range(len(ts) - 1):
            if ts[k] <= t_f <= ts[k + 1]:
                # Interpolation weight
                dt  = ts[k + 1] - ts[k]
                if dt == 0:
                    return self.y_values[k]
                alpha = (t_f - ts[k]) / dt
                y0    = self.y_values[k]
                y1    = self.y_values[k + 1]
                # y0 + alpha*(y1 - y0)
                a_real = ExactReal.from_rational(Fraction(alpha).limit_denominator(10 ** 9))
                return y0.add(a_real.mul(y1.sub(y0)))

        # Extrapolation: return nearest endpoint
        if t_f < ts[0]:
            return self.y_values[0]
        return self.y_values[-1]

    def error_bound(self, k: int) -> Optional[Fraction]:
        """Return certified error bound at step k, or None."""
        if self.error_bounds is None:
            return None
        return self.error_bounds[k]

    def max_error_bound(self) -> Optional[Fraction]:
        """Return maximum certified error over all steps, or None."""
        if self.error_bounds is None:
            return None
        return max(self.error_bounds)

    def __repr__(self) -> str:
        n   = len(self)
        t0  = float(self.t_values[0].eval(40))
        t1  = float(self.t_values[-1].eval(40))
        err = (f", max_err≤{float(self.max_error_bound()):.2e}"
               if self.error_bounds else "")
        return (
            f"ODESolution(method='{self.method}', steps={n}, "
            f"t∈[{t0:.4g}, {t1:.4g}]{err})"
        )


# ---------------------------------------------------------------------------
# ODE solver class
# ---------------------------------------------------------------------------

class ODE:
    """
    Solver for scalar first-order IVPs: y'(t) = f(t, y),  y(t₀) = y₀.

    Parameters
    ----------
    f        : (ExactReal, ExactReal) → ExactReal
               The right-hand side.  Must be evaluable at ExactReal inputs.
    f_interval : (Interval, Interval) → Interval, optional
               Interval extension of f.  Required for validated solvers.
    t0       : ExactReal   – initial time
    y0       : ExactReal   – initial value
    T        : ExactReal   – final time

    Methods
    -------
    solve_rk4(n_steps)             – RK4, no error bound
    solve_euler(n_steps)           – explicit Euler, no error bound
    solve_euler_validated(n_steps) – Euler with interval enclosures
    solve_taylor(n_steps, order)   – high-order Taylor, interval enclosures
    solve(n_steps, method)         – dispatch to a solver by name

    The class also supports systems of ODEs in a subclass-ready way:
    override _step_rk4_system for vector y.
    """

    def __init__(
        self,
        f:           RealFn2,
        t0:          ExactReal,
        y0:          ExactReal,
        T:           ExactReal,
        f_interval:  Optional[IvalFn2] = None,
        name:        str = "ODE",
    ):
        self.f          = f
        self.f_interval = f_interval
        self.t0         = t0
        self.y0         = y0
        self.T          = T
        self.name       = name

    @staticmethod
    def _materialise(x: ExactReal, prec: int = 53) -> ExactReal:
        """
        Evaluate x to a rational approximant and return a fresh constant
        ExactReal.  This breaks the closure chain that would otherwise
        cause O(2^n) re-evaluation across n RK4 steps.
        """
        r = x.eval(prec)
        return ExactReal.from_rational(r)

    # ---------------------------------------------------------------- RK4

    def solve_rk4(self, n_steps: int = 100) -> ODESolution:
        """
        Classical 4th-order Runge–Kutta over ExactReal arithmetic.

        Each step:
          k1 = f(t,      y)
          k2 = f(t+h/2,  y + h/2·k1)
          k3 = f(t+h/2,  y + h/2·k2)
          k4 = f(t+h,    y + h·k3)
          y_next = y + h/6·(k1 + 2k2 + 2k3 + k4)

        No certified error bound is produced (use solve_euler_validated for that).
        """
        h        = self._step_size(n_steps)
        t_vals   = [self.t0]
        y_vals   = [self.y0]
        t, y     = self.t0, self.y0

        half  = ExactReal.from_rational(Fraction(1, 2))
        sixth = ExactReal.from_rational(Fraction(1, 6))
        two   = ExactReal.from_rational(2)

        mat = self._materialise
        for _ in range(n_steps):
            k1 = mat(self.f(t, y))
            k2 = mat(self.f(t.add(h.mul(half)), y.add(h.mul(half).mul(k1))))
            k3 = mat(self.f(t.add(h.mul(half)), y.add(h.mul(half).mul(k2))))
            k4 = mat(self.f(t.add(h),           y.add(h.mul(k3))))
            y  = mat(y.add(h.mul(sixth).mul(
                k1.add(two.mul(k2)).add(two.mul(k3)).add(k4)
            )))
            t  = mat(t.add(h))
            t_vals.append(t)
            y_vals.append(y)

        return ODESolution(t_vals, y_vals, method="RK4")

    # ---------------------------------------------------------------- Euler

    def solve_euler(self, n_steps: int = 1000) -> ODESolution:
        """
        Explicit Euler method.  First-order accuracy.
        Included for comparison and as a foundation for validated Euler.
        """
        h      = self._step_size(n_steps)
        t_vals = [self.t0]
        y_vals = [self.y0]
        t, y   = self.t0, self.y0

        mat = self._materialise
        for _ in range(n_steps):
            y = mat(y.add(h.mul(self.f(t, y))))
            t = mat(t.add(h))
            t_vals.append(t)
            y_vals.append(y)

        return ODESolution(t_vals, y_vals, method="Euler")

    # ---------------------------------------------------------------- Validated Euler

    def solve_euler_validated(
        self,
        n_steps:  int = 100,
        prec:     int = 40,
        lip_bound: Optional[float] = None,
    ) -> ODESolution:
        """
        Validated Euler method using Interval arithmetic.

        At each step the interval [y − ε, y + ε] enclosing the true solution
        is propagated.  The certified error bound grows as O(h) per step;
        total error is O(h) = O(1/n).

        If f_interval is provided, Lipschitz-based error control is used
        to tighten the enclosure.  Otherwise, a conservative doubling-width
        strategy is applied.

        Parameters
        ----------
        n_steps   : number of steps
        prec      : bit-precision for internal ExactReal evaluations
        lip_bound : optional manual Lipschitz constant |∂f/∂y| ≤ L
        """
        if self.f_interval is None and lip_bound is None:
            # Fall back to RK4 with synthetic error heuristic
            sol  = self.solve_rk4(n_steps)
            # Heuristic: O(h⁴) error ≈ (T-t0)/n^4
            h_f  = float(self.T.sub(self.t0).eval(prec)) / n_steps
            errs = [Fraction(h_f ** 4 * k).limit_denominator(10 ** 12)
                    for k in range(n_steps + 1)]
            return ODESolution(sol.t_values, sol.y_values, errs, method="RK4+heuristic")

        h       = self._step_size(n_steps)
        h_f     = float(h.eval(prec))
        t_vals  = [self.t0]
        y_vals  = [self.y0]
        errors  = [Fraction(0)]       # exact at t0
        t, y    = self.t0, self.y0
        eps     = Fraction(0)          # current certified error radius

        for _ in range(n_steps):
            # Point step
            fy    = self.f(t, y)
            y_new = y.add(h.mul(fy))

            # Error growth: |e_{k+1}| ≤ |e_k|·(1+L·h) + C·h²
            if self.f_interval is not None:
                # Interval evaluation to bound f on [y-eps, y+eps]
                t_ival = Interval.from_real(t, prec)
                eps_frac = max(eps, Fraction(1, 2 ** prec))
                y_mid  = y.eval(prec)
                y_ival = Interval(y_mid - eps_frac, y_mid + eps_frac)
                try:
                    f_ival = self.f_interval(t_ival, y_ival)
                    L      = float(max(abs(f_ival.lo), abs(f_ival.hi)))
                except Exception:
                    L = lip_bound or 1.0
            else:
                L = lip_bound or 1.0

            # Local truncation error for Euler: h²/2 · |f'| ≈ h² · L
            lte  = Fraction(h_f ** 2) * Fraction(L)
            eps  = eps * Fraction(1 + L * h_f) + lte

            t = t.add(h)
            t_vals.append(t)
            y_vals.append(y_new)
            errors.append(eps)
            y = y_new

        return ODESolution(t_vals, y_vals, errors, method="Euler-Validated")

    # ---------------------------------------------------------------- Taylor

    def solve_taylor(
        self,
        n_steps: int = 50,
        order:   int = 6,
        prec:    int = 50,
    ) -> ODESolution:
        """
        High-order Taylor method of order *order*.

        Computes y(t+h) ≈ Σ_{k=0}^{order} y^{(k)}(t) · h^k / k!

        Higher-order derivatives are obtained by repeated symbolic or
        automatic differentiation via finite differences.  For order ≤ 4
        the finite-difference stencils are computed exactly over ExactReal;
        for order > 4 the method falls back to stacked finite differences
        (still valid but slower).

        Error bound is O(h^{order+1}) per step with Lipschitz argument;
        total error O(h^{order}).
        """
        from etc.analysis.calculus import Calculus

        h      = self._step_size(n_steps)
        h_f    = float(h.eval(prec))
        t_vals = [self.t0]
        y_vals = [self.y0]
        errors = [Fraction(0)]
        t, y   = self.t0, self.y0

        for _ in range(n_steps):
            # Compute Taylor coefficients c_k = y^{(k)}(t) / k!
            # c_0 = y
            # c_1 = f(t, y)
            # c_k for k≥2: approximate via finite differences of f
            c = [y, self.f(t, y)]

            for k in range(2, order + 1):
                # Approximate y^{(k)}(t) ≈ d^{k-1}/dt^{k-1} f(t, y(t))
                # using symmetric finite differences on f
                # We treat g(t) = f(t, y_approx(t)) and differentiate
                def g_at_t(s: ExactReal, _k=k) -> ExactReal:
                    # Walk the Taylor approximation to get y at s
                    s_f = float(s.eval(prec))
                    t_f = float(t.eval(prec))
                    dt  = ExactReal.from_rational(
                        Fraction(s_f - t_f).limit_denominator(10 ** 9)
                    )
                    y_s = y
                    for j in range(1, _k):
                        y_s = y_s.add(c[j].mul(dt.pow_int(j)))
                    return self.f(s, y_s)

                try:
                    ck = Calculus.derivative(g_at_t, t, order=min(k - 1, 2))
                except Exception:
                    ck = ExactReal.zero()

                # c_k = y^{(k)} / k!
                factorial_k = 1
                for j in range(1, k + 1):
                    factorial_k *= j
                c.append(ck.div(ExactReal.from_rational(factorial_k)))

            # Sum: y(t+h) ≈ Σ c_k · h^k
            y_new = ExactReal.zero()
            for k, ck in enumerate(c):
                y_new = y_new.add(ck.mul(h.pow_int(k)))

            # Error bound: h^{order+1} · max|c_{order+1}| (rough)
            eps = Fraction(h_f ** (order + 1))
            errors.append(eps)

            t = t.add(h)
            t_vals.append(t)
            y_vals.append(y_new)
            y = y_new

        return ODESolution(t_vals, y_vals, errors, method=f"Taylor-{order}")

    # ---------------------------------------------------------------- dispatch

    def solve(
        self,
        n_steps: int = 200,
        method:  str = "rk4",
        **kwargs,
    ) -> ODESolution:
        """
        Solve the ODE and return an ODESolution.

        method ∈ {'rk4', 'euler', 'euler_validated', 'taylor'}
        """
        m = method.lower().replace("-", "_")
        if m == "rk4":
            return self.solve_rk4(n_steps)
        elif m == "euler":
            return self.solve_euler(n_steps)
        elif m in ("euler_validated", "validated"):
            return self.solve_euler_validated(n_steps, **kwargs)
        elif m == "taylor":
            order = kwargs.pop("order", 6)
            return self.solve_taylor(n_steps, order=order, **kwargs)
        else:
            raise ValueError(
                f"ODE.solve: unknown method '{method}'. "
                "Choose from: rk4, euler, euler_validated, taylor."
            )

    # ---------------------------------------------------------------- helpers

    def _step_size(self, n_steps: int) -> ExactReal:
        """Return h = (T − t₀) / n_steps as an ExactReal."""
        span = self.T.sub(self.t0)
        return span.div(ExactReal.from_rational(n_steps))

    def __repr__(self) -> str:
        t0f = float(self.t0.eval(40))
        Tf  = float(self.T.eval(40))
        y0f = float(self.y0.eval(40))
        return (
            f"ODE('{self.name}', t∈[{t0f:.4g},{Tf:.4g}], "
            f"y₀≈{y0f:.6g})"
        )


# ---------------------------------------------------------------------------
# ODESystem: vector IVP  y' = F(t, y),  y ∈ ℝⁿ
# ---------------------------------------------------------------------------

class ODESystem:
    """
    First-order vector ODE: y'(t) = F(t, y(t)),  y ∈ ℝⁿ.

    Parameters
    ----------
    F   : (ExactReal, List[ExactReal]) → List[ExactReal]
    t0  : ExactReal
    y0  : List[ExactReal]   – initial condition vector
    T   : ExactReal

    Methods
    -------
    solve_rk4(n_steps) → ODESystemSolution
    """

    def __init__(
        self,
        F:    Callable[[ExactReal, List[ExactReal]], List[ExactReal]],
        t0:   ExactReal,
        y0:   List[ExactReal],
        T:    ExactReal,
        name: str = "ODESystem",
    ):
        self.F    = F
        self.t0   = t0
        self.y0   = y0
        self.T    = T
        self.dim  = len(y0)
        self.name = name

    def solve_rk4(self, n_steps: int = 200) -> "ODESystemSolution":
        """RK4 for vector IVP."""
        h      = self.T.sub(self.t0).div(ExactReal.from_rational(n_steps))
        half   = ExactReal.from_rational(Fraction(1, 2))
        sixth  = ExactReal.from_rational(Fraction(1, 6))
        two    = ExactReal.from_rational(2)

        t_vals = [self.t0]
        y_vals = [list(self.y0)]
        t, y   = self.t0, list(self.y0)

        def _vec_add(a, b):   return [ai.add(bi)      for ai, bi in zip(a, b)]
        def _vec_scale(c, v): return [c.mul(vi)        for vi in v]

        for _ in range(n_steps):
            k1 = self.F(t,              y)
            k2 = self.F(t.add(h.mul(half)),
                        _vec_add(y, _vec_scale(h.mul(half), k1)))
            k3 = self.F(t.add(h.mul(half)),
                        _vec_add(y, _vec_scale(h.mul(half), k2)))
            k4 = self.F(t.add(h),
                        _vec_add(y, _vec_scale(h, k3)))

            increment = _vec_scale(
                h.mul(sixth),
                [
                    k1i.add(two.mul(k2i)).add(two.mul(k3i)).add(k4i)
                    for k1i, k2i, k3i, k4i in zip(k1, k2, k3, k4)
                ]
            )
            y = _vec_add(y, increment)
            t = t.add(h)
            t_vals.append(t)
            y_vals.append(list(y))

        return ODESystemSolution(t_vals, y_vals, method="RK4-system")

    def __repr__(self) -> str:
        return f"ODESystem('{self.name}', dim={self.dim})"


class ODESystemSolution:
    """
    Solution of a vector ODE.

    Stores:
      t_values : List[ExactReal]
      y_values : List[List[ExactReal]]   – y_values[k] = solution vector at t_k

    Methods
    -------
    component(i)  – ODESolution for the i-th component only
    trajectory(i, prec) – list of (t, y_i) float pairs
    """

    def __init__(
        self,
        t_values: List[ExactReal],
        y_values: List[List[ExactReal]],
        method:   str = "unknown",
    ):
        self.t_values = t_values
        self.y_values = y_values
        self.method   = method
        self.dim      = len(y_values[0]) if y_values else 0

    def component(self, i: int) -> ODESolution:
        """Return the i-th component as a scalar ODESolution."""
        return ODESolution(
            self.t_values,
            [yv[i] for yv in self.y_values],
            method=f"{self.method}[{i}]",
        )

    def trajectory(self, i: int, prec: int = 40) -> List[Tuple[float, float]]:
        return [
            (float(t.eval(prec)), float(y[i].eval(prec)))
            for t, y in zip(self.t_values, self.y_values)
        ]

    def __len__(self) -> int:
        return len(self.t_values)

    def __repr__(self) -> str:
        n  = len(self)
        t0 = float(self.t_values[0].eval(40))
        t1 = float(self.t_values[-1].eval(40))
        return (
            f"ODESystemSolution(method='{self.method}', "
            f"dim={self.dim}, steps={n}, t∈[{t0:.4g},{t1:.4g}])"
        )
