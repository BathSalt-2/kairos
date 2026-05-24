"""
Attractor Landscape Mapper
==========================
Identifies stable states (attractors), unstable fixed points, and transition
pathways in the cognitive phase space.

Capabilities:
  • Fixed-point detection via Newton–Raphson
  • Attractor basin mapping
  • Correlation dimension estimation
  • Bifurcation diagram generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist

from kairos.core.state import PhasePoint, Trajectory


@dataclass
class Attractor:
    """Descriptor of a detected attractor."""

    center: np.ndarray
    type: str  # "point", "limit_cycle", "strange"
    dimension: float  # correlation dimension
    basin_size: float  # fraction of ICs that converge here
    lyapunov_max: float = 0.0


@dataclass
class AttractorLandscapeMapper:
    """
    Maps the attractor landscape of a dynamical system.

    Parameters
    ----------
    rhs : callable
        ODE right-hand side f(t, y, **params) → dy/dt
    params : dict
        Parameters for the RHS function.
    dim : int
        State-space dimensionality.
    """

    rhs: Callable
    params: dict = field(default_factory=dict)
    dim: int = 3

    # ── fixed-point finding ───────────────────────────────────────
    def find_fixed_points(
        self,
        search_bounds: Tuple[float, float] = (-30, 30),
        n_starts: int = 200,
        tol: float = 1e-10,
        rng: np.random.Generator | None = None,
    ) -> List[np.ndarray]:
        """
        Find fixed points (where f(t, y) ≈ 0) by multi-start Newton–Raphson.
        """
        rng = rng or np.random.default_rng()
        found: List[np.ndarray] = []

        for _ in range(n_starts):
            y0 = rng.uniform(search_bounds[0], search_bounds[1], self.dim)
            try:
                sol, info, ier, _ = fsolve(
                    lambda y: self.rhs(0, y, **self.params),
                    y0,
                    full_output=True,
                )
                if ier == 1:
                    residual = np.linalg.norm(info["fvec"])
                    if residual < tol:
                        # Deduplicate
                        is_new = all(np.linalg.norm(sol - fp) > 1e-6 for fp in found)
                        if is_new:
                            found.append(sol)
            except Exception:
                continue
        return found

    def classify_fixed_point(self, fp: np.ndarray, eps: float = 1e-6) -> str:
        """
        Classify a fixed point by linearising the dynamics (Jacobian eigenvalues).

        Returns one of: "stable_node", "unstable_node", "saddle", "stable_spiral",
        "unstable_spiral", "center".
        """
        J = self._jacobian(fp, eps)
        eigenvalues = np.linalg.eigvals(J)
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag

        has_complex = np.any(np.abs(imag_parts) > 1e-8)
        all_negative = np.all(real_parts < -1e-8)
        all_positive = np.all(real_parts > 1e-8)
        mixed = not all_negative and not all_positive

        if all_negative:
            return "stable_spiral" if has_complex else "stable_node"
        elif all_positive:
            return "unstable_spiral" if has_complex else "unstable_node"
        elif mixed:
            return "saddle"
        else:
            return "center"

    # ── attractor basin mapping ───────────────────────────────────
    def map_basins(
        self,
        attractors: List[np.ndarray],
        grid_bounds: Tuple[float, float] = (-30, 30),
        grid_resolution: int = 40,
        evolve_time: float = 100.0,
        dt: float = 0.01,
        dims: Tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        """
        For a 2D slice of initial conditions, determine which attractor
        each IC converges to.

        Returns (grid_resolution, grid_resolution) int array of attractor indices
        (-1 = escaped / no convergence).
        """
        from kairos.engines.chaos import ChaosDynamicsEngine

        engine = ChaosDynamicsEngine(system=self.rhs, params=self.params)

        lo, hi = grid_bounds
        xs = np.linspace(lo, hi, grid_resolution)
        ys = np.linspace(lo, hi, grid_resolution)
        basin = np.full((grid_resolution, grid_resolution), -1, dtype=int)

        base_ic = np.zeros(self.dim)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                ic = base_ic.copy()
                ic[dims[0]] = x
                ic[dims[1]] = y
                traj = engine.evolve(ic, evolve_time, dt)
                final = traj.points[-1].position

                # Which attractor is closest?
                min_dist = float("inf")
                min_idx = -1
                for a_idx, a_center in enumerate(attractors):
                    d = np.linalg.norm(final - a_center)
                    if d < min_dist:
                        min_dist = d
                        min_idx = a_idx
                basin[j, i] = min_idx  # note: j=row(y), i=col(x)

        return basin

    # ── correlation dimension ─────────────────────────────────────
    @staticmethod
    def correlation_dimension(
        trajectory: Trajectory,
        max_points: int = 2000,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Estimate the correlation dimension D2 of an attractor using
        the Grassberger–Procaccia algorithm.
        """
        rng = rng or np.random.default_rng()
        pos = trajectory.positions_array()
        if len(pos) > max_points:
            idx = rng.choice(len(pos), max_points, replace=False)
            pos = pos[idx]

        dists = pdist(pos)
        dists = dists[dists > 0]

        log_r = np.linspace(np.log(dists.min()), np.log(dists.max()), 30)
        log_C = np.zeros_like(log_r)
        n = len(pos)
        total_pairs = n * (n - 1) / 2

        for k, lr in enumerate(log_r):
            r = np.exp(lr)
            count = np.sum(dists < r)
            log_C[k] = np.log(count / total_pairs) if count > 0 else -30

        # Linear fit to the scaling region (middle 60%)
        n_pts = len(log_r)
        start = n_pts // 5
        end = 4 * n_pts // 5
        if end - start < 3:
            return 0.0
        coeffs = np.polyfit(log_r[start:end], log_C[start:end], 1)
        return float(coeffs[0])

    # ── bifurcation diagram ───────────────────────────────────────
    def bifurcation_diagram(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        n_values: int = 200,
        initial: np.ndarray | None = None,
        duration: float = 100.0,
        discard: float = 80.0,
        dt: float = 0.01,
        observe_dim: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate a bifurcation diagram by sweeping a parameter and
        recording the local maxima of one state variable.

        Returns (param_values, list_of_maxima_arrays).
        """
        from kairos.engines.chaos import ChaosDynamicsEngine

        rng = np.random.default_rng()
        param_values = np.linspace(param_range[0], param_range[1], n_values)
        maxima_list: List[np.ndarray] = []

        if initial is None:
            initial = rng.standard_normal(self.dim)

        for pval in param_values:
            params = {**self.params, param_name: pval}
            engine = ChaosDynamicsEngine(system=self.rhs, params=params)
            traj = engine.evolve(initial, duration, dt)
            pos = traj.positions_array()
            # Discard transient
            discard_idx = int(discard / dt)
            signal = pos[discard_idx:, observe_dim]

            # Find local maxima
            if len(signal) < 3:
                maxima_list.append(np.array([]))
                continue
            dx = np.diff(signal)
            peaks = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
            maxima_list.append(signal[peaks])

        return param_values, maxima_list

    # ── internal ──────────────────────────────────────────────────
    def _jacobian(self, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        J = np.zeros((self.dim, self.dim))
        f0 = self.rhs(0, y, **self.params)
        for i in range(self.dim):
            y_p = y.copy()
            y_p[i] += eps
            fi = self.rhs(0, y_p, **self.params)
            J[:, i] = (fi - f0) / eps
        return J
