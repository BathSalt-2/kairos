"""
Chaos Dynamics Engine
=====================
Models cognitive states as points in chaotic dynamical systems with sensitive
dependence on initial conditions.

Implements:
  • Lorenz attractor
  • Rössler attractor
  • Custom KAIROS cognitive dynamics
  • Lyapunov exponent computation
  • Sensitive-dependence diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from kairos.core.state import PhasePoint, Trajectory


# ── Built-in systems ─────────────────────────────────────────────

def lorenz(t: float, state: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def rossler(t: float, state: np.ndarray, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> np.ndarray:
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)])


def kairos_cognitive(
    t: float,
    state: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.6,
    delta: float = 0.3,
    omega: float = 2.0,
) -> np.ndarray:
    """
    KAIROS cognitive dynamics — a 4D chaotic system designed for modelling
    cognitive state trajectories.

    Dimensions: [exploration, exploitation, integration, meta-awareness]
    """
    e, x, i, m = state
    de = alpha * (x - e) + gamma * np.sin(omega * t) * m
    dx = -beta * x + alpha * e * i - delta * m
    di = gamma * (e * x - i) + delta * np.cos(omega * t)
    dm = -delta * m + beta * np.tanh(e - x + i)
    return np.array([de, dx, di, dm])


BUILTIN_SYSTEMS: Dict[str, Callable] = {
    "lorenz": lorenz,
    "rossler": rossler,
    "kairos_cognitive": kairos_cognitive,
}


@dataclass
class ChaosDynamicsEngine:
    """
    Engine for evolving chaotic dynamical systems and computing
    chaos-theoretic diagnostics.

    Parameters
    ----------
    system : str | Callable
        Name of a built-in system or a user-supplied ODE RHS function
        ``f(t, state, **params) -> np.ndarray``.
    params : dict
        Keyword parameters forwarded to the ODE function.
    """

    system: str | Callable = "lorenz"
    params: dict = field(default_factory=dict)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.system, str):
            if self.system not in BUILTIN_SYSTEMS:
                raise ValueError(f"Unknown system '{self.system}'. Choose from {list(BUILTIN_SYSTEMS)}")
            self._rhs = BUILTIN_SYSTEMS[self.system]
        else:
            self._rhs = self.system

    # ── simulation ────────────────────────────────────────────────
    def evolve(
        self,
        initial: np.ndarray,
        duration: float = 50.0,
        dt: float = 0.01,
        method: str = "RK45",
    ) -> Trajectory:
        """
        Integrate the system from *initial* for *duration* seconds.

        Returns a Trajectory of PhasePoints.
        """
        t_span = (0.0, duration)
        t_eval = np.arange(0, duration, dt)

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, **self.params),
            t_span,
            initial,
            method=method,
            t_eval=t_eval,
            max_step=dt,
        )

        traj = Trajectory(metadata={"system": str(self.system), "params": self.params})
        for idx in range(len(sol.t)):
            traj.append(PhasePoint(position=sol.y[:, idx], time=sol.t[idx]))
        return traj

    # ── Lyapunov exponent ─────────────────────────────────────────
    def max_lyapunov(
        self,
        initial: np.ndarray,
        duration: float = 100.0,
        dt: float = 0.01,
        eps: float = 1e-8,
    ) -> float:
        """
        Estimate the maximal Lyapunov exponent (MLE) via the tangent-vector
        renormalisation method.

        Positive MLE → chaos.
        """
        dim = len(initial)
        y = initial.copy()
        d = self._rng.standard_normal(dim)
        d = d / np.linalg.norm(d) * eps

        lyap_sum = 0.0
        steps = int(duration / dt)

        for _ in range(steps):
            y_new = self._rk4_step(y, dt)
            y_pert = self._rk4_step(y + d, dt)
            d_new = y_pert - y_new
            d_norm = np.linalg.norm(d_new)
            if d_norm < 1e-30:
                break
            lyap_sum += np.log(d_norm / eps)
            d = d_new / d_norm * eps
            y = y_new

        return lyap_sum / (steps * dt)

    def lyapunov_spectrum(
        self,
        initial: np.ndarray,
        duration: float = 100.0,
        dt: float = 0.01,
    ) -> np.ndarray:
        """
        Estimate the full Lyapunov spectrum using QR decomposition of the
        tangent map.
        """
        dim = len(initial)
        y = initial.copy()
        Q = np.eye(dim)
        lyap = np.zeros(dim)
        steps = int(duration / dt)

        for step in range(steps):
            # Evolve base trajectory
            y_new = self._rk4_step(y, dt)

            # Evolve perturbation columns
            Z = np.zeros((dim, dim))
            for j in range(dim):
                y_pert = self._rk4_step(y + 1e-8 * Q[:, j], dt)
                Z[:, j] = (y_pert - y_new) / 1e-8

            Q_new, R = np.linalg.qr(Z)
            for j in range(dim):
                diag = abs(R[j, j])
                if diag > 1e-30:
                    lyap[j] += np.log(diag)
            Q = Q_new
            y = y_new

        return lyap / (steps * dt)

    # ── sensitive dependence ──────────────────────────────────────
    def sensitivity_test(
        self,
        initial: np.ndarray,
        eps: float = 1e-6,
        duration: float = 50.0,
        dt: float = 0.01,
    ) -> Tuple[Trajectory, Trajectory, np.ndarray]:
        """
        Evolve two nearby trajectories and return both plus the divergence
        time-series.
        """
        perturbed = initial.copy()
        perturbed[0] += eps

        traj1 = self.evolve(initial, duration, dt)
        traj2 = self.evolve(perturbed, duration, dt)

        pos1 = traj1.positions_array()
        pos2 = traj2.positions_array()
        n = min(len(pos1), len(pos2))
        divergence = np.linalg.norm(pos1[:n] - pos2[:n], axis=1)

        return traj1, traj2, divergence

    # ── internal RK4 ─────────────────────────────────────────────
    def _rk4_step(self, y: np.ndarray, dt: float) -> np.ndarray:
        f = lambda t, s: self._rhs(t, s, **self.params)
        k1 = f(0, y)
        k2 = f(0, y + dt / 2 * k1)
        k3 = f(0, y + dt / 2 * k2)
        k4 = f(0, y + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
