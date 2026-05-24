"""
Bifurcation Detector
====================
Identifies critical decision points where system behaviour qualitatively
changes — transitions between attractor basins, period-doubling, symmetry
breaking, etc.

Methods:
  • Real-time variance-based early warning signals
  • Eigenvalue tracking across parameter sweeps
  • Critical slowing-down detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kairos.core.state import PhasePoint, Trajectory


@dataclass
class BifurcationEvent:
    """Record of a detected bifurcation."""

    time: float
    index: int
    type: str  # "fold", "hopf", "period_doubling", "crisis", "unknown"
    parameter_value: float | None = None
    old_attractor: np.ndarray | None = None
    new_attractor: np.ndarray | None = None
    severity: float = 0.0  # 0–1 scale


@dataclass
class BifurcationDetector:
    """
    Online detector for bifurcation events in a running trajectory.

    Uses variance, autocorrelation, and spectral changes as early-warning
    signals for critical transitions (Scheffer et al. 2009).
    """

    window_size: int = 200
    stride: int = 50
    variance_threshold: float = 2.0   # multiplicative jump in variance
    acf_threshold: float = 0.3        # absolute increase in lag-1 autocorrelation
    spectral_threshold: float = 0.5   # fractional shift in dominant frequency

    # ── main detection ────────────────────────────────────────────
    def detect(self, trajectory: Trajectory) -> List[BifurcationEvent]:
        """
        Scan a trajectory for bifurcation events using early-warning
        signal analysis.
        """
        pos = trajectory.positions_array()
        times = trajectory.times()
        n = len(pos)
        events: List[BifurcationEvent] = []

        if n < 2 * self.window_size:
            return events

        prev_var = self._windowed_variance(pos[: self.window_size])
        prev_acf = self._lag1_autocorrelation(pos[: self.window_size])

        for start in range(self.stride, n - self.window_size, self.stride):
            window = pos[start : start + self.window_size]
            curr_var = self._windowed_variance(window)
            curr_acf = self._lag1_autocorrelation(window)

            # Check variance spike (critical slowing down)
            var_ratio = curr_var / (prev_var + 1e-12)
            acf_delta = curr_acf - prev_acf

            if var_ratio > self.variance_threshold or acf_delta > self.acf_threshold:
                bif_type = self._classify_bifurcation(pos, start, self.window_size)
                event = BifurcationEvent(
                    time=float(times[start + self.window_size // 2]),
                    index=start + self.window_size // 2,
                    type=bif_type,
                    severity=min(1.0, max(var_ratio / 5.0, acf_delta)),
                )
                # Avoid duplicate events too close together
                if not events or (event.index - events[-1].index) > self.window_size:
                    events.append(event)

            prev_var = curr_var
            prev_acf = curr_acf

        return events

    # ── early-warning signals ─────────────────────────────────────
    def early_warning_signals(
        self,
        trajectory: Trajectory,
    ) -> Dict[str, np.ndarray]:
        """
        Compute sliding-window early-warning indicators.

        Returns dict with keys: "times", "variance", "acf1", "skewness", "kurtosis".
        """
        pos = trajectory.positions_array()
        times = trajectory.times()
        n = len(pos)

        t_out, var_out, acf_out, skew_out, kurt_out = [], [], [], [], []

        for start in range(0, n - self.window_size, self.stride):
            window = pos[start : start + self.window_size]
            t_mid = times[start + self.window_size // 2]
            flat = window.flatten()

            t_out.append(t_mid)
            var_out.append(float(np.var(flat)))
            acf_out.append(self._lag1_autocorrelation(window))

            m = flat.mean()
            s = flat.std()
            if s > 1e-12:
                skew_out.append(float(np.mean(((flat - m) / s) ** 3)))
                kurt_out.append(float(np.mean(((flat - m) / s) ** 4) - 3))
            else:
                skew_out.append(0.0)
                kurt_out.append(0.0)

        return {
            "times": np.array(t_out),
            "variance": np.array(var_out),
            "acf1": np.array(acf_out),
            "skewness": np.array(skew_out),
            "kurtosis": np.array(kurt_out),
        }

    # ── eigenvalue tracking ───────────────────────────────────────
    @staticmethod
    def track_eigenvalues(
        rhs: Callable,
        param_name: str,
        param_range: Tuple[float, float],
        fixed_point_fn: Callable,
        n_values: int = 100,
        dim: int = 3,
        eps: float = 1e-6,
        base_params: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track Jacobian eigenvalues of a fixed point across a parameter sweep.

        Parameters
        ----------
        fixed_point_fn : callable(params) → np.ndarray
            Returns the fixed point for a given parameter set.

        Returns
        -------
        param_values : (n_values,)
        eigenvalues : (n_values, dim) complex array
        """
        base_params = base_params or {}
        param_values = np.linspace(param_range[0], param_range[1], n_values)
        eigenvalues = np.zeros((n_values, dim), dtype=complex)

        for k, pval in enumerate(param_values):
            params = {**base_params, param_name: pval}
            fp = fixed_point_fn(params)

            # Numerical Jacobian
            J = np.zeros((dim, dim))
            f0 = rhs(0, fp, **params)
            for i in range(dim):
                fp_p = fp.copy()
                fp_p[i] += eps
                J[:, i] = (rhs(0, fp_p, **params) - f0) / eps

            eigenvalues[k] = np.linalg.eigvals(J)

        return param_values, eigenvalues

    # ── internal helpers ──────────────────────────────────────────
    @staticmethod
    def _windowed_variance(window: np.ndarray) -> float:
        return float(np.var(window))

    @staticmethod
    def _lag1_autocorrelation(window: np.ndarray) -> float:
        """Mean lag-1 autocorrelation across dimensions."""
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        acf_vals = []
        for d in range(window.shape[1]):
            sig = window[:, d]
            sig = sig - sig.mean()
            s = sig.std()
            if s < 1e-12:
                acf_vals.append(0.0)
                continue
            n = len(sig)
            acf = np.sum(sig[:-1] * sig[1:]) / ((n - 1) * s * s)
            acf_vals.append(acf)
        return float(np.mean(acf_vals))

    @staticmethod
    def _classify_bifurcation(pos: np.ndarray, idx: int, window: int) -> str:
        """Heuristic classification based on trajectory topology change."""
        before = pos[max(0, idx - window) : idx]
        after = pos[idx : idx + window]

        if len(before) < 10 or len(after) < 10:
            return "unknown"

        var_before = np.var(before)
        var_after = np.var(after)

        # Period detection via zero-crossings
        def zero_crossings(sig: np.ndarray) -> int:
            centered = sig - sig.mean(axis=0)
            if centered.ndim > 1:
                centered = centered[:, 0]
            return int(np.sum(np.diff(np.sign(centered)) != 0))

        zc_before = zero_crossings(before)
        zc_after = zero_crossings(after)

        if var_after < 0.1 * var_before:
            return "fold"  # collapsed to fixed point
        elif zc_after > 1.8 * zc_before:
            return "period_doubling"
        elif var_after > 5 * var_before:
            return "crisis"  # sudden expansion
        elif abs(zc_after - zc_before) > 0.5 * zc_before:
            return "hopf"
        return "unknown"
