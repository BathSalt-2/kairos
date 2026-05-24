"""
Phase Space Navigator
=====================
Traverses the high-dimensional space of possible cognitive trajectories.

Provides:
  • Time-delay embedding (Takens' theorem)
  • Recurrence analysis
  • Phase-space density estimation
  • Trajectory distance metrics (Hausdorff, Fréchet)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from kairos.core.state import PhasePoint, Trajectory


@dataclass
class PhaseSpaceNavigator:
    """
    Tools for embedding, analysing, and navigating phase-space structures.
    """

    # ── time-delay embedding (Takens) ────────────────────────────
    @staticmethod
    def time_delay_embed(
        signal: np.ndarray,
        dim: int = 3,
        tau: int = 10,
    ) -> np.ndarray:
        """
        Reconstruct a phase-space attractor from a scalar time-series
        using Takens' delay embedding theorem.

        Parameters
        ----------
        signal : (N,) array
        dim : embedding dimension
        tau : time delay in samples

        Returns
        -------
        embedded : (M, dim) array   where M = N - (dim-1)*tau
        """
        N = len(signal)
        M = N - (dim - 1) * tau
        if M <= 0:
            raise ValueError("Signal too short for requested dim/tau")
        embedded = np.zeros((M, dim))
        for d in range(dim):
            embedded[:, d] = signal[d * tau : d * tau + M]
        return embedded

    @staticmethod
    def optimal_tau(signal: np.ndarray, max_tau: int = 100) -> int:
        """
        Estimate optimal delay via first minimum of mutual information
        (simplified: use first zero-crossing of autocorrelation).
        """
        sig = signal - signal.mean()
        acf = np.correlate(sig, sig, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / acf[0]
        for i in range(1, min(max_tau, len(acf))):
            if acf[i] <= 0:
                return i
        return max_tau

    @staticmethod
    def optimal_dim(signal: np.ndarray, tau: int, max_dim: int = 10, threshold: float = 0.01) -> int:
        """
        Estimate embedding dimension via False Nearest Neighbors (FNN).
        Returns the first dimension where FNN ratio drops below *threshold*.
        """
        for d in range(1, max_dim + 1):
            emb = PhaseSpaceNavigator.time_delay_embed(signal, d, tau)
            emb_next = PhaseSpaceNavigator.time_delay_embed(signal, d + 1, tau)
            n = min(len(emb), len(emb_next))
            if n < 10:
                return d

            # For each point, find nearest neighbour
            dists = cdist(emb[:n], emb[:n])
            np.fill_diagonal(dists, np.inf)
            nn_idx = dists.argmin(axis=1)
            nn_dist = dists[np.arange(n), nn_idx]

            # Check if neighbours stay close in d+1
            dist_d1 = np.linalg.norm(emb_next[:n] - emb_next[nn_idx[:n]], axis=1)
            mask = nn_dist > 1e-12
            fnn_ratio = np.mean(((dist_d1[mask] / nn_dist[mask]) > 10))

            if fnn_ratio < threshold:
                return d + 1
        return max_dim

    # ── recurrence analysis ───────────────────────────────────────
    @staticmethod
    def recurrence_matrix(
        trajectory: Trajectory | np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Compute binary recurrence matrix.

        Parameters
        ----------
        trajectory : Trajectory or (N, D) array
        threshold : distance threshold; if None, use 10% of max distance

        Returns
        -------
        R : (N, N) boolean array  —  R[i,j] = True if points i,j are close.
        """
        if isinstance(trajectory, Trajectory):
            pos = trajectory.positions_array()
        else:
            pos = np.asarray(trajectory)

        dists = cdist(pos, pos)
        if threshold is None:
            threshold = 0.1 * dists.max()
        return dists <= threshold

    @staticmethod
    def recurrence_rate(R: np.ndarray) -> float:
        """Fraction of recurrent points (excluding diagonal)."""
        n = R.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(R[mask].mean())

    @staticmethod
    def determinism(R: np.ndarray, min_line: int = 2) -> float:
        """
        Determinism (DET): fraction of recurrence points forming
        diagonal lines of at least *min_line* length.
        """
        n = R.shape[0]
        diag_lengths: list[int] = []
        for k in range(-n + 1, n):
            if k == 0:
                continue
            d = np.diag(R, k)
            length = 0
            for val in d:
                if val:
                    length += 1
                else:
                    if length >= min_line:
                        diag_lengths.append(length)
                    length = 0
            if length >= min_line:
                diag_lengths.append(length)
        total_recurrence = R.sum() - np.trace(R)
        if total_recurrence == 0:
            return 0.0
        return float(sum(diag_lengths) / total_recurrence)

    # ── trajectory distances ──────────────────────────────────────
    @staticmethod
    def hausdorff_distance(traj_a: Trajectory, traj_b: Trajectory) -> float:
        """Directed Hausdorff distance between two trajectories."""
        A = traj_a.positions_array()
        B = traj_b.positions_array()
        dists = cdist(A, B)
        h_ab = dists.min(axis=1).max()
        h_ba = dists.min(axis=0).max()
        return float(max(h_ab, h_ba))

    # ── density estimation ────────────────────────────────────────
    @staticmethod
    def phase_density(
        trajectory: Trajectory,
        grid_resolution: int = 50,
        dims: Tuple[int, int] = (0, 1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D histogram density of trajectory projected onto two dimensions.

        Returns (density, x_edges, y_edges).
        """
        pos = trajectory.positions_array()
        x = pos[:, dims[0]]
        y = pos[:, dims[1]]
        density, xe, ye = np.histogram2d(x, y, bins=grid_resolution)
        return density, xe, ye

    # ── navigation ────────────────────────────────────────────────
    def find_nearest_return(
        self,
        trajectory: Trajectory,
        target: PhasePoint,
        tolerance: float = 1.0,
    ) -> list[int]:
        """
        Find indices where the trajectory passes within *tolerance*
        of a target point — i.e. Poincaré return times.
        """
        pos = trajectory.positions_array()
        dists = np.linalg.norm(pos - target.position, axis=1)
        return list(np.where(dists < tolerance)[0])
