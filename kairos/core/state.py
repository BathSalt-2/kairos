"""
Cognitive state representations for the KAIROS framework.

A CognitiveState is a snapshot of the system at one instant — a point in
high-dimensional phase space plus metadata (energy, entropy, mode label).
Trajectories are ordered sequences of PhasePoints that record system evolution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PhasePoint:
    """A single point in phase space."""

    position: np.ndarray  # (D,) coordinate vector
    velocity: np.ndarray | None = None  # optional derivative
    time: float = 0.0

    # ── helpers ───────────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self.position.shape[0]

    def distance_to(self, other: "PhasePoint") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def copy(self) -> "PhasePoint":
        return PhasePoint(
            position=self.position.copy(),
            velocity=self.velocity.copy() if self.velocity is not None else None,
            time=self.time,
        )


@dataclass
class CognitiveState:
    """
    Full cognitive state — phase-space coordinates plus emergent metadata.

    Attributes
    ----------
    phase : PhasePoint
        Current location in phase space.
    energy : float
        Scalar energy / activation level.
    entropy : float
        Measure of internal disorder / uncertainty.
    mode : str
        Current cognitive mode — one of ``divergent``, ``convergent``,
        ``integrative``.
    label : str
        Optional human-readable tag (e.g. "exploring", "deciding").
    id : str
        Unique identifier for tracking.
    """

    phase: PhasePoint
    energy: float = 1.0
    entropy: float = 0.5
    mode: str = "integrative"
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ── derived properties ────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self.phase.dim

    @property
    def position(self) -> np.ndarray:
        return self.phase.position

    def dominance_vector(self) -> np.ndarray:
        """Return unit vector in the direction of highest activation."""
        p = self.phase.position
        norm = np.linalg.norm(p)
        return p / norm if norm > 1e-12 else np.zeros_like(p)

    def perturb(self, sigma: float = 0.01, rng: np.random.Generator | None = None) -> "CognitiveState":
        """Return a new state with Gaussian noise added to position."""
        rng = rng or np.random.default_rng()
        new_pos = self.phase.position + rng.normal(0, sigma, size=self.dim)
        new_phase = PhasePoint(position=new_pos, velocity=self.phase.velocity, time=self.phase.time)
        return CognitiveState(
            phase=new_phase,
            energy=self.energy,
            entropy=self.entropy,
            mode=self.mode,
            label=self.label,
        )


@dataclass
class Trajectory:
    """Ordered collection of PhasePoints representing system evolution."""

    points: List[PhasePoint] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ── core API ──────────────────────────────────────────────────
    def append(self, point: PhasePoint) -> None:
        self.points.append(point)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> PhasePoint:
        return self.points[idx]

    # ── numpy helpers ─────────────────────────────────────────────
    def positions_array(self) -> np.ndarray:
        """Return (N, D) array of positions."""
        return np.array([p.position for p in self.points])

    def times(self) -> np.ndarray:
        return np.array([p.time for p in self.points])

    # ── analysis shortcuts ────────────────────────────────────────
    def total_length(self) -> float:
        """Arc length of the trajectory in phase space."""
        if len(self.points) < 2:
            return 0.0
        pos = self.positions_array()
        diffs = np.diff(pos, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.positions_array()
        return pos.min(axis=0), pos.max(axis=0)

    def tail(self, n: int = 100) -> "Trajectory":
        """Return trajectory of last *n* points."""
        return Trajectory(points=self.points[-n:], metadata=self.metadata)
