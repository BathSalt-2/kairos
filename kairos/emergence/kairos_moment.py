"""
Kairos Moment Recognition
==========================
Detects opportune moments for intervention — when small actions yield
maximum impact.  This is the system's eponymous capability.

A "kairos moment" occurs when:
  1. The system is near a bifurcation point (sensitivity is high)
  2. Oscillatory coherence is elevated (the system is "listening")
  3. Entropy is in a sweet spot (not too ordered, not too chaotic)
  4. Energy reserves are sufficient for action

The detector fuses these signals into a composite readiness score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from kairos.core.state import CognitiveState, Trajectory
from kairos.emergence.bifurcation import BifurcationDetector, BifurcationEvent


@dataclass
class KairosMoment:
    """A detected opportune moment."""

    time: float
    index: int
    readiness: float  # composite score 0–1
    sensitivity: float  # how responsive the system is to perturbation
    coherence: float  # oscillatory coherence at this instant
    entropy: float  # current entropy
    energy: float  # current energy
    nearby_bifurcation: Optional[BifurcationEvent] = None
    recommended_action: str = ""
    action_magnitude: float = 0.0


@dataclass
class KairosMomentDetector:
    """
    Detector for kairos moments in a running or completed trajectory.

    Parameters
    ----------
    sensitivity_weight : float
        Weight of bifurcation proximity in readiness score.
    coherence_weight : float
        Weight of oscillatory coherence.
    entropy_weight : float
        Weight of entropy optimality (peaks at target_entropy).
    energy_weight : float
        Weight of energy availability.
    target_entropy : float
        Optimal entropy for a kairos moment (edge of chaos).
    readiness_threshold : float
        Minimum composite readiness to flag as kairos moment.
    min_separation : int
        Minimum sample separation between detected moments.
    """

    sensitivity_weight: float = 0.35
    coherence_weight: float = 0.30
    entropy_weight: float = 0.20
    energy_weight: float = 0.15
    target_entropy: float = 0.5
    readiness_threshold: float = 0.6
    min_separation: int = 100

    # ── main detection ────────────────────────────────────────────
    def detect(
        self,
        trajectory: Trajectory,
        states: List[CognitiveState] | None = None,
        energy_trace: List[float] | None = None,
        entropy_trace: List[float] | None = None,
        synchrony_trace: List[float] | None = None,
        bifurcation_events: List[BifurcationEvent] | None = None,
    ) -> List[KairosMoment]:
        """
        Scan for kairos moments.

        Accepts pre-computed traces (from OscillatorySynthesisCore) or
        derives them from the trajectory.
        """
        n = len(trajectory)
        if n < 20:
            return []

        times = trajectory.times()

        # Derive missing traces
        if energy_trace is None:
            energy_trace = [1.0] * n
        if entropy_trace is None:
            entropy_trace = list(self._estimate_entropy(trajectory))
        if synchrony_trace is None:
            synchrony_trace = [0.5] * n

        # Detect bifurcations if not provided
        if bifurcation_events is None:
            detector = BifurcationDetector()
            bifurcation_events = detector.detect(trajectory)

        # Compute per-sample sensitivity (proximity to bifurcation)
        sensitivity = self._sensitivity_from_bifurcations(n, bifurcation_events)

        # Compute readiness scores
        readiness = np.zeros(n)
        for i in range(n):
            # Entropy score: Gaussian around target_entropy
            ent_val = entropy_trace[i] if i < len(entropy_trace) else 0.5
            ent_score = np.exp(-((ent_val - self.target_entropy) ** 2) / 0.08)

            # Energy: normalised [0, 1]
            e_val = energy_trace[i] if i < len(energy_trace) else 1.0
            e_score = np.clip(e_val / 2.0, 0, 1)

            # Coherence (synchrony)
            coh = synchrony_trace[i] if i < len(synchrony_trace) else 0.5

            readiness[i] = (
                self.sensitivity_weight * sensitivity[i]
                + self.coherence_weight * coh
                + self.entropy_weight * ent_score
                + self.energy_weight * e_score
            )

        # Extract peaks above threshold
        moments: List[KairosMoment] = []
        candidates = np.where(readiness > self.readiness_threshold)[0]

        for idx in candidates:
            # Enforce minimum separation
            if moments and (idx - moments[-1].index) < self.min_separation:
                # Keep the higher-readiness one
                if readiness[idx] > moments[-1].readiness:
                    moments[-1] = self._build_moment(
                        idx, times, readiness, sensitivity, synchrony_trace,
                        entropy_trace, energy_trace, bifurcation_events,
                    )
                continue

            moments.append(
                self._build_moment(
                    idx, times, readiness, sensitivity, synchrony_trace,
                    entropy_trace, energy_trace, bifurcation_events,
                )
            )

        return moments

    # ── readiness time-series ─────────────────────────────────────
    def readiness_trace(
        self,
        trajectory: Trajectory,
        energy_trace: List[float] | None = None,
        entropy_trace: List[float] | None = None,
        synchrony_trace: List[float] | None = None,
    ) -> np.ndarray:
        """Return per-sample readiness score (without thresholding)."""
        n = len(trajectory)
        if energy_trace is None:
            energy_trace = [1.0] * n
        if entropy_trace is None:
            entropy_trace = list(self._estimate_entropy(trajectory))
        if synchrony_trace is None:
            synchrony_trace = [0.5] * n

        detector = BifurcationDetector()
        events = detector.detect(trajectory)
        sensitivity = self._sensitivity_from_bifurcations(n, events)

        scores = np.zeros(n)
        for i in range(n):
            ent_val = entropy_trace[i] if i < len(entropy_trace) else 0.5
            ent_score = np.exp(-((ent_val - self.target_entropy) ** 2) / 0.08)
            e_val = energy_trace[i] if i < len(energy_trace) else 1.0
            e_score = np.clip(e_val / 2.0, 0, 1)
            coh = synchrony_trace[i] if i < len(synchrony_trace) else 0.5
            scores[i] = (
                self.sensitivity_weight * sensitivity[i]
                + self.coherence_weight * coh
                + self.entropy_weight * ent_score
                + self.energy_weight * e_score
            )
        return scores

    # ── helpers ───────────────────────────────────────────────────
    def _sensitivity_from_bifurcations(
        self, n: int, events: List[BifurcationEvent], decay: float = 0.01,
    ) -> np.ndarray:
        """
        Convert discrete bifurcation events to continuous sensitivity signal
        via Gaussian kernels centered on each event.
        """
        sensitivity = np.zeros(n)
        for event in events:
            idx = event.index
            kernel = event.severity * np.exp(-decay * (np.arange(n) - idx) ** 2)
            sensitivity += kernel
        return np.clip(sensitivity, 0, 1)

    @staticmethod
    def _estimate_entropy(trajectory: Trajectory, window: int = 50) -> np.ndarray:
        """Sliding-window sample entropy estimate."""
        pos = trajectory.positions_array()
        n = len(pos)
        entropy = np.full(n, 0.5)
        for i in range(window, n):
            w = pos[i - window : i]
            v = np.var(w)
            entropy[i] = np.clip(np.log1p(v) / 5.0, 0, 1)
        return entropy

    def _build_moment(
        self,
        idx: int,
        times: np.ndarray,
        readiness: np.ndarray,
        sensitivity: np.ndarray,
        synchrony_trace: list,
        entropy_trace: list,
        energy_trace: list,
        bifurcation_events: List[BifurcationEvent],
    ) -> KairosMoment:
        # Find nearest bifurcation event
        nearest_bif = None
        min_dist = float("inf")
        for ev in bifurcation_events:
            d = abs(ev.index - idx)
            if d < min_dist:
                min_dist = d
                nearest_bif = ev

        # Recommend action
        if sensitivity[idx] > 0.7:
            action = "strong_perturbation"
            magnitude = 0.8
        elif readiness[idx] > 0.8:
            action = "moderate_nudge"
            magnitude = 0.5
        else:
            action = "gentle_influence"
            magnitude = 0.2

        return KairosMoment(
            time=float(times[idx]) if idx < len(times) else 0.0,
            index=idx,
            readiness=float(readiness[idx]),
            sensitivity=float(sensitivity[idx]),
            coherence=float(synchrony_trace[idx]) if idx < len(synchrony_trace) else 0.5,
            entropy=float(entropy_trace[idx]) if idx < len(entropy_trace) else 0.5,
            energy=float(energy_trace[idx]) if idx < len(energy_trace) else 1.0,
            nearby_bifurcation=nearest_bif,
            recommended_action=action,
            action_magnitude=magnitude,
        )
