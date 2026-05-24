"""
Oscillatory Synthesis Core — Recursive Oscillatory Synthesis (ROS)
===================================================================
The central engine of emergent cognition in KAIROS.

Implements the recursive interplay between three cognitive modes:
  • Divergent  — explore phase space, generate possibilities (Default Mode Network)
  • Convergent — contract to attractors, crystallize decisions (Task-Positive Network)
  • Integrative — synthesize across scales and modalities (Salience Network)

The ROS algorithm cycles through these modes, using oscillatory coupling to
drive emergence of novel cognitive structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kairos.core.oscillator import CoupledOscillators, OscillatorBank
from kairos.core.state import CognitiveState, PhasePoint, Trajectory

logger = logging.getLogger(__name__)


class CognitiveMode(str, Enum):
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"
    INTEGRATIVE = "integrative"


@dataclass
class SynthesisResult:
    """Output of a single ROS cycle."""

    trajectory: Trajectory
    states: List[CognitiveState]
    mode_sequence: List[CognitiveMode]
    energy_trace: List[float]
    entropy_trace: List[float]
    synchrony_trace: List[float]
    emergent_patterns: List[np.ndarray]
    n_cycles: int = 0


@dataclass
class OscillatorySynthesisCore:
    """
    Recursive Oscillatory Synthesis engine.

    Parameters
    ----------
    dim : int
        Dimensionality of cognitive state space.
    n_oscillators : int
        Number of coupled oscillators driving mode transitions.
    coupling : float
        Coupling strength for the Kuramoto model.
    divergent_noise : float
        Noise scale during divergent exploration.
    convergent_rate : float
        Contraction rate toward attractors during convergent mode.
    integration_strength : float
        Strength of cross-scale integration during integrative mode.
    mode_duration : float
        Time spent in each mode before transitioning (seconds).
    """

    dim: int = 4
    n_oscillators: int = 12
    coupling: float = 2.0
    divergent_noise: float = 0.5
    convergent_rate: float = 0.3
    integration_strength: float = 0.4
    mode_duration: float = 5.0
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def __post_init__(self) -> None:
        # Oscillator frequencies span theta–gamma range for biological plausibility
        freqs = self._rng.uniform(4, 80, self.n_oscillators)
        self.oscillators = CoupledOscillators(freqs, coupling=self.coupling, rng=self._rng)

    # ── main ROS loop ─────────────────────────────────────────────
    def run(
        self,
        initial_state: CognitiveState | None = None,
        n_cycles: int = 10,
        dt: float = 0.01,
        attractor_points: List[np.ndarray] | None = None,
    ) -> SynthesisResult:
        """
        Execute *n_cycles* of Recursive Oscillatory Synthesis.

        Each cycle: divergent → convergent → integrative.
        """
        if initial_state is None:
            initial_state = CognitiveState(
                phase=PhasePoint(
                    position=self._rng.standard_normal(self.dim),
                    velocity=np.zeros(self.dim),
                ),
                energy=1.0,
                entropy=0.5,
                mode="integrative",
            )

        if attractor_points is None:
            # Generate some default attractor basins
            attractor_points = [self._rng.standard_normal(self.dim) * 3 for _ in range(3)]

        trajectory = Trajectory()
        states: List[CognitiveState] = [initial_state]
        mode_sequence: List[CognitiveMode] = []
        energy_trace: List[float] = [initial_state.energy]
        entropy_trace: List[float] = [initial_state.entropy]
        synchrony_trace: List[float] = [self.oscillators.synchrony()]
        emergent_patterns: List[np.ndarray] = []

        current = initial_state
        t = 0.0
        steps_per_mode = int(self.mode_duration / dt)

        for cycle in range(n_cycles):
            for mode in CognitiveMode:
                current = CognitiveState(
                    phase=current.phase.copy(),
                    energy=current.energy,
                    entropy=current.entropy,
                    mode=mode.value,
                    label=f"cycle_{cycle}_{mode.value}",
                )
                mode_sequence.append(mode)

                for step in range(steps_per_mode):
                    # Advance oscillators
                    self.oscillators.step(dt)
                    sync = self.oscillators.synchrony()

                    # Mode-specific dynamics
                    if mode == CognitiveMode.DIVERGENT:
                        current = self._divergent_step(current, sync, dt)
                    elif mode == CognitiveMode.CONVERGENT:
                        current = self._convergent_step(current, attractor_points, sync, dt)
                    else:
                        current = self._integrative_step(current, sync, dt)

                    t += dt
                    current.phase.time = t
                    trajectory.append(current.phase.copy())
                    energy_trace.append(current.energy)
                    entropy_trace.append(current.entropy)
                    synchrony_trace.append(sync)

                states.append(current)

            # After each full cycle, extract emergent pattern
            pattern = self._extract_pattern(trajectory.tail(steps_per_mode * 3))
            emergent_patterns.append(pattern)

        return SynthesisResult(
            trajectory=trajectory,
            states=states,
            mode_sequence=mode_sequence,
            energy_trace=energy_trace,
            entropy_trace=entropy_trace,
            synchrony_trace=synchrony_trace,
            emergent_patterns=emergent_patterns,
            n_cycles=n_cycles,
        )

    # ── mode dynamics ─────────────────────────────────────────────
    def _divergent_step(self, state: CognitiveState, sync: float, dt: float) -> CognitiveState:
        """Explore: add noise scaled inversely by synchrony."""
        noise_scale = self.divergent_noise * (1.0 + (1.0 - sync))
        pos = state.phase.position + self._rng.normal(0, noise_scale * dt, self.dim)
        energy = state.energy * (1.0 - 0.01 * dt)  # slight decay during exploration
        entropy = min(1.0, state.entropy + 0.05 * dt * noise_scale)
        return CognitiveState(
            phase=PhasePoint(position=pos, time=state.phase.time),
            energy=energy,
            entropy=entropy,
            mode="divergent",
            label=state.label,
        )

    def _convergent_step(
        self,
        state: CognitiveState,
        attractors: List[np.ndarray],
        sync: float,
        dt: float,
    ) -> CognitiveState:
        """Contract toward nearest attractor, strength modulated by synchrony."""
        # Find nearest attractor
        dists = [np.linalg.norm(state.phase.position - a) for a in attractors]
        nearest = attractors[int(np.argmin(dists))]

        direction = nearest - state.phase.position
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm

        pull = self.convergent_rate * sync * dt
        pos = state.phase.position + direction * pull
        energy = min(2.0, state.energy + 0.02 * dt * sync)
        entropy = max(0.0, state.entropy - 0.08 * dt * sync)

        return CognitiveState(
            phase=PhasePoint(position=pos, time=state.phase.time),
            energy=energy,
            entropy=entropy,
            mode="convergent",
            label=state.label,
        )

    def _integrative_step(self, state: CognitiveState, sync: float, dt: float) -> CognitiveState:
        """Synthesize: non-linear mixing driven by oscillatory coupling."""
        pos = state.phase.position.copy()

        # Cross-dimension coupling (non-linear mixing)
        for i in range(self.dim):
            j = (i + 1) % self.dim
            pos[i] += self.integration_strength * np.tanh(pos[j] - pos[i]) * sync * dt

        # Add weak oscillatory perturbation from Kuramoto phases
        osc_phase = self.oscillators.phases[:self.dim] if self.n_oscillators >= self.dim else np.zeros(self.dim)
        pos += 0.05 * np.sin(osc_phase[:len(pos)]) * dt

        energy = state.energy * (1.0 - 0.005 * dt)
        entropy = state.entropy + 0.01 * (0.5 - state.entropy) * dt  # relax toward 0.5

        return CognitiveState(
            phase=PhasePoint(position=pos, time=state.phase.time),
            energy=energy,
            entropy=entropy,
            mode="integrative",
            label=state.label,
        )

    def _extract_pattern(self, recent: Trajectory) -> np.ndarray:
        """Extract an emergent pattern as the principal component of recent trajectory."""
        if len(recent) < 10:
            return np.zeros(self.dim)
        pos = recent.positions_array()
        centered = pos - pos.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            return Vt[0] * S[0]  # first principal component scaled by singular value
        except np.linalg.LinAlgError:
            return pos.mean(axis=0)
