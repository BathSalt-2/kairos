"""
Neural oscillator models for the KAIROS framework.

Implements Kuramoto-style coupled oscillators with biologically-inspired
frequency bands (gamma, theta, alpha, beta) and cross-frequency coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Frequency band constants (Hz) ────────────────────────────────
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
}


@dataclass
class Oscillator:
    """
    Single phase oscillator.

    Parameters
    ----------
    freq : float
        Natural frequency (Hz).
    phase : float
        Current phase (radians).
    amplitude : float
        Oscillation amplitude.
    band : str
        Frequency band label.
    """

    freq: float
    phase: float = 0.0
    amplitude: float = 1.0
    band: str = "gamma"
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def step(self, dt: float) -> None:
        self.phase += 2 * np.pi * self.freq * dt
        self.phase %= 2 * np.pi

    def signal(self, t: float) -> float:
        return float(self.amplitude * np.sin(2 * np.pi * self.freq * t + self.phase))

    @classmethod
    def from_band(cls, band: str, rng: np.random.Generator | None = None) -> "Oscillator":
        rng = rng or np.random.default_rng()
        lo, hi = BANDS[band]
        freq = rng.uniform(lo, hi)
        phase = rng.uniform(0, 2 * np.pi)
        return cls(freq=freq, phase=phase, band=band, _rng=rng)


class OscillatorBank:
    """
    Collection of independent oscillators — convenience wrapper for
    generating multi-band signals.
    """

    def __init__(self, oscillators: List[Oscillator] | None = None):
        self.oscillators: List[Oscillator] = oscillators or []

    # ── construction helpers ──────────────────────────────────────
    @classmethod
    def default_brain(cls, rng: np.random.Generator | None = None) -> "OscillatorBank":
        """Create a bank with oscillators in each biological band."""
        rng = rng or np.random.default_rng()
        oscs = [Oscillator.from_band(b, rng=rng) for b in BANDS]
        return cls(oscs)

    # ── dynamics ──────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        for o in self.oscillators:
            o.step(dt)

    def composite_signal(self, t: float) -> float:
        return sum(o.signal(t) for o in self.oscillators)

    def band_power(self, band: str, t_array: np.ndarray) -> float:
        """RMS power of a given band over a time window."""
        band_oscs = [o for o in self.oscillators if o.band == band]
        if not band_oscs:
            return 0.0
        sig = np.array([sum(o.signal(t) for o in band_oscs) for t in t_array])
        return float(np.sqrt(np.mean(sig ** 2)))


class CoupledOscillators:
    """
    Kuramoto-model coupled oscillator network.

    Each oscillator i evolves as:
        dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)

    Parameters
    ----------
    freqs : np.ndarray
        Natural frequencies (N,).
    coupling : float
        Global coupling constant K.
    """

    def __init__(
        self,
        freqs: np.ndarray,
        coupling: float = 1.0,
        phases: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.N = len(freqs)
        self.freqs = np.asarray(freqs, dtype=float)
        self.coupling = coupling
        rng = rng or np.random.default_rng()
        self.phases = phases if phases is not None else rng.uniform(0, 2 * np.pi, self.N)

    # ── simulation ────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        """Euler step of the Kuramoto model."""
        diffs = self.phases[None, :] - self.phases[:, None]  # (N, N)
        interaction = np.sin(diffs).mean(axis=1)
        dtheta = self.freqs + self.coupling * interaction
        self.phases = (self.phases + dtheta * dt) % (2 * np.pi)

    def run(self, duration: float, dt: float = 0.001) -> np.ndarray:
        """Run simulation and return phase history (T, N)."""
        steps = int(duration / dt)
        history = np.zeros((steps, self.N))
        for i in range(steps):
            history[i] = self.phases.copy()
            self.step(dt)
        return history

    # ── order parameter ───────────────────────────────────────────
    def order_parameter(self) -> complex:
        """Kuramoto order parameter r·exp(iψ).  |r| ∈ [0, 1]."""
        return complex(np.mean(np.exp(1j * self.phases)))

    def synchrony(self) -> float:
        """Synchrony level |r| ∈ [0, 1].  1 = fully synchronised."""
        return abs(self.order_parameter())

    # ── cross-frequency coupling ──────────────────────────────────
    @staticmethod
    def phase_amplitude_coupling(
        phase_signal: np.ndarray,
        amplitude_signal: np.ndarray,
        n_bins: int = 18,
    ) -> float:
        """
        Modulation index (Tort et al. 2010) measuring how much the
        amplitude of one oscillation is modulated by the phase of another.

        Returns MI ∈ [0, +∞) where 0 = no coupling.
        """
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        phase_wrapped = np.angle(np.exp(1j * phase_signal))
        digitized = np.digitize(phase_wrapped, bin_edges) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)
        mean_amp = np.array([amplitude_signal[digitized == b].mean() if np.any(digitized == b) else 0 for b in range(n_bins)])
        total = mean_amp.sum()
        if total == 0:
            return 0.0
        p = mean_amp / total
        p = p[p > 0]
        h = -np.sum(p * np.log(p))
        h_max = np.log(n_bins)
        return float((h_max - h) / h_max)
