"""Core dynamical-system primitives used across all engines."""

from .state import CognitiveState, PhasePoint, Trajectory
from .oscillator import Oscillator, OscillatorBank, CoupledOscillators

__all__ = [
    "CognitiveState",
    "PhasePoint",
    "Trajectory",
    "Oscillator",
    "OscillatorBank",
    "CoupledOscillators",
]
