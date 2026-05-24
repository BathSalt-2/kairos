"""KAIROS engine modules — the core computational machinery."""

from .chaos import ChaosDynamicsEngine
from .phase_space import PhaseSpaceNavigator
from .attractor import AttractorLandscapeMapper
from .synthesis import OscillatorySynthesisCore

__all__ = [
    "ChaosDynamicsEngine",
    "PhaseSpaceNavigator",
    "AttractorLandscapeMapper",
    "OscillatorySynthesisCore",
]
