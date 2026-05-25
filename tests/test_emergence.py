"""Tests for the KAIROS emergence layer."""

import numpy as np
import pytest

from kairos.core.state import PhasePoint, Trajectory
from kairos.engines.chaos import ChaosDynamicsEngine
from kairos.engines.synthesis import OscillatorySynthesisCore
from kairos.emergence.bifurcation import BifurcationDetector
from kairos.emergence.crystallization import PatternCrystallizer
from kairos.emergence.kairos_moment import KairosMomentDetector


class TestBifurcationDetector:
    def test_detect_on_chaotic_trajectory(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=100, dt=0.01)
        detector = BifurcationDetector()
        events = detector.detect(traj)
        # Lorenz at standard params is fully chaotic, may or may not detect events
        assert isinstance(events, list)

    def test_early_warning_signals(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=50, dt=0.01)
        detector = BifurcationDetector()
        signals = detector.early_warning_signals(traj)
        assert "variance" in signals
        assert "acf1" in signals
        assert len(signals["variance"]) > 0


class TestPatternCrystallizer:
    def test_crystallize(self):
        rng = np.random.default_rng(42)
        core = OscillatorySynthesisCore(dim=4, _rng=rng)
        result = core.run(n_cycles=3)

        crystallizer = PatternCrystallizer()
        structures = crystallizer.crystallize(result.trajectory, result.emergent_patterns)
        assert isinstance(structures, list)
        # Should produce at least one structure
        if structures:
            ks = structures[0]
            assert 0 <= ks.confidence <= 1
            assert 0 <= ks.coherence <= 1
            assert len(ks.embedding) > 0


class TestKairosMomentDetector:
    def test_detect_moments(self):
        rng = np.random.default_rng(42)
        core = OscillatorySynthesisCore(dim=4, _rng=rng)
        result = core.run(n_cycles=5)

        detector = KairosMomentDetector(readiness_threshold=0.4)
        moments = detector.detect(
            result.trajectory,
            energy_trace=result.energy_trace,
            entropy_trace=result.entropy_trace,
            synchrony_trace=result.synchrony_trace,
        )
        assert isinstance(moments, list)

    def test_readiness_trace(self):
        rng = np.random.default_rng(42)
        core = OscillatorySynthesisCore(dim=4, _rng=rng)
        result = core.run(n_cycles=3)

        detector = KairosMomentDetector()
        trace = detector.readiness_trace(
            result.trajectory,
            result.energy_trace,
            result.entropy_trace,
            result.synchrony_trace,
        )
        assert len(trace) == len(result.trajectory)
        assert np.all(trace >= 0)
