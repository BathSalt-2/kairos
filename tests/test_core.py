"""Tests for kairos.core — state primitives and oscillator models."""

import numpy as np
import pytest

from kairos.core.state import CognitiveState, PhasePoint, Trajectory
from kairos.core.oscillator import (
    CoupledOscillators,
    Oscillator,
    OscillatorBank,
)


class TestPhasePoint:
    def test_creation(self):
        p = PhasePoint(position=np.array([1.0, 2.0, 3.0]))
        assert p.dim == 3

    def test_distance(self):
        a = PhasePoint(position=np.zeros(3))
        b = PhasePoint(position=np.ones(3))
        assert abs(a.distance_to(b) - np.sqrt(3)) < 1e-10


class TestCognitiveState:
    def test_perturb(self):
        p = PhasePoint(position=np.zeros(4))
        state = CognitiveState(phase=p, energy=1.0, entropy=0.5)
        perturbed = state.perturb(sigma=0.1)
        assert not np.allclose(perturbed.position, state.position)

    def test_dominance_vector(self):
        p = PhasePoint(position=np.array([3.0, 0.0, 0.0]))
        state = CognitiveState(phase=p)
        dv = state.dominance_vector()
        assert abs(dv[0] - 1.0) < 1e-10


class TestTrajectory:
    def test_append_and_length(self):
        t = Trajectory()
        for i in range(10):
            t.append(PhasePoint(position=np.random.randn(3), time=float(i)))
        assert len(t) == 10
        assert t.positions_array().shape == (10, 3)

    def test_total_length(self):
        t = Trajectory()
        t.append(PhasePoint(position=np.array([0.0, 0.0]), time=0.0))
        t.append(PhasePoint(position=np.array([3.0, 4.0]), time=1.0))
        assert abs(t.total_length() - 5.0) < 1e-10


class TestOscillator:
    def test_step(self):
        o = Oscillator(freq=10.0)
        old_phase = o.phase
        o.step(0.01)
        assert o.phase != old_phase

    def test_from_band(self):
        o = Oscillator.from_band("gamma")
        assert 30 <= o.freq <= 100


class TestOscillatorBank:
    def test_default_brain(self):
        bank = OscillatorBank.default_brain()
        assert len(bank.oscillators) == 5


class TestCoupledOscillators:
    def test_synchrony_bounds(self):
        co = CoupledOscillators(np.ones(10), coupling=0.0)
        co.phases = np.zeros(10)  # all in sync
        assert abs(co.synchrony() - 1.0) < 1e-10

    def test_run(self):
        co = CoupledOscillators(np.linspace(1, 5, 10), coupling=2.0)
        history = co.run(duration=1.0, dt=0.001)
        assert history.shape == (1000, 10)

    def test_phase_amplitude_coupling(self):
        t = np.linspace(0, 10, 5000)
        phase_sig = np.sin(2 * np.pi * 6 * t)
        amp_sig = 1 + 0.5 * np.sin(2 * np.pi * 6 * t)
        mi = CoupledOscillators.phase_amplitude_coupling(phase_sig, amp_sig)
        assert mi >= 0
