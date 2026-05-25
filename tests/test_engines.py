"""Tests for KAIROS engine modules."""

import numpy as np
import pytest

from kairos.engines.chaos import ChaosDynamicsEngine
from kairos.engines.phase_space import PhaseSpaceNavigator
from kairos.engines.attractor import AttractorLandscapeMapper
from kairos.engines.chaos import lorenz
from kairos.engines.synthesis import OscillatorySynthesisCore


class TestChaosDynamicsEngine:
    def test_lorenz_evolve(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=10, dt=0.01)
        assert len(traj) > 900
        assert traj.positions_array().shape[1] == 3

    def test_rossler_evolve(self):
        engine = ChaosDynamicsEngine(system="rossler")
        traj = engine.evolve(np.array([1.0, 1.0, 0.0]), duration=10, dt=0.01)
        assert len(traj) > 900

    def test_kairos_cognitive(self):
        engine = ChaosDynamicsEngine(system="kairos_cognitive")
        traj = engine.evolve(np.array([0.5, 0.5, 0.5, 0.5]), duration=10, dt=0.01)
        assert traj.positions_array().shape[1] == 4

    def test_max_lyapunov_lorenz(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        mle = engine.max_lyapunov(np.array([1.0, 1.0, 1.0]), duration=50)
        assert mle > 0  # Lorenz is chaotic

    def test_lyapunov_spectrum(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        spectrum = engine.lyapunov_spectrum(np.array([1.0, 1.0, 1.0]), duration=50)
        assert len(spectrum) == 3
        assert spectrum[0] > 0  # at least one positive

    def test_sensitivity(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        t1, t2, div = engine.sensitivity_test(np.array([1.0, 1.0, 1.0]), duration=20)
        assert len(div) > 0
        assert div[-1] > div[0]  # divergence should increase


class TestPhaseSpaceNavigator:
    def test_time_delay_embed(self):
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
        embedded = PhaseSpaceNavigator.time_delay_embed(signal, dim=3, tau=10)
        assert embedded.shape[1] == 3
        assert embedded.shape[0] == 1000 - 2 * 10

    def test_optimal_tau(self):
        signal = np.sin(np.linspace(0, 20 * np.pi, 2000))
        tau = PhaseSpaceNavigator.optimal_tau(signal)
        assert 1 <= tau <= 100

    def test_recurrence_matrix(self):
        from kairos.core.state import PhasePoint, Trajectory
        traj = Trajectory()
        for i in range(50):
            traj.append(PhasePoint(position=np.array([np.sin(i * 0.1), np.cos(i * 0.1)]), time=float(i)))
        R = PhaseSpaceNavigator.recurrence_matrix(traj)
        assert R.shape == (50, 50)
        assert R.dtype == bool

    def test_recurrence_rate(self):
        R = np.eye(10, dtype=bool)
        rate = PhaseSpaceNavigator.recurrence_rate(R)
        assert rate == 0.0  # only diagonal

    def test_hausdorff(self):
        from kairos.core.state import PhasePoint, Trajectory
        t1 = Trajectory(points=[PhasePoint(position=np.array([float(i), 0.0])) for i in range(10)])
        t2 = Trajectory(points=[PhasePoint(position=np.array([float(i), 1.0])) for i in range(10)])
        d = PhaseSpaceNavigator.hausdorff_distance(t1, t2)
        assert abs(d - 1.0) < 1e-10


class TestAttractorLandscapeMapper:
    def test_find_fixed_points(self):
        from kairos.engines.chaos import lorenz
        mapper = AttractorLandscapeMapper(rhs=lorenz, params={"sigma": 10, "rho": 28, "beta": 8 / 3}, dim=3)
        fps = mapper.find_fixed_points(n_starts=100)
        assert len(fps) >= 1  # Lorenz has 3 fixed points

    def test_correlation_dimension(self):
        engine = ChaosDynamicsEngine(system="lorenz")
        traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=50)
        dim = AttractorLandscapeMapper.correlation_dimension(traj)
        assert 1.0 < dim < 3.0  # Lorenz ≈ 2.06


class TestOscillatorySynthesisCore:
    def test_run(self):
        rng = np.random.default_rng(42)
        core = OscillatorySynthesisCore(dim=4, n_oscillators=8, _rng=rng)
        result = core.run(n_cycles=2)
        assert result.n_cycles == 2
        assert len(result.trajectory) > 0
        assert len(result.energy_trace) > 0
        assert len(result.mode_sequence) == 6  # 3 modes × 2 cycles
        assert len(result.emergent_patterns) == 2
