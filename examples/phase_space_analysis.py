#!/usr/bin/env python3
"""
Example: Phase-space reconstruction from scalar time-series data.

Demonstrates Takens' delay embedding, recurrence analysis, and
correlation dimension estimation — useful for analysing real-world
signals (EEG, stock prices, climate data, etc.).
"""

import numpy as np

from kairos.engines.chaos import ChaosDynamicsEngine
from kairos.engines.phase_space import PhaseSpaceNavigator
from kairos.engines.attractor import AttractorLandscapeMapper


def main():
    print("⚡ KAIROS — Phase-Space Analysis Example")
    print("=" * 60)

    # Generate a scalar time-series from the Lorenz attractor (x-component)
    print("\n▸ Generating Lorenz time-series (x-component only)…")
    engine = ChaosDynamicsEngine(system="lorenz")
    traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=100, dt=0.01)
    x_signal = traj.positions_array()[:, 0]
    print(f"  Signal length: {len(x_signal)} samples")

    nav = PhaseSpaceNavigator()

    # Optimal delay
    print("\n▸ Estimating optimal time delay…")
    tau = nav.optimal_tau(x_signal)
    print(f"  τ_opt = {tau} samples")

    # Optimal embedding dimension
    print("\n▸ Estimating embedding dimension (FNN)…")
    dim = nav.optimal_dim(x_signal, tau, max_dim=8)
    print(f"  d_opt = {dim}")

    # Embed
    print(f"\n▸ Embedding signal (d={dim}, τ={tau})…")
    embedded = nav.time_delay_embed(x_signal, dim=dim, tau=tau)
    print(f"  Embedded shape: {embedded.shape}")

    # Recurrence analysis
    print("\n▸ Computing recurrence matrix…")
    R = nav.recurrence_matrix(embedded)
    rr = nav.recurrence_rate(R)
    det = nav.determinism(R)
    print(f"  Recurrence rate: {rr:.4f}")
    print(f"  Determinism:     {det:.4f}")

    # Correlation dimension
    print("\n▸ Estimating correlation dimension (Grassberger–Procaccia)…")
    from kairos.core.state import Trajectory, PhasePoint
    embed_traj = Trajectory(
        points=[PhasePoint(position=embedded[i], time=float(i)) for i in range(len(embedded))]
    )
    d2 = AttractorLandscapeMapper.correlation_dimension(embed_traj, max_points=2000)
    print(f"  D₂ = {d2:.3f}  (expected ≈ 2.05 for Lorenz)")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
