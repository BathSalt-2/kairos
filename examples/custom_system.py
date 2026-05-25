#!/usr/bin/env python3
"""
Example: Defining and exploring a custom dynamical system with KAIROS.
"""

import numpy as np

from kairos.engines.chaos import ChaosDynamicsEngine
from kairos.engines.attractor import AttractorLandscapeMapper


def double_scroll(t, state, alpha=9.0, beta=14.286):
    """Chua's double-scroll attractor — a classic chaotic circuit."""
    x, y, z = state
    h = -1.143 * x + 0.5 * (-0.714 + 1.143) * (abs(x + 1) - abs(x - 1))
    dx = alpha * (y - x - h)
    dy = x - y + z
    dz = -beta * y
    return np.array([dx, dy, dz])


def main():
    print("⚡ KAIROS — Custom System Example (Chua's Double Scroll)")
    print("=" * 60)

    # Use the custom system with ChaosDynamicsEngine
    engine = ChaosDynamicsEngine(system=double_scroll)
    ic = np.array([0.1, 0.0, 0.0])
    traj = engine.evolve(ic, duration=100, dt=0.005)

    print(f"Trajectory: {len(traj)} points, arc length = {traj.total_length():.2f}")

    # Lyapunov
    mle = engine.max_lyapunov(ic, duration=200, dt=0.005)
    print(f"Max Lyapunov exponent: {mle:.4f}  {'(CHAOTIC)' if mle > 0 else '(non-chaotic)'}")

    # Find fixed points
    mapper = AttractorLandscapeMapper(rhs=double_scroll, params={}, dim=3)
    fps = mapper.find_fixed_points(search_bounds=(-5, 5), n_starts=200)
    print(f"\nFixed points found: {len(fps)}")
    for i, fp in enumerate(fps):
        cls = mapper.classify_fixed_point(fp)
        print(f"  FP{i}: {np.round(fp, 4)}  →  {cls}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
