#!/usr/bin/env python3
"""
KAIROS Quick-Start Example
===========================
Demonstrates the core pipeline:
  1. Evolve a chaotic system (Lorenz attractor)
  2. Compute Lyapunov exponents
  3. Run Recursive Oscillatory Synthesis (ROS)
  4. Detect kairos moments
  5. Crystallize emergent patterns
"""

import numpy as np

from kairos.engines.chaos import ChaosDynamicsEngine
from kairos.engines.synthesis import OscillatorySynthesisCore
from kairos.emergence.bifurcation import BifurcationDetector
from kairos.emergence.crystallization import PatternCrystallizer
from kairos.emergence.kairos_moment import KairosMomentDetector


def main():
    print("=" * 60)
    print("⚡ KAIROS — Quick-Start Demo")
    print("=" * 60)

    # ── 1. Chaos Dynamics ─────────────────────────────────────────
    print("\n▸ 1. Evolving Lorenz attractor…")
    engine = ChaosDynamicsEngine(system="lorenz")
    ic = np.array([1.0, 1.0, 1.0])
    traj = engine.evolve(ic, duration=50, dt=0.01)
    print(f"  Trajectory: {len(traj)} points, arc length = {traj.total_length():.2f}")

    # ── 2. Lyapunov exponents ─────────────────────────────────────
    print("\n▸ 2. Computing Lyapunov spectrum…")
    mle = engine.max_lyapunov(ic, duration=80)
    spectrum = engine.lyapunov_spectrum(ic, duration=80)
    print(f"  Max Lyapunov exponent: {mle:.4f}  {'(CHAOTIC ✓)' if mle > 0 else ''}")
    print(f"  Full spectrum: {np.round(spectrum, 4).tolist()}")

    # ── 3. Sensitive dependence ───────────────────────────────────
    print("\n▸ 3. Testing sensitive dependence on initial conditions…")
    _, _, div = engine.sensitivity_test(ic, eps=1e-9, duration=30)
    print(f"  Initial separation: 1e-9")
    print(f"  Final divergence:   {div[-1]:.4f}  (amplification: {div[-1]/1e-9:.2e}×)")

    # ── 4. Recursive Oscillatory Synthesis ────────────────────────
    print("\n▸ 4. Running Recursive Oscillatory Synthesis (10 cycles)…")
    rng = np.random.default_rng(42)
    core = OscillatorySynthesisCore(dim=4, n_oscillators=12, coupling=2.0, _rng=rng)
    result = core.run(n_cycles=10)
    print(f"  Trajectory: {len(result.trajectory)} points")
    print(f"  Mode transitions: {len(result.mode_sequence)} ({' → '.join(m.value[:3] for m in result.mode_sequence[:9])}…)")
    print(f"  Emergent patterns: {len(result.emergent_patterns)}")
    print(f"  Final energy: {result.energy_trace[-1]:.4f}")
    print(f"  Final entropy: {result.entropy_trace[-1]:.4f}")
    print(f"  Final synchrony: {result.synchrony_trace[-1]:.4f}")

    # ── 5. Kairos Moment Detection ────────────────────────────────
    print("\n▸ 5. Detecting kairos moments…")
    detector = KairosMomentDetector(readiness_threshold=0.5)
    moments = detector.detect(
        result.trajectory,
        energy_trace=result.energy_trace,
        entropy_trace=result.entropy_trace,
        synchrony_trace=result.synchrony_trace,
    )
    print(f"  Found {len(moments)} kairos moments:")
    for m in moments[:5]:
        print(
            f"    ⚡ t={m.time:7.2f}  readiness={m.readiness:.3f}  "
            f"sensitivity={m.sensitivity:.3f}  action={m.recommended_action}"
        )
    if len(moments) > 5:
        print(f"    … and {len(moments) - 5} more")

    # ── 6. Pattern Crystallization ────────────────────────────────
    print("\n▸ 6. Crystallizing emergent patterns…")
    crystallizer = PatternCrystallizer()
    structures = crystallizer.crystallize(result.trajectory, result.emergent_patterns)
    print(f"  Crystallized {len(structures)} knowledge structures:")
    for ks in structures[:5]:
        print(
            f"    📐 {ks.id}: confidence={ks.confidence:.3f}, "
            f"coherence={ks.coherence:.3f}, components={len(ks.components)}"
        )

    # ── 7. Bifurcation detection ──────────────────────────────────
    print("\n▸ 7. Scanning for bifurcation events…")
    bif_detector = BifurcationDetector()
    events = bif_detector.detect(result.trajectory)
    print(f"  Detected {len(events)} bifurcation events")
    for ev in events[:3]:
        print(f"    🔀 t={ev.time:.2f}  type={ev.type}  severity={ev.severity:.3f}")

    print("\n" + "=" * 60)
    print("✅ Demo complete! Explore more with `python -m kairos --help`")
    print("=" * 60)


if __name__ == "__main__":
    main()
