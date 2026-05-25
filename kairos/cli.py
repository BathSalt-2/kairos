"""
KAIROS Command-Line Interface
==============================
Run experiments, compute diagnostics, and generate visualizations from the
terminal.

Usage:
    python -m kairos evolve --system lorenz --duration 100 --plot
    python -m kairos lyapunov --system rossler
    python -m kairos synthesis --cycles 10 --plot
    python -m kairos moments --cycles 15 --threshold 0.55
    python -m kairos bifurcation --system lorenz --param rho --range 0 50
    python -m kairos serve
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def cmd_evolve(args: argparse.Namespace) -> None:
    from kairos.engines.chaos import ChaosDynamicsEngine
    engine = ChaosDynamicsEngine(system=args.system, params=json.loads(args.params) if args.params else {})
    initial = np.array(json.loads(args.initial)) if args.initial else _default_ic(args.system)
    traj = engine.evolve(initial, args.duration, args.dt)
    print(f"Evolved {args.system} for {args.duration}s → {len(traj)} points, arc length {traj.total_length():.2f}")

    if args.plot:
        from kairos.visualization.plots import plot_trajectory_3d
        save = args.output or f"{args.system}_trajectory.png"
        plot_trajectory_3d(traj, title=f"{args.system.title()} Attractor", save_path=save)
        print(f"Saved plot → {save}")


def cmd_lyapunov(args: argparse.Namespace) -> None:
    from kairos.engines.chaos import ChaosDynamicsEngine
    engine = ChaosDynamicsEngine(system=args.system, params=json.loads(args.params) if args.params else {})
    initial = np.array(json.loads(args.initial)) if args.initial else _default_ic(args.system)

    mle = engine.max_lyapunov(initial, duration=args.duration)
    spectrum = engine.lyapunov_spectrum(initial, duration=args.duration)

    print(f"System: {args.system}")
    print(f"Max Lyapunov exponent: {mle:.6f}  {'(CHAOTIC)' if mle > 0 else '(non-chaotic)'}")
    print(f"Full spectrum: {np.round(spectrum, 4).tolist()}")

    if args.plot:
        from kairos.visualization.plots import plot_lyapunov_spectrum
        save = args.output or f"{args.system}_lyapunov.png"
        plot_lyapunov_spectrum(spectrum, title=f"{args.system.title()} Lyapunov Spectrum", save_path=save)
        print(f"Saved plot → {save}")


def cmd_synthesis(args: argparse.Namespace) -> None:
    from kairos.engines.synthesis import OscillatorySynthesisCore
    from kairos.emergence.crystallization import PatternCrystallizer

    rng = np.random.default_rng(args.seed)
    core = OscillatorySynthesisCore(
        dim=args.dim,
        n_oscillators=args.oscillators,
        coupling=args.coupling,
        divergent_noise=args.noise,
        _rng=rng,
    )
    result = core.run(n_cycles=args.cycles)

    crystallizer = PatternCrystallizer()
    structures = crystallizer.crystallize(result.trajectory, result.emergent_patterns)

    print(f"ROS completed: {args.cycles} cycles, {len(result.trajectory)} trajectory points")
    print(f"Emergent patterns: {len(result.emergent_patterns)}")
    print(f"Crystallized structures: {len(structures)}")
    for ks in structures:
        print(f"  {ks.id}: confidence={ks.confidence:.3f}, coherence={ks.coherence:.3f}, components={len(ks.components)}")

    if args.plot:
        from kairos.visualization.plots import plot_synthesis_dashboard
        save = args.output or "synthesis_dashboard.png"
        plot_synthesis_dashboard(
            result.energy_trace, result.entropy_trace, result.synchrony_trace,
            [m.value for m in result.mode_sequence],
            title="KAIROS Synthesis Dashboard", save_path=save,
        )
        print(f"Saved plot → {save}")


def cmd_moments(args: argparse.Namespace) -> None:
    from kairos.engines.synthesis import OscillatorySynthesisCore
    from kairos.emergence.kairos_moment import KairosMomentDetector

    rng = np.random.default_rng(args.seed)
    core = OscillatorySynthesisCore(dim=4, _rng=rng)
    result = core.run(n_cycles=args.cycles)

    detector = KairosMomentDetector(readiness_threshold=args.threshold)
    moments = detector.detect(
        result.trajectory,
        energy_trace=result.energy_trace,
        entropy_trace=result.entropy_trace,
        synchrony_trace=result.synchrony_trace,
    )

    print(f"Detected {len(moments)} kairos moments (threshold={args.threshold})")
    for m in moments:
        print(
            f"  ⚡ t={m.time:.2f}  readiness={m.readiness:.3f}  "
            f"sensitivity={m.sensitivity:.3f}  coherence={m.coherence:.3f}  "
            f"action={m.recommended_action}"
        )


def cmd_bifurcation(args: argparse.Namespace) -> None:
    from kairos.engines.chaos import BUILTIN_SYSTEMS
    from kairos.engines.attractor import AttractorLandscapeMapper

    rhs = BUILTIN_SYSTEMS[args.system]
    dim = 4 if args.system == "kairos_cognitive" else 3
    mapper = AttractorLandscapeMapper(rhs=rhs, params={}, dim=dim)
    pvals, maxima = mapper.bifurcation_diagram(
        param_name=args.param,
        param_range=(args.range[0], args.range[1]),
        n_values=args.n_values,
        duration=args.duration,
    )
    total_maxima = sum(len(m) for m in maxima)
    print(f"Bifurcation diagram: {len(pvals)} parameter values, {total_maxima} total maxima captured")

    if args.plot:
        from kairos.visualization.plots import plot_bifurcation
        save = args.output or f"{args.system}_bifurcation.png"
        plot_bifurcation(pvals, maxima, param_name=args.param, save_path=save)
        print(f"Saved plot → {save}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    print(f"Starting KAIROS API on {args.host}:{args.port}")
    uvicorn.run("kairos.api.main:app", host=args.host, port=args.port, reload=args.reload)


def _default_ic(system: str) -> np.ndarray:
    if system == "kairos_cognitive":
        return np.array([0.5, 0.5, 0.5, 0.5])
    return np.array([1.0, 1.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser(prog="kairos", description="KAIROS — Cognitive Dynamics Framework")
    sub = parser.add_subparsers(dest="command")

    # evolve
    p = sub.add_parser("evolve", help="Evolve a chaotic system")
    p.add_argument("--system", default="lorenz", choices=["lorenz", "rossler", "kairos_cognitive"])
    p.add_argument("--params", default=None, help="JSON dict of parameters")
    p.add_argument("--initial", default=None, help="JSON list for initial conditions")
    p.add_argument("--duration", type=float, default=50)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--output", default=None)

    # lyapunov
    p = sub.add_parser("lyapunov", help="Compute Lyapunov exponents")
    p.add_argument("--system", default="lorenz", choices=["lorenz", "rossler", "kairos_cognitive"])
    p.add_argument("--params", default=None)
    p.add_argument("--initial", default=None)
    p.add_argument("--duration", type=float, default=100)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--output", default=None)

    # synthesis
    p = sub.add_parser("synthesis", help="Run Recursive Oscillatory Synthesis")
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--oscillators", type=int, default=12)
    p.add_argument("--coupling", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--output", default=None)

    # moments
    p = sub.add_parser("moments", help="Detect kairos moments")
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)

    # bifurcation
    p = sub.add_parser("bifurcation", help="Generate bifurcation diagram")
    p.add_argument("--system", default="lorenz", choices=["lorenz", "rossler", "kairos_cognitive"])
    p.add_argument("--param", default="rho")
    p.add_argument("--range", nargs=2, type=float, default=[0, 50])
    p.add_argument("--n-values", type=int, default=150)
    p.add_argument("--duration", type=float, default=100)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--output", default=None)

    # serve
    p = sub.add_parser("serve", help="Start the API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "evolve": cmd_evolve,
        "lyapunov": cmd_lyapunov,
        "synthesis": cmd_synthesis,
        "moments": cmd_moments,
        "bifurcation": cmd_bifurcation,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
