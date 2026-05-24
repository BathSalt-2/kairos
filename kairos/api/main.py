"""
KAIROS API Server
=================
FastAPI application exposing the full KAIROS engine via REST endpoints.

Run:
    uvicorn kairos.api.main:app --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up kairos-api
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kairos import __version__
from kairos.core.state import CognitiveState, PhasePoint, Trajectory
from kairos.engines.chaos import ChaosDynamicsEngine, BUILTIN_SYSTEMS
from kairos.engines.phase_space import PhaseSpaceNavigator
from kairos.engines.attractor import AttractorLandscapeMapper
from kairos.engines.synthesis import OscillatorySynthesisCore
from kairos.emergence.bifurcation import BifurcationDetector
from kairos.emergence.crystallization import PatternCrystallizer
from kairos.emergence.kairos_moment import KairosMomentDetector
from kairos.utils.config import KairosConfig

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="KAIROS API",
    description="Kinetic Adaptive Intelligence for Recursive Oscillatory Synthesis",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────

class EvolveRequest(BaseModel):
    system: str = "lorenz"
    params: dict = {}
    initial: List[float] = [1.0, 1.0, 1.0]
    duration: float = 50.0
    dt: float = 0.01

class TrajectoryResponse(BaseModel):
    positions: List[List[float]]
    times: List[float]
    total_length: float
    system: str

class LyapunovRequest(BaseModel):
    system: str = "lorenz"
    params: dict = {}
    initial: List[float] = [1.0, 1.0, 1.0]
    duration: float = 100.0

class LyapunovResponse(BaseModel):
    max_lyapunov: float
    spectrum: List[float]
    is_chaotic: bool

class SensitivityRequest(BaseModel):
    system: str = "lorenz"
    params: dict = {}
    initial: List[float] = [1.0, 1.0, 1.0]
    eps: float = 1e-6
    duration: float = 50.0

class SensitivityResponse(BaseModel):
    divergence: List[float]
    times: List[float]
    max_divergence: float

class SynthesisRequest(BaseModel):
    dim: int = 4
    n_cycles: int = 5
    n_oscillators: int = 12
    coupling: float = 2.0
    divergent_noise: float = 0.5
    convergent_rate: float = 0.3
    seed: Optional[int] = None

class SynthesisResponse(BaseModel):
    n_cycles: int
    trajectory_length: int
    energy_trace: List[float]
    entropy_trace: List[float]
    synchrony_trace: List[float]
    mode_sequence: List[str]
    n_emergent_patterns: int
    n_kairos_moments: int
    kairos_moments: List[dict]
    knowledge_structures: List[dict]

class EmbedRequest(BaseModel):
    signal: List[float]
    dim: int = 3
    tau: Optional[int] = None

class EmbedResponse(BaseModel):
    embedded: List[List[float]]
    tau_used: int
    n_points: int

class BifurcationRequest(BaseModel):
    system: str = "lorenz"
    params: dict = {}
    param_name: str = "rho"
    param_range: List[float] = [0.0, 50.0]
    n_values: int = 150
    duration: float = 100.0
    observe_dim: int = 2

class BifurcationResponse(BaseModel):
    param_values: List[float]
    maxima: List[List[float]]


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "KAIROS", "version": __version__, "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/systems")
def list_systems():
    return {"systems": list(BUILTIN_SYSTEMS.keys())}


@app.post("/evolve", response_model=TrajectoryResponse)
def evolve(req: EvolveRequest):
    """Evolve a chaotic system and return the trajectory."""
    engine = ChaosDynamicsEngine(system=req.system, params=req.params)
    initial = np.array(req.initial)
    traj = engine.evolve(initial, req.duration, req.dt)
    pos = traj.positions_array()
    return TrajectoryResponse(
        positions=pos.tolist(),
        times=traj.times().tolist(),
        total_length=traj.total_length(),
        system=req.system,
    )


@app.post("/lyapunov", response_model=LyapunovResponse)
def lyapunov(req: LyapunovRequest):
    """Compute Lyapunov exponents for a system."""
    engine = ChaosDynamicsEngine(system=req.system, params=req.params)
    initial = np.array(req.initial)
    mle = engine.max_lyapunov(initial, duration=req.duration)
    spectrum = engine.lyapunov_spectrum(initial, duration=req.duration)
    return LyapunovResponse(
        max_lyapunov=mle,
        spectrum=spectrum.tolist(),
        is_chaotic=mle > 0.01,
    )


@app.post("/sensitivity", response_model=SensitivityResponse)
def sensitivity(req: SensitivityRequest):
    """Test sensitive dependence on initial conditions."""
    engine = ChaosDynamicsEngine(system=req.system, params=req.params)
    initial = np.array(req.initial)
    _, _, divergence = engine.sensitivity_test(initial, req.eps, req.duration)
    times = np.linspace(0, req.duration, len(divergence)).tolist()
    return SensitivityResponse(
        divergence=divergence.tolist(),
        times=times,
        max_divergence=float(divergence.max()),
    )


@app.post("/synthesis", response_model=SynthesisResponse)
def synthesis(req: SynthesisRequest):
    """Run Recursive Oscillatory Synthesis."""
    rng = np.random.default_rng(req.seed)
    core = OscillatorySynthesisCore(
        dim=req.dim,
        n_oscillators=req.n_oscillators,
        coupling=req.coupling,
        divergent_noise=req.divergent_noise,
        convergent_rate=req.convergent_rate,
        _rng=rng,
    )
    result = core.run(n_cycles=req.n_cycles)

    # Detect kairos moments
    detector = KairosMomentDetector()
    moments = detector.detect(
        result.trajectory,
        result.states,
        result.energy_trace,
        result.entropy_trace,
        result.synchrony_trace,
    )

    # Crystallize patterns
    crystallizer = PatternCrystallizer()
    structures = crystallizer.crystallize(result.trajectory, result.emergent_patterns)

    return SynthesisResponse(
        n_cycles=result.n_cycles,
        trajectory_length=len(result.trajectory),
        energy_trace=result.energy_trace,
        entropy_trace=result.entropy_trace,
        synchrony_trace=result.synchrony_trace,
        mode_sequence=[m.value for m in result.mode_sequence],
        n_emergent_patterns=len(result.emergent_patterns),
        n_kairos_moments=len(moments),
        kairos_moments=[
            {
                "time": m.time,
                "readiness": round(m.readiness, 4),
                "sensitivity": round(m.sensitivity, 4),
                "coherence": round(m.coherence, 4),
                "entropy": round(m.entropy, 4),
                "recommended_action": m.recommended_action,
            }
            for m in moments
        ],
        knowledge_structures=[
            {
                "id": ks.id,
                "confidence": round(ks.confidence, 4),
                "coherence": round(ks.coherence, 4),
                "n_components": len(ks.components),
            }
            for ks in structures
        ],
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Time-delay embed a scalar signal into phase space."""
    signal = np.array(req.signal)
    nav = PhaseSpaceNavigator()
    tau = req.tau or nav.optimal_tau(signal)
    embedded = nav.time_delay_embed(signal, req.dim, tau)
    return EmbedResponse(
        embedded=embedded.tolist(),
        tau_used=tau,
        n_points=len(embedded),
    )


@app.post("/bifurcation", response_model=BifurcationResponse)
def bifurcation(req: BifurcationRequest):
    """Generate a bifurcation diagram by sweeping a parameter."""
    if req.system not in BUILTIN_SYSTEMS:
        raise HTTPException(400, f"Unknown system: {req.system}")

    rhs = BUILTIN_SYSTEMS[req.system]
    dim = len(req.params) or 3
    if req.system == "kairos_cognitive":
        dim = 4

    mapper = AttractorLandscapeMapper(rhs=rhs, params=req.params, dim=dim)
    param_range = (req.param_range[0], req.param_range[1])
    pvals, maxima = mapper.bifurcation_diagram(
        param_name=req.param_name,
        param_range=param_range,
        n_values=req.n_values,
        duration=req.duration,
        observe_dim=req.observe_dim,
    )
    return BifurcationResponse(
        param_values=pvals.tolist(),
        maxima=[m.tolist() for m in maxima],
    )
