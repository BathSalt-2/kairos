<p align="center">
  <img src="https://img.shields.io/badge/Or4cl3_AI_Solutions-KAIROS-FF6B35?style=for-the-badge" alt="KAIROS Badge">
  <img src="https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge" alt="Version Badge">
  <img src="https://img.shields.io/badge/Python-3.10+-00D4AA?style=for-the-badge" alt="Python Badge">
  <img src="https://img.shields.io/badge/Domain-Chaos_Theory_%7C_Neuroscience_%7C_Quantum_AI-6B5BFF?style=for-the-badge" alt="Domain Badge">
</p>

<h1 align="center">⚡ KAIROS</h1>

<p align="center">
  <strong>Kinetic Adaptive Intelligence for Recursive Oscillatory Synthesis</strong>
  <br />
  <em>An AI framework at the intersection of Chaos Theory, Quantum-Inspired Temporal Cognition, and Neuroscience</em>
  <br /><br />
  <a href="#-quick-start">Quick Start</a>
  ·
  <a href="#-architecture">Architecture</a>
  ·
  <a href="#-api-reference">API Reference</a>
  ·
  <a href="https://github.com/BathSalt-2/kairos/issues">Report Bug</a>
</p>

---

## 📖 Overview

**KAIROS** is a research framework by [Or4cl3 AI Solutions](https://github.com/BathSalt-2) that models intelligence as an emergent property of oscillatory dynamics — drawing on chaos theory, temporal neuroscience, and quantum-inspired computation.

The name KAIROS (from the ancient Greek *καιρός* — "the opportune moment") reflects the system's core thesis: intelligent behaviour emerges from the ability to recognise and act at **critical temporal junctures** — moments where small perturbations in chaotic systems lead to radically different outcomes.

> See [`docs/DESIGN.md`](docs/DESIGN.md) for the full theoretical framework.

---

## 🚀 Quick Start

### Install

```bash
# Core (numpy + scipy only)
pip install -e .

# With API server
pip install -e ".[api]"

# With visualisation dashboard
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"
```

### Run the demo

```bash
python examples/quickstart.py
```

### CLI

```bash
# Evolve a chaotic system
kairos evolve --system lorenz --duration 100 --plot

# Compute Lyapunov exponents
kairos lyapunov --system rossler

# Run Recursive Oscillatory Synthesis
kairos synthesis --cycles 10 --plot

# Detect kairos moments
kairos moments --cycles 15 --threshold 0.55

# Generate bifurcation diagram
kairos bifurcation --system lorenz --param rho --range 0 50 --plot

# Start the API server
kairos serve
```

### Docker

```bash
cp .env.example .env
docker compose up --build
# API  → http://localhost:8000
# Dashboard → http://localhost:8501
```

---

## 🏗️ Architecture

```
kairos/
├── core/                      # Foundational primitives
│   ├── state.py               #   PhasePoint, CognitiveState, Trajectory
│   └── oscillator.py          #   Kuramoto oscillators, cross-frequency coupling
│
├── engines/                   # Core computational machinery
│   ├── chaos.py               #   ChaosDynamicsEngine — Lorenz, Rössler, custom ODEs
│   ├── phase_space.py         #   PhaseSpaceNavigator — Takens embedding, recurrence analysis
│   ├── attractor.py           #   AttractorLandscapeMapper — fixed points, basins, bifurcation diagrams
│   └── synthesis.py           #   OscillatorySynthesisCore — Recursive Oscillatory Synthesis (ROS)
│
├── emergence/                 # Emergence layer
│   ├── bifurcation.py         #   BifurcationDetector — early-warning signals, critical transitions
│   ├── crystallization.py     #   PatternCrystallizer — emergent patterns → knowledge structures
│   └── kairos_moment.py       #   KairosMomentDetector — opportune moment recognition
│
├── api/                       # REST API (FastAPI)
│   └── main.py                #   Full API server with all endpoints
│
├── visualization/             # Visualisation toolkit
│   ├── plots.py               #   Publication-quality matplotlib plots
│   └── dashboard.py           #   Interactive Streamlit dashboard
│
├── utils/                     # Configuration & logging
│   ├── config.py              #   KairosConfig with JSON persistence
│   └── logging.py             #   Structured logging setup
│
├── cli.py                     # Command-line interface
└── __main__.py                # python -m kairos support
```

### Component Map

```
┌─────────────────────────────────────────────────────────┐
│                      KAIROS CORE                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          TEMPORAL COGNITION ENGINE                │   │
│  │                                                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │  Chaos     │  │  Phase     │  │  Attractor  │  │   │
│  │  │  Dynamics  │◄─┤  Space     │─►│  Landscape  │  │   │
│  │  │  Engine    │  │  Navigator │  │  Mapper     │  │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │   │
│  │        │               │               │          │   │
│  │        └───────────────┼───────────────┘          │   │
│  │                        ▼                          │   │
│  │              ┌──────────────────┐                 │   │
│  │              │  OSCILLATORY     │                 │   │
│  │              │  SYNTHESIS CORE  │                 │   │
│  │              │  (Recursive ROS) │                 │   │
│  │              └────────┬─────────┘                 │   │
│  │                       │                           │   │
│  └───────────────────────┼───────────────────────────┘   │
│                          │                               │
│  ┌───────────────────────┼───────────────────────────┐   │
│  │         EMERGENCE LAYER                           │   │
│  │                       ▼                           │   │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────────┐  │   │
│  │  │ Bifurcation│ │ Pattern  │ │ Kairos Moment  │  │   │
│  │  │ Detector   │ │ Crystall.│ │ Recognition    │  │   │
│  │  └────────────┘ └──────────┘ └────────────────┘  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Components

### Chaos Dynamics Engine

Integrates chaotic dynamical systems with full diagnostics:

```python
from kairos.engines.chaos import ChaosDynamicsEngine
import numpy as np

engine = ChaosDynamicsEngine(system="lorenz")
traj = engine.evolve(np.array([1.0, 1.0, 1.0]), duration=100)

# Lyapunov exponents (positive = chaos)
mle = engine.max_lyapunov(np.array([1.0, 1.0, 1.0]))
spectrum = engine.lyapunov_spectrum(np.array([1.0, 1.0, 1.0]))

# Sensitive dependence test
traj1, traj2, divergence = engine.sensitivity_test(np.array([1.0, 1.0, 1.0]))
```

Built-in systems: `lorenz`, `rossler`, `kairos_cognitive` (4D). Pass any callable for custom ODEs.

### Oscillatory Synthesis Core (ROS)

The heart of KAIROS — recursive cycling through divergent, convergent, and integrative cognitive modes, driven by coupled Kuramoto oscillators:

```python
from kairos.engines.synthesis import OscillatorySynthesisCore

core = OscillatorySynthesisCore(dim=4, n_oscillators=12, coupling=2.0)
result = core.run(n_cycles=10)

# result.trajectory — full phase-space trajectory
# result.energy_trace — energy over time
# result.entropy_trace — entropy over time
# result.synchrony_trace — oscillator synchrony
# result.emergent_patterns — extracted patterns per cycle
```

### Kairos Moment Detection

Detects opportune moments for intervention by fusing bifurcation proximity, oscillatory coherence, entropy, and energy:

```python
from kairos.emergence.kairos_moment import KairosMomentDetector

detector = KairosMomentDetector(readiness_threshold=0.6)
moments = detector.detect(
    result.trajectory,
    energy_trace=result.energy_trace,
    entropy_trace=result.entropy_trace,
    synchrony_trace=result.synchrony_trace,
)

for m in moments:
    print(f"⚡ t={m.time:.2f}  readiness={m.readiness:.3f}  action={m.recommended_action}")
```

### Phase-Space Navigator

Takens' delay embedding, recurrence analysis, and trajectory metrics:

```python
from kairos.engines.phase_space import PhaseSpaceNavigator

nav = PhaseSpaceNavigator()
tau = nav.optimal_tau(signal)
dim = nav.optimal_dim(signal, tau)
embedded = nav.time_delay_embed(signal, dim=dim, tau=tau)
R = nav.recurrence_matrix(embedded)
```

### Attractor Landscape Mapper

Fixed-point detection, classification, basin mapping, and bifurcation diagrams:

```python
from kairos.engines.attractor import AttractorLandscapeMapper

mapper = AttractorLandscapeMapper(rhs=my_system, params={}, dim=3)
fps = mapper.find_fixed_points()
for fp in fps:
    print(mapper.classify_fixed_point(fp))  # stable_node, saddle, etc.

pvals, maxima = mapper.bifurcation_diagram("rho", (0, 50))
```

---

## 🌐 API Reference

Start the server: `kairos serve` or `docker compose up`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Status & version |
| `/health` | GET | Health check |
| `/systems` | GET | List available dynamical systems |
| `/evolve` | POST | Evolve a chaotic system → trajectory |
| `/lyapunov` | POST | Compute Lyapunov exponents |
| `/sensitivity` | POST | Sensitive-dependence test |
| `/synthesis` | POST | Run full ROS pipeline + kairos moments + crystallization |
| `/embed` | POST | Time-delay embed a scalar signal |
| `/bifurcation` | POST | Generate bifurcation diagram |

### Example: cURL

```bash
# Evolve Lorenz attractor
curl -X POST http://localhost:8000/evolve \
  -H "Content-Type: application/json" \
  -d '{"system": "lorenz", "initial": [1,1,1], "duration": 50}'

# Run full synthesis
curl -X POST http://localhost:8000/synthesis \
  -H "Content-Type: application/json" \
  -d '{"dim": 4, "n_cycles": 5, "seed": 42}'
```

---

## 📊 Visualisation

### Static plots (matplotlib)

```python
from kairos.visualization.plots import (
    plot_trajectory_3d,
    plot_bifurcation,
    plot_synthesis_dashboard,
    plot_lyapunov_spectrum,
    plot_recurrence,
)

plot_trajectory_3d(traj, save_path="lorenz.png")
plot_synthesis_dashboard(energy, entropy, synchrony, modes, save_path="dashboard.png")
```

### Interactive dashboard (Streamlit)

```bash
streamlit run kairos/visualization/dashboard.py
# → http://localhost:8501
```

Three tabs:
- **🌀 Chaos Engine** — 3D attractor exploration with real-time Lyapunov computation
- **🔄 Oscillatory Synthesis** — ROS energy/entropy/synchrony dashboard
- **⏱ Kairos Moments** — Readiness trace with detected opportune moments

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest
```

---

## 🔗 Related Projects

| Project | Description | Link |
|---|---|---|
| **Quantum Ethics** | Ethical decision-making through quantum superposition of moral frameworks | [Quantum-Ethics-](https://github.com/BathSalt-2/Quantum-Ethics-) |
| **QSCI** | Quantum Intelligence Hub | [qsci-sigma-matrix-framework](https://github.com/BathSalt-2/qsci-sigma-matrix-framework) |
| **QuantumMind** | Mobile-first AI with integrated quantum simulator | [QuantumMind](https://github.com/BathSalt-2/QuantumMind) |
| **Daedalus** | Sentinel-class synthetic intelligence | [Daedalus](https://github.com/BathSalt-2/Daedalus) |
| **NeuroForge** | AI-powered adaptive learning platform | [NeuroForge-](https://github.com/BathSalt-2/NeuroForge-) |

---

## 🚀 Roadmap

- [x] Theoretical framework definition
- [x] Core architecture design
- [x] Chaos Dynamics Engine (Lorenz, Rössler, custom KAIROS cognitive ODE)
- [x] Phase Space Navigator (Takens embedding, recurrence, FNN)
- [x] Attractor Landscape Mapper (fixed points, basins, bifurcation diagrams)
- [x] Oscillatory Synthesis Core (Recursive Oscillatory Synthesis)
- [x] Bifurcation detection (early-warning signals, critical slowing down)
- [x] Pattern Crystallization (emergent patterns → knowledge structures)
- [x] Kairos Moment recognition (opportune-moment detection)
- [x] REST API (FastAPI)
- [x] Interactive dashboard (Streamlit + Plotly)
- [x] CLI with all commands
- [x] Docker Compose deployment
- [x] Test suite
- [ ] Integration with Or4cl3 ecosystem (QSCI, Daedalus)
- [ ] GPU-accelerated large-scale simulations
- [ ] Real-time EEG/signal ingestion pipeline
- [ ] WebSocket streaming for live attractor visualisation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b research/your-contribution`)
3. Commit your changes
4. Push and open a Pull Request

Key areas: nonlinear dynamics, computational neuroscience, visualisation, benchmarking.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📬 Contact

**Dustin Groves** — Founder, Or4cl3 AI Solutions

- GitHub: [@BathSalt-2](https://github.com/BathSalt-2)
- Email: [bathsaltone@gmail.com](mailto:bathsaltone@gmail.com)

---

<p align="center">
  <em>"The opportune moment is not found — it is recognised by a mind attuned to the rhythms of chaos."</em>
  <br />— <strong>KAIROS</strong>
</p>
