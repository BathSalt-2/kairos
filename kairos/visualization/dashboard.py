"""
Interactive Streamlit Dashboard for KAIROS
==========================================
Run:
    streamlit run kairos/visualization/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def main() -> None:
    if not HAS_STREAMLIT:
        print("Install streamlit and plotly: pip install streamlit plotly")
        return

    from kairos.engines.chaos import ChaosDynamicsEngine, BUILTIN_SYSTEMS
    from kairos.engines.synthesis import OscillatorySynthesisCore
    from kairos.emergence.bifurcation import BifurcationDetector
    from kairos.emergence.kairos_moment import KairosMomentDetector

    st.set_page_config(page_title="KAIROS Dashboard", layout="wide", page_icon="⚡")

    st.markdown("""
    <style>
    .stApp { background-color: #0D1117; }
    h1, h2, h3 { color: #FF6B35 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚡ KAIROS — Cognitive Dynamics Explorer")

    tab1, tab2, tab3 = st.tabs(["🌀 Chaos Engine", "🔄 Oscillatory Synthesis", "⏱ Kairos Moments"])

    # ── Tab 1: Chaos Engine ───────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 3])

        with col1:
            system = st.selectbox("Dynamical System", list(BUILTIN_SYSTEMS.keys()))
            duration = st.slider("Duration (s)", 10, 200, 50)
            dt = st.select_slider("Time step", [0.001, 0.005, 0.01, 0.02], value=0.01)

            if system == "lorenz":
                sigma = st.slider("σ", 1.0, 20.0, 10.0)
                rho = st.slider("ρ", 0.0, 50.0, 28.0)
                beta = st.slider("β", 0.1, 5.0, 8 / 3)
                params = {"sigma": sigma, "rho": rho, "beta": beta}
                initial = np.array([1.0, 1.0, 1.0])
            elif system == "rossler":
                a = st.slider("a", 0.0, 1.0, 0.2)
                b = st.slider("b", 0.0, 1.0, 0.2)
                c = st.slider("c", 1.0, 20.0, 5.7)
                params = {"a": a, "b": b, "c": c}
                initial = np.array([1.0, 1.0, 0.0])
            else:
                alpha = st.slider("α", 0.1, 3.0, 1.0)
                beta_p = st.slider("β_cog", 0.1, 3.0, 0.8)
                gamma = st.slider("γ", 0.1, 3.0, 0.6)
                params = {"alpha": alpha, "beta": beta_p, "gamma": gamma}
                initial = np.array([0.5, 0.5, 0.5, 0.5])

            run_chaos = st.button("▶ Evolve System", key="chaos_run")

        with col2:
            if run_chaos:
                with st.spinner("Computing trajectory…"):
                    engine = ChaosDynamicsEngine(system=system, params=params)
                    traj = engine.evolve(initial, duration, dt)
                    pos = traj.positions_array()

                    # 3D scatter
                    fig = go.Figure(data=[go.Scatter3d(
                        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                        mode="lines",
                        line=dict(width=1, color=np.linspace(0, 1, len(pos)), colorscale="Plasma"),
                    )])
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0D1117",
                        plot_bgcolor="#0D1117",
                        title=f"{system.title()} Attractor",
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Lyapunov
                    mle = engine.max_lyapunov(initial, duration=min(100, duration))
                    st.metric("Max Lyapunov Exponent", f"{mle:.4f}", delta="Chaotic ✓" if mle > 0 else "Not chaotic")

    # ── Tab 2: ROS ────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([1, 3])

        with col1:
            dim = st.slider("Cognitive Dimensions", 2, 8, 4)
            n_cycles = st.slider("ROS Cycles", 1, 20, 5)
            n_osc = st.slider("Oscillators", 4, 30, 12)
            coupling = st.slider("Coupling K", 0.1, 5.0, 2.0)
            noise = st.slider("Divergent Noise", 0.1, 2.0, 0.5)
            seed = st.number_input("Seed", value=42)
            run_ros = st.button("▶ Run Synthesis", key="ros_run")

        with col2:
            if run_ros:
                with st.spinner("Running Recursive Oscillatory Synthesis…"):
                    rng = np.random.default_rng(seed)
                    core = OscillatorySynthesisCore(
                        dim=dim, n_oscillators=n_osc, coupling=coupling,
                        divergent_noise=noise, _rng=rng,
                    )
                    result = core.run(n_cycles=n_cycles)
                    pos = result.trajectory.positions_array()

                    # Dashboard subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=["Energy", "Entropy", "Synchrony"],
                        shared_xaxes=True,
                        vertical_spacing=0.06,
                    )
                    t = list(range(len(result.energy_trace)))
                    fig.add_trace(go.Scatter(x=t, y=result.energy_trace, line=dict(color="#FF6B35", width=1), name="Energy"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=t, y=result.entropy_trace, line=dict(color="#00D4AA", width=1), name="Entropy"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=t, y=result.synchrony_trace, line=dict(color="#6B5BFF", width=1), name="Synchrony"), row=3, col=1)
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.info(f"Generated {len(result.emergent_patterns)} emergent patterns across {n_cycles} cycles")

    # ── Tab 3: Kairos Moments ─────────────────────────────────────
    with tab3:
        col1, col2 = st.columns([1, 3])

        with col1:
            threshold = st.slider("Readiness Threshold", 0.3, 0.9, 0.6)
            target_ent = st.slider("Target Entropy", 0.2, 0.8, 0.5)
            km_cycles = st.slider("Cycles", 3, 20, 8, key="km_cycles")
            km_seed = st.number_input("Seed", value=123, key="km_seed")
            run_km = st.button("▶ Detect Moments", key="km_run")

        with col2:
            if run_km:
                with st.spinner("Running synthesis and detecting kairos moments…"):
                    rng = np.random.default_rng(km_seed)
                    core = OscillatorySynthesisCore(dim=4, _rng=rng)
                    result = core.run(n_cycles=km_cycles)

                    detector = KairosMomentDetector(
                        readiness_threshold=threshold,
                        target_entropy=target_ent,
                    )
                    moments = detector.detect(
                        result.trajectory,
                        energy_trace=result.energy_trace,
                        entropy_trace=result.entropy_trace,
                        synchrony_trace=result.synchrony_trace,
                    )
                    readiness = detector.readiness_trace(
                        result.trajectory,
                        result.energy_trace,
                        result.entropy_trace,
                        result.synchrony_trace,
                    )

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=readiness, mode="lines", line=dict(color="#00D4AA", width=1), name="Readiness"))
                    fig.add_hline(y=threshold, line_dash="dash", line_color="#FF6B35", annotation_text="Threshold")

                    for m in moments:
                        fig.add_vline(x=m.index, line_dash="dot", line_color="#FFD93D", opacity=0.6)
                        fig.add_annotation(x=m.index, y=m.readiness, text=f"⚡{m.readiness:.2f}", showarrow=True, arrowcolor="#FFD93D")

                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                        title=f"Kairos Moments Detected: {len(moments)}",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if moments:
                        st.subheader("Detected Moments")
                        for m in moments:
                            st.markdown(
                                f"⚡ **t={m.time:.2f}** — readiness={m.readiness:.3f}, "
                                f"sensitivity={m.sensitivity:.3f}, coherence={m.coherence:.3f}, "
                                f"action=`{m.recommended_action}` (magnitude {m.action_magnitude:.1f})"
                            )
                    else:
                        st.warning("No kairos moments detected. Try lowering the threshold or increasing cycles.")


if __name__ == "__main__":
    main()
