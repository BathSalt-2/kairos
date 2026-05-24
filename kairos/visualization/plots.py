"""
KAIROS Visualization Library
=============================
Publication-quality plots for phase spaces, attractors, bifurcation diagrams,
oscillatory traces, and kairos-moment overlays.

Uses matplotlib with optional Plotly export for interactive 3D.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from kairos.core.state import Trajectory
from kairos.emergence.bifurcation import BifurcationEvent
from kairos.emergence.kairos_moment import KairosMoment


# ── colour palette ────────────────────────────────────────────────
KAIROS_COLORS = {
    "primary": "#FF6B35",
    "secondary": "#00D4AA",
    "accent": "#6B5BFF",
    "divergent": "#FF6B6B",
    "convergent": "#4ECDC4",
    "integrative": "#FFD93D",
    "bg_dark": "#0D1117",
    "text": "#E6EDF3",
    "grid": "#21262D",
}


def _apply_kairos_style(ax: "plt.Axes") -> None:
    ax.set_facecolor(KAIROS_COLORS["bg_dark"])
    ax.tick_params(colors=KAIROS_COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(KAIROS_COLORS["grid"])
    ax.xaxis.label.set_color(KAIROS_COLORS["text"])
    ax.yaxis.label.set_color(KAIROS_COLORS["text"])
    ax.title.set_color(KAIROS_COLORS["text"])


def plot_trajectory_3d(
    trajectory: Trajectory,
    dims: Tuple[int, int, int] = (0, 1, 2),
    title: str = "Phase-Space Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    colorby: str = "time",
) -> Optional["plt.Figure"]:
    """3D phase-space trajectory coloured by time or velocity."""
    if not HAS_MPL:
        return None

    pos = trajectory.positions_array()
    fig = plt.figure(figsize=figsize, facecolor=KAIROS_COLORS["bg_dark"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(KAIROS_COLORS["bg_dark"])

    x, y, z = pos[:, dims[0]], pos[:, dims[1]], pos[:, dims[2]]

    if colorby == "time":
        c = np.linspace(0, 1, len(x))
    else:
        c = np.linalg.norm(np.diff(pos[:, list(dims)], axis=0), axis=1)
        c = np.append(c, c[-1])
        c = c / c.max()

    scatter = ax.scatter(x, y, z, c=c, cmap="plasma", s=0.3, alpha=0.7)
    ax.set_xlabel(f"Dim {dims[0]}", color=KAIROS_COLORS["text"])
    ax.set_ylabel(f"Dim {dims[1]}", color=KAIROS_COLORS["text"])
    ax.set_zlabel(f"Dim {dims[2]}", color=KAIROS_COLORS["text"])
    ax.set_title(title, color=KAIROS_COLORS["primary"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=KAIROS_COLORS["text"])
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="Time" if colorby == "time" else "Speed")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_bifurcation(
    param_values: np.ndarray,
    maxima_list: List[np.ndarray],
    param_name: str = "ρ",
    title: str = "Bifurcation Diagram",
    save_path: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """Plot a bifurcation diagram."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=KAIROS_COLORS["bg_dark"])
    _apply_kairos_style(ax)

    for pval, maxima in zip(param_values, maxima_list):
        if len(maxima) > 0:
            ax.plot(
                [pval] * len(maxima),
                maxima,
                ".",
                color=KAIROS_COLORS["secondary"],
                markersize=0.3,
                alpha=0.5,
            )

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Local Maxima", fontsize=12)
    ax.set_title(title, color=KAIROS_COLORS["primary"], fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_recurrence(
    R: np.ndarray,
    title: str = "Recurrence Plot",
    save_path: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """Plot a recurrence matrix."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=KAIROS_COLORS["bg_dark"])
    _apply_kairos_style(ax)
    ax.imshow(R, cmap="binary", origin="lower", aspect="auto")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Time index")
    ax.set_title(title, color=KAIROS_COLORS["primary"], fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_synthesis_dashboard(
    energy_trace: List[float],
    entropy_trace: List[float],
    synchrony_trace: List[float],
    mode_sequence: List[str],
    kairos_moments: List[KairosMoment] | None = None,
    title: str = "KAIROS Synthesis Dashboard",
    save_path: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """
    Multi-panel dashboard showing energy, entropy, synchrony, mode,
    and kairos moments from an ROS run.
    """
    if not HAS_MPL:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor=KAIROS_COLORS["bg_dark"], sharex=True)
    fig.suptitle(title, color=KAIROS_COLORS["primary"], fontsize=16, fontweight="bold")

    t = np.arange(len(energy_trace))

    # Energy
    ax = axes[0]
    _apply_kairos_style(ax)
    ax.plot(t, energy_trace, color=KAIROS_COLORS["primary"], linewidth=0.5)
    ax.set_ylabel("Energy")
    ax.fill_between(t, energy_trace, alpha=0.2, color=KAIROS_COLORS["primary"])

    # Entropy
    ax = axes[1]
    _apply_kairos_style(ax)
    ax.plot(t, entropy_trace, color=KAIROS_COLORS["secondary"], linewidth=0.5)
    ax.set_ylabel("Entropy")
    ax.axhline(0.5, color=KAIROS_COLORS["grid"], linestyle="--", alpha=0.5)

    # Synchrony
    ax = axes[2]
    _apply_kairos_style(ax)
    ax.plot(t, synchrony_trace, color=KAIROS_COLORS["accent"], linewidth=0.5)
    ax.set_ylabel("Synchrony")

    # Kairos moments overlay
    if kairos_moments:
        for moment in kairos_moments:
            for a in axes[:3]:
                a.axvline(moment.index, color=KAIROS_COLORS["divergent"], alpha=0.4, linewidth=1)

    # Mode sequence (bottom)
    ax = axes[3]
    _apply_kairos_style(ax)
    mode_colors = {"divergent": KAIROS_COLORS["divergent"], "convergent": KAIROS_COLORS["convergent"], "integrative": KAIROS_COLORS["integrative"]}
    # Create colour bar for modes
    steps_per_mode = len(t) // max(len(mode_sequence), 1)
    for i, mode in enumerate(mode_sequence):
        start = i * steps_per_mode
        end = (i + 1) * steps_per_mode
        ax.axvspan(start, end, color=mode_colors.get(mode, "#888"), alpha=0.4)
    ax.set_ylabel("Mode")
    ax.set_xlabel("Step")
    ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_lyapunov_spectrum(
    spectrum: np.ndarray,
    title: str = "Lyapunov Spectrum",
    save_path: Optional[str] = None,
) -> Optional["plt.Figure"]:
    """Bar chart of the Lyapunov spectrum."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=KAIROS_COLORS["bg_dark"])
    _apply_kairos_style(ax)

    colors = [KAIROS_COLORS["divergent"] if v > 0 else KAIROS_COLORS["convergent"] for v in spectrum]
    ax.bar(range(len(spectrum)), spectrum, color=colors, edgecolor=KAIROS_COLORS["grid"])
    ax.axhline(0, color=KAIROS_COLORS["text"], linewidth=0.5)
    ax.set_xlabel("Exponent index")
    ax.set_ylabel("λ")
    ax.set_title(title, color=KAIROS_COLORS["primary"], fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig
