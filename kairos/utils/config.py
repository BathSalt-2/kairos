"""
Configuration management for KAIROS.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChaosConfig:
    system: str = "lorenz"
    params: Dict[str, float] = field(default_factory=lambda: {"sigma": 10.0, "rho": 28.0, "beta": 2.6667})
    duration: float = 50.0
    dt: float = 0.01


@dataclass
class OscillatorConfig:
    n_oscillators: int = 12
    coupling: float = 2.0
    mode_duration: float = 5.0


@dataclass
class SynthesisConfig:
    dim: int = 4
    n_cycles: int = 10
    divergent_noise: float = 0.5
    convergent_rate: float = 0.3
    integration_strength: float = 0.4


@dataclass
class DetectionConfig:
    readiness_threshold: float = 0.6
    target_entropy: float = 0.5
    min_separation: int = 100


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])
    log_level: str = "info"


@dataclass
class VisualizationConfig:
    theme: str = "dark"
    width: int = 1200
    height: int = 800
    fps: int = 30
    colormap: str = "plasma"


@dataclass
class KairosConfig:
    """Top-level KAIROS configuration."""

    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    seed: Optional[int] = None

    # ── I/O ───────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KairosConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            chaos=ChaosConfig(**data.get("chaos", {})),
            oscillator=OscillatorConfig(**data.get("oscillator", {})),
            synthesis=SynthesisConfig(**data.get("synthesis", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            api=APIConfig(**data.get("api", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
            seed=data.get("seed"),
        )

    @classmethod
    def from_env(cls) -> "KairosConfig":
        """Load config, preferring KAIROS_CONFIG env var if set."""
        config_path = os.environ.get("KAIROS_CONFIG")
        if config_path and Path(config_path).exists():
            return cls.load(config_path)
        return cls()
