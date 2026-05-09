"""Simulation configuration — tune without touching core logic."""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class Config:
    # ── Universe ────────────────────────────────────────────────
    width: int = 1280
    height: int = 720
    particle_count: int = 1_000_000

    # ── Particle types ──────────────────────────────────────────
    num_types: int = 4          # Ψ1..Ψ4  (red, blue, green, yellow)

    # ── Physics ─────────────────────────────────────────────────
    r_min: float = 2.0          # core repulsion radius
    r_max: float = 50.0         # interaction cutoff
    friction: float = 0.05      # fraction of velocity lost per step

    # ── Quantum fluctuation ─────────────────────────────────────
    vacuum_temperature: float = 0.8   # 0..1; higher → more spontaneous births
    fluctuation_interval: int = 30    # frames between fluctuation events

    @property
    def r_mid(self) -> float:
        return (self.r_max + self.r_min) / 2.0

    @property
    def spatial_cell_size(self) -> float:
        return self.r_max

    def interaction_matrix(self) -> np.ndarray:
        """Factory: fresh random 4×4 asymmetric matrix in [-1, 1]."""
        return np.random.uniform(-1, 1, size=(self.num_types, self.num_types))

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in dir(self)
                if not f.startswith('_') and not callable(getattr(self, f))}
