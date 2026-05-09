"""Simulation engine — orchestrates physics steps, fluctuation, and interaction matrix."""

import torch
import numpy as np
from typing import Optional

from .config import Config
from .particle_types import ParticleState
from .spatial_hash import SpatialHashGPU, NeighbourList
from .physics import build_neighbour_list, compute_forces, integrate


class SimulationEngine:
    """Top-level simulation driver.

    Handles:
    - initialisation / reset
    - per-frame step (build neighbours → compute forces → integrate → wrap)
    - quantum fluctuation (random particle births)
    - interaction-matrix mutation ("重置引力矩阵")
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._select_device()
        print(f"[Engine] Device: {self.device}")

        self.state: Optional[ParticleState] = None
        self.spatial_hash: Optional[SpatialHashGPU] = None
        self.interaction_matrix: Optional[torch.Tensor] = None

        # Scratch neighbour list reused every frame
        self._neighbour_list: Optional[NeighbourList] = None

        self.frame = 0

    # ── Device ─────────────────────────────────────────────────────────────────

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[Engine] GPU: {torch.cuda.get_device_name(0)}  "
                  f"({props.total_mem / 1e9:.1f} GB, "
                  f"{props.multi_processor_count} SMs)")
            return torch.device("cuda")
        print("[Engine] WARNING: CUDA unavailable — CPU fallback active.")
        return torch.device("cpu")

    # ── Init / Reset ──────────────────────────────────────────────────────────

    def initialise(self, particle_count: Optional[int] = None):
        """Create / recreate all GPU tensors and reset the universe."""
        n = particle_count or self.cfg.particle_count
        print(f"[Engine] Initialising {n:,} particles…")

        self.state = ParticleState(
            n=n,
            width=self.cfg.width,
            height=self.cfg.height,
            num_types=self.cfg.num_types,
            device=self.device,
        )

        self.spatial_hash = SpatialHashGPU(self.cfg, self.device)

        # Pre-allocate scratch neighbour list with a generous capacity estimate
        # ~30 neighbours per particle is typical for uniform distribution
        cap = n * 30
        self._neighbour_list = NeighbourList(
            i_indices=torch.empty(cap, dtype=torch.int64, device=self.device),
            j_indices=torch.empty(cap, dtype=torch.int64, device=self.device),
            offsets=torch.empty(n + 1, dtype=torch.int64, device=self.device),
            n_pairs=0,
        )

        self.mutate_matrix()
        self.frame = 0
        print("[Engine] Ready.")

    # ── Interaction Matrix ────────────────────────────────────────────────────

    def mutate_matrix(self, matrix: Optional[np.ndarray] = None):
        """Replace the interaction matrix. Pass a custom matrix or randomise."""
        raw = matrix if matrix is not None else self.cfg.interaction_matrix()
        self.interaction_matrix = torch.from_numpy(raw.astype(np.float32)).to(self.device)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self) -> int:
        """Advance one frame. Returns the frame counter."""
        if self.state is None:
            raise RuntimeError("Engine not initialised — call initialise() first")

        cfg = self.cfg
        st = self.state

        # 1. Reset scratch force buffers
        st.reset_forces()

        # 2. Build neighbour list (O(N), CPU-GPU hybrid)
        neighbour_list = build_neighbour_list(
            pos_x=st.positions_x,
            pos_y=st.positions_y,
            types=st.types,
            spatial_hash=self.spatial_hash,
        )

        # 3. Vectorised GPU force computation
        compute_forces(
            pos_x=st.positions_x,
            pos_y=st.positions_y,
            types=st.types,
            force_x=st.forces_x,
            force_y=st.forces_y,
            interaction_matrix=self.interaction_matrix,
            r_min=cfg.r_min,
            r_max=cfg.r_max,
            width=cfg.width,
            height=cfg.height,
            neighbour_list=neighbour_list,
        )

        # 4. Integrate (Euler + friction + periodic wrap)
        integrate(
            pos_x=st.positions_x,
            pos_y=st.positions_y,
            vel_x=st.velocities_x,
            vel_y=st.velocities_y,
            force_x=st.forces_x,
            force_y=st.forces_y,
            friction=cfg.friction,
            width=cfg.width,
            height=cfg.height,
        )

        # 5. Quantum fluctuation
        if cfg.vacuum_temperature > 0 and self.frame % cfg.fluctuation_interval == 0:
            self._apply_fluctuation()

        self.frame += 1
        return self.frame

    # ── Fluctuation ───────────────────────────────────────────────────────────

    def _apply_fluctuation(self):
        """Spawn new particles near existing clusters (vacuum energy local peaks)."""
        n = self.state.n
        if n == 0:
            return

        # Pick a random seed particle as the fluctuation centre
        seed_idx = torch.randint(0, n, (1,), device=self.device).item()
        cx = self.state.positions_x[seed_idx].item()
        cy = self.state.positions_y[seed_idx].item()

        count = int(self.cfg.vacuum_temperature * 100)
        if count > 0:
            self.state.spawn(
                count=count,
                width=self.cfg.width,
                height=self.cfg.height,
                num_types=self.cfg.num_types,
                cx=cx,
                cy=cy,
                radius=self.cfg.r_max * 0.5,
            )

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if self.state is None:
            return {}
        return {
            "frame": self.frame,
            "particles": self.state.n,
            "matrix_min": float(self.interaction_matrix.min()),
            "matrix_max": float(self.interaction_matrix.max()),
        }
