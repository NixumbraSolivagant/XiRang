"""Particle state management — GPU tensors for positions, velocities, types."""

import torch
from typing import Optional


class ParticleState:
    """Holds all per-particle state as GPU tensors.

    Memory layout (all 1-D flat arrays of length N):
    ── positions : (N,) each for x, y
    ── velocities: (N,) each for vx, vy
    ── forces    : (N,) each for fx, fy  (scratch, zeroed each frame)
    ── types     : (N,) int64  ∈ [0, num_types-1]
    """

    def __init__(
        self,
        n: int,
        width: float,
        height: float,
        num_types: int,
        device: torch.device,
    ):
        self.n = n
        self.device = device

        self.positions_x = torch.empty(n, dtype=torch.float32, device=device)
        self.positions_y = torch.empty(n, dtype=torch.float32, device=device)
        self.velocities_x = torch.zeros(n, dtype=torch.float32, device=device)
        self.velocities_y = torch.zeros(n, dtype=torch.float32, device=device)
        self.forces_x = torch.zeros(n, dtype=torch.float32, device=device)
        self.forces_y = torch.zeros(n, dtype=torch.float32, device=device)
        self.types = torch.empty(n, dtype=torch.int64, device=device)

        self._random_init(width, height, num_types)

    def _random_init(self, width: float, height: float, num_types: int):
        self.positions_x.uniform_(0, width)
        self.positions_y.uniform_(0, height)
        self.types = torch.randint(0, num_types, (self.n,), dtype=torch.int64, device=self.device)

    def reset_forces(self):
        self.forces_x.zero_()
        self.forces_y.zero_()

    def wrap(self, width: float, height: float):
        """Periodic boundary: fold particles back into [0, width) × [0, height)."""
        self.positions_x = torch.remainder(self.positions_x, width)
        self.positions_y = torch.remainder(self.positions_y, height)

    def spawn(
        self,
        count: int,
        width: float,
        height: float,
        num_types: int,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        radius: Optional[float] = None,
    ):
        """Add `count` new particles, optionally clustered near (cx, cy, radius)."""
        if count == 0:
            return

        new_n = self.n + count
        # Allocate new buffers
        px = torch.empty(count, dtype=torch.float32, device=self.device)
        py = torch.empty(count, dtype=torch.float32, device=self.device)
        vx = torch.zeros(count, dtype=torch.float32, device=self.device)
        vy = torch.zeros(count, dtype=torch.float32, device=self.device)
        fx = torch.zeros(count, dtype=torch.float32, device=self.device)
        fy = torch.zeros(count, dtype=torch.float32, device=self.device)
        tp = torch.randint(0, num_types, (count,), dtype=torch.int64, device=self.device)

        if radius is not None and cx is not None and cy is not None:
            r = torch.sqrt(torch.rand(count, device=self.device)) * radius
            angle = torch.rand(count, device=self.device) * 2 * 3.14159265
            px[:] = cx + r * torch.cos(angle)
            py[:] = cy + r * torch.sin(angle)
        else:
            px.uniform_(0, width)
            py.uniform_(0, height)

        # Concatenate
        self.positions_x = torch.cat([self.positions_x, px])
        self.positions_y = torch.cat([self.positions_y, py])
        self.velocities_x = torch.cat([self.velocities_x, vx])
        self.velocities_y = torch.cat([self.velocities_y, vy])
        self.forces_x = torch.cat([self.forces_x, fx])
        self.forces_y = torch.cat([self.forces_y, fy])
        self.types = torch.cat([self.types, tp])

        self.n = new_n
