"""Spatial hash grid — builds flat GPU neighbour lists for O(N) force computation.

Architecture:
  1. Partition particles into grid cells on CPU  (fast, ~10ms for 1M)
  2. Enumerate cell→cell interactions             (all 9 combinations)
  3. For each interacting cell pair, build flat (i, j) index tensors on GPU
  4. physics.py then vectorises force evaluation over all pairs at once

Memory:  O(N × K) where K ≈ avg neighbours per particle  (<< N²)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import Config


@dataclass
class NeighbourList:
    """Flat neighbour list: particle i interacts with neighbours[i : i+count[i]]."""

    # Primary buffers — (total_pairs,)
    i_indices: torch.Tensor = None      # "from" particle index
    j_indices: torch.Tensor = None      # "to"   particle index

    # Offsets into the flat arrays — (N+1,)
    offsets: torch.Tensor = None         # offsets[i] … offsets[i+1]-1 are i's neighbours

    n_pairs: int = 0


@dataclass
class _CellParticles:
    """Particles belonging to one grid cell."""

    indices: np.ndarray          # (k,) particle indices in this cell


class SpatialHashGPU:
    """GPU-friendly spatial hash that returns NeighbourList for force pipeline."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.width = cfg.width
        self.height = cfg.height
        self.cell_size = cfg.r_max
        self.r_max = cfg.r_max

        self.ncols = max(1, int(np.ceil(cfg.width / self.cell_size)))
        self.nrows = max(1, int(np.ceil(cfg.height / self.cell_size)))
        self.ncells = self.ncols * self.nrows

        # Reusable scratch buffers (allocated once, resized if needed)
        self._scratch_offsets = None
        self._scratch_i = None
        self._scratch_j = None

        # 9 directional cell offsets
        self._dc = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        self._dr = [-1, -1, -1, 0, 0, 0, 1, 1, 1]

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, pos_x: torch.Tensor, pos_y: torch.Tensor,
              types: torch.Tensor) -> NeighbourList:
        """Build neighbour list from current positions. Returns NeighbourList."""
        n = pos_x.shape[0]

        # ── CPU: assign particles to cells ────────────────────────────────
        col = (pos_x.cpu().numpy() / self.cell_size).astype(np.int32)
        row = (pos_y.cpu().numpy() / self.cell_size).astype(np.int32)
        col = np.clip(col, 0, self.ncols - 1)
        row = np.clip(row, 0, self.nrows - 1)
        cell_ids = row * self.ncols + col          # (N,)

        # Sort by cell
        order = np.argsort(cell_ids, kind='stable')
        sorted_idx = order
        sorted_cell = cell_ids[order]

        # Bincount → particles per cell
        counts = np.bincount(sorted_cell, minlength=self.ncells)   # (ncells,)
        offsets_np = np.zeros(self.ncells + 1, dtype=np.int64)
        offsets_np[1:] = np.cumsum(counts)
        offsets_np[-1] = n
        offsets_t = torch.from_numpy(offsets_np).to(self.device)

        # Flat (i, j) index pairs — accumulated in a list then concatenated
        i_list: List[torch.Tensor] = []
        j_list: List[torch.Tensor] = []

        half_w = self.width / 2.0
        half_h = self.height / 2.0

        pos_x_np = pos_x.cpu().numpy()
        pos_y_np = pos_y.cpu().numpy()

        for cell_id in range(self.ncells):
            start = offsets_np[cell_id]
            end = offsets_np[cell_id + 1]
            primary = sorted_idx[start:end]          # particle indices in this cell

            if len(primary) == 0:
                continue

            for direction in range(9):
                dc = self._dc[direction]
                dr = self._dr[direction]
                nc_col = (cell_id % self.ncols + dc) % self.ncols
                nc_row = (cell_id // self.ncols + dr) % self.nrows
                neighbour_cell = nc_row * self.ncols + nc_col

                n_start = offsets_np[neighbour_cell]
                n_end = offsets_np[neighbour_cell + 1]
                secondary = sorted_idx[n_start:n_end]

                if len(secondary) == 0:
                    continue

                # Compute pairwise distances
                pi_x = pos_x_np[primary]           # (k,)
                pi_y = pos_y_np[primary]
                pj_x = pos_x_np[secondary]         # (m,)
                pj_y = pos_y_np[secondary]

                dx = pj_x[np.newaxis, :] - pi_x[:, np.newaxis]  # (k, m)
                dy = pj_y[np.newaxis, :] - pi_y[:, np.newaxis]

                # Periodic wrap
                dx = np.where(dx > half_w, dx - self.width, dx)
                dx = np.where(dx < -half_w, dx + self.width, dx)
                dy = np.where(dy > half_h, dy - self.height, dy)
                dy = np.where(dy < -half_h, dy + self.height, dy)

                r2 = dx * dx + dy * dy

                if direction == 4:  # self cell: skip i==j and r<=0
                    r2.flat[::r2.shape[1] + 1] = self.r_max * self.r_max  # mask diagonal

                mask = (0 < r2) & (r2 < self.r_max * self.r_max)
                rows, cols = np.where(mask)

                if len(rows) == 0:
                    continue

                i_flat = primary[rows]
                j_flat = secondary[cols]

                i_list.append(torch.from_numpy(i_flat.astype(np.int64)).to(self.device))
                j_list.append(torch.from_numpy(j_flat.astype(np.int64)).to(self.device))

        if i_list:
            i_all = torch.cat(i_list)
            j_all = torch.cat(j_list)
        else:
            i_all = torch.zeros(0, dtype=torch.int64, device=self.device)
            j_all = torch.zeros(0, dtype=torch.int64, device=self.device)

        return NeighbourList(
            i_indices=i_all,
            j_indices=j_all,
            offsets=offsets_t,
            n_pairs=i_all.shape[0],
        )
