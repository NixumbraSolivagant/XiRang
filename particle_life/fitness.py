"""Fitness evaluation — measures structural complexity of a particle configuration.

Metrics (all computed on GPU):
  1. Local spatial entropy  — clustering strength
  2. Velocity variance       — ordered motion (dissipative structures)
  3. Type diversity          — avoids all-one-type "dead" crystals
  4. Inter-type correlation  — cross-species organization

Fitness = weighted sum of normalised metrics.
"""

import torch
import numpy as np
from typing import Tuple

from .spatial_hash import SpatialHashGPU, NeighbourList


def evaluate(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    vel_x: torch.Tensor,
    vel_y: torch.Tensor,
    types: torch.Tensor,
    spatial_hash: SpatialHashGPU,
    r_max: float,
    num_types: int,
) -> Tuple[float, dict]:
    """Compute fitness and sub-metrics for the current particle state.

    All tensors are on the same device. Returns (fitness, dict of sub-metrics).
    """
    device = pos_x.device
    n = pos_x.shape[0]

    # ── 1. Local spatial entropy (clustering) ─────────────────────────────────
    # For each particle: count same-type neighbours within r_max
    # High entropy in these counts → diverse cluster sizes → complex
    nl = spatial_hash.build(pos_x, pos_y, types)
    if nl.n_pairs == 0:
        return 0.0, {"entropy": 0.0, "vel_var": 0.0, "type_div": 0.0, "inter_type": 0.0}

    i_idx = nl.i_indices
    j_idx = nl.j_indices

    # same-type neighbour fraction per particle
    type_i = types[i_idx]
    type_j = types[j_idx]
    same = (type_i == type_j).float()           # (P,)

    # scatter-add to get per-particle same-type neighbour count
    count = torch.zeros(n, dtype=torch.float32, device=device)
    count.index_add_(0, i_idx, torch.ones_like(same))
    denom = torch.zeros(n, dtype=torch.float32, device=device)
    denom.index_add_(0, i_idx, torch.ones_like(same))
    count = count / denom.clamp(min=1.0)         # fraction of same-type neighbours

    # entropy of distribution of these fractions
    frac = count.float()
    frac_safe = torch.where(frac > 0.01, frac, torch.ones_like(frac) * 0.01)
    entropy = -(frac_safe * torch.log(frac_safe)).mean().item()
    entropy = float(np.clip(entropy, 0.0, 2.0))

    # ── 2. Velocity variance (ordered motion) ─────────────────────────────────
    speed_sq = vel_x * vel_x + vel_y * vel_y
    speed_mean_sq = speed_sq.mean()
    speed_var = speed_sq.var().item()
    vel_var = float(np.clip(speed_var / (speed_mean_sq.item() + 1e-6), 0.0, 10.0))

    # ── 3. Type diversity (per-region Shannon entropy) ─────────────────────────
    # Divide space into grid; compute type entropy per cell, then average
    cell_size = r_max
    ncols = max(1, int(np.ceil(spatial_hash.width / cell_size)))
    nrows = max(1, int(np.ceil(spatial_hash.height / cell_size)))

    col = (pos_x / cell_size).long().clamp(0, ncols - 1)
    row = (pos_y / cell_size).long().clamp(0, nrows - 1)
    cell = row * ncols + col

    # Per-cell type histogram → entropy
    ncells = ncols * nrows
    hist = torch.zeros(ncells * num_types, dtype=torch.float32, device=device)
    # one-hot encode each particle type into the histogram
    idx = cell * num_types + types
    hist.index_add_(0, idx, torch.ones(n, dtype=torch.float32, device=device))

    hist = hist.view(ncells, num_types)          # (ncells, num_types)
    total = hist.sum(dim=1, keepdim=True).clamp(min=1.0)
    prob = hist / total                            # (ncells, num_types)
    # Shannon entropy per cell; mean over cells with particles
    safe_prob = torch.where(prob > 1e-6, prob, torch.ones_like(prob) * 1e-6)
    cell_ent = -(safe_prob * torch.log(safe_prob)).sum(dim=1)
    type_div = cell_ent[total.squeeze() > 1.0].mean().item()
    type_div = float(np.clip(type_div, 0.0, np.log(num_types)))

    # ── 4. Inter-type spatial correlation ─────────────────────────────────────
    # Between-type edge density / within-type edge density
    # "edges" = pairs in neighbour list
    n_same = same.sum().item()
    n_total = same.shape[0]
    n_diff = n_total - n_same
    inter_type = float(n_diff / max(n_total, 1))  # fraction of cross-type interactions

    # ── Combine ────────────────────────────────────────────────────────────────
    # Normalise to [0, 1] ranges (raw values already in roughly [0, 1–2])
    e_norm = entropy / 2.0
    v_norm = min(vel_var / 3.0, 1.0)
    t_norm = type_div / np.log(num_types) if num_types > 1 else 0.0
    i_norm = inter_type

    fitness = 0.35 * e_norm + 0.25 * v_norm + 0.25 * t_norm + 0.15 * i_norm

    metrics = {
        "entropy": entropy,
        "vel_var": vel_var,
        "type_div": type_div,
        "inter_type": inter_type,
        "fitness": float(fitness),
    }
    return float(fitness), metrics
