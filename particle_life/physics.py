"""Physics kernel — vectorised GPU force computation and integration.

Algorithm:
  1. spatial_hash.build()  →  NeighbourList (i_indices, j_indices)
  2. Gather positions of each pair  →  (pos_i, pos_j)  [GPU tensor scatter]
  3. Vectorised distance & force evaluation  →  fx, fy  (GPU)
  4. Scatter-add into per-particle force buffers
  5. Semi-implicit Euler integration with friction + periodic wrap

Complexity: O(total_pairs) ≈ O(N × K)  where K ≈ neighbours per particle.
"""

import torch
from .spatial_hash import SpatialHashGPU, NeighbourList


def build_neighbour_list(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    types: torch.Tensor,
    spatial_hash: SpatialHashGPU,
) -> NeighbourList:
    """Alias — delegates to SpatialHashGPU.build."""
    return spatial_hash.build(pos_x, pos_y, types)


def compute_forces(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    types: torch.Tensor,
    force_x: torch.Tensor,
    force_y: torch.Tensor,
    interaction_matrix: torch.Tensor,
    r_min: float,
    r_max: float,
    width: float,
    height: float,
    neighbour_list: NeighbourList,
) -> None:
    """Vectorised GPU force computation over all pairs in neighbour_list.

    Interaction law (asymmetric matrix M[i,j]):
      r < r_min         →  core repulsion  F = -(r/r_min - 1)
      r_min ≤ r ≤ r_max →  bell-shaped     F = M[i,j] * (1 - |r - r_mid| / half_span)
      r > r_max         →  0  (already filtered by spatial hash)
    Newton's 3rd law applied: F_ij on j = -F_ij on i.
    """
    if neighbour_list.n_pairs == 0:
        return

    device = pos_x.device
    i_idx = neighbour_list.i_indices          # (P,)
    j_idx = neighbour_list.j_indices          # (P,)

    # Gather positions for each pair
    xi = pos_x[i_idx]                          # (P,)
    yi = pos_y[i_idx]
    xj = pos_x[j_idx]
    yj = pos_y[j_idx]

    # Delta vectors with periodic wrap
    dx = xj - xi
    dy = yj - yi
    half_w = width / 2.0
    half_h = height / 2.0
    dx = dx + width  * ((dx < -half_w).float() - (dx > half_w).float())
    dy = dy + height * ((dy < -half_h).float() - (dy > half_h).float())

    # Distances
    r = torch.sqrt(dx * dx + dy * dy)         # (P,)  r > 0 guaranteed
    r_safe = torch.where(r > 1e-8, r, torch.ones_like(r))

    # Unit direction vectors
    nx = dx / r_safe
    ny = dy / r_safe

    # Gather interaction coefficients  M[types[i], types[j]]
    ti = types[i_idx]                          # (P,)
    tj = types[j_idx]
    coef = interaction_matrix[ti, tj]        # (P,)  asymmetric

    # ── Force magnitude ───────────────────────────────────────────────────
    r_mid = (r_max + r_min) / 2.0
    half_span = r_mid - r_min

    # Core repulsion zone  r < r_min
    core_mask = r < r_min
    core_f = torch.where(core_mask, (r / r_min) - 1.0, torch.zeros_like(r))  # always ≤ 0
    core_f = -core_f                            # positive → repulsive direction

    # Bell-shaped interaction zone  r_min ≤ r ≤ r_max
    bell_mask = (r >= r_min) & (r <= r_max)
    bell_f = torch.where(
        bell_mask,
        coef * (1.0 - torch.abs(r - r_mid) / half_span),
        torch.zeros_like(r),
    )

    f_scalar = core_f + bell_f                  # (P,)  signed: +attract, -repel

    # ── Force vectors ─────────────────────────────────────────────────────
    fx = f_scalar * nx                          # (P,)
    fy = f_scalar * ny

    # ── Scatter-add into per-particle buffers (Newton's 3rd law) ───────────
    # force on i from j: +fx, +fy
    # force on j from i: -fx, -fy
    force_x.index_add_(0, i_idx, fx)
    force_y.index_add_(0, i_idx, fy)
    force_x.index_add_(0, j_idx, -fx)
    force_y.index_add_(0, j_idx, -fy)


def integrate(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    vel_x: torch.Tensor,
    vel_y: torch.Tensor,
    force_x: torch.Tensor,
    force_y: torch.Tensor,
    friction: float,
    width: float,
    height: float,
) -> None:
    """Semi-implicit Euler with friction and periodic boundary wrap (in-place)."""
    vel_x.add_(force_x)
    vel_y.add_(force_y)

    vel_x.mul_(1.0 - friction)
    vel_y.mul_(1.0 - friction)

    pos_x.add_(vel_x)
    pos_y.add_(vel_y)

    # Periodic wrap — done in-place via masked assignment
    pos_x.rem_(width)
    pos_y.rem_(height)
    # Clamp any negative remainders that rem_ can produce on some PyTorch versions
    pos_x.clamp_(min=0.0)
    pos_y.clamp_(min=0.0)
