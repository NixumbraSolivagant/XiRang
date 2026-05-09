"""Renderer — renders particle field to a Pygame window.

Backed by a PyGame Surface whose buffer is shared with a PyTorch tensor
(GPU → CPU zero-copy via pygame.surfarray or direct numpy view when on CPU,
or via a pinned-memory texture upload when on GPU).
"""

import numpy as np
import pygame
import torch
from typing import Optional

from .config import Config

# Colour per particle type — high-saturation RGB
TYPE_COLORS = [
    (255, 60, 60),   # Ψ1 red     —凝聚态基子
    (60, 120, 255),  # Ψ2 blue    —活跃态基子
    (60, 220, 80),   # Ψ3 green   —不对称基子
    (255, 220, 60), # Ψ4 yellow  —催化基子
]


class Renderer:
    """Pygame-based real-time renderer with optional headless mode."""

    def __init__(self, cfg: Config, headless: bool = False):
        self.cfg = cfg
        self.headless = headless

        if headless:
            self.screen: Optional[pygame.Surface] = None
            self._pixel_buf: Optional[np.ndarray] = None
            return

        pygame.init()
        pygame.display.set_caption("XiRang — 粒子生命宇宙")
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        self.clock = pygame.time.Clock()

        # Pixel buffer — we'll draw into this numpy array each frame
        self._pixel_buf = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)

        # Font for HUD
        try:
            self._font = pygame.font.SysFont("monospace", 14)
        except Exception:
            self._font = None

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, pos_x: torch.Tensor, pos_y: torch.Tensor, types: torch.Tensor):
        """Blit particle field to the pygame surface.

        Uses a pixel buffer for maximum throughput — each particle paints one
        pixel; no rectangles or sprites overhead.
        """
        if self.headless or self.screen is None:
            return

        W, H = self.cfg.width, self.cfg.height
        self._pixel_buf.fill(0)   # clear to black

        # Move tensors to CPU and convert to numpy (non-blocking when pinned)
        px_np = pos_x.cpu().numpy()
        py_np = pos_y.cpu().numpy()
        tp_np = types.cpu().numpy()

        buf = self._pixel_buf

        for k in range(len(px_np)):
            ix = int(px_np[k])
            iy = int(py_np[k])
            if 0 <= ix < W and 0 <= iy < H:
                t = tp_np[k] % len(TYPE_COLORS)
                buf[iy, ix] = TYPE_COLORS[t]

        # Upload pixel buffer as a Pygame surface
        surf = pygame.surfarray.make_surface(buf.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        self._draw_hud()

        pygame.display.flip()

    def _draw_hud(self):
        if self._font is None:
            return
        lines = [
            f"FPS : {self.clock.get_fps():.0f}",
            f"Particles: {self.cfg.particle_count:,}",
            f"[R] Reset matrix  [Space] Pause",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (8, 8 + i * 18))

    def tick(self, fps: float = 0):
        """Call once per frame to update the clock."""
        if not self.headless:
            self.clock.tick(fps if fps > 0 else 9999)

    def close(self):
        if not self.headless:
            pygame.quit()

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def screenshot(self, path: str, pos_x, pos_y, types):
        """Write current frame to a PNG file."""
        self.render(pos_x, pos_y, types)
        if self.screen is not None:
            pygame.image.save(self.screen, path)
