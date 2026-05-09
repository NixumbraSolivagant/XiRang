"""Headless renderer — pure-PIL PNG writer, zero GUI dependencies.

Writes every N-th frame to disk as a PNG, so you can run the simulation
on a GPU server with no display and encode the sequence with ffmpeg afterwards:

    ffmpeg -framerate 60 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
"""

import os
from typing import Optional

import numpy as np
import torch

from .config import Config

# Colour per particle type — high-saturation RGB
TYPE_COLORS = [
    (255, 60, 60),   # Ψ1 red     —凝聚态基子
    (60, 120, 255),  # Ψ2 blue    —活跃态基子
    (60, 220, 80),   # Ψ3 green   —不对称基子
    (255, 220, 60),  # Ψ4 yellow  —催化基子
]


class HeadlessRecorder:
    """Pure-PIL frame writer — no pygame, no display required."""

    def __init__(self, cfg: Config, output_dir: str = "frames", every: int = 1):
        self.cfg = cfg
        self.output_dir = output_dir
        self.every = every
        self.frame_count = 0

        os.makedirs(output_dir, exist_ok=True)

        # Scratch pixel buffer reused every frame
        self._buf = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)

        self._color_array = np.array(TYPE_COLORS, dtype=np.uint8)  # (4, 3)

    def render(self, pos_x: torch.Tensor, pos_y: torch.Tensor, types: torch.Tensor):
        """Write a PNG frame to disk if this frame should be captured."""
        if self.frame_count % self.every != 0:
            self.frame_count += 1
            return

        W, H = self.cfg.width, self.cfg.height
        self._buf.fill(0)

        px = pos_x.cpu().numpy()
        py = pos_y.cpu().numpy()
        tp = types.cpu().numpy()

        # Clip to screen bounds
        ix = np.clip(px.astype(np.int32), 0, W - 1)
        iy = np.clip(py.astype(np.int32), 0, H - 1)

        # Vectorised colour lookup
        colors = self._color_array[tp % len(TYPE_COLORS)]      # (N, 3)
        self._buf[iy, ix] = colors

        from PIL import Image
        img = Image.fromarray(self._buf, mode="RGB")
        path = os.path.join(self.output_dir, f"frame_{self.frame_count // self.every:06d}.png")
        img.save(path, optimize=False)

    def tick(self, fps: float = 0):
        self.frame_count += 1

    def close(self):
        pass
