#!/usr/bin/env python3
"""XiRang — GPU Particle Life Simulator

Controls (GUI mode):
  R          — randomise interaction matrix
  Space      — pause / resume
  F          — decrease vacuum temperature
  S          — screenshot
  Q / Esc    — quit

Headless / server:
  python main.py --headless -n 1000000 --steps 500
  ffmpeg -framerate 60 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4

Evolution:
  python main.py --evolve -n 100000
"""

import os
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import argparse
import time

import torch

from particle_life.config import Config
from particle_life.engine import SimulationEngine


# ── GUI renderer (optional) ─────────────────────────────────────────────────

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def _gui_renderer(cfg: Config):
    from particle_life.renderer import Renderer
    return Renderer(cfg, headless=False)


# ── Headless recorder (pure PIL, zero GUI deps) ─────────────────────────────

def _headless_recorder(cfg: Config, output_dir: str, every: int):
    from particle_life.headless_recorder import HeadlessRecorder
    return HeadlessRecorder(cfg, output_dir=output_dir, every=every)


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="XiRang — GPU Particle Life Simulator")

    # Simulation
    p.add_argument("-n", "--particles", type=int, default=1_000_000)
    p.add_argument("-W", "--width", type=int, default=1280)
    p.add_argument("-H", "--height", type=int, default=720)
    p.add_argument("-r", "--r-max", type=float, default=50.0)
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("-f", "--friction", type=float, default=0.05)
    p.add_argument("-v", "--vacuum", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=60)

    # Headless
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="frames")
    p.add_argument("--record-every", type=int, default=1)

    # Matrix persistence
    p.add_argument("--save-matrix", type=str, default=None)
    p.add_argument("--load-matrix", type=str, default=None)

    # Evolution (CMA-ES)
    g = p.add_argument_group("evolution (--evolve)")
    g.add_argument("--evolve", action="store_true",
                   help="Run CMA-ES optimisation (implicitly --headless)")
    g.add_argument("--evolve-steps", type=int, default=120)
    g.add_argument("--evolve-gens", type=int, default=200)
    g.add_argument("--evolve-popsize", type=int, default=0)
    g.add_argument("--evolve-patience", type=int, default=30)
    g.add_argument("--evolve-seed", type=int, default=42)
    g.add_argument("--evolve-sigma", type=float, default=0.5)
    g.add_argument("--evolve-best", type=str, default="best_matrix.npy")
    g.add_argument("--evolve-log", type=str, default="evolve_log.json")
    g.add_argument("--evolve-record", action="store_true",
                   help="Record frames of best matrix at end of evolution")

    return p.parse_args()


# ── GUI loop ───────────────────────────────────────────────────────────────

def run_gui(cfg: Config, args):
    if not PYGAME_AVAILABLE:
        sys.exit("pygame not available — use --headless")

    engine = SimulationEngine(cfg)
    renderer = _gui_renderer(cfg)
    engine.initialise(particle_count=args.particles)

    if args.load_matrix:
        import numpy as np
        engine.mutate_matrix(np.load(args.load_matrix))
        print(f"[Main] Loaded: {args.load_matrix}")

    paused = False
    running = True
    times = []

    while running:
        t0 = time.perf_counter()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_r:
                    engine.mutate_matrix()
                    if args.save_matrix:
                        import numpy as np
                        np.save(args.save_matrix, engine.interaction_matrix.cpu().numpy())
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_f:
                    cfg.vacuum_temperature = max(0.0, cfg.vacuum_temperature - 0.1)
                elif k == pygame.K_s:
                    renderer.screenshot(
                        f"screenshot_{engine.frame:06d}.png",
                        engine.state.positions_x,
                        engine.state.positions_y,
                        engine.state.types,
                    )

        if not paused:
            engine.step()

        renderer.render(
            engine.state.positions_x,
            engine.state.positions_y,
            engine.state.types,
        )
        renderer.tick(fps=args.fps)

        dt = time.perf_counter() - t0
        times.append(dt)
        if len(times) % 120 == 0:
            avg = sum(times[-120:]) / 120 * 1000
            print(f"[Main] frame={engine.frame}  particles={engine.state.n:,}  "
                  f"step={avg:.1f}ms")

    renderer.close()


# ── Headless loop ────────────────────────────────────────────────────────────

def run_headless(cfg: Config, args):
    engine = SimulationEngine(cfg)
    recorder = _headless_recorder(cfg, args.output_dir, args.record_every)
    engine.initialise(particle_count=args.particles)

    if args.load_matrix:
        import numpy as np
        engine.mutate_matrix(np.load(args.load_matrix))
        print(f"[Main] Loaded: {args.load_matrix}")

    print(f"[Headless] {args.particles:,} particles, "
          f"every {args.record_every} frames → {args.output_dir}/")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in range(args.steps):
        engine.step()
        recorder.render(
            engine.state.positions_x,
            engine.state.positions_y,
            engine.state.types,
        )
        recorder.tick()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    print(f"[Headless] {args.steps} steps in {elapsed:.2f}s  "
          f"= {elapsed*1000/args.steps:.2f}ms/step")
    recorder.close()


# ── Evolution loop ───────────────────────────────────────────────────────────

def run_evolve(cfg: Config, args):
    from particle_life.evolver import Evolver, EvolverConfig

    ev_cfg = EvolverConfig(
        eval_steps=args.evolve_steps,
        max_generations=args.evolve_gens,
        patience=args.evolve_patience,
        best_matrix_path=args.evolve_best,
        log_path=args.evolve_log,
        seed=args.evolve_seed,
        sigma=args.evolve_sigma,
    )

    # Override vacuum for evolution — off by default
    cfg.vacuum_temperature = 0.0

    evolver = Evolver(cfg, ev_cfg)
    evolver.evolve()

    # Record best matrix
    if args.evolve_record:
        print(f"[Evolve] Recording best matrix → {args.evolve_best}")
        engine = SimulationEngine(cfg)
        recorder = _headless_recorder(cfg, "evolve_best_frames", 1)
        engine.initialise(particle_count=args.particles)
        engine.mutate_matrix(evolver.get_best_matrix())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(args.steps):
            engine.step()
            recorder.render(
                engine.state.positions_x,
                engine.state.positions_y,
                engine.state.types,
            )
            recorder.tick()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[Evolve] Recorded {args.steps} frames in "
              f"{time.perf_counter() - t_start:.1f}s")
        recorder.close()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = Config(
        width=args.width,
        height=args.height,
        particle_count=args.particles,
        r_min=args.r_min,
        r_max=args.r_max,
        friction=args.friction,
        vacuum_temperature=args.vacuum,
    )

    if args.evolve:
        run_evolve(cfg, args)
    elif args.headless:
        run_headless(cfg, args)
    else:
        run_gui(cfg, args)


if __name__ == "__main__":
    main()
