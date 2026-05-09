"""Evolutionary optimiser — CMA-ES driving the particle-life simulation.

Ties together:
  1. CMA-ES  — black-box optimiser for the 4×4 interaction matrix
  2. SimulationEngine — evaluates a given matrix
  3. fitness.evaluate — measures structural complexity

Each generation:
  1. CMA-ES asks for λ candidate matrices
  2. Each matrix is evaluated by running the simulation for `eval_steps` frames
  3. CMA-ES is told the fitnesses; distribution shifts toward complexity

Memory footprint is constant — only one simulation engine is kept alive.
"""

import os
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from .config import Config
from .engine import SimulationEngine
from .fitness import evaluate
from .cmaes import CMAES


@dataclass
class EvolverConfig:
    popsize: int = 0        # 0 = auto (CMA-ES default)
    eval_steps: int = 120    # simulation steps per fitness evaluation
    max_generations: int = 200
    patience: int = 30       # generations without improvement before restart
    best_matrix_path: str = "best_matrix.npy"
    log_path: str = "evolve_log.json"
    seed: int = 42
    sigma: float = 0.5       # initial CMA-ES step size


class Evolver:
    """CMA-ES-driven interaction-matrix optimiser for structural complexity."""

    def __init__(self, cfg: Config, ev_cfg: EvolverConfig):
        self.cfg = cfg
        self.ev_cfg = ev_cfg
        self.device = self._select_device()
        print(f"[Evolver] Device: {self.device}")

        # Simulation engine (shared, re-initialised per evaluation batch)
        self.engine = SimulationEngine(cfg)
        self.engine.initialise(particle_count=cfg.particle_count)

        # Spatial hash is reused each step inside fitness.evaluate
        self._spatial_hash = self.engine.spatial_hash

        # CMA-ES on the flattened 4×4 interaction matrix (16 parameters)
        self.matrix_dim = cfg.num_types ** 2          # 16
        self.cma = CMAES(
            dim=self.matrix_dim,
            sigma=ev_cfg.sigma,
            seed=ev_cfg.seed,
            popsize=ev_cfg.popsize,
        )
        # Overwrite mean with a bounded initial distribution
        rng = np.random.default_rng(ev_cfg.seed)
        self.cma.state.mean = rng.uniform(-0.5, 0.5, size=self.matrix_dim)

        self.history: List[dict] = []
        self.best_fitness = -np.inf
        self.best_matrix: Optional[np.ndarray] = None

    # ── Device ─────────────────────────────────────────────────────────────────

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            print(f"[Evolver] GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        print("[Evolver] WARNING: running on CPU")
        return torch.device("cpu")

    # ── Core loop ─────────────────────────────────────────────────────────────

    def evolve(self):
        """Run CMA-ES for max_generations or until convergence."""
        ev = self.ev_cfg
        engine = self.engine
        fitness_fn = self._make_fitness_fn()

        gen = 0
        stagnant = 0
        popsize = ev.popsize or self.cma.popsize

        print(f"[Evolver] Starting: {ev.max_generations} gens × {popsize} pop = "
              f"{ev.max_generations * popsize} total evaluations")
        print(f"[Evolver] eval_steps={ev.eval_steps}, patience={ev.patience}")

        t0 = time.perf_counter()

        while gen < ev.max_generations:
            # 1. Sample population
            matrices, z_pop = self.cma.ask()

            # 2. Evaluate all candidates
            fitnesses: List[float] = []
            for i, mat in enumerate(matrices):
                # Reset particle positions before each evaluation
                engine.reset_state()
                f, metrics = fitness_fn(mat)
                fitnesses.append(f)

            # 3. CMA-ES update
            self.cma.tell(matrices, fitnesses, z_pop)

            best_idx = int(np.argmin(fitnesses))      # minimiser
            gen_best = fitnesses[best_idx]
            gen_best_mat = matrices[best_idx]

            # 4. Track best (CMA-ES minimises negative-fitness, so > means better)
            improved = False
            if gen_best > self.best_fitness:   # less negative = higher real fitness
                self.best_fitness = gen_best
                self.best_matrix = gen_best_mat.copy()
                improved = True
                stagnant = 0
                np.save(ev.best_matrix_path, self._vec_to_mat(self.best_matrix))
            else:
                stagnant += 1

            # 5. Log
            entry = {
                "gen": gen,
                "pop_best": float(gen_best),
                "pop_mean": float(np.mean(fitnesses)),
                "sigma": float(self.cma.state.sigma),
                "counteval": self.cma.state.counteval,
                "improved": improved,
                "stagnant": stagnant,
            }
            self.history.append(entry)

            # 6. Progress report
            if gen % max(1, ev.max_generations // 20) == 0 or improved:
                elapsed = time.perf_counter() - t0
                print(f"[Evolver] gen={gen:3d}  "
                      f"best={self.best_fitness:.4f}  "
                      f"pop_mean={entry['pop_mean']:.4f}  "
                      f"σ={entry['sigma']:.3f}  "
                      f"evals={entry['counteval']}  "
                      f"{elapsed:.0f}s")

            # 7. Restart if converged
            if stagnant >= ev.patience:
                print(f"[Evolver] Stagnant for {stagnant} generations — restarting CMA-ES")
                self.cma = CMAES(
                    dim=self.matrix_dim,
                    sigma=ev.sigma,
                    seed=ev.seed + gen,
                    popsize=ev.popsize,
                )
                rng = np.random.default_rng(ev.seed + gen)
                self.cma.state.mean = self.best_matrix.copy() + rng.uniform(
                    -0.1, 0.1, size=self.matrix_dim
                )
                stagnant = 0

            gen += 1

        # Save log
        with open(ev.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        total = time.perf_counter() - t0
        print(f"\n[Evolver] Done in {total:.1f}s  "
              f"best_fitness={self.best_fitness:.4f}  "
              f"total_evals={self.cma.state.counteval}")
        print(f"[Evolver] Best matrix → {ev.best_matrix_path}")
        print(f"[Evolver] Log → {ev.log_path}")

    # ── Fitness function ─────────────────────────────────────────────────────

    def _make_fitness_fn(self):
        """Return a callable fitness(matrix_vec) → (fitness, metrics)."""
        cfg = self.cfg
        engine = self.engine
        ev_cfg = self.ev_cfg
        spatial_hash = self._spatial_hash

        def _fitness(mat_vec: np.ndarray) -> Tuple[float, dict]:
            # Convert flat vector → 4×4 matrix and apply to engine
            matrix = self._vec_to_mat(mat_vec)
            engine.mutate_matrix(matrix)

            # Run simulation for eval_steps
            for _ in range(ev_cfg.eval_steps):
                engine.step()

            # Evaluate
            st = engine.state
            f, metrics = evaluate(
                pos_x=st.positions_x,
                pos_y=st.positions_y,
                vel_x=st.velocities_x,
                vel_y=st.velocities_y,
                types=st.types,
                spatial_hash=spatial_hash,
                r_max=cfg.r_max,
                num_types=cfg.num_types,
            )
            return -f, metrics              # CMA-ES minimises

        return _fitness

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _vec_to_mat(self, vec: np.ndarray) -> np.ndarray:
        """Flat 16-D vector → 4×4 asymmetric matrix in [-1, 1]."""
        mat = vec.reshape(self.cfg.num_types, self.cfg.num_types)
        return np.clip(mat, -1.0, 1.0)

    def get_best_matrix(self) -> np.ndarray:
        return self._vec_to_mat(self.best_matrix)
