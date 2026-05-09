"""CMA-ES — Covariance Matrix Adaptation Evolution Strategy.

Pure NumPy implementation; no dependencies beyond NumPy.
Reference: Hansen & Ostermeier 2001, "Completely Derandomized Self-Adaptation in Evolution Strategies".

Usage:
    opt = CMAES(dim=16, sigma=0.3, seed=42)
    while not done:
        pop, z_pop = opt.ask()
        fitnesses = [fitness(x) for x in pop]
        opt.tell(pop, fitnesses, z_pop)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CMAState:
    mean: np.ndarray          # (dim,)
    sigma: float
    pc: np.ndarray            # (dim,)
    ps: np.ndarray            # (dim,)
    C: np.ndarray             # (dim, dim)
    B: np.ndarray             # (dim, dim)
    D: np.ndarray             # (dim,)
    invsqrtC: np.ndarray      # (dim, dim)
    counteval: int


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy optimiser."""

    def __init__(
        self,
        dim: int,
        sigma: float = 0.3,
        seed: Optional[int] = None,
        popsize: int = 0,
    ):
        self.dim = dim
        self.popsize = popsize or max(4, int(4 + 3 * np.log(dim)))
        self.rng = np.random.default_rng(seed)

        # Learning rates
        self.mu = self.popsize // 2
        weights = np.array([
            max(np.log(self.mu + 0.5) - np.log(i + 1), 0.0)
            for i in range(self.mu)
        ], dtype=np.float64)
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / (weights ** 2).sum()

        self.cc  = 4.0 / (dim + 4.0)
        self.cs  = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.c1  = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff)
                        / ((dim + 2) ** 2 + 2 * self.mu_eff))
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        self.state = CMAState(
            mean=self.rng.standard_normal(dim),
            sigma=sigma,
            pc=np.zeros(dim),
            ps=np.zeros(dim),
            C=np.eye(dim),
            B=np.eye(dim),
            D=np.ones(dim),
            invsqrtC=np.eye(dim),
            counteval=0,
        )
        self._z_pop: Optional[List[np.ndarray]] = None   # noise vectors from ask()

    def ask(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample population. Returns (solutions, noise_vectors)."""
        s = self.state
        pop = []
        z_pop = []
        for _ in range(self.popsize):
            z = self.rng.standard_normal(self.dim)
            z_pop.append(z.copy())
            pop.append(s.mean + s.sigma * (s.B @ (s.D * z)))
        self._z_pop = z_pop
        return pop, z_pop

    def tell(self, solutions: List[np.ndarray], fitnesses: List[float],
             z_pop: Optional[List[np.ndarray]] = None):
        """Update distribution. z_pop from the matching ask() call is used if not given."""
        if z_pop is None:
            z_pop = self._z_pop
        if z_pop is None:
            raise ValueError("z_pop required — call ask() before tell() or pass z_pop explicitly")

        s = self.state
        dim = self.dim

        # Sort by fitness (minimiser)
        order = np.argsort(fitnesses)
        sorted_sols = np.stack([solutions[i] for i in order])           # (lam, dim)
        sorted_z    = np.stack([z_pop[i]    for i in order])           # (lam, dim)
        best_sols   = sorted_sols[:self.mu]                            # (mu, dim)
        best_z      = sorted_z[:self.mu]                               # (mu, dim)

        old_mean = s.mean.copy()
        s.mean = self.weights @ best_sols                              # (dim,)

        # Rank-μ covariance update
        delta_mean = (s.mean - old_mean) / s.sigma                    # (dim,)
        rank_mu = sum(w * np.outer(d, d) for w, d in zip(self.weights, best_z))
        s.C = ((1 - self.c1 - self.cmu) * s.C
               + self.c1 * np.outer(s.pc, s.pc)
               + self.cmu * rank_mu)

        # pc update
        s.pc = (1 - self.cc) * s.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * delta_mean

        # ps update — use weighted mean of noise vectors
        z_mean = self.weights @ best_z                                 # (dim,)
        s.ps = (1 - self.cs) * s.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (s.invsqrtC @ z_mean)

        # Adaptive step size
        sigma_delta = self.cs * (np.linalg.norm(s.ps) / self.chi_n - 1)
        s.sigma = float(np.clip(s.sigma * np.exp(np.clip(sigma_delta, -10, 10)), 1e-10, 10.0))

        # Eigendecomposition
        s.D, s.B = np.linalg.eigh(s.C)
        s.D = np.sqrt(np.maximum(s.D, 1e-10))
        s.invsqrtC = s.B @ np.diag(1.0 / s.D) @ s.B.T

        s.counteval += self.popsize

    @property
    def best(self) -> np.ndarray:
        return self.state.mean.copy()

    def save(self) -> dict:
        s = self.state
        return {
            "mean": s.mean.copy(), "sigma": float(s.sigma),
            "C": s.C.copy(), "pc": s.pc.copy(),
            "ps": s.ps.copy(), "counteval": s.counteval,
        }

    def load(self, state: dict):
        s = self.state
        s.mean = state["mean"].copy()
        s.sigma = float(state["sigma"])
        s.C = state["C"].copy()
        s.pc = state["pc"].copy()
        s.ps = state["ps"].copy()
        s.counteval = state["counteval"]
        s.D, s.B = np.linalg.eigh(s.C)
        s.D = np.sqrt(np.maximum(s.D, 1e-10))
        s.invsqrtC = s.B @ np.diag(1.0 / s.D) @ s.B.T
