# XiRang — GPU Particle Life Simulator

A CUDA-accelerated particle-life simulation engine. Emergent structures (motile clusters, crystals, gaseous states) arise from a 4×4 asymmetric interaction matrix — no biological priors.

---

## Physics

| Component | Specification |
|---|---|
| **Genesis** | Stochastic particle birth via quantum-fluctuation model |
| **Particle types** | 4 species (Ψ₁–Ψ₄), distinguished solely by interaction parameters |
| **Interaction law** | Asymmetric 4×4 matrix `M[i,j] ∈ [-1, 1]`; forces vanish beyond `r_max` |
| **Integration** | Semi-implicit Euler with friction coefficient `μ` |
| **Neighbour search** | Spatial hash grid → O(N) per step (not O(N²)) |
| **Boundary** | Periodic toroidal wrap |

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0.0`, `numpy>=1.24.0`, `pygame>=2.5.0`, `Pillow>=10.0.0`

---

## Usage

**GUI (requires display):**

```bash
python main.py
```

| Key | Action |
|---|---|
| `R` | Randomise interaction matrix |
| `Space` | Pause / resume |
| `F` | Decrease vacuum temperature |
| `S` | Screenshot |
| `Q` / `Esc` | Quit |

**Headless (server / no-GPU):**

```bash
python main.py --headless --particles 1000000 --steps 500 --record-every 1
```

Frames are written to `frames/frame_%06d.png`. Encode to video with:

```bash
ffmpeg -framerate 60 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `-n`, `--particles` | `1000000` | Particle count |
| `-W` | `1280` | Simulation width (px) |
| `-H` | `720` | Simulation height (px) |
| `-r`, `--r-max` | `50.0` | Max interaction radius |
| `--r-min` | `2.0` | Core repulsion radius |
| `-f`, `--friction` | `0.05` | Friction coefficient |
| `-v`, `--vacuum` | `0.8` | Vacuum temperature (0–1) |
| `--fps` | `60` | Target frame rate |
| `--headless` | — | Disable GUI |
| `--steps` | `500` | Steps in headless mode |
| `--output-dir` | `frames` | Frame output directory |
| `--record-every` | `1` | Capture every Nth frame |
| `--save-matrix` | — | Save interaction matrix as `.npy` |
| `--load-matrix` | — | Load interaction matrix from `.npy` |

---

## Architecture

```
XiRang/
├── main.py
├── requirements.txt
└── particle_life/
    ├── config.py            # Physical constants
    ├── particle_types.py    # GPU tensor state
    ├── spatial_hash.py      # O(N) neighbour lookup
    ├── physics.py           # Vectorised force kernel
    ├── fitness.py           # Complexity metrics
    ├── cmaes.py             # CMA-ES optimiser
    ├── evolver.py           # Evolution driver
    ├── engine.py            # Simulation driver
    ├── renderer.py          # Pygame GUI renderer
    └── headless_recorder.py # PIL frame writer
```

---

## Evolution (CMA-ES)

Optimise the 4×4 interaction matrix for maximal structural complexity using
CMA-ES — no neural networks, no gradients, ~16 parameters.

```bash
# Fast evaluation on CPU (100k particles, ~30min on 1 GPU)
python main.py --evolve -n 100000 --evolve-gens 200

# Resume from checkpoint
python main.py --evolve --evolve-best best_matrix.npy
```

**Fitness function** (weighted sum, all on GPU):

| Metric | Weight | Measures |
|---|---|---|
| Local spatial entropy | 0.35 | Clustering strength |
| Velocity variance | 0.25 | Ordered motion |
| Type diversity | 0.25 | Cross-type mixing |
| Inter-type correlation | 0.15 | Cross-species edges |

Record best matrix after evolution:

```bash
python main.py --evolve -n 100000 --evolve-record --steps 300
ffmpeg -framerate 60 -i evolve_best_frames/frame_%06d.png \
       -c:v libx264 -pix_fmt yuv420p evolve_best.mp4
```

---

## Deploying to a GPU Server

```bash
pip install -r requirements.txt
# ensure CUDA / cuDNN drivers are present for PyTorch CUDA support
python main.py --headless --particles 1000000 --steps 500 --record-every 1
```

`pygame` is optional in headless mode; `Pillow` is required.
