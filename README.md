# Diffusion Simulation Project

A Python-based implementation of the **DDPM (Denoising Diffusion Probabilistic Model)** forward diffusion process. Originally built from a buggy script (`second.py`) that was fixed and enhanced into a full diffusion simulation pipeline.

## Project Structure

```
├── second_fixed.py          # Enhanced DDPM forward diffusion + reverse process + 3-panel plot
├── run_simulations.py       # Run 100 independent simulations → outputs CSV
├── analyze_simulations.py   # Analyze CSV → outputs 4-panel analysis plot
├── diffusion_simulations.csv # (generated) 600-row CSV with all run data
├── diffusion_plot*.png      # (generated) Diffusion trajectory plots
├── analysis_plot*.png       # (generated) Analysis plots from CSV
│
├── Dockerfile               # Docker image definition
├── entrypoint.sh            # Container entrypoint with command dispatch
├── docker-compose.yml       # One-command orchestration
├── requirements.txt         # Python dependencies
├── .dockerignore            # Files excluded from Docker build context.
└── README.md                # This file
```

## Running Locally (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual scripts
python second_fixed.py            # Forward diffusion + reverse + plot
python run_simulations.py         # 100-run simulation → CSV
python analyze_simulations.py     # CSV analysis → plot

# Or run all at once:
python second_fixed.py && python run_simulations.py && python analyze_simulations.py
```

All PNG and CSV files are saved with **automatic serial numbering** — existing files are never overwritten.

## Running with Docker

### Build the image

```bash
docker build -t diffusion-sim .
```

### Run individual scripts

| Command | Description |
|---|---|
| `docker run --rm -v "%cd%:/app/output" diffusion-sim second_fixed` | Forward diffusion + reverse + plot |
| `docker run --rm -v "%cd%:/app/output" diffusion-sim run_simulations` | 100 simulations → CSV |
| `docker run --rm -v "%cd%:/app/output" diffusion-sim analyze_simulations` | CSV analysis → plot |
| `docker run --rm -v "%cd%:/app/output" diffusion-sim all` | Run all three in sequence |

> **PowerShell users**: Replace `"%cd%"` with `${PWD}`

### Using Docker Compose

```bash
# Run all three scripts (default)
docker compose up

# Run a specific script
docker compose run diffusion-sim second_fixed
docker compose run --rm diffusion-sim run_simulations
docker compose run --rm diffusion-sim analyze_simulations
```

## What Each Script Does

### `second_fixed.py`
- Implements the DDPM forward diffusion: `x_t = √(1-β_t)·x_{t-1} + √(β_t)·ε`
- Uses a **linear beta schedule** from 0.02 → 0.5
- Tracks α_t, ᾱ_t (cumulative product), and SNR at each step
- Demonstrates a **reverse (denoising) process**
- Includes the **original noise schedule** from `second.py` for comparison
- Saves a **3-panel plot**: trajectories, schedule quantities, signal degradation

### `run_simulations.py`
- Runs **100 independent forward diffusion simulations**
- Uses deterministic seeds (1–100) for reproducibility
- Saves all data to `diffusion_simulations.csv` (600 rows)

### `analyze_simulations.py`
- Reads the CSV and computes aggregate statistics (mean, std, min, max per step)
- Saves a **4-panel analysis plot**:
  1. MSE aggregated over 100 runs (+ theoretical expectation)
  2. L2 distance aggregated over 100 runs
  3. All 100 trajectories overlaid (dimension x[2])
  4. Histogram of final MSE at step t=5

## Mathematical Background

The forward diffusion process corrupts a clean signal x₀ over T steps by gradually adding Gaussian noise:

```
x_t = √(1-β_t) · x_{t-1} + √(β_t) · ε        ε ~ N(0, I)
```

Equivalently, in closed form:

```
x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε̄            ε̄ ~ N(0, I)
```

where β_t is the noise schedule, α_t = 1-β_t, and ᾱ_t = ∏_{s=1}^{t} α_s.

The reverse (denoising) process removes noise step by step using:

```
x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z
```

## Output File Naming

All saved files use automatic serial numbering:

| Base Name | Example Files | Description |
|---|---|---|
| `diffusion_plot.png` | `diffusion_plot.png`, `diffusion_plot_1.png`, ... | Diffusion trajectory plots |
| `analysis_plot.png` | `analysis_plot.png`, `analysis_plot_1.png`, ... | Analysis plots from CSV |
| `diffusion_simulations.csv` | `diffusion_simulations.csv` | Simulation data (single file, overwritten) |