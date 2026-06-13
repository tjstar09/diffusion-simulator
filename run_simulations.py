import numpy as np
import os
import csv

# ============================================================
# 100-run simulation of the DDPM forward diffusion process
# ============================================================

# Initial signal x_0
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
steps = 5

# Beta schedule (linear)
beta_start = 0.02
beta_end   = 0.5
beta = np.linspace(beta_start, beta_end, steps)
alpha      = 1.0 - beta
alpha_bar  = np.cumprod(alpha)

num_runs = 100

# Prepare CSV output
csv_filename = "diffusion_simulations.csv"
with open(csv_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow([
        "run_id", "step",
        "x0", "x1", "x2", "x3", "x4",
        "mse", "l2_dist",
        "beta_t", "alpha_bar_t", "snr_t"
    ])

    for run_id in range(1, num_runs + 1):
        # Deterministic seed for reproducibility
        np.random.seed(run_id)

        current_x = x.copy()
        # Write step 0 (initial state)
        writer.writerow([
            run_id, 0,
            *np.round(current_x, 6),
            0.0, 0.0,
            0.0, 1.0, float("inf")
        ])

        for t in range(1, steps + 1):
            t_idx = t - 1

            # Sample noise
            noise = np.random.normal(0, 1, size=x.shape)

            # Forward diffusion step
            current_x = np.sqrt(1.0 - beta[t_idx]) * current_x + np.sqrt(beta[t_idx]) * noise

            # Compute degradation metrics
            mse_val = np.mean((current_x - x) ** 2)
            l2_val  = np.linalg.norm(current_x - x)

            writer.writerow([
                run_id, t,
                *np.round(current_x, 6),
                round(mse_val, 6),
                round(l2_val, 6),
                round(beta[t_idx], 6),
                round(alpha_bar[t_idx], 6),
                round(alpha_bar[t_idx] / (1.0 - alpha_bar[t_idx]), 6) if alpha_bar[t_idx] < 1 else "inf"
            ])

print(f"Simulation complete. {num_runs} runs written to '{csv_filename}'")