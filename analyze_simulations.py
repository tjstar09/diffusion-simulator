import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# Analyze 100-run diffusion simulation from CSV
# ============================================================

csv_filename = "diffusion_simulations.csv"
if not os.path.exists(csv_filename):
    print(f"Error: '{csv_filename}' not found. Run 'run_simulations.py' first.")
    exit(1)

df = pd.read_csv(csv_filename)

print(f"Loaded {len(df)} rows from '{csv_filename}'")
print(f"Columns: {list(df.columns)}\n")

# -----------------------------------------------------------
# 1) Aggregate statistics per step across all runs
# -----------------------------------------------------------
agg = df.groupby("step").agg(
    mse_mean   = ("mse", "mean"),
    mse_std    = ("mse", "std"),
    mse_min    = ("mse", "min"),
    mse_max    = ("mse", "max"),
    l2_mean    = ("l2_dist", "mean"),
    l2_std     = ("l2_dist", "std"),
    l2_min     = ("l2_dist", "min"),
    l2_max     = ("l2_dist", "max"),
).reset_index()

steps_arr = agg["step"].values

# Pre-compute theoretical MSE (same as in second_fixed.py)
beta_start = 0.02
beta_end   = 0.5
steps_total = 5
x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
d = x0.shape[0]
x0_norm_sq = np.linalg.norm(x0) ** 2

beta = np.linspace(beta_start, beta_end, steps_total)
alpha = 1.0 - beta
alpha_bar = np.cumprod(alpha)
alpha_bar_full = np.concatenate([[1.0], alpha_bar])

expected_mse = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) \
               + (1 - alpha_bar_full)

# -----------------------------------------------------------
# 2) Create analysis plot (3 panels) with serial numbering
# -----------------------------------------------------------
plt.figure(figsize=(14, 10))

# --- Subplot 1: MSE distribution across steps ---
plt.subplot(2, 2, 1)
plt.errorbar(steps_arr, agg["mse_mean"], yerr=agg["mse_std"],
             fmt='o-', capsize=5, color='C1', label='Mean ± Std')
plt.fill_between(steps_arr, agg["mse_min"], agg["mse_max"],
                 alpha=0.15, color='C1', label='Min–Max range')
plt.plot(steps_arr, expected_mse, '--', color='gray', alpha=0.7,
         label='Expected MSE (theory)')
plt.xlabel("Diffusion Step t")
plt.ylabel("MSE(x_t, x_0)")
plt.title("MSE Aggregated Over 100 Runs")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 2: L2 distance distribution across steps ---
plt.subplot(2, 2, 2)
plt.errorbar(steps_arr, agg["l2_mean"], yerr=agg["l2_std"],
             fmt='s-', capsize=5, color='C5', label='Mean ± Std')
plt.fill_between(steps_arr, agg["l2_min"], agg["l2_max"],
                 alpha=0.15, color='C5', label='Min–Max range')
plt.xlabel("Diffusion Step t")
plt.ylabel("L2 Distance(x_t, x_0)")
plt.title("L2 Distance Aggregated Over 100 Runs")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 3: Trajectories — overlay of all 100 runs (single dimension) ---
plt.subplot(2, 2, 3)
# Pick dimension 2 (x2) as a representative
for run_id in df["run_id"].unique():
    run_df = df[df["run_id"] == run_id]
    plt.plot(run_df["step"], run_df["x2"], alpha=0.08, color='C0')
# Overlay mean trajectory
mean_x2 = df.groupby("step")["x2"].mean()
plt.plot(mean_x2.index, mean_x2.values, 'o-', color='red', linewidth=2, label='Mean trajectory')
plt.axhline(y=x0[2], color='green', linestyle='--', alpha=0.6, label=f'x_0[2] = {x0[2]}')
plt.xlabel("Diffusion Step t")
plt.ylabel("x[2] value")
plt.title("All 100 Trajectories (dimension x[2])")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 4: Final step (t=5) histogram of MSE ---
plt.subplot(2, 2, 4)
final_df = df[df["step"] == 5]
plt.hist(final_df["mse"], bins=15, color='C1', edgecolor='white', alpha=0.7, density=True)
plt.axvline(final_df["mse"].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {final_df["mse"].mean():.3f}')
plt.axvline(expected_mse[-1], color='gray', linestyle=':',
            linewidth=2, label=f'Theory = {expected_mse[-1]:.3f}')
plt.xlabel("MSE at final step (t=5)")
plt.ylabel("Density")
plt.title("Distribution of Final MSE Over 100 Runs")
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle("Diffusion Simulation Analysis — 100 Runs", fontsize=14, y=1.01)
plt.tight_layout()

# Serial filename
base = "analysis_plot"
ext = ".png"
filename = base + ext
serial = 1
while os.path.exists(filename):
    filename = f"{base}_{serial}{ext}"
    serial += 1

plt.savefig(filename, dpi=150)
print(f"Analysis plot saved as '{filename}'")