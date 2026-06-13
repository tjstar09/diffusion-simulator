import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Enhanced Forward Diffusion Process (DDPM-style)
# ============================================================

# Initial signal x_0
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
steps = 5

# -----------------------------------------------------------
# 1) Time-varying beta schedule (linear schedule from DDPM)
# -----------------------------------------------------------
beta_start = 0.02
beta_end   = 0.5
beta = np.linspace(beta_start, beta_end, steps)

# Derived quantities
alpha      = 1.0 - beta                     # α_t = 1 - β_t
alpha_bar  = np.cumprod(alpha)              # ᾱ_t = ∏_{s=1}^{t} α_s
snr        = alpha_bar / (1.0 - alpha_bar)  # Signal-to-noise ratio at each step


# -----------------------------------------------------------
# 2) Forward diffusion with tracked intermediate states
# -----------------------------------------------------------
x_history = [x.copy()]  # store all states (including x_0)

print("=" * 72)
print("   ENHANCED FORWARD DIFFUSION PROCESS")
print("=" * 72)
print(f"\nInitial state x_0: {x}")
print(f"Beta schedule (linear from {beta_start} → {beta_end}):")
for t_idx, b in enumerate(beta, start=1):
    print(f"   β_{t_idx} = {b:.4f}")
print()

current_x = x.copy()

for t in range(1, steps + 1):
    t_idx = t - 1  # 0-based index

    # Sample Gaussian noise
    noise = np.random.normal(0, 1, size=x.shape)

    # Forward diffusion step: x_t = √(1 - β_t) · x_{t-1} + √(β_t) · ε
    current_x = np.sqrt(1.0 - beta[t_idx]) * current_x + np.sqrt(beta[t_idx]) * noise
    x_history.append(current_x.copy())

    # --- Logging ---
    print(f"Step {t}:")
    print(f"   β_{t}  = {beta[t_idx]:.4f}    ᾱ_{t} = {alpha_bar[t_idx]:.4f}    SNR = {snr[t_idx]:.2f}")
    print(f"   Action: x_{t} = √(1-β_{t})·x_{t-1} + √(β_{t})·ε")
    print(f"   Result x_{t}: {np.round(current_x, 3)}")
    print()

print("--- Forward Diffusion Complete ---")
print()


# ============================================================
# 3) Reverse (Denoising) Process  —  simple demonstration
# ============================================================
print("=" * 72)
print("   REVERSE (DENOISING) PROCESS  (single trajectory)")
print("=" * 72)
print(f"\nStarting from x_{steps} (the fully noised state):")
print(f"   x_{steps} = {np.round(x_history[-1], 3)}\n")

x_rev = x_history[-1].copy()

for t in range(steps, 0, -1):
    t_idx = t - 1  # 0-based index

    if t > 1:
        # Predict the noise from x_t (in a real model this would be a learned network)
        noise_pred = np.random.normal(0, 1, size=x.shape)

        # Simple reverse step (assuming we know the true noise, for demonstration):
        # x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_pred)  +  σ_t · z
        # where σ_t² = β_t (simplified) and z ~ N(0,I)
        coef1 = 1.0 / np.sqrt(alpha[t_idx])
        coef2 = beta[t_idx] / np.sqrt(1.0 - alpha_bar[t_idx])
        z     = np.random.normal(0, 1, size=x.shape)

        x_rev = coef1 * (x_rev - coef2 * noise_pred) + np.sqrt(beta[t_idx]) * z
    else:
        # Final step: t = 1 → t = 0  (no additional noise)
        noise_pred = np.random.normal(0, 1, size=x.shape)
        coef1 = 1.0 / np.sqrt(alpha[t_idx])
        coef2 = beta[t_idx] / np.sqrt(1.0 - alpha_bar[t_idx])
        x_rev = coef1 * (x_rev - coef2 * noise_pred)

    print(f"Reverse step t={t} → t={t-1}:  x_{t-1} = {np.round(x_rev, 3)}")

print("\n--- Reverse Denoising Complete ---")
print()


# ============================================================
# 4) Original noise schedule from second.py  (for comparison)
# ============================================================
print("=" * 72)
print("   ALTERNATIVE NOISE SCHEDULE  (from second.py)")
print("=" * 72)
print(r"   σ²(t) = θ₀ · exp(-t / T_max)^λ" + "\n")

theta_0 = 0.01 * 0.1  # step_size * 0.01  (from original code)
lambda_ = 1.0
T_max   = 1.0

time_points = np.linspace(0, 1, steps + 1)
sigma_sq    = theta_0 * np.exp(-time_points / T_max) ** lambda_

print(f"   θ₀ = {theta_0:.4e}    λ = {lambda_}    T_max = {T_max}")
print(f"   Variance schedule: {np.round(sigma_sq, 6)}\n")
print("   Note: This schedule decays variance exponentially with time.\n")


# ============================================================
# 5) Visualization of the diffusion trajectory
# ============================================================
x_history_arr = np.array(x_history)  # shape (steps+1, 5)

# Compute degradation metrics
mse = np.mean((x_history_arr - x[np.newaxis, :]) ** 2, axis=1)  # MSE vs original at each step
l2_dist = np.linalg.norm(x_history_arr - x[np.newaxis, :], axis=1)  # L2 distance

plt.figure(figsize=(14, 4))

# --- Subplot 1: Trajectories ---
plt.subplot(1, 3, 1)
for dim_idx in range(5):
    plt.plot(range(steps + 1), x_history_arr[:, dim_idx],
             marker='o', label=f'x[{dim_idx}]')
plt.xlabel("Diffusion Step t")
plt.ylabel("Value")
plt.title("Forward Diffusion Trajectory\n(each dimension)")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 2: Schedule quantities ---
plt.subplot(1, 3, 2)
plt.plot(range(1, steps + 1), beta, marker='s', color='C2', label='β_t')
plt.plot(range(1, steps + 1), alpha_bar, marker='^', color='C3', label='ᾱ_t')
plt.plot(range(1, steps + 1), snr, marker='d', color='C4', label='SNR')
plt.xlabel("Step t")
plt.ylabel("Value")
plt.title("Schedule Quantities")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Subplot 3: Signal degradation ---
plt.subplot(1, 3, 3)
plt.plot(range(steps + 1), mse, marker='o', color='C1', label='MSE(x_t, x_0)')
plt.plot(range(steps + 1), l2_dist, marker='s', color='C5', label='L2 dist(x_t, x_0)')
# Theoretical expected L2 distance (for reference): sqrt(1 - ᾱ_t) * ||x_0||
# Since x_t ~ N(√(ᾱ_t)·x_0, (1-ᾱ_t)·I), the expected squared L2 is:
#   E[||x_t - x_0||²] = (1-√(ᾱ_t))²·||x_0||² + d·(1-ᾱ_t)
d = x.shape[0]
x0_norm_sq = np.linalg.norm(x) ** 2
expected_mse = ((1 - np.sqrt(np.concatenate([[1.0], alpha_bar]))) ** 2) * (x0_norm_sq / d) \
               + (1 - np.concatenate([[1.0], alpha_bar]))
plt.plot(range(steps + 1), expected_mse, '--', color='gray', alpha=0.6,
         label='Expected MSE (theory)')
plt.xlabel("Step t")
plt.ylabel("Degradation Metric")
plt.title("Signal Degradation Over Time")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Generate filename with serial number if file already exists
base_filename = "diffusion_plot"
extension = ".png"
filename = base_filename + extension
serial = 1
while os.path.exists(filename):
    filename = f"{base_filename}_{serial}{extension}"
    serial += 1

plt.savefig(filename, dpi=150)
print("=" * 72)
print(f"   Plot saved as '{filename}'")
print("=" * 72)

# Show interactive window (comment out if running headless)
# plt.show()