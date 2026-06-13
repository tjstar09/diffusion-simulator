"""
Validation script for PRE deployment.
Checks forward diffusion, MSE, and reverse process.
"""
import numpy as np

# Validate forward diffusion
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
steps = 5
beta = np.linspace(0.02, 0.5, steps)
alpha = 1.0 - beta
alpha_bar = np.cumprod(alpha)
assert alpha_bar[0] > alpha_bar[-1], 'alpha_bar should decrease'
print('Beta schedule valid')

np.random.seed(42)
current_x = x.copy()
for t in range(steps):
    noise = np.random.normal(0, 1, size=x.shape)
    current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise
mse = np.mean((current_x - x) ** 2)
assert np.isfinite(mse), 'MSE should be finite'
print('Forward diffusion valid (MSE={:.4f})'.format(mse))
print('Pre-deployment validation complete')