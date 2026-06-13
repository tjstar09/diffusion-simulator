"""
Smoke test for CI pipeline. Verifies forward diffusion works.
"""
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
steps = 5
beta = np.linspace(0.02, 0.5, steps)

current_x = x.copy()
np.random.seed(42)
for t in range(steps):
    noise = np.random.normal(0, 1, size=x.shape)
    current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise

assert np.all(np.isfinite(current_x)), 'Forward diffusion failed'
print('Smoke test passed')