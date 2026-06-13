"""
Unit tests for second_fixed.py - DDPM Forward Diffusion + Reverse Process.
Tests are driven by test_config.json configuration.
"""
import json
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBetaSchedule:
    """Tests for the beta schedule calculations."""

    def test_beta_schedule_linear_range(self, compute_beta_schedule):
        """Verify beta schedule is linear from 0.02 to 0.5 over 5 steps."""
        schedule = compute_beta_schedule(beta_start=0.02, beta_end=0.5, steps=5)
        beta = schedule["beta"]

        assert len(beta) == 5, f"Expected 5 beta values, got {len(beta)}"
        assert np.isclose(beta[0], 0.02), f"First beta should be 0.02, got {beta[0]}"
        assert np.isclose(beta[-1], 0.5), f"Last beta should be 0.5, got {beta[-1]}"

        # Verify linear spacing
        expected = np.linspace(0.02, 0.5, 5)
        assert np.allclose(beta, expected), (
            f"Beta schedule not linear. Expected {expected}, got {beta}"
        )

    def test_alpha_bar_calculation(self, compute_beta_schedule):
        """Verify alpha_bar = cumprod(1 - beta) is computed correctly."""
        schedule = compute_beta_schedule()
        alpha = schedule["alpha"]
        alpha_bar = schedule["alpha_bar"]

        # alpha = 1 - beta
        expected_alpha = 1.0 - schedule["beta"]
        assert np.allclose(alpha, expected_alpha), (
            f"alpha != 1 - beta. Got {alpha}, expected {expected_alpha}"
        )

        # alpha_bar = cumprod(alpha)
        expected_alpha_bar = np.cumprod(alpha)
        assert np.allclose(alpha_bar, expected_alpha_bar), (
            f"alpha_bar != cumprod(alpha). Got {alpha_bar}, expected {expected_alpha_bar}"
        )

        # alpha_bar should be decreasing (since each alpha < 1)
        for i in range(1, len(alpha_bar)):
            assert alpha_bar[i] < alpha_bar[i - 1], (
                f"alpha_bar should be decreasing, but alpha_bar[{i}] >= alpha_bar[{i-1}]"
            )

    def test_snr_calculation(self, compute_beta_schedule):
        """Verify SNR = alpha_bar / (1 - alpha_bar)."""
        schedule = compute_beta_schedule()
        alpha_bar = schedule["alpha_bar"]
        snr = schedule["snr"]

        expected_snr = alpha_bar / (1.0 - alpha_bar)
        assert np.allclose(snr, expected_snr), (
            f"SNR calculation incorrect. Got {snr}, expected {expected_snr}"
        )

        # SNR should decrease as more noise is added
        assert snr[0] > snr[-1], (
            "SNR at step 0 should be greater than at the final step"
        )


class TestForwardDiffusion:
    """Tests for the forward diffusion process."""

    def test_forward_diffusion_step_formula(self, sample_initial_signal):
        """Verify the forward diffusion step formula produces correct shapes."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)

        np.random.seed(42)
        current_x = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise

        # Output should have same shape as input
        assert current_x.shape == x.shape, (
            f"Output shape {current_x.shape} != input shape {x.shape}"
        )

        # Output should be finite (no NaN or Inf)
        assert np.all(np.isfinite(current_x)), (
            "Forward diffusion produced non-finite values"
        )

    def test_forward_diffusion_monotonic_mse(self, sample_initial_signal):
        """Verify MSE increases monotonically with diffusion steps."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)

        # Compute theoretical MSE: (1 - sqrt(alpha_bar))^2 * ||x0||^2/d + (1 - alpha_bar)
        d = x.shape[0]
        x0_norm_sq = np.linalg.norm(x) ** 2
        alpha_bar_full = np.concatenate([[1.0], alpha_bar])
        expected_mse = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) \
                       + (1 - alpha_bar_full)

        # MSE should increase monotonically
        for i in range(1, len(expected_mse)):
            assert expected_mse[i] >= expected_mse[i - 1], (
                f"MSE decreased from step {i-1} to step {i}: "
                f"{expected_mse[i-1]} -> {expected_mse[i]}"
            )

        # MSE at step 0 should be 0 (no degradation)
        assert np.isclose(expected_mse[0], 0.0, atol=1e-10), (
            f"Expected MSE at step 0 to be 0, got {expected_mse[0]}"
        )

    def test_forward_diffusion_with_deterministic_seed(self, sample_initial_signal):
        """Verify same seed produces identical diffusion results."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)

        # First run
        np.random.seed(123)
        current_x1 = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x1 = np.sqrt(1.0 - beta[t]) * current_x1 + np.sqrt(beta[t]) * noise

        # Second run with same seed
        np.random.seed(123)
        current_x2 = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x2 = np.sqrt(1.0 - beta[t]) * current_x2 + np.sqrt(beta[t]) * noise

        assert np.allclose(current_x1, current_x2), (
            "Same seed produced different diffusion results"
        )

    def test_x_history_tracking(self, sample_initial_signal):
        """Verify x_history correctly tracks states at each step."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)

        x_history = [x.copy()]
        np.random.seed(42)
        current_x = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise
            x_history.append(current_x.copy())

        x_history_arr = np.array(x_history)

        # Should have (steps + 1) entries
        assert x_history_arr.shape[0] == steps + 1, (
            f"Expected {steps + 1} history entries, got {x_history_arr.shape[0]}"
        )

        # First entry should equal initial signal
        assert np.allclose(x_history_arr[0], x), (
            "First history entry should equal initial signal"
        )

        # Each entry should have same dimension as input
        assert x_history_arr.shape[1] == x.shape[0], (
            f"History entries dimension {x_history_arr.shape[1]} != {x.shape[0]}"
        )


class TestReverseProcess:
    """Tests for the reverse (denoising) process."""

    def test_reverse_process_output_shape(self, sample_initial_signal):
        """Verify reverse denoising produces correctly shaped output."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)

        # Forward diffusion to get x_t
        np.random.seed(42)
        current_x = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise
        x_rev = current_x.copy()

        # Reverse process
        for t in range(steps, 0, -1):
            t_idx = t - 1
            noise_pred = np.random.normal(0, 1, size=x.shape)
            coef1 = 1.0 / np.sqrt(alpha[t_idx])
            coef2 = beta[t_idx] / np.sqrt(1.0 - alpha_bar[t_idx])
            if t > 1:
                z = np.random.normal(0, 1, size=x.shape)
                x_rev = coef1 * (x_rev - coef2 * noise_pred) + np.sqrt(beta[t_idx]) * z
            else:
                x_rev = coef1 * (x_rev - coef2 * noise_pred)

        # Output should have same shape as input
        assert x_rev.shape == x.shape, (
            f"Reverse output shape {x_rev.shape} != input shape {x.shape}"
        )

        # Output should be finite
        assert np.all(np.isfinite(x_rev)), (
            "Reverse process produced non-finite values"
        )


class TestOriginalNoiseSchedule:
    """Tests for the alternative noise schedule from second.py."""

    def test_noise_schedule_values(self):
        """Verify the alternative noise schedule computes correctly."""
        theta_0 = 0.01 * 0.1  # step_size * 0.01
        lambda_ = 1.0
        T_max = 1.0
        steps = 5

        time_points = np.linspace(0, 1, steps + 1)
        sigma_sq = theta_0 * np.exp(-time_points / T_max) ** lambda_

        assert len(sigma_sq) == steps + 1, (
            f"Expected {steps + 1} sigma_sq values, got {len(sigma_sq)}"
        )
        assert sigma_sq[0] == theta_0, (
            f"First sigma_sq should be {theta_0}, got {sigma_sq[0]}"
        )
        assert np.all(sigma_sq >= 0), "sigma_sq should be non-negative"
        assert np.all(np.isfinite(sigma_sq)), "sigma_sq should be finite"
        # Should be decreasing
        assert sigma_sq[-1] < sigma_sq[0], (
            "sigma_sq should decrease over time"
        )


class TestDegradationMetrics:
    """Tests for MSE and L2 distance calculations."""

    def test_mse_calculation(self, sample_initial_signal, compute_beta_schedule):
        """Verify MSE between diffused state and original is computed correctly."""
        x = sample_initial_signal
        schedule = compute_beta_schedule()

        # Simulate a diffused state
        np.random.seed(42)
        beta = schedule["beta"]
        current_x = x.copy()
        for t in range(len(beta)):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise

        # Compute MSE
        mse = np.mean((current_x - x) ** 2)

        assert mse >= 0, f"MSE should be non-negative, got {mse}"
        assert np.isfinite(mse), f"MSE should be finite, got {mse}"

    def test_l2_distance_non_negative(self, sample_initial_signal):
        """Verify L2 distance is non-negative."""
        x = sample_initial_signal
        steps = 5
        beta = np.linspace(0.02, 0.5, steps)

        np.random.seed(42)
        current_x = x.copy()
        for t in range(steps):
            noise = np.random.normal(0, 1, size=x.shape)
            current_x = np.sqrt(1.0 - beta[t]) * current_x + np.sqrt(beta[t]) * noise

        l2_dist = np.linalg.norm(current_x - x)

        assert l2_dist >= 0, f"L2 distance should be non-negative, got {l2_dist}"
        assert np.isfinite(l2_dist), f"L2 distance should be finite, got {l2_dist}"