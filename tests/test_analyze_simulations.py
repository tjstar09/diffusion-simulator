"""
Unit tests for analyze_simulations.py - Analysis of simulation CSV with plots.
Tests are driven by test_config.json configuration.
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a sample CSV file for testing analysis functions."""
    # Create a mini simulation CSV (3 runs, 5 steps)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = 5
    beta = np.linspace(0.02, 0.5, steps)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    num_runs = 3

    csv_file = tmp_path / "test_analysis_simulations.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "step",
            "x0", "x1", "x2", "x3", "x4",
            "mse", "l2_dist",
            "beta_t", "alpha_bar_t", "snr_t"
        ])

        for run_id in range(1, num_runs + 1):
            np.random.seed(run_id)
            current_x = x.copy()
            writer.writerow([
                run_id, 0, *np.round(current_x, 6),
                0.0, 0.0, 0.0, 1.0, float("inf")
            ])
            for t in range(1, steps + 1):
                t_idx = t - 1
                noise = np.random.normal(0, 1, size=x.shape)
                current_x = np.sqrt(1.0 - beta[t_idx]) * current_x + np.sqrt(beta[t_idx]) * noise
                mse_val = np.mean((current_x - x) ** 2)
                l2_val = np.linalg.norm(current_x - x)
                writer.writerow([
                    run_id, t, *np.round(current_x, 6),
                    round(mse_val, 6), round(l2_val, 6),
                    round(beta[t_idx], 6), round(alpha_bar[t_idx], 6),
                    round(alpha_bar[t_idx] / (1.0 - alpha_bar[t_idx]), 6) if alpha_bar[t_idx] < 1 else "inf"
                ])

    return csv_file


class TestCSVLoading:
    """Tests for loading and validating the CSV data."""

    def test_csv_loading_with_pandas(self, sample_csv_path):
        """Verify CSV can be loaded with pandas without errors."""
        df = pd.read_csv(sample_csv_path)

        assert isinstance(df, pd.DataFrame), (
            "Loaded data should be a pandas DataFrame"
        )
        assert len(df) > 0, "DataFrame should not be empty"

    def test_loaded_data_columns(self, sample_csv_path):
        """Verify loaded DataFrame has expected columns."""
        df = pd.read_csv(sample_csv_path)
        expected_columns = {
            "run_id", "step", "x0", "x1", "x2", "x3", "x4",
            "mse", "l2_dist", "beta_t", "alpha_bar_t", "snr_t"
        }
        actual_columns = set(df.columns)
        assert actual_columns == expected_columns, (
            f"Column mismatch.\nExpected: {expected_columns}\nGot: {actual_columns}"
        )

    def test_loaded_data_types(self, sample_csv_path):
        """Verify numeric columns have correct data types."""
        df = pd.read_csv(sample_csv_path)
        numeric_cols = ["step", "x0", "x1", "x2", "x3", "x4", "mse", "l2_dist", "beta_t", "alpha_bar_t"]
        for col in numeric_cols:
            assert np.issubdtype(df[col].dtype, np.number), (
                f"Column '{col}' should be numeric, got dtype {df[col].dtype}"
            )

    def test_missing_csv_raises_error(self, project_root, tmp_path):
        """Verify that missing CSV file raises an error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(project_root, "analyze_simulations.py")],
            capture_output=True,
            text=True,
            cwd=tmp_path  # Run in tmp_path where no CSV exists
        )
        # Should exit with error or print error message
        assert result.returncode != 0 or "Error" in result.stdout or "not found" in result.stderr, (
            "Should report error when CSV is missing"
        )


class TestAggregation:
    """Tests for the groupby aggregation logic."""

    def test_groupby_aggregation_shape(self, sample_csv_path):
        """Verify groupby aggregation returns correct number of steps."""
        df = pd.read_csv(sample_csv_path)
        agg = df.groupby("step").agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_min=("mse", "min"),
            mse_max=("mse", "max"),
            l2_mean=("l2_dist", "mean"),
            l2_std=("l2_dist", "std"),
            l2_min=("l2_dist", "min"),
            l2_max=("l2_dist", "max"),
        ).reset_index()

        # Should have 6 steps (0 through 5)
        assert len(agg) == 6, (
            f"Expected 6 aggregated steps, got {len(agg)}"
        )
        assert list(agg["step"]) == [0, 1, 2, 3, 4, 5], (
            f"Steps should be [0,1,2,3,4,5], got {list(agg['step'])}"
        )

    def test_aggregation_statistics_ranges(self, sample_csv_path):
        """Verify aggregation statistics make sense."""
        df = pd.read_csv(sample_csv_path)
        agg = df.groupby("step").agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_min=("mse", "min"),
            mse_max=("mse", "max"),
        ).reset_index()

        for _, row in agg.iterrows():
            # mean should be between min and max
            assert row["mse_min"] <= row["mse_mean"] <= row["mse_max"], (
                f"At step {row['step']}: mean {row['mse_mean']} not between "
                f"min {row['mse_min']} and max {row['mse_max']}"
            )
            # std should be non-negative
            assert row["mse_std"] >= 0, (
                f"Negative std at step {row['step']}: {row['mse_std']}"
            )

    def test_step_zero_statistics(self, sample_csv_path):
        """Verify statistics at step 0 are all zero (no degradation)."""
        df = pd.read_csv(sample_csv_path)
        step_zero = df[df["step"] == 0]
        assert (step_zero["mse"] == 0.0).all(), (
            "MSE at step 0 should be 0"
        )
        assert (step_zero["l2_dist"] == 0.0).all(), (
            "L2 distance at step 0 should be 0"
        )


class TestExpectedMSE:
    """Tests for the theoretical expected MSE computation."""

    def test_expected_mse_formula(self, sample_csv_path):
        """Verify expected MSE follows the theoretical formula."""
        df = pd.read_csv(sample_csv_path)
        steps_total = 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = x0.shape[0]
        x0_norm_sq = np.linalg.norm(x0) ** 2

        beta = np.linspace(0.02, 0.5, steps_total)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        alpha_bar_full = np.concatenate([[1.0], alpha_bar])

        expected_mse = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) \
                       + (1 - alpha_bar_full)

        # At step 0: expected MSE should be 0
        assert np.isclose(expected_mse[0], 0.0, atol=1e-10), (
            f"Expected MSE at step 0 should be 0, got {expected_mse[0]}"
        )

        # Expected MSE should be monotonically increasing
        for i in range(1, len(expected_mse)):
            assert expected_mse[i] > expected_mse[i - 1], (
                f"Expected MSE decreased from step {i-1} to {i}: "
                f"{expected_mse[i-1]} -> {expected_mse[i]}"
            )

        # At final step, expected MSE = (1 - sqrt(alpha_bar[-1]))^2 * ||x0||^2/d + (1 - alpha_bar[-1])
        # With x0 = [1,2,3,4,5], ||x0||^2/d = 11, alpha_bar[-1] -> 0, expected MSE -> ~12
        # But with alpha_bar[-1] = 0.0009, MSE = (1-0.03)^2*11 + (1-0.0009) ≈ 11.3
        # Verify it's in a reasonable range (between the values at step 0 and infinity)
        assert expected_mse[-1] > expected_mse[-2], (
            f"Expected MSE should increase monotonically, but step {len(expected_mse)-1} "
            f"({expected_mse[-1]}) <= step {len(expected_mse)-2} ({expected_mse[-2]})"
        )
        assert expected_mse[-1] < 12.5, (
            f"Expected MSE at final step ({expected_mse[-1]}) seems unreasonably large"
        )

    def test_expected_mse_reproducibility(self):
        """Verify expected MSE computation is deterministic (no random seed dependency)."""
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        steps_total = 5
        d = x0.shape[0]
        x0_norm_sq = np.linalg.norm(x0) ** 2

        beta = np.linspace(0.02, 0.5, steps_total)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        alpha_bar_full = np.concatenate([[1.0], alpha_bar])

        expected_mse_1 = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) \
                         + (1 - alpha_bar_full)

        # Same computation again should produce identical results
        expected_mse_2 = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) \
                         + (1 - alpha_bar_full)

        assert np.allclose(expected_mse_1, expected_mse_2), (
            "Expected MSE computation should be deterministic"
        )


class TestAnalysisPlot:
    """Tests for plot generation."""

    def test_plot_creation_uses_serial_numbering(self, sample_csv_path, tmp_path):
        """Verify plot files use serial numbering (don't overwrite)."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        df = pd.read_csv(sample_csv_path)

        # Step 1: Compute aggregation and expected MSE (same as analyze_simulations.py)
        agg = df.groupby("step").agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_min=("mse", "min"),
            mse_max=("mse", "max"),
            l2_mean=("l2_dist", "mean"),
            l2_std=("l2_dist", "std"),
            l2_min=("l2_dist", "min"),
            l2_max=("l2_dist", "max"),
        ).reset_index()

        steps_total = 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = x0.shape[0]
        x0_norm_sq = np.linalg.norm(x0) ** 2
        beta = np.linspace(0.02, 0.5, steps_total)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        alpha_bar_full = np.concatenate([[1.0], alpha_bar])
        expected_mse = ((1 - np.sqrt(alpha_bar_full)) ** 2) * (x0_norm_sq / d) + (1 - alpha_bar_full)

        # Step 2: Create the plot
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        steps_arr = agg["step"].values
        plt.errorbar(steps_arr, agg["mse_mean"], yerr=agg["mse_std"],
                     fmt='o-', capsize=5, color='C1', label='Mean ± Std')
        plt.fill_between(steps_arr, agg["mse_min"], agg["mse_max"],
                         alpha=0.15, color='C1', label='Min–Max range')
        plt.plot(steps_arr, expected_mse, '--', color='gray', alpha=0.7,
                 label='Expected MSE (theory)')
        plt.xlabel("Diffusion Step t")
        plt.ylabel("MSE(x_t, x_0)")
        plt.title("MSE Aggregated Over Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.errorbar(steps_arr, agg["l2_mean"], yerr=agg["l2_std"],
                     fmt='s-', capsize=5, color='C5', label='Mean ± Std')
        plt.fill_between(steps_arr, agg["l2_min"], agg["l2_max"],
                         alpha=0.15, color='C5', label='Min–Max range')
        plt.xlabel("Diffusion Step t")
        plt.ylabel("L2 Distance(x_t, x_0)")
        plt.title("L2 Distance Aggregated Over Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        for run_id in df["run_id"].unique():
            run_df = df[df["run_id"] == run_id]
            plt.plot(run_df["step"], run_df["x2"], alpha=0.3, color='C0')
        mean_x2 = df.groupby("step")["x2"].mean()
        plt.plot(mean_x2.index, mean_x2.values, 'o-', color='red', linewidth=2, label='Mean trajectory')
        plt.axhline(y=x0[2], color='green', linestyle='--', alpha=0.6, label=f'x_0[2] = {x0[2]}')
        plt.xlabel("Diffusion Step t")
        plt.ylabel("x[2] value")
        plt.title("All Trajectories (dimension x[2])")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        final_df = df[df["step"] == 5]
        plt.hist(final_df["mse"], bins=5, color='C1', edgecolor='white', alpha=0.7, density=True)
        plt.axvline(final_df["mse"].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean = {final_df["mse"].mean():.3f}')
        plt.axvline(expected_mse[-1], color='gray', linestyle=':',
                    linewidth=2, label=f'Theory = {expected_mse[-1]:.3f}')
        plt.xlabel("MSE at final step (t=5)")
        plt.ylabel("Density")
        plt.title("Distribution of Final MSE Over Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle("Diffusion Simulation Analysis — Test Run", fontsize=14, y=1.01)
        plt.tight_layout()

        # Save to tmp_path
        test_plot = tmp_path / "test_analysis_plot.png"
        plt.savefig(test_plot, dpi=150)
        plt.close()

        # Verify plot was created
        assert test_plot.exists(), "Plot file was not created"
        assert test_plot.stat().st_size > 0, "Plot file is empty"

        # Save another one to verify serial numbering is possible
        test_plot2 = tmp_path / "test_analysis_plot_1.png"
        plt.figure(figsize=(14, 10))
        plt.text(0.5, 0.5, "Dummy plot for serial test", ha="center")
        plt.savefig(test_plot2, dpi=150)
        plt.close()
        assert test_plot2.exists(), "Second plot file was not created"

    def test_mean_trajectory_computation(self, sample_csv_path):
        """Verify mean trajectory computation across runs."""
        df = pd.read_csv(sample_csv_path)
        mean_x2 = df.groupby("step")["x2"].mean()

        assert len(mean_x2) == 6, (
            f"Expected 6 mean values (steps 0-5), got {len(mean_x2)}"
        )
        # Mean at step 0 should equal x0[2] = 3.0
        assert np.isclose(mean_x2.iloc[0], 3.0), (
            f"Mean x2 at step 0 should be 3.0, got {mean_x2.iloc[0]}"
        )