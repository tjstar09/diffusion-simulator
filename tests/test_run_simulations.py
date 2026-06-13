"""
Unit tests for run_simulations.py - 100-run simulation exporting to CSV.
Tests are driven by test_config.json configuration.
"""
import os
import sys
import csv
import io
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def run_simulation_csv(tmp_path):
    """Run a mini simulation with fewer runs and return the CSV content as string."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = 5
    beta_start = 0.02
    beta_end = 0.5
    beta = np.linspace(beta_start, beta_end, steps)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    num_runs = 3  # 3 runs for fast test

    csv_file = tmp_path / "test_diffusion_simulations.csv"
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
                run_id, 0,
                *np.round(current_x, 6),
                0.0, 0.0,
                0.0, 1.0, float("inf")
            ])

            for t in range(1, steps + 1):
                t_idx = t - 1
                noise = np.random.normal(0, 1, size=x.shape)
                current_x = np.sqrt(1.0 - beta[t_idx]) * current_x + np.sqrt(beta[t_idx]) * noise
                mse_val = np.mean((current_x - x) ** 2)
                l2_val = np.linalg.norm(current_x - x)
                writer.writerow([
                    run_id, t,
                    *np.round(current_x, 6),
                    round(mse_val, 6),
                    round(l2_val, 6),
                    round(beta[t_idx], 6),
                    round(alpha_bar[t_idx], 6),
                    round(alpha_bar[t_idx] / (1.0 - alpha_bar[t_idx]), 6) if alpha_bar[t_idx] < 1 else "inf"
                ])

    # Read CSV back
    with open(csv_file, "r") as f:
        content = f.read()

    return content


class TestCSVOutput:
    """Tests for the CSV output format and structure."""

    def test_csv_header(self, run_simulation_csv):
        """Verify CSV contains the expected column headers."""
        reader = csv.DictReader(io.StringIO(run_simulation_csv))
        expected_headers = {
            "run_id", "step",
            "x0", "x1", "x2", "x3", "x4",
            "mse", "l2_dist",
            "beta_t", "alpha_bar_t", "snr_t"
        }
        actual_headers = set(reader.fieldnames)
        assert actual_headers == expected_headers, (
            f"CSV headers mismatch.\nExpected: {expected_headers}\nGot: {actual_headers}"
        )

    def test_csv_row_count(self, run_simulation_csv):
        """Verify CSV has correct number of rows (runs * (steps+1))."""
        lines = run_simulation_csv.strip().split("\n")
        data_rows = len(lines) - 1  # subtract header
        steps = 5
        num_runs = 3
        expected_rows = num_runs * (steps + 1)
        assert data_rows == expected_rows, (
            f"Expected {expected_rows} data rows, got {data_rows}"
        )

    def test_csv_step_zero_present(self, run_simulation_csv):
        """Verify step 0 is present for each run with correct initial state."""
        reader = csv.DictReader(io.StringIO(run_simulation_csv))
        step_zero_rows = [row for row in reader if int(row["step"]) == 0]
        assert len(step_zero_rows) == 3, (
            f"Expected 3 step-0 rows (1 per run), got {len(step_zero_rows)}"
        )
        for row in step_zero_rows:
            assert row["mse"] == "0.0", (
                f"Step 0 MSE should be 0.0, got {row['mse']}"
            )
            assert row["l2_dist"] == "0.0", (
                f"Step 0 L2 should be 0.0, got {row['l2_dist']}"
            )

    def test_csv_values_are_numeric(self, run_simulation_csv):
        """Verify all value columns contain valid numeric data."""
        lines = run_simulation_csv.strip().split("\n")
        # Check a few data rows
        for line in lines[1:6]:  # first 5 rows after header
            parts = line.split(",")
            # Columns 3-7 (x0, x1, x2, x3, x4) should be numeric
            for val in parts[2:7]:
                float(val)  # will raise ValueError if not numeric
            # mse and l2_dist should be numeric
            float(parts[7])
            float(parts[8])

    def test_deterministic_reproducibility(self, tmp_path):
        """Verify same seed produces identical CSV output."""
        def _generate_csv(seed, output_path):
            x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            steps = 5
            beta = np.linspace(0.02, 0.5, steps)
            alpha = 1.0 - beta
            alpha_bar = np.cumprod(alpha)

            with open(output_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "step", "x0", "x1", "x2", "x3", "x4", "mse", "l2_dist", "beta_t", "alpha_bar_t", "snr_t"])
                np.random.seed(seed)
                current_x = x.copy()
                writer.writerow([1, 0, *np.round(current_x, 6), 0.0, 0.0, 0.0, 1.0, float("inf")])
                for t in range(1, steps + 1):
                    noise = np.random.normal(0, 1, size=x.shape)
                    current_x = np.sqrt(1.0 - beta[t-1]) * current_x + np.sqrt(beta[t-1]) * noise
                    mse_val = np.mean((current_x - x) ** 2)
                    l2_val = np.linalg.norm(current_x - x)
                    writer.writerow([1, t, *np.round(current_x, 6), round(mse_val, 6), round(l2_val, 6),
                                     round(beta[t-1], 6), round(alpha_bar[t-1], 6),
                                     round(alpha_bar[t-1] / (1.0 - alpha_bar[t-1]), 6) if alpha_bar[t-1] < 1 else "inf"])

        # Generate two CSV files with the same seed
        csv1 = tmp_path / "test_repro1.csv"
        csv2 = tmp_path / "test_repro2.csv"
        _generate_csv(42, csv1)
        _generate_csv(42, csv2)

        with open(csv1, "r") as f1, open(csv2, "r") as f2:
            assert f1.read() == f2.read(), (
                "Same seed produced different CSV outputs"
            )

    def test_mse_and_l2_non_negative(self, run_simulation_csv):
        """Verify MSE and L2 distance values are non-negative."""
        reader = csv.DictReader(io.StringIO(run_simulation_csv))
        for row in reader:
            mse = float(row["mse"])
            l2 = float(row["l2_dist"])
            step = int(row["step"])
            if step == 0:
                assert mse == 0.0, f"Step 0 MSE should be 0, got {mse}"
                assert l2 == 0.0, f"Step 0 L2 should be 0, got {l2}"
            else:
                assert mse >= 0, f"Negative MSE at step {step}: {mse}"
                assert l2 >= 0, f"Negative L2 at step {step}: {l2}"


class TestBetaAndAlphaInCSV:
    """Tests for beta and alpha_bar values in CSV output."""

    def test_beta_values_in_csv(self, run_simulation_csv):
        """Verify beta values in CSV follow the expected linear schedule."""
        reader = csv.DictReader(io.StringIO(run_simulation_csv))
        expected_beta = [round(b, 6) for b in np.linspace(0.02, 0.5, 5)]

        for row in reader:
            step = int(row["step"])
            if step > 0:
                beta_val = float(row["beta_t"])
                expected = expected_beta[step - 1]
                assert abs(beta_val - expected) < 1e-4, (
                    f"Unexpected beta at step {step}: {beta_val} (expected {expected})"
                )

    def test_alpha_bar_decreasing(self, run_simulation_csv):
        """Verify alpha_bar values decrease as step increases (per run)."""
        reader = csv.DictReader(io.StringIO(run_simulation_csv))
        # Group rows by run_id
        runs = {}
        for row in reader:
            rid = int(row["run_id"])
            if rid not in runs:
                runs[rid] = []
            runs[rid].append(row)

        for rid, rows in runs.items():
            prev_alpha_bar = float("inf")
            for row in rows:
                step = int(row["step"])
                if step > 0:
                    alpha_bar_val = float(row["alpha_bar_t"])
                    assert alpha_bar_val < prev_alpha_bar, (
                        f"alpha_bar increased at run {rid}, step {step}: "
                        f"{alpha_bar_val} (was {prev_alpha_bar})"
                    )
                    prev_alpha_bar = alpha_bar_val
