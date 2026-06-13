"""
Pytest configuration for the diffusion-simulator test suite.
Loads test_config.json and makes it available to all test modules.
"""
import json
import os
import sys
import pytest

# Add the project root to sys.path so we can import the simulation modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def test_config():
    """Load and return the test configuration from test_config.json."""
    config_path = os.path.join(PROJECT_ROOT, "test_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


@pytest.fixture(scope="session")
def project_root():
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_initial_signal():
    """Return the standard initial signal x_0 used across all scripts."""
    import numpy as np
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def default_beta_params():
    """Return the default beta schedule parameters."""
    return {
        "beta_start": 0.02,
        "beta_end": 0.5,
        "steps": 5
    }


@pytest.fixture
def compute_beta_schedule():
    """Return a function that computes the beta schedule for given parameters."""
    import numpy as np

    def _compute(beta_start=0.02, beta_end=0.5, steps=5):
        beta = np.linspace(beta_start, beta_end, steps)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)
        snr = alpha_bar / (1.0 - alpha_bar)
        return {
            "beta": beta,
            "alpha": alpha,
            "alpha_bar": alpha_bar,
            "snr": snr
        }
    return _compute