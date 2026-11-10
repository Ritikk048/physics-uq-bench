"""
Unit tests for evaluation metrics.

Tests:
  - Gaussian NLL is minimised when sigma matches the true noise level
  - Coverage is correct for a perfectly calibrated Gaussian
  - ECE is 0 for a perfectly calibrated predictor
  - CRPS is 0 for a perfect point prediction (sigma -> 0)
  - evaluate_model returns a UQMetrics with all fields finite
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from evaluation.metrics import (
    gaussian_nll,
    rmse,
    crps,
    coverage_at_level,
    ece_regression,
    evaluate_model,
)


N = 2000
rng = np.random.default_rng(0)


def test_gaussian_nll_minimised_at_true_sigma():
    mu = np.zeros(N)
    sigma_true = 1.0
    y = rng.normal(mu, sigma_true, N)
    nll_true = gaussian_nll(y, mu, np.full(N, sigma_true))
    nll_bad = gaussian_nll(y, mu, np.full(N, 2.0))
    assert nll_true < nll_bad, "NLL should be lower at the correct sigma"


def test_rmse_zero_for_perfect_prediction():
    y = rng.normal(0, 1, N)
    assert rmse(y, y) < 1e-12


def test_coverage_at_90():
    mu = np.zeros(N)
    sigma = np.ones(N)
    y = rng.normal(mu, sigma, N)
    cov = coverage_at_level(y, mu, sigma, level=0.90)
    assert abs(cov - 0.90) < 0.03, f"Expected ~0.90, got {cov:.4f}"


def test_coverage_at_50():
    mu = np.zeros(N)
    sigma = np.ones(N)
    y = rng.normal(mu, sigma, N)
    cov = coverage_at_level(y, mu, sigma, level=0.50)
    assert abs(cov - 0.50) < 0.05, f"Expected ~0.50, got {cov:.4f}"


def test_ece_near_zero_for_calibrated_predictor():
    mu = np.zeros(N)
    sigma = np.ones(N)
    y = rng.normal(mu, sigma, N)
    ece = ece_regression(y, mu, sigma, n_bins=20)
    assert ece < 0.05, f"ECE should be near 0 for calibrated Gaussian, got {ece:.4f}"


def test_crps_decreases_with_better_sigma():
    y = rng.normal(0, 1, N)
    mu = np.zeros(N)
    crps_good = crps(y, mu, np.ones(N))
    crps_bad = crps(y, mu, np.full(N, 5.0))
    assert crps_good < crps_bad


def test_evaluate_model_all_finite():
    y = rng.normal(0, 1, N)
    mu = rng.normal(0, 1, N)
    sigma = np.abs(rng.normal(1, 0.2, N)) + 0.01
    metrics = evaluate_model(y, mu, sigma)
    for name, val in metrics.to_dict().items():
        assert np.isfinite(val), f"Metric '{name}' is not finite: {val}"


def test_evaluate_model_deterministic():
    y = rng.normal(0, 1, N)
    mu = rng.normal(0, 1, N)
    sigma = np.abs(rng.normal(1, 0.2, N)) + 0.01
    m1 = evaluate_model(y, mu, sigma)
    m2 = evaluate_model(y, mu, sigma)
    for key in m1.to_dict():
        assert abs(m1.to_dict()[key] - m2.to_dict()[key]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
