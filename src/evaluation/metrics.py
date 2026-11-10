"""
Evaluation metrics for probabilistic forecasts.

Implements:
  - Gaussian NLL (negative log-likelihood under Gaussian predictive)
  - RMSE
  - CRPS (Continuous Ranked Probability Score) via properscoring
  - ECE (Expected Calibration Error) for regression
  - Coverage at 50% and 90% credible intervals
  - Sharpness (mean predictive std)
  - Interval Score (penalises wide intervals)
  - Skill Score relative to climatological baseline

All metrics accept numpy arrays for compatibility with both
PyTorch and PyMC-based models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings


@dataclass
class UQMetrics:
    """Container for all computed metrics."""

    gaussian_nll: float = np.nan
    rmse: float = np.nan
    mae: float = np.nan
    crps: float = np.nan
    ece: float = np.nan
    coverage_50: float = np.nan
    coverage_90: float = np.nan
    sharpness: float = np.nan
    interval_score_90: float = np.nan

    def to_dict(self) -> Dict[str, float]:
        return {
            "gaussian_nll": self.gaussian_nll,
            "rmse": self.rmse,
            "mae": self.mae,
            "crps": self.crps,
            "ece": self.ece,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
            "sharpness": self.sharpness,
            "interval_score_90": self.interval_score_90,
        }

    def __repr__(self) -> str:
        lines = ["UQMetrics:"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:<22}: {v:.6f}")
        return "\n".join(lines)


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Mean Gaussian negative log-likelihood.

    NLL = 0.5 * [ log(2*pi*sigma^2) + (y - mu)^2 / sigma^2 ]
    """
    sigma = np.maximum(sigma, 1e-8)
    nll = 0.5 * (np.log(2 * np.pi * sigma**2) + ((y_true - mu) ** 2) / sigma**2)
    return float(nll.mean())


def rmse(y_true: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - mu) ** 2)))


def mae(y_true: np.ndarray, mu: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - mu)))


def crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    CRPS under a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [ (y-mu)/sigma * (2*Phi((y-mu)/sigma) - 1)
                                        + 2*phi((y-mu)/sigma) - 1/sqrt(pi) ]

    where Phi is the standard normal CDF and phi its PDF.
    """
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    crps_vals = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
    return float(crps_vals.mean())


def coverage_at_level(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.9
) -> float:
    """
    Fraction of true values falling inside the [level] credible interval.

    Ideal: coverage == level (perfectly calibrated).
    Over-confident: coverage < level.
    Under-confident: coverage > level.
    """
    from scipy.stats import norm

    alpha = 1.0 - level
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = np.logical_and(y_true >= lower, y_true <= upper)
    return float(covered.mean())


def interval_score(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.9
) -> float:
    """
    Winkler interval score: rewards sharp intervals that still cover the truth.

    IS = (upper - lower) + 2/alpha * [lower - y]_+ + 2/alpha * [y - upper]_+
    """
    from scipy.stats import norm

    alpha = 1.0 - level
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - upper, 0)
    return float((width + penalty_low + penalty_high).mean())


def ece_regression(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Expected Calibration Error for regression via confidence-coverage plot.

    For each confidence level p in [0,1], compute coverage of the central
    p-interval and measure deviation from ideal diagonal.
    """
    from scipy.stats import norm

    levels = np.linspace(0.05, 0.95, n_bins)
    expected_cov = levels
    observed_cov = np.array([coverage_at_level(y_true, mu, sigma, lv) for lv in levels])
    return float(np.mean(np.abs(observed_cov - expected_cov)))


def sharpness(sigma: np.ndarray) -> float:
    """Mean predictive standard deviation (lower is sharper, all else equal)."""
    return float(sigma.mean())


def evaluate_model(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_calibration_bins: int = 20,
) -> UQMetrics:
    """
    Compute the full UQMetrics suite from (true, predicted mean, predicted std).

    Args:
        y_true: (N,) ground-truth values
        mu: (N,) predictive means
        sigma: (N,) predictive standard deviations (must be > 0)
        n_calibration_bins: bins for ECE computation

    Returns:
        UQMetrics dataclass
    """
    y_true = y_true.ravel()
    mu = mu.ravel()
    sigma = np.maximum(sigma.ravel(), 1e-8)

    return UQMetrics(
        gaussian_nll=gaussian_nll(y_true, mu, sigma),
        rmse=rmse(y_true, mu),
        mae=mae(y_true, mu),
        crps=crps(y_true, mu, sigma),
        ece=ece_regression(y_true, mu, sigma, n_calibration_bins),
        coverage_50=coverage_at_level(y_true, mu, sigma, 0.50),
        coverage_90=coverage_at_level(y_true, mu, sigma, 0.90),
        sharpness=sharpness(sigma),
        interval_score_90=interval_score(y_true, mu, sigma, 0.90),
    )
