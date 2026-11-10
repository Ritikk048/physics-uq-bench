"""
Calibration analysis and plots.

Produces:
  - Reliability diagram (observed coverage vs expected coverage)
  - Sharpness histogram
  - PIT (Probability Integral Transform) histogram — uniform PIT = perfect calibration
  - Rank histogram for ensemble-based models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, uniform
from typing import Optional, Tuple
from pathlib import Path


class CalibrationAnalyzer:
    """
    Calibration diagnostics for a probabilistic regression model.

    Parameters
    ----------
    y_true: (N,) ground-truth values
    mu: (N,) predictive means
    sigma: (N,) predictive standard deviations
    model_name: label used in plot titles
    """

    def __init__(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        model_name: str = "model",
    ):
        self.y = y_true.ravel()
        self.mu = mu.ravel()
        self.sigma = np.maximum(sigma.ravel(), 1e-8)
        self.name = model_name

        # Standardised residuals
        self.z = (self.y - self.mu) / self.sigma

        # PIT values
        self.pit = norm.cdf(self.z)

    def coverage_curve(self, n_levels: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (expected_coverage, observed_coverage) arrays for reliability diagram.
        A perfectly calibrated model lies on the diagonal.
        """
        levels = np.linspace(0.01, 0.99, n_levels)
        observed = np.array(
            [
                np.mean(np.abs(self.z) <= norm.ppf(0.5 + lv / 2))
                for lv in levels
            ]
        )
        return levels, observed

    def pit_histogram(self, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        PIT values should follow U[0,1] for a calibrated model.
        Returns (bin_centres, counts / expected_count).
        """
        counts, edges = np.histogram(self.pit, bins=n_bins, range=(0, 1), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return centres, counts

    def plot_full_diagnostics(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 4),
    ) -> plt.Figure:
        """
        3-panel calibration figure:
          1. Reliability diagram
          2. PIT histogram
          3. Residual distribution vs standard normal
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # --- Reliability diagram ---
        ax1 = fig.add_subplot(gs[0])
        levels, observed = self.coverage_curve()
        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        ax1.plot(levels, observed, color="#185FA5", lw=2, label=self.name)
        ax1.fill_between(levels, levels, observed, alpha=0.15, color="#185FA5")
        ax1.set_xlabel("Expected coverage")
        ax1.set_ylabel("Observed coverage")
        ax1.set_title("Reliability diagram")
        ax1.legend(fontsize=9)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # --- PIT histogram ---
        ax2 = fig.add_subplot(gs[1])
        centres, density = self.pit_histogram()
        ax2.bar(centres, density, width=0.9 / len(centres), color="#1D9E75", alpha=0.7)
        ax2.axhline(1.0, color="k", lw=1, ls="--", label="Ideal U[0,1]")
        ax2.set_xlabel("PIT value")
        ax2.set_ylabel("Density")
        ax2.set_title("PIT histogram")
        ax2.set_xlim(0, 1)

        # --- Standardised residual distribution ---
        ax3 = fig.add_subplot(gs[2])
        z_clip = np.clip(self.z, -4, 4)
        ax3.hist(z_clip, bins=40, density=True, color="#7F77DD", alpha=0.7, label="Empirical")
        z_range = np.linspace(-4, 4, 200)
        ax3.plot(z_range, norm.pdf(z_range), "k--", lw=1.5, label="N(0,1)")
        ax3.set_xlabel("Standardised residual")
        ax3.set_ylabel("Density")
        ax3.set_title("Residual distribution")
        ax3.legend(fontsize=9)

        fig.suptitle(f"Calibration diagnostics — {self.name}", fontsize=11, y=1.02)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def summary_stats(self) -> dict:
        """Scalar calibration summary: D_KL(PIT || U[0,1]), ECE, over/under-confidence."""
        levels, observed = self.coverage_curve()
        ece = float(np.mean(np.abs(observed - levels)))
        over = float(np.mean(observed - levels))  # positive = overconfident
        pit_std = float(self.pit.std())
        return {
            "ece": ece,
            "mean_calibration_bias": over,
            "pit_std": pit_std,
            "z_mean": float(self.z.mean()),
            "z_std": float(self.z.std()),
        }
