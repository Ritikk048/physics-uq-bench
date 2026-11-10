"""
Ablation study runner.

Systematically varies one hyperparameter at a time (keeping others at best-found values)
to quantify each component's contribution to UQ performance.

Produces:
  - Per-ablation metric tables (CSV)
  - Sensitivity plots (metric vs ablation variable)
  - Bootstrap confidence intervals across ablation runs
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
import matplotlib.pyplot as plt

from ..data.generators import PhysicsDataBatch
from ..models import MODEL_REGISTRY
from ..training.trainer import Trainer
from ..evaluation.metrics import evaluate_model


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


class AblationRunner:
    """
    One-at-a-time (OAT) ablation study for a single model type.

    Parameters
    ----------
    model_type: str — 'pinn' | 'mc_dropout' | 'deep_gp' | 'refann'
    base_params: dict — baseline hyperparameter config (e.g. from HPO)
    ablation_grid: dict — {param_name: [v1, v2, v3, ...]}
    data: PhysicsDataBatch
    pde_type: str
    pde_params: dict
    n_epochs: int — training epochs per ablation
    n_seeds: int — number of random seeds per ablation run
    metric: str — primary metric
    results_dir: str
    device: str
    """

    def __init__(
        self,
        model_type: str,
        base_params: dict,
        ablation_grid: dict,
        data: PhysicsDataBatch,
        pde_type: str = "heat",
        pde_params: Optional[dict] = None,
        n_epochs: int = 5000,
        n_seeds: int = 3,
        metric: str = "gaussian_nll",
        results_dir: str = "results/ablation",
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.base_params = base_params
        self.ablation_grid = ablation_grid
        self.data = data
        self.pde_type = pde_type
        self.pde_params = pde_params or {}
        self.n_epochs = n_epochs
        self.n_seeds = n_seeds
        self.metric = metric
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _run_single(self, params: dict, seed: int) -> float:
        torch.manual_seed(seed)
        np.random.seed(seed)

        lr = params.pop("lr", 1e-3)
        ModelClass = MODEL_REGISTRY[self.model_type]

        if self.model_type == "pinn":
            model = ModelClass(
                input_dim=self.data.x_col.shape[1],
                pde_type=self.pde_type,
                pde_params=self.pde_params,
                **{k: v for k, v in params.items() if k not in ("lambda_pde", "lambda_bc")},
            )
            trainer = Trainer(
                model, model_type="pinn", lr=lr, epochs=self.n_epochs,
                lambda_pde=params.get("lambda_pde", 1.0),
                lambda_bc=params.get("lambda_bc", 10.0),
                device=self.device,
            )
        else:
            model = ModelClass(input_dim=self.data.x_col.shape[1], **params)
            trainer = Trainer(
                model, model_type=self.model_type, lr=lr,
                epochs=self.n_epochs, device=self.device,
            )

        trainer.fit(self.data)

        x_test = self.data.x_test.to(self.device)
        mean, std = model.predict_with_uncertainty(x_test, n_samples=50)
        mean_np = mean.detach().cpu().numpy().ravel()
        std_np = std.detach().cpu().numpy().ravel()
        y_true = self.data.u_test_noisy.numpy().ravel()

        return evaluate_model(y_true, mean_np, std_np).to_dict()[self.metric]

    def run(self) -> pd.DataFrame:
        """
        Run the full OAT ablation study.

        Returns a DataFrame with columns:
            param_name | param_value | metric_mean | metric_lo | metric_hi
        """
        rows = []

        for param_name, values in self.ablation_grid.items():
            print(f"\nAblating '{param_name}': {values}")

            for val in values:
                scores = []
                for seed in range(self.n_seeds):
                    params = deepcopy(self.base_params)
                    params[param_name] = val
                    try:
                        score = self._run_single(params, seed)
                        scores.append(score)
                    except Exception as e:
                        print(f"  seed {seed} failed: {e}")

                if scores:
                    scores = np.array(scores)
                    lo, hi = _bootstrap_ci(scores)
                    rows.append({
                        "param_name": param_name,
                        "param_value": val,
                        f"{self.metric}_mean": scores.mean(),
                        f"{self.metric}_lo": lo,
                        f"{self.metric}_hi": hi,
                        "n_runs": len(scores),
                    })
                    print(f"  {param_name}={val}: {scores.mean():.4f} [{lo:.4f}, {hi:.4f}]")

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / f"ablation_{self.model_type}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nAblation results saved to {csv_path}")

        self._plot_ablation(df)
        return df

    def _plot_ablation(self, df: pd.DataFrame):
        """Sensitivity plots: one panel per ablated parameter."""
        param_names = df["param_name"].unique()
        n = len(param_names)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        metric_col = f"{self.metric}_mean"
        lo_col = f"{self.metric}_lo"
        hi_col = f"{self.metric}_hi"

        for ax, pname in zip(axes, param_names):
            sub = df[df["param_name"] == pname].sort_values("param_value")
            xs = sub["param_value"].values.astype(float)
            ys = sub[metric_col].values
            los = sub[lo_col].values
            his = sub[hi_col].values

            ax.plot(xs, ys, "o-", color="#185FA5", lw=2)
            ax.fill_between(xs, los, his, alpha=0.2, color="#185FA5")
            ax.axhline(ys[0], color="gray", ls="--", lw=1, label="Baseline")
            ax.set_xlabel(pname)
            ax.set_ylabel(self.metric)
            ax.set_title(f"Sensitivity: {pname}")

        fig.suptitle(f"OAT Ablation — {self.model_type}", fontsize=11)
        fig.tight_layout()

        fig_path = self.results_dir / f"ablation_{self.model_type}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Ablation plot saved to {fig_path}")
