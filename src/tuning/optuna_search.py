"""
Optuna-based hyperparameter optimisation for all benchmark models.

Each model type exposes a search space and an objective function.
Training-stability pruning (MedianPruner) is used to terminate bad trials early.

Supported search spaces:
  - PINN:       hidden_dim, n_layers, lr, lambda_pde, lambda_bc, omega
  - MC Dropout: hidden_dim, n_layers, lr, p_drop
  - Deep GP:    hidden_dim (GP latent dim), num_inducing, lr
  - ReFANN:     hidden_dim, n_layers, lr, n_refine
"""

import optuna
import torch
import numpy as np
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import json

from ..data.generators import PhysicsDataBatch
from ..models import MODEL_REGISTRY
from ..training.trainer import Trainer
from ..evaluation.metrics import evaluate_model


def _suggest_pinn(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 256, step=32),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "omega": trial.suggest_float("omega", 10.0, 60.0, step=5.0),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "lambda_pde": trial.suggest_float("lambda_pde", 0.1, 10.0, log=True),
        "lambda_bc": trial.suggest_float("lambda_bc", 1.0, 100.0, log=True),
    }


def _suggest_mc_dropout(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "p_drop": trial.suggest_float("p_drop", 0.05, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }


def _suggest_deep_gp(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 2, 8),
        "num_inducing": trial.suggest_int("num_inducing", 64, 256, step=32),
        "lr": trial.suggest_float("lr", 1e-3, 5e-2, log=True),
    }


def _suggest_refann(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "n_refine": trial.suggest_int("n_refine", 1, 3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }


SEARCH_SPACES = {
    "pinn": _suggest_pinn,
    "mc_dropout": _suggest_mc_dropout,
    "deep_gp": _suggest_deep_gp,
    "refann": _suggest_refann,
}


class HyperparameterSearch:
    """
    Runs Optuna HPO for a given model type on a PhysicsDataBatch.

    Parameters
    ----------
    model_type: str — one of 'pinn', 'mc_dropout', 'deep_gp', 'refann'
    data: PhysicsDataBatch
    pde_type: str — 'heat' | 'wave' | 'poisson' (for PINN residual)
    pde_params: dict — PDE parameters (e.g. {'alpha': 0.01})
    n_trials: int — number of Optuna trials
    timeout: int — max seconds for the study
    n_epochs_fast: int — reduced epoch budget used during HPO
    metric: str — Optuna objective ('gaussian_nll' | 'rmse' | 'crps')
    device: str
    results_dir: str — where to save study artifacts
    """

    def __init__(
        self,
        model_type: str,
        data: PhysicsDataBatch,
        pde_type: str = "heat",
        pde_params: Optional[dict] = None,
        n_trials: int = 50,
        timeout: int = 3600,
        n_epochs_fast: int = 3000,
        metric: str = "gaussian_nll",
        device: str = "cpu",
        results_dir: str = "results",
    ):
        self.model_type = model_type
        self.data = data
        self.pde_type = pde_type
        self.pde_params = pde_params or {}
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_epochs_fast = n_epochs_fast
        self.metric = metric
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_params: Optional[dict] = None
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        suggest_fn = SEARCH_SPACES[self.model_type]
        params = suggest_fn(trial)

        try:
            lr = params.pop("lr")
            ModelClass = MODEL_REGISTRY[self.model_type]

            if self.model_type == "pinn":
                model = ModelClass(
                    input_dim=self.data.x_col.shape[1],
                    pde_type=self.pde_type,
                    pde_params=self.pde_params,
                    **{k: v for k, v in params.items() if k not in ("lambda_pde", "lambda_bc")},
                )
                trainer = Trainer(
                    model, model_type="pinn", lr=lr, epochs=self.n_epochs_fast,
                    lambda_pde=params.get("lambda_pde", 1.0),
                    lambda_bc=params.get("lambda_bc", 10.0),
                    device=self.device,
                )
            else:
                model = ModelClass(
                    input_dim=self.data.x_col.shape[1], **params
                )
                trainer = Trainer(
                    model, model_type=self.model_type, lr=lr,
                    epochs=self.n_epochs_fast, device=self.device,
                )

            trainer.fit(self.data)

            x_test = self.data.x_test.to(self.device)
            mean, std = model.predict_with_uncertainty(x_test, n_samples=50)
            mean_np = mean.detach().cpu().numpy().ravel()
            std_np = std.detach().cpu().numpy().ravel()
            y_true = self.data.u_test_noisy.numpy().ravel()

            from ..evaluation.metrics import evaluate_model
            metrics = evaluate_model(y_true, mean_np, std_np)
            score = metrics.to_dict()[self.metric]

            if np.isnan(score) or np.isinf(score):
                raise optuna.exceptions.TrialPruned()

            return score

        except Exception as e:
            raise optuna.exceptions.TrialPruned()

    def run(self) -> dict:
        """
        Execute the Optuna study.

        Returns best hyperparameters dict and saves JSON to results_dir.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        out = {
            "model_type": self.model_type,
            "metric": self.metric,
            "best_value": self.study.best_value,
            "best_params": self.best_params,
            "n_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }

        json_path = self.results_dir / f"best_params_{self.model_type}.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        return out
