"""
MCMC-based Bayesian Neural Network via PyMC.

Uses NUTS (No-U-Turn Sampler) on a shallow MLP with horseshoe priors.
Predictive uncertainty is computed from the posterior predictive distribution.

For large-scale use, the model is wrapped in a lightweight PyTorch interface
that delegates sampling to PyMC and stores posterior samples for fast prediction.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Optional
import warnings


class MCMCBayesianNN:
    """
    BNN trained with NUTS via PyMC.

    Parameters
    ----------
    input_dim: int
        Dimension of input features
    hidden_dim: int
        Hidden units per layer (1-layer MLP for tractability)
    n_samples: int
        Number of MCMC draws to keep
    n_tune: int
        Number of tuning (warmup) steps
    target_accept: float
        NUTS target acceptance rate
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        n_samples: int = 500,
        n_tune: int = 500,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.trace = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Run NUTS on (X, y) data.

        X: (N, input_dim) numpy array
        y: (N,) numpy array of targets
        """
        try:
            import pymc as pm
        except ImportError:
            raise ImportError("Install pymc: pip install pymc")

        n, d = X.shape
        h = self.hidden_dim

        with pm.Model() as self._model:
            # Horseshoe prior on weights
            tau = pm.HalfNormal("tau", sigma=1.0)

            W1 = pm.Normal("W1", mu=0, sigma=tau, shape=(d, h))
            b1 = pm.Normal("b1", mu=0, sigma=1, shape=(h,))
            W2 = pm.Normal("W2", mu=0, sigma=tau, shape=(h, 1))
            b2 = pm.Normal("b2", mu=0, sigma=1, shape=(1,))
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)

            X_data = pm.Data("X_data", X)
            hidden = pm.math.tanh(X_data @ W1 + b1)
            mu_pred = (hidden @ W2 + b2).squeeze(-1)

            y_obs = pm.Normal("y_obs", mu=mu_pred, sigma=sigma_obs, observed=y)

            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=True,
                return_inferencedata=True,
            )

        self._is_fitted = True

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 500
    ) -> Tuple[Tensor, Tensor]:
        """
        Posterior predictive using stored MCMC samples.
        Returns mean and std as tensors.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            raise ImportError("Install pymc and arviz.")

        X_np = x.numpy()
        with self._model:
            pm.set_data({"X_data": X_np})
            ppc = pm.sample_posterior_predictive(
                self.trace, random_seed=self.random_seed, progressbar=False
            )

        samples = ppc.posterior_predictive["y_obs"].values  # (chains, draws, N)
        samples = samples.reshape(-1, samples.shape[-1])    # (chains*draws, N)

        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        return (
            torch.tensor(mean, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(std, dtype=torch.float32).unsqueeze(-1),
        )

    def predict_mean(self, x: Tensor) -> Tensor:
        mean, _ = self.predict_with_uncertainty(x)
        return mean

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
