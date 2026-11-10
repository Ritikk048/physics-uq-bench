"""
MC Dropout network for approximate Bayesian inference.

Dropout is kept ON at test time; running N forward passes gives
an empirical predictive distribution (epistemic + aleatoric).

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class MCDropoutNet(nn.Module):
    """
    Fully-connected network with MC Dropout.

    At inference, call predict_with_uncertainty(x, n_samples=100) to get
    predictive mean and total uncertainty (std over samples).

    Args:
        input_dim: input feature dimension
        hidden_dim: neurons per hidden layer
        n_layers: number of hidden layers
        p_drop: dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.p_drop = p_drop

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=p_drop))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-6, 2)
        return mu, log_sigma

    def predict_mean(self, x: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            mu, _ = self.forward(x)
        return mu

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        Stochastic forward passes with dropout active.

        Returns:
            mean: (N, 1) predictive mean
            std:  (N, 1) total predictive std (epistemic + aleatoric)
        """
        self.train()  # keep dropout active
        samples = []
        sigmas = []

        with torch.no_grad():
            for _ in range(n_samples):
                mu_s, log_sig_s = self.forward(x)
                samples.append(mu_s)
                sigmas.append(torch.exp(log_sig_s))

        samples = torch.stack(samples, dim=0)      # (S, N, 1)
        sigmas = torch.stack(sigmas, dim=0)        # (S, N, 1)

        mean = samples.mean(0)
        # Total variance = epistemic + mean aleatoric
        epistemic_var = samples.var(0)
        aleatoric_var = (sigmas**2).mean(0)
        total_std = (epistemic_var + aleatoric_var).sqrt()

        return mean, total_std

    def epistemic_uncertainty(self, x: Tensor, n_samples: int = 100) -> Tensor:
        """Pure epistemic uncertainty (variance of mean predictions)."""
        self.train()
        means = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, _ = self.forward(x)
                means.append(mu)
        means = torch.stack(means, dim=0)
        return means.var(0).sqrt()
