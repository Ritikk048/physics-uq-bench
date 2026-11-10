"""
ReFANN — Residual-Feedback Artificial Neural Network for uncertainty quantification.

ReFANN uses a two-head architecture where the second head predicts the residual
of the first, iteratively refining both mean and variance estimates.

Reference: Inspired by "ReFANN: A Refined Feedforward Artificial Neural Network
           for Uncertainty Estimation" approach combining ensemble-like residuals
           with a single forward pass.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.Linear(dim, dim),
        )
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class ReFANN(nn.Module):
    """
    ReFANN: shared backbone → primary head (mu1, log_sigma1)
                            → residual correction via secondary backbone
                            → refined output (mu, log_sigma)

    The key idea: the secondary backbone sees the primary residual prediction
    concatenated with the input, allowing it to model structured residuals.

    Args:
        input_dim: dimensionality of spatial-temporal input
        hidden_dim: neurons per layer
        n_layers: depth of each sub-network
        n_refine: number of refinement iterations
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_refine: int = 2,
    ):
        super().__init__()
        self.n_refine = n_refine

        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            *[ResidualBlock(hidden_dim) for _ in range(n_layers - 1)],
        )

        # Primary prediction head
        self.primary_mu = nn.Linear(hidden_dim, 1)
        self.primary_log_sigma = nn.Linear(hidden_dim, 1)

        # Refinement modules: each takes (features || previous_mu || previous_log_sigma)
        refine_in = hidden_dim + 2
        self.refinement_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(refine_in, hidden_dim),
                    nn.Tanh(),
                    *[ResidualBlock(hidden_dim) for _ in range(n_layers - 1)],
                )
                for _ in range(n_refine)
            ]
        )
        self.refine_mu_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_refine)]
        )
        self.refine_log_sigma_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_refine)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns final (mu, log_sigma) after n_refine iterations.
        """
        features = self.encoder(x)

        mu = self.primary_mu(features)
        log_sigma = self.primary_log_sigma(features).clamp(-6, 2)

        for i in range(self.n_refine):
            context = torch.cat([features, mu.detach(), log_sigma.detach()], dim=-1)
            h = self.refinement_nets[i](context)
            # Additive residual corrections
            mu = mu + self.refine_mu_heads[i](h)
            log_sigma = (log_sigma + self.refine_log_sigma_heads[i](h)).clamp(-6, 2)

        return mu, log_sigma

    def predict_mean(self, x: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            mu, _ = self.forward(x)
        return mu

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """
        ReFANN is deterministic — returns aleatoric uncertainty only.
        n_samples is ignored; kept for interface compatibility.
        """
        self.eval()
        with torch.no_grad():
            mu, log_sigma = self.forward(x)
        return mu, torch.exp(log_sigma)
