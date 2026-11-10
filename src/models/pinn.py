"""
Physics-Informed Neural Network (PINN) with heteroscedastic uncertainty.

Architecture: fully-connected MLP with sinusoidal activations (Siren-style)
that outputs both mean and log-variance, trained with:
  - PDE residual loss (computed via autograd)
  - Boundary/IC loss
  - Gaussian NLL on observed data
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class SineActivation(nn.Module):
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * x)


class PINN(nn.Module):
    """
    Heteroscedastic PINN: outputs (mu, log_sigma) at every spatial-temporal point.

    Args:
        input_dim: dimensionality of (x, t) or (x, y) inputs
        hidden_dim: neurons per hidden layer
        n_layers: number of hidden layers
        omega: frequency for sinusoidal activations
        pde_type: 'heat' | 'wave' | 'poisson'  — determines residual computation
        pde_params: dict of PDE parameters (e.g. {'alpha': 0.01})
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 4,
        omega: float = 30.0,
        pde_type: str = "heat",
        pde_params: Optional[dict] = None,
    ):
        super().__init__()
        self.pde_type = pde_type
        self.pde_params = pde_params or {}

        layers = [nn.Linear(input_dim, hidden_dim), SineActivation(omega)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), SineActivation(omega)]
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

        self._init_weights(omega)

    def _init_weights(self, omega: float):
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    nn.init.uniform_(layer.weight, -1 / layer.in_features, 1 / layer.in_features)
                else:
                    bound = (6 / layer.in_features) ** 0.5 / omega
                    nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-6, 2)
        return mu, log_sigma

    def predict_mean(self, x: Tensor) -> Tensor:
        mu, _ = self.forward(x)
        return mu

    def predict_with_uncertainty(self, x: Tensor, n_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """Returns predictive mean and std (aleatoric only for PINN)."""
        mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def pde_residual(self, x_col: Tensor) -> Tensor:
        """
        Compute PDE residual at collocation points via automatic differentiation.
        x_col: (N, 2) tensor of (x, t) or (x, y) points, requires_grad=True
        """
        x_col = x_col.requires_grad_(True)
        mu, _ = self.forward(x_col)

        if self.pde_type == "heat":
            return self._heat_residual(x_col, mu)
        elif self.pde_type == "wave":
            return self._wave_residual(x_col, mu)
        elif self.pde_type == "poisson":
            return self._poisson_residual(x_col, mu)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

    def _heat_residual(self, xt: Tensor, u: Tensor) -> Tensor:
        alpha = self.pde_params.get("alpha", 0.01)
        grad_u = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        return u_t - alpha * u_xx

    def _wave_residual(self, xt: Tensor, u: Tensor) -> Tensor:
        c = self.pde_params.get("c", 1.0)
        grad_u = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        u_tt = torch.autograd.grad(u_t.sum(), xt, create_graph=True)[0][:, 1:2]
        return u_tt - c**2 * u_xx

    def _poisson_residual(self, xy: Tensor, u: Tensor) -> Tensor:
        import numpy as np
        grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
        x, y = xy[:, 0:1], xy[:, 1:2]
        f = 2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return -u_xx - u_yy - f
