"""
Loss functions for physics-informed and probabilistic training.

PINNLoss:
  Total = lambda_pde * MSE(residual) + lambda_bc * MSE(BC) + lambda_nll * Gaussian_NLL(obs)

GaussianNLLLoss:
  Standard Gaussian NLL for heteroscedastic regression.

ELBOLoss:
  ELBO for variational GP training via GPyTorch's VariationalELBO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import gpytorch


class GaussianNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss.

    NLL = 0.5 * [ log(sigma^2) + (y - mu)^2 / sigma^2 ]
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu: Tensor, log_sigma: Tensor, target: Tensor) -> Tensor:
        sigma_sq = torch.exp(2 * log_sigma)
        loss = 0.5 * (log_sigma + 0.5 * torch.log(torch.tensor(2 * 3.14159))
                      + (target - mu) ** 2 / sigma_sq)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PINNLoss(nn.Module):
    """
    Composite PINN loss:
      L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_nll * L_nll

    Args:
        lambda_pde: weight for PDE residual
        lambda_bc: weight for boundary / initial conditions
        lambda_nll: weight for Gaussian NLL on observed data
                    (set 0 to train with pure MSE)
    """

    def __init__(
        self,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_nll: float = 1.0,
    ):
        super().__init__()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_nll = lambda_nll
        self.nll_loss = GaussianNLLLoss()

    def forward(
        self,
        pde_residual: Tensor,
        mu_bc: Tensor,
        u_bc_true: Tensor,
        log_sigma_bc: Tensor,
        mu_obs: Optional[Tensor] = None,
        log_sigma_obs: Optional[Tensor] = None,
        u_obs: Optional[Tensor] = None,
    ) -> dict:
        l_pde = torch.mean(pde_residual**2)
        l_bc = F.mse_loss(mu_bc, u_bc_true)

        loss = self.lambda_pde * l_pde + self.lambda_bc * l_bc

        l_nll = torch.tensor(0.0)
        if mu_obs is not None and u_obs is not None:
            l_nll = self.nll_loss(mu_obs, log_sigma_obs, u_obs)
            loss = loss + self.lambda_nll * l_nll

        return {
            "loss": loss,
            "l_pde": l_pde.item(),
            "l_bc": l_bc.item(),
            "l_nll": l_nll.item() if isinstance(l_nll, Tensor) else l_nll,
        }


class ELBOLoss:
    """
    Variational ELBO loss for GPyTorch Deep GP training.
    Wraps gpytorch.mlls.VariationalELBO.
    """

    def __init__(self, model, likelihood, num_data: int):
        self.mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
        )

    def __call__(self, output, target: Tensor) -> Tensor:
        return -self.mll(output, target.squeeze(-1))
