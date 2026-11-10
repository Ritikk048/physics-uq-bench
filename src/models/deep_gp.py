"""
Deep Kernel Learning (DKL) via GPyTorch's approximate GP inference.

Architecture: DNN feature extractor + Sparse Variational GP
  - Feature extractor: 2 → 64 → 32 → 8 (Tanh activations)
  - SVGP: RBF kernel in 8-D feature space, 128 inducing points
  - Trained end-to-end with VariationalELBO

Note: The benchmark uses the DKL implementation in scripts/quick_results.py.
      This file contains the original DeepGP skeleton (Salimbeni & Deisenroth,
      NeurIPS 2017) which had intractable shape errors with output_dims=1.
"""

import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from typing import Tuple


class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims: int, output_dims: int, num_inducing: int = 128):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]),
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean(batch_shape=torch.Size([output_dims]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([output_dims]), ard_num_dims=input_dims),
            batch_shape=torch.Size([output_dims]),
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class DeepGPModel(DeepGP):
    """
    2-layer Deep GP with variational inference.

    Use DeepGPModel.likelihood (a GaussianLikelihood) with ELBO loss during training.
    Call predict_with_uncertainty for mean + std at test time.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 3, num_inducing: int = 128):
        super().__init__()
        self.hidden_layer = DGPHiddenLayer(input_dim, hidden_dim, num_inducing)
        self.last_layer = DGPHiddenLayer(hidden_dim, 1, num_inducing)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x: Tensor) -> MultivariateNormal:
        hidden_rep = self.hidden_layer(x)
        output = self.last_layer(hidden_rep)
        return output

    def predict_mean(self, x: Tensor) -> Tensor:
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self(x))
        return preds.mean.unsqueeze(-1)

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns predictive mean and std using posterior sampling.
        """
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(n_samples):
            preds = self.likelihood(self(x))
            mean = preds.mean.mean(0).unsqueeze(-1)
            std = preds.variance.mean(0).sqrt().unsqueeze(-1)
        return mean, std
