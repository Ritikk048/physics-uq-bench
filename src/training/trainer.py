"""
Unified training loop for PINN, MC Dropout, Deep GP, and ReFANN models.

Each model type uses a different loss but the same outer loop structure,
enabling fair head-to-head comparison.

Training stability features:
  - Cosine annealing LR schedule
  - Gradient clipping
  - Early stopping on validation NLL
  - Loss component logging
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional, Dict, Callable
import time

from ..data.generators import PhysicsDataBatch
from .losses import PINNLoss, GaussianNLLLoss, ELBOLoss
from ..evaluation.metrics import evaluate_model


class EarlyStopping:
    def __init__(self, patience: int = 2000, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = np.inf
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class Trainer:
    """
    Unified trainer supporting PINN, MC Dropout, Deep GP, and ReFANN.

    Args:
        model: one of the models from src.models
        model_type: 'pinn' | 'mc_dropout' | 'deep_gp' | 'refann'
        lr: initial learning rate
        epochs: maximum training epochs
        patience: early stopping patience
        lambda_pde: PINN PDE loss weight (ignored for non-PINN)
        lambda_bc: PINN BC loss weight (ignored for non-PINN)
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model,
        model_type: str = "pinn",
        lr: float = 1e-3,
        epochs: int = 20000,
        patience: int = 2000,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ):
        self.model_type = model_type
        self.epochs = epochs
        self.device = torch.device(device)
        self.grad_clip = grad_clip

        if model_type == "mcmc_bnn":
            self.model = model
            return

        self.model = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.early_stop = EarlyStopping(patience=patience)

        if model_type == "pinn":
            self.loss_fn = PINNLoss(lambda_pde=lambda_pde, lambda_bc=lambda_bc)
        elif model_type in ("mc_dropout", "refann"):
            self.loss_fn = GaussianNLLLoss()
        elif model_type == "deep_gp":
            pass  # set during fit()

        self.history: Dict[str, list] = {
            "loss": [], "val_nll": [], "l_pde": [], "l_bc": [], "l_nll": []
        }

    def fit(self, data: PhysicsDataBatch) -> Dict[str, list]:
        """
        Train on a PhysicsDataBatch.

        Returns:
            history dict with training curves
        """
        if self.model_type == "mcmc_bnn":
            return self._fit_mcmc(data)
        elif self.model_type == "deep_gp":
            return self._fit_deep_gp(data)
        elif self.model_type == "pinn":
            return self._fit_pinn(data)
        else:
            return self._fit_nn(data)

    def _fit_pinn(self, data: PhysicsDataBatch) -> dict:
        x_col = data.x_col.to(self.device)
        x_bc = data.x_bc.to(self.device)
        u_bc = data.u_bc.to(self.device)
        x_test = data.x_test.to(self.device)
        u_test_noisy = data.u_test_noisy.to(self.device)

        bar = tqdm(range(self.epochs), desc="PINN", ncols=90)
        for epoch in bar:
            self.model.train()
            self.optimizer.zero_grad()

            residual = self.model.pde_residual(x_col.clone().requires_grad_(True))
            mu_bc, log_sig_bc = self.model(x_bc)

            loss_dict = self.loss_fn(
                pde_residual=residual,
                mu_bc=mu_bc,
                u_bc_true=u_bc,
                log_sigma_bc=log_sig_bc,
            )
            loss = loss_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 200 == 0:
                self.model.eval()
                with torch.no_grad():
                    mu_val, log_sig_val = self.model(x_test)
                    val_nll = GaussianNLLLoss()(mu_val, log_sig_val, u_test_noisy).item()

                self.history["loss"].append(loss.item())
                self.history["val_nll"].append(val_nll)
                self.history["l_pde"].append(loss_dict["l_pde"])
                self.history["l_bc"].append(loss_dict["l_bc"])

                bar.set_postfix(loss=f"{loss.item():.4f}", val_nll=f"{val_nll:.4f}")

                if self.early_stop.step(val_nll):
                    bar.write(f"Early stop at epoch {epoch}")
                    break

        return self.history

    def _fit_nn(self, data: PhysicsDataBatch) -> dict:
        x_train = data.x_bc.to(self.device)
        u_train = data.u_bc.to(self.device)
        x_val = data.x_test.to(self.device)
        u_val = data.u_test_noisy.to(self.device)

        bar = tqdm(range(self.epochs), desc=self.model_type.upper(), ncols=90)
        for epoch in bar:
            self.model.train()
            self.optimizer.zero_grad()
            mu, log_sigma = self.model(x_train)
            loss = self.loss_fn(mu, log_sigma, u_train)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 200 == 0:
                self.model.eval()
                with torch.no_grad():
                    mu_val, log_sig_val = self.model(x_val)
                    val_nll = self.loss_fn(mu_val, log_sig_val, u_val).item()

                self.history["loss"].append(loss.item())
                self.history["val_nll"].append(val_nll)
                bar.set_postfix(loss=f"{loss.item():.4f}", val_nll=f"{val_nll:.4f}")

                if self.early_stop.step(val_nll):
                    bar.write(f"Early stop at epoch {epoch}")
                    break

        return self.history

    def _fit_deep_gp(self, data: PhysicsDataBatch) -> dict:
        import gpytorch

        x_train = data.x_bc.to(self.device)
        u_train = data.u_bc.squeeze(-1).to(self.device)
        self.loss_fn = ELBOLoss(self.model, self.model.likelihood, num_data=len(x_train))

        bar = tqdm(range(self.epochs), desc="DeepGP", ncols=90)
        for epoch in bar:
            self.model.train()
            self.model.likelihood.train()
            self.optimizer.zero_grad()

            with gpytorch.settings.num_likelihood_samples(10):
                output = self.model(x_train)
                loss = self.loss_fn(output, u_train)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 200 == 0:
                self.history["loss"].append(loss.item())
                bar.set_postfix(elbo=f"{-loss.item():.4f}")

        return self.history

    def _fit_mcmc(self, data: PhysicsDataBatch) -> dict:
        X = data.x_bc.numpy()
        y = data.u_bc.numpy().ravel()
        self.model.fit(X, y)
        return {}
