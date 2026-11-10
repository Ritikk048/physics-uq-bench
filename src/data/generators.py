"""
Physics dataset generators.

Each dataset returns:
  - collocation points (PDE residual training)
  - boundary/initial condition points
  - test points with exact solution
  - optional additive Gaussian observation noise
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple, Dict
import yaml


@dataclass
class PhysicsDataBatch:
    x_col: torch.Tensor         # collocation (PDE interior)
    x_bc: torch.Tensor          # boundary / initial
    u_bc: torch.Tensor          # exact u at BC/IC
    x_test: torch.Tensor        # test grid
    u_test: torch.Tensor        # exact u at test points
    u_test_noisy: torch.Tensor  # noisy observations (for likelihood eval)
    noise_std: float
    name: str


class HeatEquationDataset:
    """
    1-D heat equation: u_t = alpha * u_xx,  x in [0,1], t in [0,1]
    IC: u(x,0) = sin(pi*x)
    BC: u(0,t) = u(1,t) = 0
    Exact: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi*x)
    """

    def __init__(self, cfg: dict, seed: int = 42):
        rng = np.random.default_rng(seed)
        alpha = cfg["alpha"]
        n_col = cfg["n_collocation"]
        n_bc = cfg["n_boundary"]
        n_ic = cfg["n_initial"]
        n_test = cfg["n_test"]
        noise_std = cfg["noise_std"]

        # Exact solution
        def u_exact(x, t):
            return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

        # Collocation (interior)
        x_col = rng.uniform(0, 1, n_col)
        t_col = rng.uniform(0, 1, n_col)
        X_col = np.stack([x_col, t_col], axis=1)

        # Boundary: x=0 and x=1, t random
        t_bc = rng.uniform(0, 1, n_bc)
        X_bc0 = np.stack([np.zeros(n_bc // 2), t_bc[: n_bc // 2]], axis=1)
        X_bc1 = np.stack([np.ones(n_bc // 2), t_bc[n_bc // 2 :]], axis=1)

        # Initial condition: t=0, x random
        x_ic = rng.uniform(0, 1, n_ic)
        X_ic = np.stack([x_ic, np.zeros(n_ic)], axis=1)

        X_bc_all = np.vstack([X_bc0, X_bc1, X_ic])
        u_bc_all = u_exact(X_bc_all[:, 0], X_bc_all[:, 1])

        # Test grid
        x_t = np.linspace(0, 1, int(np.sqrt(n_test)))
        t_t = np.linspace(0, 1, int(np.sqrt(n_test)))
        XX, TT = np.meshgrid(x_t, t_t)
        X_test = np.stack([XX.ravel(), TT.ravel()], axis=1)
        u_test = u_exact(X_test[:, 0], X_test[:, 1])
        u_noisy = u_test + rng.normal(0, noise_std, u_test.shape)

        self.data = PhysicsDataBatch(
            x_col=torch.tensor(X_col, dtype=torch.float32),
            x_bc=torch.tensor(X_bc_all, dtype=torch.float32),
            u_bc=torch.tensor(u_bc_all, dtype=torch.float32).unsqueeze(-1),
            x_test=torch.tensor(X_test, dtype=torch.float32),
            u_test=torch.tensor(u_test, dtype=torch.float32).unsqueeze(-1),
            u_test_noisy=torch.tensor(u_noisy, dtype=torch.float32).unsqueeze(-1),
            noise_std=noise_std,
            name="heat_equation",
        )

    def get(self) -> PhysicsDataBatch:
        return self.data


class WaveEquationDataset:
    """
    1-D wave equation: u_tt = c^2 * u_xx,  x in [0,1], t in [0,1]
    IC: u(x,0) = sin(pi*x),  u_t(x,0) = 0
    BC: u(0,t) = u(1,t) = 0
    Exact: u(x,t) = cos(c*pi*t)*sin(pi*x)
    """

    def __init__(self, cfg: dict, seed: int = 42):
        rng = np.random.default_rng(seed)
        c = cfg["c"]
        n_col = cfg["n_collocation"]
        n_bc = cfg["n_boundary"]
        n_ic = cfg["n_initial"]
        n_test = cfg["n_test"]
        noise_std = cfg["noise_std"]

        def u_exact(x, t):
            return np.cos(c * np.pi * t) * np.sin(np.pi * x)

        x_col = rng.uniform(0, 1, n_col)
        t_col = rng.uniform(0, 1, n_col)
        X_col = np.stack([x_col, t_col], axis=1)

        t_bc = rng.uniform(0, 1, n_bc)
        X_bc0 = np.stack([np.zeros(n_bc // 2), t_bc[: n_bc // 2]], axis=1)
        X_bc1 = np.stack([np.ones(n_bc // 2), t_bc[n_bc // 2 :]], axis=1)

        x_ic = rng.uniform(0, 1, n_ic // 2)
        X_ic_u = np.stack([x_ic, np.zeros(n_ic // 2)], axis=1)
        X_ic_ut = np.stack([x_ic, np.zeros(n_ic // 2)], axis=1)

        X_bc_all = np.vstack([X_bc0, X_bc1, X_ic_u, X_ic_ut])
        u_bc_all = u_exact(X_bc_all[:, 0], X_bc_all[:, 1])

        x_t = np.linspace(0, 1, int(np.sqrt(n_test)))
        t_t = np.linspace(0, 1, int(np.sqrt(n_test)))
        XX, TT = np.meshgrid(x_t, t_t)
        X_test = np.stack([XX.ravel(), TT.ravel()], axis=1)
        u_test = u_exact(X_test[:, 0], X_test[:, 1])
        u_noisy = u_test + rng.normal(0, noise_std, u_test.shape)

        self.data = PhysicsDataBatch(
            x_col=torch.tensor(X_col, dtype=torch.float32),
            x_bc=torch.tensor(X_bc_all, dtype=torch.float32),
            u_bc=torch.tensor(u_bc_all, dtype=torch.float32).unsqueeze(-1),
            x_test=torch.tensor(X_test, dtype=torch.float32),
            u_test=torch.tensor(u_test, dtype=torch.float32).unsqueeze(-1),
            u_test_noisy=torch.tensor(u_noisy, dtype=torch.float32).unsqueeze(-1),
            noise_std=noise_std,
            name="wave_equation",
        )

    def get(self) -> PhysicsDataBatch:
        return self.data


class PoissonDataset:
    """
    2-D Poisson: -u_xx - u_yy = f(x,y),  (x,y) in [0,1]^2
    f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    BC: u = 0 on all walls
    Exact: u(x,y) = sin(pi*x)*sin(pi*y) / (2*pi^2)
    """

    def __init__(self, cfg: dict, seed: int = 42):
        rng = np.random.default_rng(seed)
        n_col = cfg["n_collocation"]
        n_bc = cfg["n_boundary"]
        n_test = cfg["n_test"]
        noise_std = cfg["noise_std"]

        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)

        x_col = rng.uniform(0, 1, n_col)
        y_col = rng.uniform(0, 1, n_col)
        X_col = np.stack([x_col, y_col], axis=1)

        n_side = n_bc // 4
        t_side = rng.uniform(0, 1, n_side)
        bc_bottom = np.stack([t_side, np.zeros(n_side)], axis=1)
        bc_top = np.stack([t_side, np.ones(n_side)], axis=1)
        bc_left = np.stack([np.zeros(n_side), t_side], axis=1)
        bc_right = np.stack([np.ones(n_side), t_side], axis=1)
        X_bc_all = np.vstack([bc_bottom, bc_top, bc_left, bc_right])
        u_bc_all = u_exact(X_bc_all[:, 0], X_bc_all[:, 1])

        n_side_test = int(np.sqrt(n_test))
        xs = np.linspace(0, 1, n_side_test)
        ys = np.linspace(0, 1, n_side_test)
        XX, YY = np.meshgrid(xs, ys)
        X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
        u_test = u_exact(X_test[:, 0], X_test[:, 1])
        u_noisy = u_test + rng.normal(0, noise_std, u_test.shape)

        self.data = PhysicsDataBatch(
            x_col=torch.tensor(X_col, dtype=torch.float32),
            x_bc=torch.tensor(X_bc_all, dtype=torch.float32),
            u_bc=torch.tensor(u_bc_all, dtype=torch.float32).unsqueeze(-1),
            x_test=torch.tensor(X_test, dtype=torch.float32),
            u_test=torch.tensor(u_test, dtype=torch.float32).unsqueeze(-1),
            u_test_noisy=torch.tensor(u_noisy, dtype=torch.float32).unsqueeze(-1),
            noise_std=noise_std,
            name="poisson",
        )

    def get(self) -> PhysicsDataBatch:
        return self.data


_REGISTRY = {
    "heat_equation": HeatEquationDataset,
    "wave_equation": WaveEquationDataset,
    "poisson": PoissonDataset,
}


def get_dataset(cfg_path: str, seed: int = 42) -> PhysicsDataBatch:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    task_name = cfg["task"]["name"]
    cls = _REGISTRY[task_name]
    return cls(cfg["task"], seed=seed).get()
