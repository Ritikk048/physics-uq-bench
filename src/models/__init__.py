from .pinn import PINN
from .mc_dropout import MCDropoutNet
from .deep_gp import DeepGPModel
from .mcmc_bnn import MCMCBayesianNN
from .refann import ReFANN

MODEL_REGISTRY = {
    "pinn": PINN,
    "mc_dropout": MCDropoutNet,
    "deep_gp": DeepGPModel,
    "mcmc_bnn": MCMCBayesianNN,
    "refann": ReFANN,
}

__all__ = ["PINN", "MCDropoutNet", "DeepGPModel", "MCMCBayesianNN", "ReFANN", "MODEL_REGISTRY"]
