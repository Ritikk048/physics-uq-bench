"""
Main benchmark runner.

Usage:
    python scripts/run_benchmark.py --task heat_equation --models pinn mc_dropout
    python scripts/run_benchmark.py --task all --models all --tune --device cuda

Workflow:
  1. Load dataset from config
  2. (Optional) HPO with Optuna
  3. Train each model with best params
  4. Evaluate on test set
  5. Produce leaderboard + calibration plots
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import get_dataset
from models import MODEL_REGISTRY
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from evaluation.calibration import CalibrationAnalyzer
from tuning.optuna_search import HyperparameterSearch
from visualization.leaderboard import Leaderboard

TASKS = ["heat_equation", "wave_equation", "poisson"]
MODELS_PYTORCH = ["pinn", "mc_dropout", "deep_gp", "refann"]
ALL_MODELS = MODELS_PYTORCH + ["mcmc_bnn"]

PDE_TYPES = {
    "heat_equation": "heat",
    "wave_equation": "wave",
    "poisson": "poisson",
}

DEFAULT_PARAMS = {
    "pinn": {"hidden_dim": 64, "n_layers": 4, "omega": 30.0},
    "mc_dropout": {"hidden_dim": 128, "n_layers": 4, "p_drop": 0.1},
    "deep_gp": {"hidden_dim": 3, "num_inducing": 128},
    "refann": {"hidden_dim": 128, "n_layers": 3, "n_refine": 2},
    "mcmc_bnn": {"hidden_dim": 32, "n_samples": 500, "n_tune": 500},
}

TRAIN_LR = {
    "pinn": 1e-3, "mc_dropout": 1e-3, "deep_gp": 1e-2,
    "refann": 1e-3, "mcmc_bnn": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="physics-uq-bench runner")
    parser.add_argument("--task", default="heat_equation",
                        choices=TASKS + ["all"], help="PDE task to benchmark")
    parser.add_argument("--models", nargs="+", default=["pinn", "mc_dropout"],
                        choices=ALL_MODELS + ["all"])
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO before training")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--n_mc_samples", type=int, default=100)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_one(task_name: str, model_name: str, args, leaderboard: Leaderboard):
    cfg_path = f"configs/{task_name}.yaml"
    pde_type = PDE_TYPES[task_name]

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}  |  Model: {model_name}")
    print(f"{'='*60}")

    data = get_dataset(cfg_path, seed=args.seed)
    params = DEFAULT_PARAMS[model_name].copy()

    if args.tune and model_name in MODELS_PYTORCH:
        print("  Running Optuna HPO ...")
        searcher = HyperparameterSearch(
            model_type=model_name,
            data=data,
            pde_type=pde_type,
            n_trials=args.n_trials,
            n_epochs_fast=min(args.epochs // 3, 3000),
            results_dir=args.results_dir,
            device=args.device,
        )
        result = searcher.run()
        params.update(result["best_params"])
        lr = params.pop("lr", TRAIN_LR[model_name])
        print(f"  Best params: {params}")
    else:
        lr = TRAIN_LR[model_name]

    ModelClass = MODEL_REGISTRY[model_name]
    input_dim = data.x_col.shape[1]

    if model_name == "pinn":
        from data.generators import get_dataset as _
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        pde_params = {k: v for k, v in cfg["task"].items()
                      if k in ("alpha", "c")}
        model = ModelClass(input_dim=input_dim, pde_type=pde_type,
                           pde_params=pde_params, **params)
        trainer = Trainer(model, model_type="pinn", lr=lr or 1e-3,
                          epochs=args.epochs, device=args.device)
    elif model_name == "mcmc_bnn":
        model = ModelClass(input_dim=input_dim, **params)
        trainer = Trainer(model, model_type="mcmc_bnn")
    else:
        model = ModelClass(input_dim=input_dim, **params)
        trainer = Trainer(model, model_type=model_name, lr=lr or 1e-3,
                          epochs=args.epochs, device=args.device)

    t0 = time.time()
    trainer.fit(data)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    device = torch.device(args.device)
    x_test = data.x_test.to(device)
    if model_name == "mcmc_bnn":
        mean, std = model.predict_with_uncertainty(data.x_test, n_samples=args.n_mc_samples)
    else:
        mean, std = model.predict_with_uncertainty(x_test, n_samples=args.n_mc_samples)
        mean, std = mean.detach().cpu(), std.detach().cpu()

    y_true = data.u_test_noisy.numpy().ravel()
    mean_np, std_np = mean.numpy().ravel(), std.numpy().ravel()

    metrics = evaluate_model(y_true, mean_np, std_np)
    print(metrics)

    # Calibration diagnostics
    cal = CalibrationAnalyzer(y_true, mean_np, std_np, model_name=model_name)
    cal_path = f"{args.results_dir}/figures/{task_name}_{model_name}_calibration.png"
    cal.plot_full_diagnostics(save_path=cal_path)

    leaderboard.add_result(model_name, task_name, metrics.to_dict())
    leaderboard.show(task=task_name)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tasks = TASKS if args.task == "all" else [args.task]
    models = ALL_MODELS if "all" in args.models else args.models

    Path(args.results_dir + "/figures").mkdir(parents=True, exist_ok=True)
    lb = Leaderboard()

    for task in tasks:
        for model in models:
            run_one(task, model, args, lb)

    lb.save(f"{args.results_dir}/leaderboard.csv")
    for task in tasks:
        lb.plot_radar(task, save_path=f"{args.results_dir}/figures/radar_{task}.png")
    lb.plot_bar_comparison(save_path=f"{args.results_dir}/figures/bar_gaussian_nll.png")

    print("\n\nFinal Leaderboard:")
    lb.show()
    print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
