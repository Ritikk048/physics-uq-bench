<div align="center">

# physics-uq-bench

A benchmark for comparing uncertainty quantification methods on physics-driven PDE tasks.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.11%2B-7B68EE)](https://gpytorch.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-5A67D8)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Comparing UQ methods across papers is awkward — each one uses a different PDE, different metrics, and different training budgets. This repo puts PINNs, MC Dropout, Deep Kernel Learning, ReFANN, and BNNs (VI) on the same data, same training loop, and same evaluation pipeline so the numbers are actually comparable.

Three PDEs: heat equation, wave equation, 2D Poisson. Six metrics: Gaussian NLL, RMSE, CRPS, ECE, coverage at 90%, interval score.

---

## Leaderboard

> `python scripts/quick_results.py` — 5 000 epochs, seed 42, 500 labeled interior points.  
> PINN is trained on boundary + collocation points only (no interior labels), so its higher RMSE is expected.

### Heat Equation (α = 0.01)

| Rank | Method | NLL ↓ | RMSE ↓ | CRPS ↓ | ECE ↓ | Cov 90% | IS-90 ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ReFANN** | **−3.107** | 0.0100 | **0.0058** | **0.101** | 0.974 | **0.047** |
| 2 | Deep GP | −2.956 | 0.0100 | 0.0062 | 0.184 | **0.995** | 0.059 |
| 3 | MC Dropout | −1.893 | 0.0148 | 0.0152 | 0.379 | 1.000 | 0.195 |
| 4 | BNN (VI) | −1.461 | 0.0304 | 0.0248 | 0.354 | 1.000 | 0.309 |
| 5 | PINN | 1.222 | 0.6224 | 0.4017 | 0.163 | 1.000 | 3.887 |

### Wave Equation (c = 1.0)

| Rank | Method | NLL ↓ | RMSE ↓ | CRPS ↓ | ECE ↓ | Cov 90% | IS-90 ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ReFANN** | **−3.107** | 0.0102 | **0.0059** | **0.088** | 0.960 | **0.047** |
| 2 | Deep GP | −2.944 | 0.0102 | 0.0063 | 0.182 | **0.996** | 0.059 |
| 3 | MC Dropout | −1.603 | 0.0219 | 0.0205 | 0.377 | 1.000 | 0.261 |
| 4 | BNN (VI) | −1.298 | 0.0455 | 0.0306 | 0.280 | 1.000 | 0.337 |
| 5 | PINN | 1.269 | 0.6533 | 0.4336 | 0.163 | 0.978 | 4.186 |

### Poisson Equation (2D)

| Rank | Method | NLL ↓ | RMSE ↓ | CRPS ↓ | ECE ↓ | Cov 90% | IS-90 ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ReFANN** | **−3.109** | 0.0101 | **0.0058** | **0.089** | 0.959 | **0.047** |
| 2 | MC Dropout | −3.002 | 0.0102 | 0.0061 | 0.155 | **0.990** | 0.054 |
| 3 | Deep GP | −2.972 | 0.0100 | 0.0062 | 0.176 | **0.994** | 0.057 |
| 4 | BNN (VI) | −2.084 | 0.0185 | 0.0136 | 0.257 | 1.000 | 0.150 |
| 5 | PINN | 1.066 | 0.3864 | 0.3213 | 0.288 | 0.998 | 3.751 |

---

## Project Structure

```
physics-uq-bench/
├── src/
│   ├── data/
│   │   └── generators.py         # Exact PDE solutions + noisy observations
│   ├── models/
│   │   ├── pinn.py               # Heteroscedastic PINN (Siren + autograd)
│   │   ├── mc_dropout.py         # MC Dropout
│   │   ├── deep_gp.py            # Deep Kernel Learning (GPyTorch)
│   │   ├── refann.py             # ReFANN: residual refinement
│   │   └── mcmc_bnn.py           # BNN skeleton (VI version in quick_results.py)
│   ├── training/
│   │   ├── losses.py             # PINNLoss, GaussianNLLLoss, ELBOLoss
│   │   └── trainer.py            # Unified loop: cosine LR, grad clip, early stop
│   ├── evaluation/
│   │   ├── metrics.py            # NLL, RMSE, CRPS, ECE, coverage, IS
│   │   └── calibration.py        # Reliability diagrams, PIT, residuals
│   ├── tuning/
│   │   └── optuna_search.py      # TPE sampler + MedianPruner, per-model search spaces
│   ├── ablation/
│   │   └── ablation_runner.py    # OAT ablation, bootstrap 95% CIs
│   └── visualization/
│       └── leaderboard.py        # Console table, LaTeX, radar/bar charts
├── configs/
│   ├── heat_equation.yaml
│   ├── wave_equation.yaml
│   └── poisson.yaml
├── scripts/
│   ├── run_benchmark.py          # Full runner: HPO + training + evaluation
│   └── quick_results.py          # Self-contained runner (no install needed)
├── slurm/
│   └── run_benchmark.sh          # SLURM array job
├── tests/
│   └── test_metrics.py           # 8 unit tests
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
git clone https://github.com/Ritikk048/physics-uq-bench
cd physics-uq-bench
pip install -e .
```

The fastest way to reproduce the leaderboard numbers — no package install required:

```bash
python3 scripts/quick_results.py
```

For the full benchmark with HPO:

```bash
# Two models, one task (~5–10 min on CPU)
python scripts/run_benchmark.py \
    --task heat_equation \
    --models pinn mc_dropout \
    --epochs 10000

# All models, all tasks, with Optuna tuning (GPU recommended)
python scripts/run_benchmark.py \
    --task all --models all \
    --tune --n_trials 50 \
    --epochs 20000 --device cuda
```

### Docker

```bash
docker build -t physics-uq-bench .
docker run --gpus all -v $(pwd)/results:/workspace/results physics-uq-bench
```

### SLURM

```bash
mkdir -p logs
sbatch --array=0-3 slurm/run_benchmark.sh
```

---

## PDE Tasks

### 1D Heat Equation
$$u_t = \alpha \, u_{xx}, \quad x \in [0,1],\ t \in [0,1], \quad \alpha = 0.01$$

| | |
|---|---|
| Initial condition | $u(x,0) = \sin(\pi x)$ |
| Boundary condition | $u(0,t) = u(1,t) = 0$ |
| Exact solution | $u(x,t) = e^{-\alpha \pi^2 t} \sin(\pi x)$ |

### 1D Wave Equation
$$u_{tt} = c^2 \, u_{xx}, \quad c = 1.0$$

| | |
|---|---|
| Initial displacement | $u(x,0) = \sin(\pi x)$ |
| Initial velocity | $u_t(x,0) = 0$ |
| Exact solution | $u(x,t) = \cos(c\pi t)\sin(\pi x)$ |

### 2D Poisson Equation
$$-u_{xx} - u_{yy} = f(x,y), \quad (x,y) \in [0,1]^2$$

| | |
|---|---|
| Forcing | $f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$ |
| Boundary | $u = 0$ on all walls (Dirichlet) |
| Exact solution | $u(x,y) = \dfrac{\sin(\pi x)\sin(\pi y)}{2\pi^2}$ |

---

## Methods

### PINN
> Raissi et al., *Journal of Computational Physics*, 2019

Fully-connected network with Siren sinusoidal activations, outputting $\mu$ and $\log\sigma$ at every point. PDE residuals are enforced via autograd. No labeled interior data — only boundary conditions and collocation points.

$$\mathcal{L} = \lambda_\text{PDE} \|\mathcal{N}[u]\|^2 + \lambda_\text{BC} \|u - u_\text{BC}\|^2 + \lambda_\text{NLL} \mathcal{L}_\text{NLL}$$

Uncertainty: **aleatoric only** (heteroscedastic).

### MC Dropout
> Gal & Ghahramani, *ICML 2016*

Standard MLP with dropout active at test time. 100 stochastic forward passes give an empirical predictive distribution. Epistemic uncertainty comes from variance across the passes; aleatoric uncertainty comes from the average predicted sigma.

Uncertainty: **epistemic + aleatoric**.

### Deep Kernel Learning
> Wilson et al., *ICML 2016*

A 3-layer DNN projects inputs into 8-dimensional feature space, then a sparse variational GP (128 inducing points) models the output in that space. DNN and GP kernel are trained end-to-end by maximising the variational ELBO.

Uncertainty: **epistemic** (variational GP posterior).

### ReFANN
> Residual-Feedback Artificial Neural Network

Two-stage architecture. The primary head outputs $\mu_0$, $\log\sigma_0$. A secondary network takes the primary predictions as additional input and outputs additive corrections to both. This residual refinement happens in a single forward pass.

Uncertainty: **aleatoric** (refined heteroscedastic).

### BNN (Variational Inference)
> Blundell et al., *ICML 2015*

Every weight is a mean-field Gaussian. Training minimises ELBO = NLL + β·KL. At inference, 200 stochastic forward passes are drawn; total variance decomposes into epistemic (variance across samples) and aleatoric (average predicted σ²). Implemented directly in PyTorch — no external sampler.

Uncertainty: **epistemic + aleatoric** (mean-field VI).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Gaussian NLL** | Average negative log-likelihood under a Gaussian predictive. Lower = better joint fit. |
| **RMSE** | Root mean squared error on the predictive mean. Point accuracy only. |
| **CRPS** | Continuous Ranked Probability Score — penalises both miscalibration and sharpness jointly. |
| **ECE** | Expected Calibration Error — mean deviation between observed and nominal coverage across confidence levels. |
| **Coverage @ 90%** | Fraction of test targets that fall inside the 90% predictive interval. Should be close to 0.90. |
| **Interval Score** | Width of the 90% interval plus a penalty for each target that falls outside it. |

Calibration diagnostics produced per model: reliability diagram, PIT histogram, standardised residual plot.

---

## Hyperparameter Optimisation

Each model is tuned independently with Optuna (TPE sampler + MedianPruner):

```python
from src.tuning import HyperparameterSearch
from src.data import get_dataset

data = get_dataset("configs/heat_equation.yaml")
searcher = HyperparameterSearch(
    model_type="mc_dropout",
    data=data,
    n_trials=50,
    timeout=3600,
    metric="gaussian_nll",
)
result = searcher.run()
```

| Model | Search space |
|---|---|
| PINN | `hidden_dim`, `n_layers`, `lr`, `omega`, `lambda_pde`, `lambda_bc` |
| MC Dropout | `hidden_dim`, `n_layers`, `lr`, `p_drop` |
| Deep GP | `hidden_dim`, `num_inducing`, `lr` |
| ReFANN | `hidden_dim`, `n_layers`, `lr`, `n_refine` |

---

## Ablation Studies

One-at-a-time ablation with bootstrap 95% CIs:

```python
from src.ablation import AblationRunner

runner = AblationRunner(
    model_type="mc_dropout",
    base_params={"hidden_dim": 128, "n_layers": 4, "p_drop": 0.1},
    ablation_grid={
        "p_drop":    [0.05, 0.10, 0.20, 0.30, 0.40],
        "n_layers":  [2, 3, 4, 5, 6],
        "hidden_dim":[64, 128, 256, 512],
    },
    data=data,
    n_seeds=5,
    metric="gaussian_nll",
)
df = runner.run()
```

---

## Adding a New Method

1. Add `src/models/your_model.py` with `forward(x) → (mu, log_sigma)` and `predict_with_uncertainty(x, n_samples) → (mean, std)`

2. Register it:
   ```python
   # src/models/__init__.py
   MODEL_REGISTRY["your_model"] = YourModel
   ```

3. Add a search space in `src/tuning/optuna_search.py`

4. Run:
   ```bash
   python scripts/run_benchmark.py --models your_model --task heat_equation
   ```

---

## Tests

```bash
pytest tests/ -v
```

All 8 unit tests cover NLL minimisation, RMSE, coverage at 90% and 50%, ECE, CRPS, metric finiteness, and determinism.

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | Neural network models + autograd |
| `gpytorch` | ≥ 1.11 | Deep Kernel Learning (SVGP) |
| `optuna` | ≥ 3.5.0 | Hyperparameter optimisation |
| `scipy` | ≥ 1.11.0 | CRPS, PIT, calibration statistics |
| `pandas` | ≥ 2.1.0 | Results aggregation |
| `rich` | ≥ 13.7.0 | Console output |
| `tqdm` | ≥ 4.66 | Progress bars |

```bash
pip install -r requirements.txt
```

---

## Output Structure

```
results/
├── leaderboard.csv
├── best_params_pinn.json
├── best_params_mc_dropout.json
├── best_params_deep_gp.json
├── best_params_refann.json
├── ablation/
│   ├── ablation_pinn.csv
│   └── ablation_pinn.png
└── figures/
    ├── heat_equation_pinn_calibration.png
    ├── radar_heat_equation.png
    └── bar_gaussian_nll.png
```

---

## Uncertainty Decomposition

Predictive uncertainty has two components: **aleatoric** (noise in the data, irreducible) and **epistemic** (uncertainty in the model weights, reducible with more data).

| Method | Aleatoric | Epistemic |
|---|:---:|:---:|
| PINN | ✅ | ❌ |
| ReFANN | ✅ | ❌ |
| MC Dropout | ✅ | ✅ |
| Deep GP | ❌ | ✅ |
| BNN (VI) | ✅ | ✅ |

---

MIT © 2024 Guru
