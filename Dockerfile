FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="hrithikk048@gmail.com"
LABEL description="physics-uq-bench: UQ benchmark for physics-driven tasks"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install package in editable mode
COPY . .
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import torch; import gpytorch; import optuna; print('All imports OK')"

# Default: run heat equation benchmark
CMD ["python", "scripts/run_benchmark.py", \
     "--task", "heat_equation", \
     "--models", "pinn", "mc_dropout", "deep_gp", "refann", \
     "--epochs", "10000", \
     "--results_dir", "/workspace/results"]
