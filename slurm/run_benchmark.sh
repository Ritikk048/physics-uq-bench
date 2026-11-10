#!/bin/bash
#SBATCH --job-name=uq-bench
#SBATCH --output=logs/uq_bench_%j.out
#SBATCH --error=logs/uq_bench_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ---- Environment ----
module load python/3.11
module load cuda/11.8

source activate uq-bench  # or: conda activate uq-bench

PROJECT_DIR="$SLURM_SUBMIT_DIR/.."
cd "$PROJECT_DIR"

# ---- Parallel model runs via SLURM array ----
# Submit as: sbatch --array=0-3 slurm/run_benchmark.sh

MODELS=("pinn" "mc_dropout" "deep_gp" "refann")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL  on task: heat_equation"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python scripts/run_benchmark.py \
    --task heat_equation \
    --models "$MODEL" \
    --tune \
    --n_trials 50 \
    --epochs 20000 \
    --device cuda \
    --results_dir "results/slurm_${SLURM_JOB_ID}" \
    --seed 42

echo "Done: $MODEL"
