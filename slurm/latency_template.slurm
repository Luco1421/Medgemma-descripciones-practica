#!/bin/bash
#SBATCH --job-name=latency
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Uso (desde la raíz del repo): PASSES=2 sbatch slurm/latency_template.slurm
# Mide la latencia por imagen de A, B, C y el modelo base en una misma GPU y un mismo proceso (results/audit/latencia.txt).

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
nvidia-smi --query-gpu=name,memory.used --format=csv
"$PYTHON" -m scripts.analysis.latency --passes "${PASSES:-1}"
