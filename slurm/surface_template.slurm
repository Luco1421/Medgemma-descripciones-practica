#!/bin/bash
#SBATCH --job-name=surface
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Métricas de superficie de la rejilla de prosa y su meta-evaluación (scripts/eval/surface.py).
# Uso (desde la raíz del repo): sbatch slurm/surface_template.slurm LISTA   (p. ej. runs/chaksu_prose_v1_grid.txt)
#                               sbatch slurm/surface_template.slurm --stage2   (B y la etapa 2 de C)

set -euo pipefail

ARGS=("${@:?falta la lista de corridas o --stage2}")

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
"$PYTHON" -m scripts.eval.surface "${ARGS[@]}"
