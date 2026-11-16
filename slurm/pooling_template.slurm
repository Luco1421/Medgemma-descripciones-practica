#!/bin/bash
#SBATCH --job-name=pooling
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Sondas sobre la torre de visión del modelo base con cuatro resúmenes de los parches (promedio, máximo, gradiente
# morfológico y su pico): mide si el promedio 4×4 del conector pierde información del borde (scripts/analysis/probe.py pooling).
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/pooling_template.slurm DATASET RECETA

set -euo pipefail

DATASET=${1:?falta DATASET (por ejemplo chaksu)}
RECIPE=${2:?falta la receta (por ejemplo json_v1)}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
"$PYTHON" -m scripts.analysis.probe pooling \
  --targets "data/targets/${DATASET}__${RECIPE}.csv" --splits "data/splits/${DATASET}_seed42.csv" \
  --prompt-file "data/targets/${DATASET}__${RECIPE}.prompt.txt" \
  --datasets-root ../datasets \
  --output "results/probes/${DATASET}_${RECIPE}_pooling"
