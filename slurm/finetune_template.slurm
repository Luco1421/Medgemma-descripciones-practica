#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir, SLURM abre el log antes de correr):
#   sbatch slurm/finetune_template.slurm DATASET RECETA CAPAS SEMILLA
# CAPAS: 'all-linear', un grupo de `lora.py` (ver `python -m scripts.model.lora preview`) o un regex.

set -euo pipefail

DATASET=${1:?falta DATASET (por ejemplo chaksu)}
RECIPE=${2:?falta la receta (por ejemplo json_v1)}
LAYERS=${3:?falta CAPAS}
SEED=${4:?falta la semilla}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

OUTPUT_DIR="experiments/${DATASET}_${RECIPE}/${LAYERS//[^A-Za-z0-9_-]/_}_s${SEED}"

cd "$REPO"

"$PYTHON" -m scripts.model.lora train \
  --targets "data/targets/${DATASET}__${RECIPE}.csv" \
  --splits "data/splits/${DATASET}_seed42.csv" \
  --split train \
  --prompt-file "data/targets/${DATASET}__${RECIPE}.prompt.txt" \
  --datasets-root ../datasets \
  --layers "$LAYERS" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
