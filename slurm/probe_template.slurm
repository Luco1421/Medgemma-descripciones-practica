#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --output=experiments/%x_%j.out
#SBATCH --error=experiments/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Sondas por profundidad sobre el modelo base (scripts/analysis/probe.py): extrae las activaciones de cada capa
# de las imágenes de entrenamiento y validación y ajusta las sondas y su tarea de control.
# Uso (desde la raíz del repo; `experiments/` tiene que existir):
#   sbatch slurm/probe_template.slurm DATASET RECETA

set -euo pipefail

DATASET=${1:?falta DATASET (por ejemplo chaksu)}
RECIPE=${2:?falta la receta (por ejemplo json_v1)}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

TARGETS="data/targets/${DATASET}__${RECIPE}.csv"
SPLITS="data/splits/${DATASET}_seed42.csv"

cd "$REPO"
# La extracción (GPU) se salta si las activaciones ya existen; `fit` se puede repetir sobre ellas sin GPU.
if [[ ! -f "results/probes/${DATASET}_${RECIPE}.npz" ]]; then
  "$PYTHON" -m scripts.analysis.probe extract \
    --targets "$TARGETS" --splits "$SPLITS" \
    --prompt-file "data/targets/${DATASET}__${RECIPE}.prompt.txt" \
    --datasets-root ../datasets \
    --output "results/probes/${DATASET}_${RECIPE}.npz"
fi
"$PYTHON" -m scripts.analysis.probe fit \
  --features "results/probes/${DATASET}_${RECIPE}.npz" \
  --targets "$TARGETS" --splits "$SPLITS" \
  --output "results/probes/${DATASET}_${RECIPE}_sondas"
