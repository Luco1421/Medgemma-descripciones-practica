#!/bin/bash
#SBATCH --job-name=prototype
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Prototipo de inferencia (scripts/model/medgemma.py): una fotografía de fondo de ojo y un prompt entran, la descripción sale.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/prototype_template.slurm IMAGEN [ADAPTADOR] [RECETA]
#   sbatch slurm/prototype_template.slurm ../datasets/chaksu/raw/Train/Train/1.0_Original_Fundus_Images/Bosch/Image151.JPG experiments/chaksu_json_v1/all-linear_s42 json_v1
# Sin ADAPTADOR usa el modelo sin adaptar. El resultado queda en el log de SLURM.

set -euo pipefail

IMAGE=${1:?falta la imagen}
ADAPTER=${2:-}
RECIPE=${3:-json_v1}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
adapter_args=()
if [[ -n "$ADAPTER" ]]; then adapter_args=(--adapter "$ADAPTER"); fi
"$PYTHON" -m scripts.model.medgemma --image "$IMAGE" --recipe "$RECIPE" "${adapter_args[@]}"
