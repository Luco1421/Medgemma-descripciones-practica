#!/bin/bash
#SBATCH --job-name=prototipo
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Ejecuta el prototipo (prototype/prototipo.ipynb, que contiene toda su lógica) de punta a punta con nbconvert y deja las salidas en el mismo
# cuaderno, sin el estado de los widgets (que lo engordaría varios MB). Construye la interfaz y llama a las funciones de los dos modos desde
# código, sin navegador (con imágenes de validación de Chákṣu, una de ellas tratada como desconocida), y comprueba
# que el selector cambia de adaptador.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/notebook_template.slurm

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
BIN=/data/jperez/app_data/miniconda/envs/ia/bin

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
"$BIN/jupyter-nbconvert" --to notebook --execute --inplace prototype/prototipo.ipynb \
  --ExecutePreprocessor.store_widget_state=False \
  --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=ia
