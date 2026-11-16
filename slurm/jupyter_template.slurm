#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Abre JupyterLab en un nodo con GPU para usar el prototipo de forma interactiva. El servidor escucha solo en el
# nodo (127.0.0.1); el .out dice el nodo y el puerto, y el .err (donde JupyterLab escribe su log) trae la URL con el token.
# El túnel SSH está en prototype/README.md.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/jupyter_template.slurm            (PORT=8899 por defecto; PORT=… sbatch … para otro)

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
BIN=/data/jperez/app_data/miniconda/envs/ia/bin
PORT=${PORT:-8899}

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
echo "nodo: $(hostname)  puerto: $PORT"
echo "túnel (desde tu equipo): ssh -J <usuario>@<login-node> -L $PORT:127.0.0.1:$PORT <usuario>@$(hostname)   y abre la URL con token de experiments/logs/jupyter_${SLURM_JOB_ID}.err, cambiando el host por localhost"
"$BIN/jupyter-lab" --no-browser --ip=127.0.0.1 --port="$PORT" --notebook-dir="$REPO"
