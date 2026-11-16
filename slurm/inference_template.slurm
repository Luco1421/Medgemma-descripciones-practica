#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir, SLURM abre el log antes de correr):
#   sbatch slurm/inference_template.slurm DATASET RECETA PARTICION [CARPETA_DE_LA_CORRIDA]
#   LIMIT=3 sbatch slurm/inference_template.slurm ...      # prueba de humo con las 3 primeras imágenes
#   DUP_BOS=1 IMAGE_MODES=real sbatch ...            # entrada con <bos> doble como en el entrenamiento; escribe en <partición>_bos2/
#   MAX_NEW_TOKENS=512 sbatch ...                    # respuestas largas (razonamiento); por defecto 128
# Con la carpeta de una corrida (la que dejó `lora.py train`) se evalúa su adapter y las predicciones
# se escriben en <carpeta>/<partición>/. Sin ella se evalúa el modelo base, en
# experiments/<dataset>_<receta>/base/<partición>/.

set -euo pipefail

DATASET=${1:?falta DATASET (por ejemplo chaksu)}
RECIPE=${2:?falta la receta (por ejemplo json_v1)}
SPLIT=${3:?falta la partición (train, validation o test)}
RUN=${4:-experiments/${DATASET}_${RECIPE}/base}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"

# Tres controles de imagen en un solo proceso: la correcta, otra según una permutación
# fija y un rectángulo negro. Si el rendimiento no cae al romper el vínculo imagen-respuesta, el modelo no
# está mirando la imagen y ninguna comparación entre capas significaría nada.
"$PYTHON" -m scripts.model.lora generate \
  --targets "data/targets/${DATASET}__${RECIPE}.csv" \
  --splits "data/splits/${DATASET}_seed42.csv" \
  --split "$SPLIT" \
  --prompt-file "data/targets/${DATASET}__${RECIPE}.prompt.txt" \
  --datasets-root ../datasets \
  --output-dir "$RUN/${SPLIT}${DUP_BOS:+_bos2}" \
  ${4:+--adapter "$RUN"} \
  ${DUP_BOS:+--duplicate-bos} \
  ${IMAGE_MODES:+--image-modes $IMAGE_MODES} \
  ${LIMIT:+--limit "$LIMIT"} \
  ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"}
