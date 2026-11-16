#!/bin/bash
#SBATCH --job-name=batch
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir, SLURM abre el log antes de correr):
#   sbatch slurm/batch_template.slurm LISTA PRIMERA CANTIDAD
# LISTA es un archivo con una corrida por línea ("DATASET RECETA CAPAS SEMILLA"); '#' abre un comentario.
# Ejecuta, una tras otra, CANTIDAD corridas desde la PRIMERA: entrena cada una (slurm/finetune_template.slurm) y
# evalúa su adaptador sobre validación una imagen a la vez (slurm/inference_template.slurm). Lo que ya existe se
# salta, así que repetir el job tras un fallo continúa donde quedó. Un fallo no detiene las demás corridas:
# se anota y el job termina con error al final, listándolas.

set -euo pipefail

LIST=${1:?falta LISTA}
FIRST=${2:?falta PRIMERA}
COUNT=${3:?falta CANTIDAD}

cd "${SLURM_SUBMIT_DIR:?run sbatch from the repository root}"

mapfile -t RUNS < <(grep -v '^[[:space:]]*\(#\|$\)' "$LIST" | sed -n "${FIRST},$((FIRST + COUNT - 1))p")
echo "${#RUNS[@]} corridas desde la línea $FIRST de $LIST"

FAILED=()
for RUN in "${RUNS[@]}"; do
  read -r DATASET RECIPE LAYERS SEED <<< "$RUN"
  RUN_DIR="experiments/${DATASET}_${RECIPE}/${LAYERS//[^A-Za-z0-9_-]/_}_s${SEED}"
  echo "=== $(date '+%F %T')  $RUN"

  if [ -f "$RUN_DIR/adapter_model.safetensors" ]; then
    echo "entrenamiento ya hecho: $RUN_DIR"
  else
    bash slurm/finetune_template.slurm "$DATASET" "$RECIPE" "$LAYERS" "$SEED" || { FAILED+=("entrenamiento: $RUN"); continue; }
  fi

  if [ -f "$RUN_DIR/validation/generate_config.json" ]; then
    echo "evaluación ya hecha: $RUN_DIR"
  else
    bash slurm/inference_template.slurm "$DATASET" "$RECIPE" validation "$RUN_DIR" || { FAILED+=("evaluación: $RUN"); continue; }
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FALLARON ${#FAILED[@]} de ${#RUNS[@]}:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "terminadas ${#RUNS[@]} corridas"
