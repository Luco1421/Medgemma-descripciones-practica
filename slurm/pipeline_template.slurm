#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Etapa 2 de la configuración C (scripts/studies/pipeline.py): recorre una lista de generaciones, una por línea,
# `TARGETS ADAPTER SALIDA` (ADAPTER = `base` para el modelo sin adaptar), cada una en su proceso, saltando las
# que ya tienen `generate_config.json`. Inferencia determinista, 128 tokens, partición de validación.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/pipeline_template.slurm runs/chaksu_C_etapa2_s42.txt     (SPLITS=… SPLIT=… para otra partición, p. ej. MM-ODIR-129)
#   sbatch --export=ALL,RECIPE=verbalize_img_v1,IMAGE_MODES=real slurm/pipeline_template.slurm runs/chaksu_C_img.txt   (etapa 2 con imagen)

set -euo pipefail

LIST=${1:?falta la lista de generaciones}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
failed=()
while read -r targets adapter output; do
  [[ -z "$targets" || "$targets" == \#* ]] && continue
  if [[ -f "$output/generate_config.json" ]]; then echo "ya hecho: $output"; continue; fi
  echo "=== $output"
  adapter_args=()
  if [[ "$adapter" != base ]]; then adapter_args=(--adapter "$adapter"); fi
  image_args=()
  if [[ -n "${IMAGE_MODES:-}" ]]; then image_args=(--image-modes $IMAGE_MODES); fi
  if ! "$PYTHON" -m scripts.model.lora generate \
      --targets "$targets" \
      --splits "${SPLITS:-data/splits/chaksu_seed42.csv}" \
      --split "${SPLIT:-validation}" \
      --prompt-file "data/targets/chaksu__${RECIPE:-verbalize_v1}.prompt.txt" \
      --datasets-root ../datasets \
      --output-dir "$output" \
      "${adapter_args[@]}" "${image_args[@]}" \
      --max-new-tokens 128; then
    failed+=("$output")
  fi
done < "$LIST"
if (( ${#failed[@]} )); then echo "FALLARON: ${failed[*]}"; exit 1; fi
