#!/bin/bash
#SBATCH --job-name=external
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Inferencia sobre MM-ODIR-129 (validación externa, scripts/studies/external.py) de las mismas corridas que el Test: una corrida por línea,
# `RECETA CARPETA_DE_LA_CORRIDA` (la carpeta `.../base` evalúa el modelo sin adaptar), con los tres controles de imagen. Los prompts son los de Chákṣu.
# Salta las que ya tienen `external/generate_config.json`: la validación externa se corre una sola vez por corrida.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/external_template.slurm runs/chaksu_test_1.txt     (MAX_NEW_TOKENS=512 para la cadena de razonamiento)

set -euo pipefail

LIST=${1:?falta la lista de corridas}

REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
PYTHON=/data/jperez/app_data/miniconda/envs/ia/bin/python

CACHE_ROOT="/data/$USER/app_data"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="/data/$USER/models/huggingface/hub"  # pesos: fuera de app_data
export XDG_CACHE_HOME="$CACHE_ROOT/cache"

cd "$REPO"
failed=()
while read -r recipe run; do
  [[ -z "$recipe" || "$recipe" == \#* ]] && continue
  if [[ -f "$run/external/generate_config.json" ]]; then echo "ya hecho: $run"; continue; fi
  echo "=== $run"
  adapter_args=()
  if [[ "$run" != */base ]]; then adapter_args=(--adapter "$run"); fi
  if ! "$PYTHON" -m scripts.model.lora generate \
      --targets "data/targets/mmodir129__external.csv" \
      --splits data/splits/mmodir129_externa.csv \
      --split external \
      --prompt-file "data/targets/chaksu__${recipe}.prompt.txt" \
      --datasets-root ../datasets \
      --output-dir "$run/external" \
      "${adapter_args[@]}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-128}"; then
    failed+=("$run")
  fi
done < "$LIST"
if (( ${#failed[@]} )); then echo "FALLARON: ${failed[*]}"; exit 1; fi
