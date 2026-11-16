#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Inferencia sobre la partición de Test de los finalistas (scripts/analysis/finalists.py): una corrida por línea,
# `RECETA CARPETA_DE_LA_CORRIDA` (la carpeta `.../base` evalúa el modelo sin adaptar), con los tres controles de imagen.
# Salta las que ya tienen `test/generate_config.json`: el Test se abre una sola vez y no se repite.
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/test_template.slurm runs/chaksu_test_1.txt

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
  if [[ -f "$run/test/generate_config.json" ]]; then echo "ya hecho: $run"; continue; fi
  echo "=== $run"
  adapter_args=()
  if [[ "$run" != */base ]]; then adapter_args=(--adapter "$run"); fi
  if ! "$PYTHON" -m scripts.model.lora generate \
      --targets "data/targets/chaksu__${recipe}.csv" \
      --splits data/splits/chaksu_seed42.csv \
      --split test \
      --prompt-file "data/targets/chaksu__${recipe}.prompt.txt" \
      --datasets-root ../datasets \
      --output-dir "$run/test" \
      "${adapter_args[@]}" \
      --max-new-tokens 128; then
    failed+=("$run")
  fi
done < "$LIST"
if (( ${#failed[@]} )); then echo "FALLARON: ${failed[*]}"; exit 1; fi
