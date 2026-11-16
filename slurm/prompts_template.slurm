#!/bin/bash
#SBATCH --job-name=prompts
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

# Prompting del modelo base (scripts/studies/baseline_prompts.py): recorre una lista de generaciones, una por línea,
# `PROMPT SPLITS PARTICION EJEMPLOS|- CONTROLES SALIDA` (CONTROLES = real o real,shuffled,blank; EJEMPLOS = JSON de P5 o `-`),
# cada una en su proceso, saltando las que ya tienen `generate_config.json`. Modelo base, determinista, 128 tokens (MAX_NEW_TOKENS=1024 para el análisis de P4).
# Uso (desde la raíz del repo; `experiments/logs/` tiene que existir):
#   sbatch slurm/prompts_template.slurm runs/chaksu_prompts_base_1.txt     (TARGETS=… para otro conjunto; MAX_NEW_TOKENS=1024 para P4 largo)

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
while read -r prompt splits split shots modes output; do
  [[ -z "$prompt" || "$prompt" == \#* ]] && continue
  if [[ -f "$output/generate_config.json" ]]; then echo "ya hecho: $output"; continue; fi
  echo "=== $output"
  shot_args=()
  if [[ "$shots" != "-" ]]; then shot_args=(--shots "$shots"); fi
  if ! "$PYTHON" -m scripts.model.lora generate \
      --targets "${TARGETS:-data/targets/chaksu__json_v1.csv}" \
      --splits "$splits" \
      --split "$split" \
      --prompt-file "$prompt" \
      --datasets-root ../datasets \
      --output-dir "$output" \
      --image-modes ${modes//,/ } \
      "${shot_args[@]}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-128}"; then
    failed+=("$output")
  fi
done < "$LIST"
if (( ${#failed[@]} )); then echo "FALLARON: ${failed[*]}"; exit 1; fi
