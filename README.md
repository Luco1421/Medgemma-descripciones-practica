# Ubicación de los adaptadores LoRA en MedGemma para fondo de ojo

Estudio de qué capas de [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) conviene adaptar con
LoRA para que el modelo lea el disco óptico en fotografías de fondo de ojo, con foco en glaucoma. Se compara
dónde se colocan los adaptadores (torre de visión o de lenguaje, profundidad, atención o MLP) con el mismo
presupuesto de parámetros entrenables, y se mide el error del cociente copa/disco vertical (vCDR) que el
modelo enuncia frente a la referencia. Dataset: Chákṣu (1345 fotografías de fondo de ojo con anotación de cinco expertos).

## Contenido

```
scripts/      instrumentos del estudio; se ejecutan desde la raíz con `python -m scripts.<paquete>.<nombre>`
  data/       del dataset crudo a particiones y texto objetivo: manifest, audit, splits, targets, agreement
  model/      carga de MedGemma y LoRA (inspect, preview, train, generate): medgemma, lora
  eval/       lectura de predicciones: evaluate, fields, results, surface, text_report, normalize, extractor_sample
  studies/    experimentos que combinan los anteriores: baseline_prompts (prompting del base), pipeline (config. C), external
  analysis/   predictores de capas y reportes: probe (sondas), rgn (Auto-RGN), grid_analysis, finalists, test_report, figures, external_report
prototype/    prototipo (un solo cuaderno con toda la lógica y la interfaz de ipywidgets): imagen de Chákṣu con referencia o imagen propia sin ella → descripción clínica, con selector de adaptador; `prototype/README.md` dice cómo lanzarlo
runs/         listas de corridas, una por línea (`DATASET RECETA CAPAS SEMILLA`)
data/         datos derivados del dataset: manifests, splits, targets, prompts, stage2 (entradas de la etapa 2 de C)
results/      lo que se lee y se cita: grids/ (una tabla por rejilla), figures/, rgn/, probes/, audit/ (informes)
adapters/     los cinco adaptadores LoRA del prototipo (no se incluyen en el repositorio por su tamaño; se entregan por solicitud al autor)
experiments/  una carpeta por corrida: adapter, run_config.json y predicciones crudas
```

`data/`, `experiments/` y partes de `results/` (`audit/*.csv`, `normalized/`, `probes/*.npz`) no se versionan.
Los datasets de terceros (Chákṣu, MM-ODIR-129) no se incluyen: se descargan
aparte y el flujo los lee de `../datasets/<nombre>/raw/`. Cada uno conserva su propia licencia.

## Uso

MedGemma es un modelo restringido en Hugging Face: hace falta `HF_TOKEN` o `hf auth login`. Las versiones del
entorno están en `requirements.txt`. Los comandos que cargan el modelo necesitan GPU; los entrenamientos del estudio se lanzaron como trabajos de SLURM en una GPU L40S de 46 GB.

```bash
# Dataset crudo → particiones y texto objetivo (sin GPU)
python -m scripts.data.manifest chaksu
python -m scripts.data.audit chaksu
python -m scripts.data.splits chaksu --seed 42
python -m scripts.data.targets chaksu --recipe json_v1

# Anatomía del modelo y qué módulos cubre un grupo de capas (requieren GPU)
python -m scripts.model.lora inspect
python -m scripts.model.lora preview --layers attention_language

# Una corrida: entrenar y evaluar sobre validación
python -m scripts.model.lora train \
    --targets data/targets/chaksu__json_v1.csv --splits data/splits/chaksu_seed42.csv --split train \
    --prompt-file data/targets/chaksu__json_v1.prompt.txt --datasets-root ../datasets \
    --layers block_language --seed 42 --output-dir experiments/chaksu_json_v1/block_language_s42
python -m scripts.model.lora generate \
    --targets data/targets/chaksu__json_v1.csv --splits data/splits/chaksu_seed42.csv --split validation \
    --prompt-file data/targets/chaksu__json_v1.prompt.txt --datasets-root ../datasets \
    --adapter experiments/chaksu_json_v1/block_language_s42 \
    --output-dir experiments/chaksu_json_v1/block_language_s42/validation

# La rejilla completa: cada línea de runs/chaksu_json_v1_grid.txt es una corrida (DATASET RECETA CAPAS SEMILLA)
# que se entrena y evalúa como arriba

# Métricas (las predicciones crudas quedan en experiments/)
python -m scripts.eval.evaluate --predictions experiments/<corrida>/pred_real.jsonl \
    --compare experiments/<corrida>/pred_shuffled.jsonl experiments/<corrida>/pred_blank.jsonl
```

El prototipo (`prototype/prototipo.ipynb`, con toda la lógica y la interfaz) necesita una GPU y los adaptadores de `adapters/`; cómo abrirlo está en `prototype/README.md`.

Cada corrida de entrenamiento escribe un `run_config.json` con el modelo, el commit, los datos, los módulos
adaptados, los hiperparámetros y la semilla.

## Licencia

Código bajo licencia MIT (`LICENSE`). Los pesos de MedGemma y los datasets se rigen por sus propias licencias y condiciones de uso.
