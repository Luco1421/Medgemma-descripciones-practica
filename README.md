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
slurm/        plantillas de job; `batch_template.slurm` recorre una lista de `runs/`
prototype/    prototipo (`prototipo.py` + cuaderno + interfaz de ipywidgets): imagen de Chákṣu con referencia o imagen propia sin ella → descripción clínica, con selector de adaptador; `prototype/README.md` dice cómo lanzarlo
runs/         una corrida por línea (`DATASET RECETA CAPAS SEMILLA`)
data/         datos derivados del dataset: manifests, splits, targets, prompts, stage2 (entradas de la etapa 2 de C)
results/      lo que se lee y se cita: grids/ (una tabla por rejilla), figures/, rgn/, probes/, audit/ (informes)
experiments/  una carpeta por corrida: adapter, run_config.json y predicciones crudas
```

`data/`, `experiments/` y partes de `results/` (`audit/*.csv`, `normalized/`, `probes/*.npz`) no se versionan.
Los datasets de terceros (Chákṣu, MM-ODIR-129) no se incluyen: se descargan
aparte y el flujo los lee de `../datasets/<nombre>/raw/`. Cada uno conserva su propia licencia.

## Uso

MedGemma es un modelo restringido en Hugging Face: hace falta `HF_TOKEN` o `hf auth login`. Las versiones del
entorno están en `requirements.txt`. Los comandos que cargan el modelo necesitan GPU y se lanzan con `sbatch`.

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
sbatch slurm/finetune_template.slurm chaksu json_v1 block_language 42
sbatch slurm/inference_template.slurm chaksu json_v1 validation experiments/chaksu_json_v1_block_language_s42

# Varias corridas de una lista, en un solo job
sbatch slurm/batch_template.slurm runs/chaksu_json_v1_grid.txt 1 10

# Métricas (las predicciones crudas quedan en experiments/)
python -m scripts.eval.evaluate --predictions experiments/<corrida>/pred_real.jsonl \
    --compare experiments/<corrida>/pred_shuffled.jsonl experiments/<corrida>/pred_blank.jsonl
```

El prototipo (`prototype/prototipo.py`, interfaz en `prototype/prototipo.ipynb`) se verifica de punta a punta con `sbatch slurm/notebook_template.slurm`, o se usa de forma interactiva con `slurm/jupyter_template.slurm` (ver `prototype/README.md`).

Cada corrida de entrenamiento escribe un `run_config.json` con el modelo, el commit, los datos, los módulos
adaptados, los hiperparámetros y la semilla.

## Licencia

Código bajo licencia MIT (`LICENSE`). Los pesos de MedGemma y los datasets se rigen por sus propias licencias y condiciones de uso.
