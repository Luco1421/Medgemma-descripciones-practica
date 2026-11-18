# results/

Lo que se lee y se cita. Todo se regenera desde `experiments/` y `data/` con el comando de la tabla (desde la raíz del repo, con `python -m`);
nada se edita a mano. Las cifras que se citen deben salir de estos archivos.

| Archivo | Qué contiene | Comando |
|---|---|---|
| `grids/chaksu_json_v1/results.csv` | Rejilla de la receta A en validación: una fila por celda (88 + base + constante), cada semilla, media, desviación, diferencia emparejada con `all-linear` | `scripts.eval.results runs/chaksu_json_v1_grid.txt` |
| `grids/chaksu_prose_v1/results.csv` | Lo mismo para la receta B (prosa) | `scripts.eval.results runs/chaksu_prose_v1_grid.txt` |
| `grids/*/results_test.csv` | Test de los 14 finalistas (abierto una vez) | `scripts.eval.results runs/chaksu_<receta>_finalistas.txt --split test` |
| `grids/chaksu_cot_v1/results.csv` | Cadena de razonamiento | `scripts.eval.results runs/chaksu_cot_semillas.txt` |
| `grids/chaksu_json_v1/results_sorteos.csv` | Sorteos adicionales del control aleatorio | `scripts.eval.results runs/chaksu_json_v1_sorteos.txt --tag sorteos` |
| `audit/chaksu.txt`, `audit/mmodir129.txt` | Auditoría de los datasets | `scripts.data.audit chaksu`, `scripts.data.audit mmodir129` |
| `audit/chaksu_acuerdo.txt` | Acuerdo entre los cinco expertos de Chákṣu | `scripts.data.agreement chaksu` |
| `audit/chaksu_rejilla_hipotesis.txt` | Hipótesis H1 a H7 y predictores de capas | `scripts.analysis.grid_analysis --rgn results/rgn/chaksu_json_v1.csv results/rgn/chaksu_prose_v1.csv` |
| `audit/chaksu_test.txt` | Los seis criterios preestablecidos en el Test | `scripts.analysis.test_report` |
| `audit/chaksu_prompts_base.txt` | Ocho prompts del modelo base | `scripts.studies.baseline_prompts report` |
| `audit/chaksu_C.txt` | Configuración C, etapa 2 solo texto | `scripts.studies.pipeline report` |
| `audit/chaksu_C_img.txt` | Configuración C, etapa 2 con imagen | `scripts.studies.pipeline report_img` |
| `audit/chaksu_superficie.txt` | Métricas de superficie de B y su meta-evaluación; GPU | `scripts.eval.surface runs/chaksu_prose_v1_grid.txt` |
| `audit/chaksu_superficie_C.txt` | Métricas de superficie de B y de la etapa 2 de C; GPU | `scripts.eval.surface --stage2` |
| `audit/mmodir129_externa.txt`, `.csv` | Validación externa en MM-ODIR-129 | `scripts.analysis.external_report` |
| `audit/recursos.txt` | Recursos usados: entrenamientos y horas de GPU, almacenamiento, memoria por adaptador y proyección de una calibración por ubicación | `scripts.analysis.resources` |
| `audit/latencia.txt`, `.csv` | Latencia de inferencia por imagen de A, B, C (redactor LoRA y base) y el modelo base: media, desviación, mediana, p95, tokens y memoria pico, 40 imágenes de validación en una L40S, dos pasadas | `scripts.analysis.latency` |
| `audit/chaksu_bos.txt` | Efecto del `<bos>` duplicado en la generación de `all-linear` | `scripts.studies.bos` (tras generar con `lora.py generate --duplicate-bos --image-modes real`) |
| `probes/chaksu_json_v1_sondas.txt`, `.csv` | Sondas lineales por capa | `scripts.analysis.probe extract` (GPU) y `fit` |
| `probes/chaksu_json_v1_pooling.txt`, `.csv` | Sondas de la torre de visión con cuatro resúmenes de los parches (promedio, máximo, gradiente morfológico, su pico); GPU | `scripts.analysis.probe pooling` |
| `rgn/chaksu_*.csv` | Norma de gradiente relativa por bloque | `scripts.analysis.rgn` |
| `figures/` | Figuras para el informe y el paper | `scripts.analysis.figures` |
| `figures/configuraciones.png`, `.pdf` | Esquema de las configuraciones A, B y C | `scripts.analysis.figures` |
| `figures/particion_camaras.png`, `.pdf` | Imágenes por partición y cámara, con la prevalencia de «sospechoso» | `scripts.analysis.figures` |
| `figures/acuerdo_expertos.png`, `.pdf` | Discrepancia de los expertos en el vCDR y dispersión experto 1 frente a experto 2 | `scripts.analysis.figures` |
| `figures/prediccion_vs_referencia.png`, `.pdf` | vCDR predicho frente al de referencia en validación (modelo base y all-linear) | `scripts.analysis.figures` |
| `figures/controles_imagen.png`, `.pdf` | Error con imagen correcta, de otro paciente y en blanco (base y all-linear) | `scripts.analysis.figures` |
| `figures/sondas_profundidad.png`, `.pdf` | Sondas de vCDR y decisión por profundidad, con su control | `scripts.analysis.figures` |
| `figures/rgn_bloques.png`, `.pdf` | RGN por bloque del modelo base, A y B | `scripts.analysis.figures` |
| `figures/sorteos_control.png`, `.pdf` | Bloque frente a cuatro sorteos del mismo presupuesto | `scripts.analysis.figures` |
| `figures/validacion_externa.png`, `.pdf` | AUC con IC en MM-ODIR-129 por ubicación, A y B | `scripts.analysis.figures` |
| `figures/etapa2_C.png`, `.pdf` | Error, fidelidad y BLEU-4 de B y de las cuatro variantes de la etapa 2 de C | `scripts.analysis.figures` |
| `figures/superficie_vs_error.png`, `.pdf` | BLEU-4 y ROUGE-L por imagen frente al error del vCDR en B | `scripts.analysis.figures` |
| `figures/pooling_sondas.png`, `.pdf` | Sondas de la torre de visión según el resumen de los parches | `scripts.analysis.figures pooling` |
| `audit/costo_beneficio.txt`, `.csv`; `figures/costo_beneficio.png`, `.pdf` | Costo-beneficio de las 88 ubicaciones frente a all-linear: parámetros, tiempo de entrenamiento, MiB del adaptador, error en validación y Test, fracción de la mejora conservada y frontera de Pareto | `scripts.analysis.costbenefit` |

El prototipo (`prototype/prototipo.ipynb`, que se guarda ejecutado) es una demostración sobre imágenes de validación de Chákṣu y una imagen tratada como desconocida; no produce cifras del estudio. Las cifras que muestra para cada adaptador se leen de `grids/*/results.csv` y `results_test.csv`.

Lo que no se versiona: `audit/*.csv` (medidas por imagen de datasets de terceros), `normalized/` (predicciones con la referencia) y `probes/*.npz` (activaciones, cientos de MB).
