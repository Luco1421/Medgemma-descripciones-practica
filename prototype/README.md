# Prototipo

Una fotografía de fondo de ojo entra y sale una descripción clínica de una o dos frases, con MedGemma (`google/medgemma-1.5-4b-it`) y los adaptadores LoRA del estudio. La lógica está en `prototipo.py` (importable); `prototipo.ipynb` solo carga el modelo, muestra la interfaz (ipywidgets) y trae un ejemplo ejecutable.

**No es un sistema clínico.** La fidelidad del texto es a los campos medidos (vCDR, hCDR, decisión), no a la imagen. La interfaz lo repite en un aviso fijo.

## Qué hace

Con un solo modelo en memoria se montan los adaptadores y `set_adapter` elige cuál actúa (antes de cada generación se comprueba con un `assert` que el adaptador activo es el pedido).

- **Camino C** (adaptadores A): la etapa 1 lee la imagen y da `{vcdr, hcdr, decision}` en JSON; la etapa 2 (LoRA de redacción `chaksu_verbalize_v1/block_language`) escribe el texto solo con esos campos y **no ve la imagen**.
  Costo medido (`results/audit/latencia.txt`; 40 imágenes de validación, GPU L40S): A y B tardan cerca de 1.1 a 1.2 s por imagen; C, con el redactor LoRA, 2.1 s (1.77 veces B), porque la etapa 2 es otra generación completa; cambiar de adaptador cuesta unos 16 ms. C no mejora el error del vCDR respecto de A; su aporte es que el texto repite por construcción los campos medidos. Conviene A para medir, B para prosa en una pasada y C solo si importa esa trazabilidad o redactar desde anotaciones de un oftalmólogo.
- **Camino B** (prosa directa): la imagen produce la frase sin pasar por campos.
- Opcionalmente se repite el mismo camino con el modelo base sin adaptar.

### Adaptadores del selector

Semilla de entrenamiento 42 (se cambia en la celda de carga). Error absoluto del vCDR: media ± desviación de las tres semillas, en la validación de Chákṣu (171 imágenes) y en el Test (reservado), leído de `results/grids/*/results.csv` y `results_test.csv`.

| Opción | Ubicación | Parámetros entrenables | Error vCDR Test | Error vCDR validación |
|---|---|---|---|---|
| A · all-linear | `experiments/chaksu_json_v1/all-linear_s42` | 38 497 792 | 0.0379 | 0.0403 |
| A · block_language | `experiments/chaksu_json_v1/block_language_s42` | 29 802 496 | 0.0404 | 0.0401 |
| A · mlp_language | `experiments/chaksu_json_v1/mlp_language_s42` | 20 889 600 | 0.0408 | 0.0412 |
| B · prosa directa, all-linear | `experiments/chaksu_prose_v1/all-linear_s42` | 38 497 792 | 0.0386 | 0.0405 |
| Etapa 2 · redacción (siempre en el camino C) | `experiments/chaksu_verbalize_v1/block_language_s42` | 29 802 496 | no mide el vCDR | no mide el vCDR |

Criterio de elección (cifras de `results_test.csv` de la receta A, 14 celdas finalistas): `all-linear` es la de menor error en el Test. Las dos siguientes son las de menor error entre las finalistas que no son controles aleatorios a igual presupuesto (`random_block_language` 0.0385 y `random_block_vision` 0.0397 quedan fuera porque son controles, no una elección de capas): `block_language` 0.0404 y `mlp_language` 0.0408. Las diferencias entre estas tres son de milésimas, del orden de la desviación entre semillas; el selector sirve para ver cómo se comportan, no para afirmar que una gana. La opción B es la de menor error de la receta de prosa (`all-linear`, Test 0.0386) y muestra el otro formato. Para comparar: el modelo base tiene 0.0599 en el Test de A y la constante 0.0584. Una opción de interés que no se ofrece: `block_language_layer01` (una sola capa, 876 544 parámetros, Test 0.0410).

## Cómo usarlo, paso a paso

1. Lanza JupyterLab en un nodo con GPU, desde la raíz del repo (el login node no tiene GPU):

        mkdir -p experiments/logs
        sbatch slurm/jupyter_template.slurm

2. Cuando el job esté en ejecución (`squeue -u $USER`), el `.out` dice el nodo y el puerto y el `.err` (donde JupyterLab escribe su log) trae la URL con el token:

        cat experiments/logs/jupyter_<job>.out
        grep token experiments/logs/jupyter_<job>.err

3. En tu equipo, un túnel que pasa por el login node (la primera vez SSH pregunta por la huella del nodo):

        ssh -J <usuario>@<login-node> -L 8899:127.0.0.1:8899 <usuario>@<nodo>

   Requiere que tu clave pública esté autorizada también en el nodo (con `-J` el cliente usa su clave en los dos saltos); desde el login node, sin clave propia, `ssh <nodo>` da `Permission denied`. Con otro puerto: `PORT=9000 sbatch slurm/jupyter_template.slurm` y se cambia en el `-L`. Se abre en el navegador la URL del `.err`, sustituyendo el host por `localhost`.
4. Se abre `prototype/prototipo.ipynb`, kernel `ia`. Se ejecuta la celda 1 (carga; unos minutos) y la celda 2 (muestra la interfaz).
5. En la interfaz: se elige el adaptador, se marca si se quiere el modelo base al lado, y se elige el modo:
   - **Imagen con referencia conocida.** Seis ejemplos de la validación de Chákṣu (por cámara, la primera imagen normal y la primera sospechosa, por id) o cualquier imagen de validación por cámara. Tras el resultado se muestra la referencia de los expertos y el error del vCDR y del hCDR.
   - **Imagen propia, sin referencia.** Se sube un JPG, PNG o TIFF (de un TIFF se lee la primera página; se espera 8 bits por canal). El resultado no muestra ninguna referencia. Opcional: marcar «anotaciones de un oftalmólogo» (vCDR, hCDR, decisión) para que la etapa 2 los redacte sin pasar por la imagen (no aplica a la opción B), o «referencia propia» para comparar el resultado con lo que escribas.
6. El resultado trae la imagen, los campos medidos, el texto, el tiempo de cada etapa y, si se pidió, lo mismo con el modelo base.
7. Al terminar: `scancel <job>`.

Una imagen con campos ilegibles (la etapa 1 no dio los tres campos) se informa como tal, con la salida cruda; no se rellena nada.

## Verificación sin navegador

    sbatch slurm/notebook_template.slurm

Ejecuta el cuaderno de punta a punta con `jupyter nbconvert` y guarda `prototype/prototipo_ejecutado.ipynb`. Construye la interfaz y llama a lo que hacen los botones: dos imágenes de validación de Chákṣu con referencia, una imagen de validación subida como archivo y tratada como desconocida (se comprueba que el resultado no contiene referencia), el camino con anotaciones y referencia propia, y un recorrido de las cuatro opciones del selector comprobando el adaptador activo y que las salidas no son todas iguales.

## Límites

- Lo que no se puede comprobar sin navegador es la interacción: clics, la subida real y el arrastre. El selector de archivos de ipywidgets 8 (`FileUpload`) es un botón; arrastrar y soltar sobre él depende del navegador y de JupyterLab y no está verificado. El botón es la vía soportada.
- Una sola imagen no demuestra nada: los errores de la tabla son medias sobre decenas de imágenes y semillas.
- Entrenado con Chákṣu (Bosch, Forus, Remidio); la decisión es «normal» frente a «sospechoso». Con fotografías de otras cámaras o no centradas en el disco no hay garantía.
- Sin referencia no se puede decir si un resultado es correcto.

Requisitos: los adaptadores en `experiments/` y los targets y particiones en `data/` (están en `.gitignore`), el token de Hugging Face de MedGemma y las imágenes de Chákṣu en `../datasets/`.
