# Prototipo

Una fotografía de fondo de ojo entra y sale una descripción clínica de una o dos frases, con MedGemma (`google/medgemma-1.5-4b-it`) y los adaptadores LoRA del estudio. Todo el prototipo es `prototipo.ipynb`: la lógica (carga, inferencia, lectura de campos, error frente a la referencia), la interfaz (ipywidgets) y un ejemplo ejecutable están en sus celdas, en orden y explicadas. Solo reutiliza los instrumentos del estudio (`scripts.model`, `scripts.eval`, `scripts.studies.pipeline`) para que muestre lo que se evaluó.

**No es un sistema clínico.** La fidelidad del texto es a los campos medidos (vCDR, hCDR, decisión), no a la imagen. La interfaz lo repite en un aviso fijo.

## Qué hace

Con un solo modelo en memoria se montan los adaptadores y `set_adapter` elige cuál actúa (antes de cada generación se comprueba con un `assert` que el adaptador activo es el pedido).

- **Camino C** (adaptadores A): la etapa 1 lee la imagen y da `{vcdr, hcdr, decision}` en JSON; la etapa 2 (LoRA de redacción `redaccion_block_language`) escribe el texto solo con esos campos y **no ve la imagen**.
  Costo medido (`results/audit/latencia.txt`; 40 imágenes de validación, GPU L40S): A y B tardan cerca de 1.1 a 1.2 s por imagen; C, con el redactor LoRA, 2.1 s (1.77 veces B), porque la etapa 2 es otra generación completa; cambiar de adaptador cuesta unos 16 ms. C no mejora el error del vCDR respecto de A; su aporte es que el texto repite por construcción los campos medidos. Conviene A para medir, B para prosa en una pasada y C solo si importa esa trazabilidad o redactar desde anotaciones de un oftalmólogo.
- **Camino B** (prosa directa): la imagen produce la frase sin pasar por campos.
- Opcionalmente se repite el mismo camino con el modelo base sin adaptar.

### Adaptadores del selector

Semilla de entrenamiento 42, fija. Los adaptadores van en `adapters/`; no se incluyen en el repositorio por su tamaño y se entregan por solicitud al autor. Error absoluto del vCDR: media ± desviación de las tres semillas, en la validación de Chákṣu (171 imágenes) y en el Test (reservado), leído de `results/grids/*/results.csv` y `results_test.csv`.

| Opción | Ubicación | Parámetros entrenables | Error vCDR Test | Error vCDR validación |
|---|---|---|---|---|
| A · all-linear | `adapters/A_all-linear` | 38 497 792 | 0.0379 | 0.0403 |
| A · block_language | `adapters/A_block_language` | 29 802 496 | 0.0404 | 0.0401 |
| A · mlp_language | `adapters/A_mlp_language` | 20 889 600 | 0.0408 | 0.0412 |
| B · prosa directa, all-linear | `adapters/B_all-linear` | 38 497 792 | 0.0386 | 0.0405 |
| Etapa 2 · redacción (siempre en el camino C) | `adapters/redaccion_block_language` | 29 802 496 | no mide el vCDR | no mide el vCDR |

Criterio de elección (cifras de `results_test.csv` de la receta A, 14 celdas finalistas): `all-linear` es la de menor error en el Test. Las dos siguientes son las de menor error entre las finalistas que no son controles aleatorios a igual presupuesto (`random_block_language` 0.0385 y `random_block_vision` 0.0397 quedan fuera porque son controles, no una elección de capas): `block_language` 0.0404 y `mlp_language` 0.0408. Las diferencias entre estas tres son de milésimas, del orden de la desviación entre semillas; el selector sirve para ver cómo se comportan, no para afirmar que una gana. La opción B es la de menor error de la receta de prosa (`all-linear`, Test 0.0386) y muestra el otro formato. Para comparar: el modelo base tiene 0.0599 en el Test de A y la constante 0.0584. Una opción de interés que no se ofrece: `block_language_layer01` (una sola capa, 876 544 parámetros, Test 0.0410).

## Cómo usarlo, paso a paso

1. Lanza JupyterLab, desde la raíz del repo, en una máquina con GPU (MedGemma tiene 4.3 B parámetros, unos 8.6 GB de pesos en bfloat16; las cifras de latencia del estudio son de una L40S). Si la GPU está en un servidor remoto, abre JupyterLab allí sin navegador y accede por un túnel SSH al puerto elegido:

        jupyter lab --no-browser --ip=127.0.0.1 --port=8899
        ssh -L 8899:127.0.0.1:8899 <usuario>@<servidor>      # en tu equipo

   Se abre en el navegador la URL con token que imprime `jupyter lab`, con el host `localhost`.
2. Se abre `prototype/prototipo.ipynb`, kernel `ia`. Se ejecutan las celdas de definiciones, la celda de carga (sección 11; unos minutos) y la de la interfaz (sección 12).
3. En la interfaz: se elige el adaptador, se marca si se quiere el modelo base al lado, y se elige el modo:
   - **Imagen con referencia conocida.** Seis ejemplos de la validación de Chákṣu (por cámara, la primera imagen normal y la primera sospechosa, por id) o cualquier imagen de validación por cámara. Tras el resultado se muestra la referencia de los expertos y el error del vCDR y del hCDR.
   - **Imagen propia, sin referencia.** Se sube un JPG, PNG o TIFF (de un TIFF se lee la primera página; se espera 8 bits por canal). El resultado no muestra ninguna referencia. Opcional: marcar «anotaciones de un oftalmólogo» (vCDR, hCDR, decisión) para que la etapa 2 los redacte sin pasar por la imagen (no aplica a la opción B), o «referencia propia» para comparar el resultado con lo que escribas.
4. El resultado trae la imagen, los campos medidos, el texto, el tiempo de cada etapa y, si se pidió, lo mismo con el modelo base.
5. Al terminar se detiene JupyterLab (Ctrl+C en su terminal).

Una imagen con campos ilegibles (la etapa 1 no dio los tres campos) se informa como tal, con la salida cruda; no se rellena nada.

## Verificación sin navegador

    jupyter nbconvert --to notebook --execute --inplace prototype/prototipo.ipynb \
        --ExecutePreprocessor.store_widget_state=False --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=ia

Ejecuta `prototype/prototipo.ipynb` de punta a punta (necesita GPU) y deja las salidas en el mismo cuaderno, sin el estado de los widgets. Construye la interfaz y llama a lo que hacen los botones: una imagen de validación de Chákṣu con referencia, una imagen de validación subida como archivo y tratada como desconocida (se comprueba que el resultado no contiene referencia), el camino con anotaciones, y un recorrido de las cuatro opciones del selector comprobando el adaptador activo y que las salidas no son todas iguales.

## Límites

- Lo que no se puede comprobar sin navegador es la interacción: clics, la subida real y el arrastre. El selector de archivos de ipywidgets 8 (`FileUpload`) es un botón; arrastrar y soltar sobre él depende del navegador y de JupyterLab y no está verificado. El botón es la vía soportada.
- Una sola imagen no demuestra nada: los errores de la tabla son medias sobre decenas de imágenes y semillas.
- Entrenado con Chákṣu (Bosch, Forus, Remidio); la decisión es «normal» frente a «sospechoso». Con fotografías de otras cámaras o no centradas en el disco no hay garantía.
- Sin referencia no se puede decir si un resultado es correcto.

Requisitos: los adaptadores en `adapters/` (se piden al autor) y los targets y particiones en `data/` (están en `.gitignore`), el token de Hugging Face de MedGemma y las imágenes de Chákṣu en `../datasets/`.
