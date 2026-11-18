"""Finalistas del Test: su definición y las listas de corridas que se derivan de ella.

    python -m scripts.analysis.finalists

Es la única fuente de qué celdas se abren en el Test. Escribe, para cada receta, `runs/chaksu_<receta>_finalistas.txt` (el formato de
`scripts.eval.results`, con `--split test`) y `runs/chaksu_test_<n>.txt`, las listas de generación del Test
(`RECETA CARPETA_DE_LA_CORRIDA`, una por línea, repartidas por turnos entre cuatro jobs; el modelo base va primero).

Criterio de la lista, fijado antes de abrir el Test:
  - a priori: `all-linear`, el bloque, la atención y el MLP de cada torre, y los dos controles aleatorios a igual presupuesto;
  - eficientes: las cinco celdas de menos de 15 M de parámetros entrenables (el 39 % de `all-linear`) y fuera de las anteriores con menor
    error medio de validación entre A y B (media de las tres semillas de cada receta).
"""

from pathlib import Path

SEEDS = (42, 43, 44)
RECIPES = ("json_v1", "prose_v1")
A_PRIORI = ["all-linear", "block_vision", "block_language", "attention_vision", "attention_language", "mlp_vision", "mlp_language",
            "random:block_vision", "random:block_language"]
EFFICIENT = ["block_language_early", "block_language_mid", "mlp_language_early", "block_language_layer00", "block_language_layer01"]
JOBS = 4


def folder_name(cell):
    return cell.replace(":", "_")


def main():
    cells = A_PRIORI + EFFICIENT
    for recipe in RECIPES:
        lines = [f"chaksu {recipe} {cell} {seed}" for cell in cells for seed in SEEDS]
        Path("runs", f"chaksu_{recipe}_finalistas.txt").write_text(
            "# Finalistas del Test, tres semillas. Formato: DATASET RECETA CAPAS SEMILLA\n" + "\n".join(lines) + "\n", encoding="utf-8")
    runs = [(recipe, f"experiments/chaksu_{recipe}/base") for recipe in RECIPES]
    runs += [(recipe, f"experiments/chaksu_{recipe}/{folder_name(cell)}_s{seed}") for cell in cells for recipe in RECIPES for seed in SEEDS]
    for job in range(JOBS):
        mine = runs[job::JOBS]
        Path("runs", f"chaksu_test_{job + 1}.txt").write_text(
            "# Test: RECETA CARPETA_DE_LA_CORRIDA; las predicciones van a <carpeta>/test/\n"
            + "\n".join(f"{recipe} {run}" for recipe, run in mine) + "\n", encoding="utf-8")
        print(f"runs/chaksu_test_{job + 1}.txt: {len(mine)} corridas")
    print(f"{len(cells)} finalistas ({len(A_PRIORI)} a priori, {len(EFFICIENT)} eficientes) × {len(RECIPES)} recetas × {len(SEEDS)} semillas, más el base")


if __name__ == "__main__":
    main()
