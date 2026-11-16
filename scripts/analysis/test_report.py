"""Aplica a los resultados del Test los seis criterios, tal como se fijaron antes de abrirlo.

    python -m scripts.analysis.test_report

Lee `results/grids/chaksu_{json_v1,prose_v1}/results.csv` (validación) y `results_test.csv` (Test, `scripts.eval.results --split test`), y las
predicciones de cada corrida para las diferencias emparejadas. Imprime y escribe `results/audit/chaksu_test.txt`.
Diferencia relevante 0.004; intervalos por bloques de captura sobre la media de las tres semillas (`scripts.results.block_interval`).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.eval.evaluate import training_medians
from scripts.analysis.finalists import A_PRIORI, EFFICIENT, SEEDS, folder_name
from scripts.eval.results import GRIDS, block_interval, run_metrics
from scripts.data.splits import chaksu_groups, read_manifest

RECIPES = {"A": "json_v1", "B": "prose_v1"}
RELEVANT = 0.004
PAIRS = {"visión": ("block_vision", "random_block_vision"), "lenguaje": ("block_language", "random_block_language")}


def main():
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    cells = [folder_name(c) for c in A_PRIORI + EFFICIENT]
    efficient = [folder_name(c) for c in EFFICIENT]
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    for label, recipe in RECIPES.items():
        folder = Path("experiments") / f"chaksu_{recipe}"
        validation = pd.read_csv(GRIDS / f"chaksu_{recipe}" / "results.csv").set_index("cell")
        test = pd.read_csv(GRIDS / f"chaksu_{recipe}" / "results_test.csv").set_index("cell")
        constant = training_medians(f"data/targets/chaksu__{recipe}.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
        error, ids = {}, None
        for cell in cells:
            runs = [run_metrics(folder / f"{cell}_s{s}", constant, "test") for s in SEEDS]
            error[cell] = np.mean([r[1] for r in runs], axis=0)
            ids = ids or runs[0][2]
            assert runs[0][2] == ids
        blocks = np.array([groups[i] for i in ids])

        def paired(a, b):
            mean, low, high = block_interval(error[a] - error[b], blocks)
            return mean, low, high

        say(f"=================== Receta {label} ({recipe}); Test de {len(ids)} imágenes; constante {test.loc['constant', 'vcdr_error_mean']:.4f}, "
            f"base {test.loc['base', 'vcdr_error_mean']:.4f}, all-linear {test.loc['all-linear', 'vcdr_error_mean']:.4f}")
        say(f"{'ubicación':<26}{'parám. M':>9}{'error Test':>11}{'dif. con all-linear [IC95]':>34}{'IC en ±0.004':>14}{'dif. validación':>17}")
        flips, inside = 0, {}
        for cell in cells:
            if cell == "all-linear":
                continue
            mean, low, high = paired(cell, "all-linear")
            inside[cell] = low > -RELEVANT and high < RELEVANT
            flipped = np.sign(validation.loc[cell, "diff_vs_ref"]) != np.sign(mean)
            flips += int(flipped)
            say(f"{cell:<26}{test.loc[cell, 'trainable_params'] / 1e6:>9.2f}{test.loc[cell, 'vcdr_error_mean']:>11.4f}"
                f"{mean:>+14.4f} [{low:+.4f}, {high:+.4f}]{'sí' if inside[cell] else 'no':>10}{validation.loc[cell, 'diff_vs_ref']:>+17.4f}{'  (cambia de lado)' if flipped else ''}")

        say("\nCriterio 1, eficiencia (IC95 de celda − all-linear dentro de ±0.004):")
        say(f"  eficientes que cumplen: {[c for c in efficient if inside[c]]} de {efficient}")
        say("Criterio 2, bloque − control aleatorio a igual presupuesto (validación en A: −0.0006 visión y −0.0007 lenguaje; en B: +0.0132 y −0.0005):")
        for tower, (block, control) in PAIRS.items():
            mean, low, high = paired(block, control)
            say(f"  {tower:<9}{block} − {control}: {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
        better = [c for c in cells if c != "all-linear" and paired(c, "all-linear")[2] < 0]
        worse = [c for c in cells if c != "all-linear" and paired(c, "all-linear")[1] > RELEVANT]
        say(f"Criterio 3, replicación: ubicaciones mejores que all-linear (IC por debajo de 0): {better or 'ninguna'}; peores en más de {RELEVANT} con IC por encima: {worse or 'ninguna'}")
        adapted = [c for c in cells if c != "base"]
        gains = test.loc[adapted, ["image_gain_mean", "blank_gain_mean"]]
        say(f"Criterio 4, controles de imagen: ganancia mínima sobre la barajada {gains.image_gain_mean.min():+.4f} ({gains.image_gain_mean.idxmin()}), "
            f"sobre el blanco {gains.blank_gain_mean.min():+.4f} ({gains.blank_gain_mean.idxmin()}); base: {test.loc['base', 'image_gain_mean']:+.4f} y {test.loc['base', 'blank_gain_mean']:+.4f}")
        rho = spearmanr(validation.loc[cells, "vcdr_error_mean"], test.loc[cells, "vcdr_error_mean"]).statistic
        rho_d = spearmanr(validation.loc[[c for c in cells if c != "all-linear"], "diff_vs_ref"],
                          test.loc[[c for c in cells if c != "all-linear"], "diff_vs_ref"]).statistic
        say(f"Criterio 5, transferencia: Spearman del error de validación con el del Test en las {len(cells)} ubicaciones {rho:+.3f}; de la diferencia con all-linear {rho_d:+.3f}; "
            f"ubicaciones que cambian de lado respecto de all-linear: {flips} de {len(cells) - 1}")
        best_val, best_test = validation.loc[cells, "vcdr_error_mean"].idxmin(), test.loc[cells, "vcdr_error_mean"].idxmin()
        say(f"  mejor en validación: {best_val} ({validation.loc[best_val, 'vcdr_error_mean']:.4f}), en Test {test.loc[best_val, 'vcdr_error_mean']:.4f}; mejor en Test: {best_test} ({test.loc[best_test, 'vcdr_error_mean']:.4f})")
        say(f"Criterio 6, referencias: constante {test.loc['constant', 'vcdr_error_mean']:.4f} (validación {validation.loc['constant', 'vcdr_error_mean']:.4f}); "
            f"base {test.loc['base', 'vcdr_error_mean']:.4f} (validación {validation.loc['base', 'vcdr_error_mean']:.4f}); all-linear {test.loc['all-linear', 'vcdr_error_mean']:.4f} "
            f"(validación {validation.loc['all-linear', 'vcdr_error_mean']:.4f}), {100 * (1 - test.loc['all-linear', 'vcdr_error_mean'] / test.loc['constant', 'vcdr_error_mean']):.0f} % por debajo de la constante\n")

    Path("results/audit/chaksu_test.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
