"""Aplica a las rejillas de A y B los criterios de las hipótesis H1–H7, tal como se fijaron de antemano.

    python -m scripts.analysis.grid_analysis [--rgn results/rgn/chaksu_json_v1.csv results/rgn/chaksu_prose_v1.csv]

Lee `results/grids/chaksu_{json_v1,prose_v1}/results.csv` y, para las comparaciones emparejadas, las
predicciones de cada corrida (`scripts.eval.results`). Imprime y escribe `results/audit/chaksu_rejilla_hipotesis.txt`.
Diferencia relevante: 0.004. Los intervalos son los de `scripts.results.block_interval`: remuestreo de
bloques de captura de la validación, sobre la media de las tres semillas imagen a imagen.
Con `--rgn` (el CSV de A y el de B, en ese orden), añade el predictor de Auto-RGN (`scripts.analysis.rgn`) a H7 con el mismo criterio que la sonda.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.eval.evaluate import training_medians
from scripts.eval.results import GRIDS, block_interval, parse_runs, run_metrics
from scripts.data.splits import chaksu_groups, read_manifest

RECIPES = {"A": "json_v1", "B": "prose_v1"}
RELEVANT = 0.004
PROBES = "results/probes/chaksu_json_v1_sondas.csv"
TOWERS = ("vision", "language")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgn", nargs=2, metavar=("A", "B"), help="CSV de scripts.analysis.rgn de A y de B para añadir el predictor Auto-RGN a H7")
    args = parser.parse_args()
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    groups, _ = chaksu_groups(read_manifest("chaksu"))
    probes = pd.read_csv(PROBES)
    rgn = dict(zip(RECIPES, map(pd.read_csv, args.rgn))) if args.rgn else None
    tables, errors, blocks = {}, {}, None
    for label, recipe in RECIPES.items():
        folder = Path("experiments") / f"chaksu_{recipe}"
        tables[label] = pd.read_csv(GRIDS / f"chaksu_{recipe}" / "results.csv").set_index("cell")
        _, _, cells = parse_runs(f"runs/chaksu_{recipe}_grid.txt")
        seeds = next(iter(cells.values()))
        constant = training_medians(f"data/targets/chaksu__{recipe}.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
        errors[label] = {}
        needed = [c for c in cells if c.startswith(("block_", "random_")) or c == "all-linear"]
        for cell in needed:
            per_seed = []
            for seed in seeds:
                _, error, ids, _ = run_metrics(folder / f"{cell}_s{seed}", constant)
                per_seed.append(error)
            errors[label][cell] = np.mean(per_seed, axis=0)
        blocks = np.array([groups[i] for i in ids])

    def paired(label, a, b):
        mean, low, high = block_interval(errors[label][a] - errors[label][b], blocks)
        return mean, low, high

    def fmt(t):
        return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"

    say(f"Criterios de las hipótesis H1–H7 sobre las rejillas (diferencia relevante {RELEVANT}). Error del vCDR, media de tres semillas.\n")
    for label in RECIPES:
        t = tables[label]
        say(f"=== {label} ({RECIPES[label]}): constante {t.loc['constant', 'vcdr_error_mean']:.4f}, base "
            f"{t.loc['base', 'vcdr_error_mean']:.4f}, all-linear {t.loc['all-linear', 'vcdr_error_mean']:.4f}")
        say("H1  bloque − control aleatorio de igual presupuesto")
        for tower in TOWERS:
            d = paired(label, f"block_{tower}", f"random_block_{tower}")
            say(f"    {tower:<9}{fmt(d)}   {'se cumple' if d[0] < -RELEVANT and d[2] < 0 else 'no se cumple'}")
        for h, tower in (("H2", "vision"), ("H3", "language")):
            say(f"{h}  {tower}: tercio medio frente al temprano y al tardío")
            for other in ("early", "late"):
                d = paired(label, f"block_{tower}_mid", f"block_{tower}_{other}")
                say(f"    medio − {other:<6}{fmt(d)}")
        singles = {tower: t[[c.startswith(f"block_{tower}_layer") for c in t.index]]["vcdr_error_mean"] for tower in TOWERS}
        say("H4  mejor capa suelta frente a all-linear (elegida entre las de su torre: sesgo de selección)")
        for tower in TOWERS:
            best = singles[tower].idxmin()
            d = paired(label, best, "all-linear")
            say(f"    {tower:<9}{best:<24}{fmt(d)}   {'se cumple' if d[0] < RELEVANT else 'no se cumple'}")
        real = t.drop(index="constant")
        c1, c2 = real["image_gain_mean"] > 0, real["image_gain_mean"] <= real["blank_gain_mean"]
        say(f"H5  {len(real)} celdas (con el base): real < barajada {int(c1.sum())}, barajada ≤ blanco {int(c2.sum())}, "
            f"las dos {int((c1 & c2).sum())}")

    cells = [c for c in tables["A"].index]
    rho = spearmanr(tables["A"].loc[cells, "vcdr_error_mean"], tables["B"].loc[cells, "vcdr_error_mean"]).statistic
    rho88 = spearmanr(*[tables[k].drop(index=["base", "constant"])["vcdr_error_mean"] for k in "AB"]).statistic
    say(f"\nH6  Spearman entre A y B sobre {len(cells)} filas (con base y constante) {rho:+.3f}; sobre las 88 celdas {rho88:+.3f}")
    for tower in TOWERS:
        singles = [c for c in cells if c.startswith(f"block_{tower}_layer")]
        say(f"    capas sueltas de {tower}: {spearmanr(tables['A'].loc[singles, 'vcdr_error_mean'], tables['B'].loc[singles, 'vcdr_error_mean']).statistic:+.3f}")

    say("\nH7  Spearman entre el predictor de cada capa y el error de su celda de una sola capa (más negativo = mejor para la sonda y Auto-RGN)")
    say("    profundidad: el signo esperado es positivo (más profundo, más error); la sonda de la capa k es la salida del bloque k")
    for label in RECIPES:
        for tower in TOWERS:
            n = 27 if tower == "vision" else 34
            error = np.array([tables[label].loc[f"block_{tower}_layer{k:02d}", "vcdr_error_mean"] for k in range(n)])
            parts = [f"profundidad {spearmanr(np.arange(n), error).statistic:+.3f}"]
            families = ("vision",) if tower == "vision" else ("image_tokens", "last_token")
            for family in families:
                score = probes[probes.family == family].set_index("layer").loc[range(1, n + 1), "vcdr_spearman"]
                parts.append(f"sonda {family} {spearmanr(score, error).statistic:+.3f}")
            if rgn is not None:
                r = rgn[label][rgn[label].tower == tower].set_index("layer").loc[range(n)]
                for kind in ("block", "attention", "mlp"):
                    parts.append(f"RGN {kind} {spearmanr(r[f'rgn_{kind}_mean'], error).statistic:+.3f}")
            say(f"    {label} {tower:<9}n={n}: " + ", ".join(parts))

    Path("results/audit/chaksu_rejilla_hipotesis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
