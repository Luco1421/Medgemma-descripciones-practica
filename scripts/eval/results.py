"""Tabla de resultados de una rejilla: una fila por celda, con cada semilla, la media y la desviación.

    python -m scripts.eval.results runs/chaksu_json_v1_grid.txt [--root experiments] [--reference all-linear] [--split validation]

Lee la lista de corridas (`DATASET RECETA CAPAS SEMILLA`, una por línea) y, de la carpeta de cada
corrida (`<root>/<dataset>_<receta>/<celda>_s<semilla>/validation/`), las predicciones crudas de los
tres controles de imagen. Escribe `<root>/<dataset>_<receta>/results.csv`, que es la única superficie de
lectura (con `--split test`, `results_test.csv`): se regenera entera y no se edita a mano. Una celda con alguna corrida sin terminar no entra y se
lista al final.

Columnas por métrica: `<m>_s<semilla>` para cada semilla, `<m>_mean` y `<m>_sd` (desviación muestral entre
semillas, ddof=1; con tres semillas está mal estimada). Es el ruido del ENTRENAMIENTO.
  vcdr_error     |vcdr enunciado − referencia| medio; una respuesta no válida cuenta como la constante de
                 entrenamiento (`scripts.eval.evaluate`).
  vcdr_spearman  correlación con la referencia sobre las respuestas válidas.
  sensitivity, specificity   de la decisión frente a la decisión por mayoría.
  format         fracción de respuestas con vcdr, hcdr y decisión válidos.
  image_gain     error con la imagen barajada menos error con la correcta; positivo = la imagen ayuda.
  blank_gain     lo mismo con la imagen en blanco.
`diff_vs_ref` es el error medio (promediado sobre las semillas, imagen a imagen) de la celda menos el de la
celda de referencia, con su intervalo del 95 %: remuestreo de bloques de captura de la validación, que es el
ruido de las IMÁGENES. Negativo = la celda tiene menos error que la referencia. Las filas `base` (modelo sin
adaptar) y `constant` (predictor constante) no tienen semillas.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from scripts.eval.evaluate import (agreement, answer_fields, decision_counts, errors, load, read_answer,
                              reference_value, training_medians)
from scripts.data.splits import chaksu_groups, read_manifest

MODES = ("real", "shuffled", "blank")
GRIDS = Path("results/grids")  # una tabla por rejilla: results/grids/<dataset>_<receta>/results.csv
METRICS = ("vcdr_error", "vcdr_spearman", "sensitivity", "specificity", "format", "image_gain", "blank_gain")
BOOTSTRAP_DRAWS = 2000


def parse_runs(path):
    runs = [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert runs and all(len(r) == 4 for r in runs), "cada línea debe ser DATASET RECETA CAPAS SEMILLA"
    assert len({tuple(r[:2]) for r in runs}) == 1, "la lista mezcla datasets o recetas"
    cells = {}
    for _, _, layers, seed in runs:
        cells.setdefault(re.sub(r"[^A-Za-z0-9_-]", "_", layers), []).append(int(seed))
    assert len({tuple(s) for s in cells.values()}) == 1, "no todas las celdas tienen las mismas semillas"
    return runs[0][0], runs[0][1], cells


def describe(cell):
    """Torre, profundidad y módulo de una celda, para filtrar la tabla."""
    if cell == "all-linear":
        return "both", "all", "all-linear"
    if cell.startswith("random_"):
        return "both", "all", "random"
    tower = "vision" if "vision" in cell else "language"
    depth = re.search(r"layer\d+|early|mid|late", cell)
    return tower, depth[0] if depth else "all", re.match(r"block|attention|mlp", cell)[0]


def run_metrics(folder, constant, split="validation"):
    """Métricas de una corrida y su error por imagen (con la imagen correcta)."""
    rows = {m: load(folder / split / f"pred_{m}.jsonl") for m in MODES}
    ids = [r["id"] for r in rows["real"]]
    assert all([r["id"] for r in rows[m]] == ids for m in MODES), f"{folder}: los controles no cubren las mismas imágenes"
    fields = answer_fields(rows["real"])
    reads = {m: [read_answer(r, fields) for r in rows[m]] for m in MODES}
    error = {m: errors(rows[m], reads[m], "vcdr", constant) for m in MODES}
    tp, positives, tn, negatives = decision_counts(rows["real"], reads["real"])
    valid = [i for i, (_, v) in enumerate(reads["real"]) if v["vcdr"] is not None]
    truth = np.array([reference_value(r, "vcdr") for r in rows["real"]])
    metrics = {
        "vcdr_error": error["real"].mean(),
        "vcdr_spearman": agreement([reads["real"][i][1]["vcdr"] for i in valid], truth[valid]),
        "sensitivity": tp / positives,
        "specificity": tn / negatives,
        "format": np.mean([all(x is not None for x in v.values()) for _, v in reads["real"]]),
        "image_gain": error["shuffled"].mean() - error["real"].mean(),
        "blank_gain": error["blank"].mean() - error["real"].mean(),
    }
    return metrics, error["real"], ids, truth


def block_interval(delta, blocks):
    """Media de la diferencia por imagen y su intervalo bootstrap del 95 %, remuestreando bloques de captura."""
    _, index = np.unique(blocks, return_inverse=True)
    sums, counts = np.bincount(index, weights=delta), np.bincount(index)
    draws = np.random.default_rng(0).integers(0, len(counts), (BOOTSTRAP_DRAWS, len(counts)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return delta.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", help="lista de corridas: DATASET RECETA CAPAS SEMILLA por línea")
    parser.add_argument("--root", default="experiments", help="carpeta con las corridas; las tablas se escriben en results/grids/")
    parser.add_argument("--reference", default="all-linear", help="celda contra la que se calcula diff_vs_ref")
    parser.add_argument("--split", default="validation", help="partición cuyas predicciones se tabulan (validation o test)")
    parser.add_argument("--tag", help="sufijo del archivo (results_<tag>.csv) para tablas que no son la rejilla completa")
    args = parser.parse_args()

    dataset, recipe, cells = parse_runs(args.runs)
    assert dataset == "chaksu", "los bloques de captura para el intervalo solo están definidos para Chákṣu"
    seeds = next(iter(cells.values()))
    folder = Path(args.root) / f"{dataset}_{recipe}"
    targets, splits = f"data/targets/{dataset}__{recipe}.csv", f"data/splits/{dataset}_seed42.csv"
    constant = training_medians(targets, splits, ["vcdr"])["vcdr"]
    groups, _ = chaksu_groups(read_manifest(dataset))

    ids, truth, runs, incomplete = None, None, {}, []
    for cell in cells:
        done = [(folder / f"{cell}_s{s}") for s in seeds]
        if not all((d / args.split / "generate_config.json").exists() for d in done):
            incomplete.append(cell)
            continue
        runs[cell] = []
        for d in done:
            metrics, error, run_ids, run_truth = run_metrics(d, constant, args.split)
            ids = ids or run_ids
            assert run_ids == ids, f"{d}: no cubre las mismas imágenes que las demás corridas"
            truth = run_truth
            config = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
            runs[cell].append((metrics, error, config["trainable_params"], config["train_metrics"]["train_loss"]))
    assert args.reference in runs, f"la celda de referencia {args.reference} no está completa"

    blocks = np.array([groups[i] for i in ids])
    mean_error = {cell: np.mean([r[1] for r in rs], axis=0) for cell, rs in runs.items()}
    base_dir = folder / "base"
    base = None
    if (base_dir / args.split / "generate_config.json").exists():
        base = run_metrics(base_dir, constant, args.split)
        assert base[2] == ids, "el modelo base no cubre las mismas imágenes"
        mean_error["base"] = base[1]
    mean_error["constant"] = np.abs(truth - constant)

    columns = ["cell", "tower", "depth", "module", "trainable_params", "train_loss_mean"]
    columns += [f"{m}_{k}" for m in METRICS for k in [f"s{s}" for s in seeds] + ["mean", "sd"]]
    columns += ["diff_vs_ref", "diff_ci_low", "diff_ci_high"]

    def diff(cell):
        if cell == args.reference:
            return {}
        mean, low, high = block_interval(mean_error[cell] - mean_error[args.reference], blocks)
        return {"diff_vs_ref": mean, "diff_ci_low": low, "diff_ci_high": high}

    table = []
    for cell, rs in runs.items():
        assert len({r[2] for r in rs}) == 1, f"{cell}: los parámetros entrenables cambian entre semillas"
        row = {"cell": cell, **dict(zip(("tower", "depth", "module"), describe(cell))),
               "trainable_params": rs[0][2], "train_loss_mean": np.mean([r[3] for r in rs]), **diff(cell)}
        for m in METRICS:
            values = [r[0][m] for r in rs]
            row.update({f"{m}_s{s}": v for s, v in zip(seeds, values)})
            row[f"{m}_mean"], row[f"{m}_sd"] = np.mean(values), np.std(values, ddof=1)
        table.append(row)
    if base:
        table.append({"cell": "base", "tower": "none", "depth": "none", "module": "none", "trainable_params": 0,
                      **{f"{m}_mean": v for m, v in base[0].items()}, **diff("base")})
    table.append({"cell": "constant", "tower": "none", "depth": "none", "module": "none", "trainable_params": 0,
                  "vcdr_error_mean": mean_error["constant"].mean(), **diff("constant")})

    stem = "results" if args.split == "validation" else f"results_{args.split}"
    out = GRIDS / f"{dataset}_{recipe}" / (f"{stem}_{args.tag}.csv" if args.tag else f"{stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in table:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    print(f"{out}: {len(runs)} celdas completas de {len(cells)}, con las semillas {seeds}")
    if incomplete:
        print(f"incompletas ({len(incomplete)}): {', '.join(incomplete)}")


if __name__ == "__main__":
    main()
