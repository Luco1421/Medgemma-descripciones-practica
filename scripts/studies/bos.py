"""Efecto del <bos> duplicado: el entrenamiento ve [2, 2, 105, ...] y la generación [2, 105, ...].

    python -m scripts.studies.bos

Compara, para `all-linear` (A, validación, semillas 42, 43 y 44), el error del vCDR de la generación estándar
(`validation/`) con el de la generación con un segundo <bos> (`validation_bos2/`, generado con
`lora.py generate --duplicate-bos`), que reproduce la entrada del entrenamiento. La diferencia se calcula por imagen
sobre el promedio de las tres semillas, con el intervalo por bloques de captura de `scripts.eval.results`.
"""

from pathlib import Path

import numpy as np

from scripts.data.splits import chaksu_groups, read_manifest
from scripts.eval.evaluate import answer_fields, errors, load, read_answer, training_medians
from scripts.eval.results import block_interval

SEEDS = (42, 43, 44)


def vcdr_error(seed, folder, constant):
    rows = load(Path(f"experiments/chaksu_json_v1/all-linear_s{seed}/{folder}/pred_real.jsonl"))
    fields = answer_fields(rows)
    return [r["id"] for r in rows], errors(rows, [read_answer(r, fields) for r in rows], "vcdr", constant)


def main():
    constant = training_medians("data/targets/chaksu__json_v1.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    single = [vcdr_error(s, "validation", constant) for s in SEEDS]
    double = [vcdr_error(s, "validation_bos2", constant) for s in SEEDS]
    ids = single[0][0]
    assert all(run[0] == ids for run in single + double), "las corridas no cubren las mismas imágenes"
    lines = ["Efecto del <bos> duplicado en la generación (all-linear, A, validación, vCDR |error| medio)",
             f"{'semilla':>8} | {'un <bos>':>9} {'dos <bos>':>10}"]
    lines += [f"{s:>8} | {a[1].mean():>9.4f} {b[1].mean():>10.4f}" for s, a, b in zip(SEEDS, single, double)]
    mean_single, mean_double = (np.mean([r[1] for r in runs], axis=0) for runs in (single, double))
    delta, low, high = block_interval(mean_double - mean_single, np.array([groups[i] for i in ids]))
    lines.append(f"media de semillas: un <bos> {mean_single.mean():.4f}, dos <bos> {mean_double.mean():.4f}")
    lines.append(f"dos − uno (emparejada, IC 95 % por bloques): {delta:+.4f} [{low:+.4f}, {high:+.4f}]")
    Path("results/audit/chaksu_bos.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
