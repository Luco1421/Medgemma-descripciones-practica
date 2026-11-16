"""Tabla normalizada de predicciones: los mismos nombres de columna significan lo mismo en todos los experimentos.

    python -m scripts.eval.normalize RUTA [RUTA ...] --output results/normalized/NOMBRE.csv

RUTA es un `pred_*.jsonl` o una carpeta (se buscan `pred_*.jsonl` dentro, recursivamente). Los `pred_*.jsonl` crudos NO son homogéneos:
en ellos `vcdr`, `hcdr` y `decision` son la referencia en las recetas A, B, cadena de razonamiento, redacción y prompts, pero en la etapa 2
de la configuración C (`experiments/chaksu_C/`) son los campos que se le dieron al modelo y la referencia está en `gold_vcdr`,
`gold_hcdr` y `gold_decision`; y en la validación externa (MM-ODIR-129) no hay referencia numérica. Esta tabla los separa.

Columnas:
  experiment, run, cell, seed, split, control   de dónde sale la fila (`control` es real, shuffled, blank o text)
  id, camera                                    imagen
  gt_vcdr, gt_hcdr, gt_decision                 REFERENCIA: mediana de los cinco expertos de Chákṣu (decimales sin redondear) y decisión por mayoría
  input_vcdr, input_hcdr, input_decision        lo que el prompt le dio al modelo (solo redacción y C), leído del texto del prompt; 2 decimales
  pred_vcdr, pred_hcdr, pred_decision           lo que el modelo dijo, leído con el mismo lector que las métricas (vacío si no se pudo leer)
  pred_readable                                 1 si los tres campos se leyeron
  err_vcdr, err_hcdr                            |pred − gt|, vacío si no se pudo leer (las métricas de `results.py` cuentan eso como la constante de entrenamiento)
  decision_correct                              1 si pred_decision == gt_decision
  ext_label, ext_grade, ext_vcdr_mask           solo MM-ODIR-129: etiqueta Normal/Pathological, grado de CDR anotado y vCDR de las máscaras
  prediction, response                          texto crudo del modelo y texto de referencia
"""

import argparse
import csv
import re
from pathlib import Path

from scripts.eval.evaluate import DECISIONS, is_number, load, parse_answer
from scripts.eval.fields import extract_prose

# carpeta de `experiments/` -> cómo se lee lo que el modelo escribió
FORMAT = {"chaksu_json_v1": "json", "chaksu_prompts_A": "json", "chaksu_prose_v1": "prose", "chaksu_verbalize_v1": "prose",
          "chaksu_verbalize_img_v1": "prose", "chaksu_C": "prose", "chaksu_cot_v1": "chain"}
SPLITS = ("validation", "test", "external", "selection")
INPUT = re.compile(r"vertical cup-to-disc ratio of ([\d.]+) and a horizontal cup-to-disc ratio of ([\d.]+), and classified the optic disc as (NORMAL|GLAUCOMA SUSPECT)")
COLUMNS = ["experiment", "run", "cell", "seed", "split", "control", "id", "camera", "gt_vcdr", "gt_hcdr", "gt_decision", "input_vcdr", "input_hcdr",
           "input_decision", "pred_vcdr", "pred_hcdr", "pred_decision", "pred_readable", "err_vcdr", "err_hcdr", "decision_correct",
           "ext_label", "ext_grade", "ext_vcdr_mask", "prediction", "response"]


def read_prediction(kind, text):
    if kind == "json":
        answer = parse_answer(text)
    elif kind == "chain":
        answer = parse_answer(text.rpartition("Answer:")[2]) if "Answer:" in text else None
    else:
        answer = extract_prose(text)
    answer = answer or {}
    return {f: (answer.get(f) if is_number(answer.get(f)) else None) for f in ("vcdr", "hcdr")} | {
        "decision": answer.get("decision") if answer.get("decision") in DECISIONS else None}


def rows_of(path):
    parts = path.parts
    assert parts[0] == "experiments" and parts[1] in FORMAT, f"{path}: no es una carpeta de experiments/ conocida ({sorted(FORMAT)})"
    experiment, run = parts[1], parts[2]
    split = parts[3] if parts[3] in SPLITS else ""
    seed = re.search(r"_s(\d+)", run)
    control = path.stem.removeprefix("pred_")
    for r in load(path):
        if "gold_vcdr" in r:
            gt = (r["gold_vcdr"], r["gold_hcdr"], r["gold_decision"])
        elif "vcdr" in r:
            gt = (r["vcdr"], r["hcdr"], r["decision"])
        else:
            gt = ("", "", "")
        given = INPUT.search(r["prompt"])
        pred = read_prediction(FORMAT[experiment], r["prediction"])
        error = {f: round(abs(pred[f] - float(g)), 6) if pred[f] is not None and g != "" else "" for f, g in zip(("vcdr", "hcdr"), gt)}
        yield {"experiment": experiment, "run": run, "cell": re.sub(r"_s\d+", "", run), "seed": seed.group(1) if seed else "", "split": split,
               "control": control, "id": r["id"], "camera": r.get("camera", ""),
               "gt_vcdr": gt[0], "gt_hcdr": gt[1], "gt_decision": gt[2],
               "input_vcdr": given.group(1) if given else "", "input_hcdr": given.group(2) if given else "", "input_decision": given.group(3) if given else "",
               "pred_vcdr": "" if pred["vcdr"] is None else pred["vcdr"], "pred_hcdr": "" if pred["hcdr"] is None else pred["hcdr"],
               "pred_decision": pred["decision"] or "", "pred_readable": int(all(v is not None for v in pred.values())),
               "err_vcdr": error["vcdr"], "err_hcdr": error["hcdr"],
               "decision_correct": int(pred["decision"] == gt[2]) if pred["decision"] and gt[2] else "",
               "ext_label": r.get("label", ""), "ext_grade": r.get("cup_to_disc_grade", ""), "ext_vcdr_mask": r.get("vcdr_mask_bbox", ""),
               "prediction": r["prediction"], "response": r["response"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="pred_*.jsonl o carpetas")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = sorted({f for p in map(Path, args.paths) for f in ([p] if p.is_file() else p.rglob("pred_*.jsonl"))})
    assert files, "ningún pred_*.jsonl en las rutas dadas"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for f in files:
            for row in rows_of(f):
                writer.writerow(row)
                n += 1
    print(f"{out}: {n} filas de {len(files)} archivos")


if __name__ == "__main__":
    main()
