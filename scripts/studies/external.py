"""Entradas de la validación externa en MM-ODIR-129: las 129 imágenes son una sola partición, `external`.

    python -m scripts.studies.external prepare     # targets y partición de las 129 imágenes
    python -m scripts.studies.external stage2      # entradas de la etapa 2 de la configuración C a partir de las predicciones de la etapa 1

Escribe `data/targets/mmodir129__external.csv` (una fila por imagen: imagen, transcripción del oftalmólogo como `text`, etiqueta, anotador,
grado de CDR anotado y vCDR de las máscaras cuando existen) y `data/splits/mmodir129_externa.csv` (todas en la partición `external`).
No elige referencia: qué se compara con qué (etiqueta, cociente escrito en la transcripción o máscaras) se decide al evaluar, con las
predicciones ya guardadas. Con estos archivos y los prompts de Chákṣu, `lora.py generate --split external` produce las predicciones de
cualquier adaptador o del modelo base; las listas de corridas del Test se generan con `scripts.analysis.finalists`.

`stage2` toma las predicciones externas del adaptador `all-linear` de A (tres semillas y los tres controles de imagen) y escribe, en
`data/stage2/`, una copia de los targets en la que `vcdr`, `hcdr` y `decision` son los campos PREDICHOS (los que entran en el prompt de la
etapa 2) y, en `runs/mmodir129_C_etapa2.txt`, la lista de generaciones de la etapa 2 (con las particiones de MM-ODIR-129).
Aborta si alguna predicción de la etapa 1 no trae los tres campos válidos: la etapa 2 no tendría entrada.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

from scripts.eval.normalize import read_prediction

STAGE1 = Path("experiments/chaksu_json_v1")
STAGE2 = Path("experiments/chaksu_C")
LORA = "experiments/chaksu_verbalize_v1/block_language_s{seed}"
SEEDS, MODES = (42, 43, 44), ("real", "shuffled", "blank")


def prepare():
    manifest = pd.read_csv("data/manifests/mmodir129.csv", dtype=str, keep_default_na=False)
    audit = pd.read_csv("results/audit/mmodir129.csv").set_index("image_id")
    assert manifest.image_id.is_unique and set(manifest.image_id) == set(audit.index), "el manifest y la auditoría no cubren las mismas imágenes"
    rows = [{"image_id": m.image_id, "image": f"mmodir129/raw/{m.image_path}", "text": m.transcription, "label": m.label,
             "doctor_name": m.doctor_name, "cup_to_disc_grade": m.cup_to_disc_ratio,
             "vcdr_mask_bbox": "" if pd.isna(audit.loc[m.image_id, "vcdr_bbox"]) else f"{audit.loc[m.image_id, 'vcdr_bbox']:.4f}",
             "patient_id": m.patient_id} for m in manifest.itertuples()]
    assert all((Path("..") / "datasets" / r["image"]).exists() for r in rows), "falta alguna imagen en ../datasets"
    for path, table in (("data/targets/mmodir129__external.csv", rows),
                        ("data/splits/mmodir129_externa.csv", [{"image_id": r["image_id"], "patient_id": r["patient_id"], "split": "external"} for r in rows])):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"{path}: {len(table)} filas")


def stage2():
    from scripts.eval.evaluate import load

    targets = list(csv.DictReader(Path("data/targets/mmodir129__external.csv").open(encoding="utf-8")))
    lists = []
    for seed in SEEDS:
        for mode in MODES:
            predicted = {}
            for row in load(STAGE1 / f"all-linear_s{seed}" / "external" / f"pred_{mode}.jsonl"):
                fields = read_prediction("json", row["prediction"])
                assert all(v is not None for v in fields.values()), f"s{seed} {mode} {row['id']}: la etapa 1 no dio los tres campos"
                predicted[row["id"]] = fields
            assert set(predicted) == {t["image_id"] for t in targets}
            rows = [{**t, "image": "", "vcdr": f"{predicted[t['image_id']]['vcdr']:.2f}", "hcdr": f"{predicted[t['image_id']]['hcdr']:.2f}",
                     "decision": predicted[t["image_id"]]["decision"]} for t in targets]
            path = Path("data/stage2") / f"mmodir129_C_all-linear_s{seed}_{mode}.csv"
            path.parent.mkdir(exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for variant, adapter in (("base", "base"), ("lora", LORA.format(seed=seed))):
                lists.append(f"{path} {adapter} {STAGE2}/ext_{variant}_all-linear_s{seed}_{mode}")
    out = Path("runs/mmodir129_C_etapa2.txt")
    out.write_text("# Etapa 2 de C sobre MM-ODIR-129: TARGETS ADAPTER SALIDA; lanzar con SPLITS=data/splits/mmodir129_externa.csv SPLIT=external\n" + "\n".join(lists) + "\n",
                   encoding="utf-8")
    print(f"{out}: {len(lists)} generaciones")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "stage2"])
    {"prepare": prepare, "stage2": stage2}[parser.parse_args().command]()
