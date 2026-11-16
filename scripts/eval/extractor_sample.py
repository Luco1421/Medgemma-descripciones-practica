"""Muestra aleatoria para medir el error de `fields.py` con un lector humano.

    python -m scripts.eval.extractor_sample draw     # escribe la hoja ciega a rellenar
    python -m scripts.eval.extractor_sample score    # compara la hoja rellenada con el extractor

`draw` toma, por fuente, textos DISTINTOS elegidos al azar (semilla fija) entre las respuestas de validación en prosa y escribe
`results/audit/extractor_muestra_aleatoria.csv` con las columnas `source`, `id`, `text` y tres columnas vacías: `manual_vcdr`,
`manual_hcdr`, `manual_decision`. No lleva lo que lee el extractor, para que el lector no lo vea. Instrucciones para el lector: escribir el
número que el TEXTO da para el cociente vertical y el horizontal (vacío si no lo da), y la decisión que el texto expresa (NORMAL o
GLAUCOMA SUSPECT; vacío si no se puede decir o es contradictoria). Sin consultar el extractor ni a otro lector.

`score` cuenta, por campo, los textos en que el extractor y el lector difieren (un None contra un valor cuenta) y da la tasa con su intervalo
de Wilson del 95 %. La unidad es el texto distinto y la muestra es estratificada: la tasa no es la de un archivo concreto.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from scripts.eval.fields import extract_prose

OUT = Path("results/audit/extractor_muestra_aleatoria.csv")
SEED = 0
# fuente -> (archivo de predicciones, cuántos textos distintos)
SOURCES = {
    "prose_v1 base, imagen correcta": ("experiments/chaksu_prose_v1/base/validation/pred_real.jsonl", 40),
    "prose_v1 base, imagen barajada": ("experiments/chaksu_prose_v1/base/validation/pred_shuffled.jsonl", 15),
    "verbalize_img_v1 base, imagen correcta": ("experiments/chaksu_verbalize_img_v1/base/validation/pred_real.jsonl", 30),
    "prose_v1 all-linear s42, imagen correcta": ("experiments/chaksu_prose_v1/all-linear_s42/validation/pred_real.jsonl", 20),
}


def draw():
    rng = random.Random(SEED)
    rows = []
    for source, (path, n) in SOURCES.items():
        predictions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        distinct = {}
        for p in predictions:
            distinct.setdefault(p["prediction"], p["id"])
        texts = sorted(distinct)
        assert len(texts) >= n, f"{source}: solo {len(texts)} textos distintos"
        rows += [{"source": source, "id": distinct[t], "text": t, "manual_vcdr": "", "manual_hcdr": "", "manual_decision": ""}
                 for t in rng.sample(texts, n)]
    rng.shuffle(rows)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{OUT}: {len(rows)} textos; rellenar manual_vcdr, manual_hcdr y manual_decision sin mirar el extractor")


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return centre - half, centre + half


def score():
    sheet = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    assert (sheet[["manual_vcdr", "manual_hcdr", "manual_decision"]] != "").any(axis=None), "la hoja está sin rellenar"
    n = len(sheet)
    for field in ("vcdr", "hcdr", "decision"):
        wrong = []
        for row in sheet.itertuples():
            extracted = extract_prose(row.text)[field]
            manual = getattr(row, f"manual_{field}")
            expected = None if manual == "" else (manual if field == "decision" else float(manual))
            if extracted != expected:
                wrong.append((row.source, extracted, expected, row.text[:140]))
        low, high = wilson(len(wrong), n)
        print(f"{field}: {len(wrong)}/{n} difieren ({len(wrong) / n:.1%}, IC95 [{low:.1%}, {high:.1%}])")
        for w in wrong:
            print("   ", w)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["draw", "score"])
    args = parser.parse_args()
    {"draw": draw, "score": score}[args.command]()


if __name__ == "__main__":
    main()
