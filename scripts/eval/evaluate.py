"""Evalúa por contenido las predicciones de un modelo contra sus referencias.

    python -m scripts.eval.evaluate --predictions pred_real.jsonl --targets T.csv --splits S.csv [--surface]

El endpoint es el contenido verificable: cada campo que el modelo enuncia (vcdr,
hcdr, decisión) se compara con la referencia, que viene en las mismas filas del JSONL (columnas de
`scripts.data.targets`). No se compara texto contra texto. Lo que se reporta y por qué:

  Formato     cuántas respuestas son legibles y con qué campos válidos. Una respuesta ilegible es una
              respuesta equivocada: en el error cuenta como si hubiera dicho la constante.
  Números     error absoluto medio, correlación de Spearman (solo sobre las respuestas válidas) y la
              fracción dentro de la tolerancia del vCDR (0.056, la diferencia mediana entre dos expertos,
              `results/audit/chaksu.txt`). Junto al error, el de un predictor CONSTANTE que dijera la
              mediana del ENTRENAMIENTO, que es lo que un modelo puede conocer sin ver las etiquetas de
              evaluación: sin esa cifra un error medio bajo engaña, porque decir siempre lo típico ya
              lo consigue.
  Decisión    sensibilidad, especificidad y exactitud balanceada frente a la decisión por mayoría.
  Colapso     cuántas respuestas distintas hay; un modelo que ignora la imagen dice siempre lo mismo.
  Superficie  con --surface, BLEU, ROUGE-L y BERTScore del texto contra el texto de referencia: se
              reportan siempre al lado y no son el endpoint. BERTScore necesita GPU.

Las funciones de este archivo las reutiliza `scripts.eval.results` para construir la tabla de una rejilla.
Imprime el informe; no escribe archivos.
"""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.eval.fields import extract_prose

VCDR_TOLERANCE = 0.056
DECISIONS = ("NORMAL", "GLAUCOMA SUSPECT")
NUMERIC = ("vcdr", "hcdr")


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_answer(text):
    """El objeto JSON de la respuesta, o None. El modelo base lo envuelve en ```json ... ```. Si el modelo razonó antes de
    responder (`<unused94>thought ... <unused95>`), solo cuenta lo que viene después del marcador de fin de razonamiento."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.rpartition("<unused95>")[2].strip())
    try:
        value = json.loads(cleaned)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_chain(row):
    """Una referencia con razonamiento: la respuesta va en la línea final `Answer: {...}`."""
    return "Answer:" in row["response"]


def is_prose(row):
    return not row["response"].lstrip().startswith("{")


def answer_fields(rows):
    """Los campos que dicen las referencias: los de la receta JSON o los tres de la prosa."""
    assert len({is_prose(r) for r in rows}) == 1, "las filas mezclan JSON y prosa"
    if is_prose(rows[0]):
        return ["vcdr", "hcdr", "decision"]
    fields = list(json.loads(rows[0]["response"]))
    assert all(list(json.loads(r["response"])) == fields for r in rows), "las filas no comparten los mismos campos"
    return fields


def read_answer(row, fields):
    """(¿respuesta legible?, valor válido de cada campo o None). La referencia dice el formato: si es JSON
    se lee JSON, y si es prosa se extraen los campos con `scripts.eval.fields`."""
    if is_chain(row):
        # Sin la línea final la respuesta no es legible; el razonamiento anterior no se lee.
        answer = parse_answer(row["prediction"].rpartition("Answer:")[2]) if "Answer:" in row["prediction"] else None
    else:
        answer = extract_prose(row["prediction"]) if is_prose(row) else parse_answer(row["prediction"])
    readable = answer is not None and (is_chain(row) or not is_prose(row) or any(v is not None for v in answer.values()))
    values = {}
    for field in fields:
        value = None if answer is None else answer.get(field)
        valid = is_number(value) if field in NUMERIC else value in DECISIONS
        values[field] = value if valid else None
    return readable, values


def reference_value(row, field):
    return row[field] if field == "decision" else float(row[field])


def training_medians(targets_path, splits_path, fields):
    """Mediana de cada campo numérico sobre las imágenes de entrenamiento: el valor del predictor constante."""
    with Path(splits_path).open(newline="", encoding="utf-8") as handle:
        split_of = {r["image_id"]: r["split"] for r in csv.DictReader(handle)}
    with Path(targets_path).open(newline="", encoding="utf-8") as handle:
        train = [r for r in csv.DictReader(handle) if split_of[r["image_id"]] == "train"]
    assert train, "no hay imágenes de entrenamiento"
    return {f: float(np.median([float(r[f]) for r in train])) for f in fields if f in NUMERIC}


def errors(rows, reads, field, constant):
    """|valor enunciado − referencia| por imagen; una respuesta sin valor válido cuenta como la constante."""
    return np.array([abs((v[field] if v[field] is not None else constant) - reference_value(r, field))
                     for r, (_, v) in zip(rows, reads)])


def decision_counts(rows, reads):
    """(sospechosos detectados, sospechosos, normales aceptados, normales). Una respuesta sin decisión válida
    no acierta en ninguna de las dos clases: es un fallo tanto en un sospechoso como en un normal."""
    suspect = np.array([r["decision"] == "GLAUCOMA SUSPECT" for r in rows])
    said_suspect = np.array([v["decision"] == "GLAUCOMA SUSPECT" for _, v in reads])
    said_normal = np.array([v["decision"] == "NORMAL" for _, v in reads])
    return (said_suspect & suspect).sum(), suspect.sum(), (said_normal & ~suspect).sum(), (~suspect).sum()


def agreement(a, b):
    return spearmanr(a, b).statistic if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")


def surface_metrics(predictions, references):
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    # BLEU de corpus, no promedio de BLEU por frase: con textos cortos el BLEU-4 de una frase es casi
    # siempre 0 y su promedio no informa.
    bleu = BLEU().corpus_score(predictions, [references]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = statistics.mean(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references))
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
    return {"BLEU-4": bleu, "ROUGE-L": rouge_l, "BERTScore-F1": f1.mean().item()}


def evaluate(rows, fields, constants, title, say):
    reads = [read_answer(r, fields) for r in rows]
    n = len(rows)
    say(f"===== {title}  (n={n}) =====")
    say(f"  respuesta legible: {sum(ok for ok, _ in reads)}/{n}; campos válidos: " + ", ".join(
        f"{f} {sum(v[f] is not None for _, v in reads)}" for f in fields))

    for field in (f for f in fields if f in NUMERIC):
        truth = np.array([reference_value(r, field) for r in rows])
        error = errors(rows, reads, field, constants[field])
        constant_error = np.abs(truth - constants[field])
        valid = [i for i, (_, v) in enumerate(reads) if v[field] is not None]
        rho = agreement([reads[i][1][field] for i in valid], truth[valid])
        line = (f"  {field:<5} |error| medio {error.mean():.4f} (constante de entrenamiento {constants[field]:.2f}: "
                f"{constant_error.mean():.4f}) | Spearman {rho:+.3f} sobre {len(valid)} válidas")
        if field == "vcdr":
            line += (f" | dentro de {VCDR_TOLERANCE}: {np.mean(error <= VCDR_TOLERANCE):.3f} "
                     f"(constante: {np.mean(constant_error <= VCDR_TOLERANCE):.3f})")
        say(line)

    if "decision" in fields:
        tp, positives, tn, negatives = decision_counts(rows, reads)
        say(f"  decisión sensibilidad {tp / positives:.3f} ({tp}/{positives}), especificidad {tn / negatives:.3f} "
            f"({tn}/{negatives}), exactitud balanceada {(tp / positives + tn / negatives) / 2:.3f}")

    answers = Counter(tuple(v[f] for f in fields) for _, v in reads)
    top, share = answers.most_common(1)[0]
    say(f"  colapso {len(answers)} respuestas distintas; la más frecuente ({dict(zip(fields, top))}) es el {share / n:.0%}")

    if "vcdr" in fields:
        say("  por cámara (|error| de vcdr, sospechosos detectados):")
        error = errors(rows, reads, "vcdr", constants["vcdr"])
        for camera in sorted({r["camera"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["camera"] == camera]
            positives = [i for i in idx if rows[i]["decision"] == "GLAUCOMA SUSPECT"]
            found = sum(reads[i][1]["decision"] == "GLAUCOMA SUSPECT" for i in positives)
            say(f"    {camera:<8} n={len(idx):<4} |error| {error[idx].mean():.4f}; sospechosos detectados {found}/{len(positives)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True, help="CSV de scripts.data.targets; de aquí sale la constante de entrenamiento")
    parser.add_argument("--splits", required=True, help="CSV de scripts.data.splits")
    parser.add_argument("--surface", action="store_true", help="añade BLEU, ROUGE-L y BERTScore (necesita GPU)")
    args = parser.parse_args()

    rows = load(args.predictions)
    fields = answer_fields(rows)
    evaluate(rows, fields, training_medians(args.targets, args.splits, fields), Path(args.predictions).name, print)
    if args.surface:
        print("\n===== Superficie (texto contra texto; no es el endpoint) =====")
        for name, value in surface_metrics([r["prediction"] for r in rows], [r["response"] for r in rows]).items():
            print(f"  {name:<14} {value:.4f}")


if __name__ == "__main__":
    main()
