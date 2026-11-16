"""Análisis del texto de la etapa de redacción: ¿repite, es fiel a los campos, inventa hallazgos?

    python -m scripts.eval.text_report experiments/chaksu_verbalize_v1/base/validation/pred_text.jsonl [...]

Cada fila trae los campos que se le dieron al modelo (`vcdr`, `hcdr`, `decision`), su respuesta
(`prediction`) y la frase de referencia (`response`). Por archivo imprime:
  fidelidad    fracción de respuestas cuyo vCDR, hCDR y decisión leídos por `fields.py` coinciden con los
               campos de entrada (extractor de reglas: su propio error no está medido).
  variedad     respuestas distintas y plantillas distintas (números y decisión sustituidos), con el peso de
               la más frecuente. Una plantilla dominante es el riesgo de memorizar una frase.
  referencia   fracción idéntica a la frase de referencia y parecido medio de palabras con ella (Jaccard).
  hallazgos    fracción que menciona algo que no estaba en los campos (lista de términos clínicos fija).
"""

import json
import re
import sys
from collections import Counter

from scripts.eval.fields import extract_prose

EXTRA_FINDINGS = ("rim", "pallor", "notch", "hemorrhage", "haemorrhage", "rnfl", "nerve fiber", "vessel", "thin",
                  "atrophy", "asymmetr", "disc size", "large disc", "small disc", "peripapillary", "drusen", "edema",
                  "bayonet", "nasal", "temporal", "superior", "inferior", "isnt", "progressi", "intraocular")


def template(text):
    text = re.sub(r"\d\.\d+", "#", text)
    return re.sub(r"(?i)glaucoma suspect|normal", "D", text)


def words(text):
    return set(re.findall(r"[a-z]+", re.sub(r"\d\.\d+", " ", text.lower())))


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        read = [extract_prose(r["prediction"]) for r in rows]
        n = len(rows)
        ok = {f: sum(r[f] is not None and (abs(r[f] - float(row[f])) < 0.0051 if f != "decision" else r[f] == row[f])
                     for r, row in zip(read, rows)) for f in ("vcdr", "hcdr", "decision")}
        all_ok = sum(all(r[f] is not None and (abs(r[f] - float(row[f])) < 0.0051 if f != "decision" else r[f] == row[f])
                         for f in ("vcdr", "hcdr", "decision")) for r, row in zip(read, rows))
        templates = Counter(template(r["prediction"]) for r in rows)
        top, top_n = templates.most_common(1)[0]
        jaccard = [len(words(r["prediction"]) & words(r["response"])) / len(words(r["prediction"]) | words(r["response"]))
                   for r in rows]
        extra = [r for r in rows if any(t in r["prediction"].lower() for t in EXTRA_FINDINGS)]
        print(f"===== {path}  (n={n})")
        print(f"  fidelidad: vcdr {ok['vcdr']}/{n}, hcdr {ok['hcdr']}/{n}, decisión {ok['decision']}/{n}, las tres {all_ok}/{n}")
        print(f"  variedad: {len({r['prediction'] for r in rows})} respuestas distintas, {len(templates)} plantillas; "
              f"la más frecuente pesa {top_n / n:.0%}: «{top[:110]}»")
        print(f"  referencia: idéntica {sum(r['prediction'] == r['response'] for r in rows)}/{n}, "
              f"Jaccard de palabras medio {sum(jaccard) / n:.2f}")
        print(f"  hallazgos fuera de los campos: {len(extra)}/{n}; longitud media {sum(len(r['prediction']) for r in rows) / n:.0f} caracteres")
        for r in extra[:3]:
            print(f"    · {r['prediction'][:200]}")


if __name__ == "__main__":
    main()
