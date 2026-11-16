"""Extractor de campos desde prosa: convierte "vertical C:D ratio of 0.63 ..." en {"vcdr": 0.63, ...}.

Es un extractor y no un juez: reglas fijas y léxico controlado, sin ningún modelo
que decida. Lo usan dos sitios, y por eso vive aparte: `targets.py` comprueba que todo texto que
genera devuelve los campos de los que salió, y `evaluate.py` lo aplica a lo que escribe el modelo.

Qué reconoce:
  vcdr, hcdr   un decimal dentro de una ventana de 40 caracteres tras la palabra "vertical" u "horizontal"
               (entera: "vertically" no cuenta), o justo antes de ella ("0.35 vertical", "0.55 in the vertical
               direction", "0.45 vertically"). Qué orden usa un texto lo decide su primera palabra de eje: si un
               decimal la precede, el texto da los números antes de la palabra; si no, después. Así
               "0.35 vertical, 0.30 horizontal" no lee 0.30 como vertical, y "vertical 0.45 horizontal 0.40"
               no lee 0.45 como horizontal.
  decision     "suspect" o "suspicious" => GLAUCOMA SUSPECT; "normal" => NORMAL. La negación se trata
               de forma explícita (el fallo documentado de CheXbert): "not a glaucoma suspect" es NORMAL
               y "not normal" es GLAUCOMA SUSPECT; también "Glaucoma suspect: no". Un texto que dice las dos cosas es ambiguo y no se
               extrae.
Lo que no reconoce devuelve None, y `evaluate.py` lo cuenta como respuesta no válida en vez de
descartarlo. Su tasa de error sobre lo que escribe el modelo no se conoce hasta medirla contra una
lectura manual de una muestra.
"""

import re

NUMBER = r"(\d\.\d+)"
AXIS = re.compile(r"\b(?:vertical|horizontal)(?:ly)?\b", re.IGNORECASE)
NUMBER_BEFORE_WORD = re.compile(NUMBER + r"\s*(?:(?:in|on|along)\s+the\s+)?$", re.IGNORECASE)
AFTER = {"vcdr": re.compile(r"\bvertical\b[^0-9]{0,40}?" + NUMBER, re.IGNORECASE),
         "hcdr": re.compile(r"\bhorizontal\b[^0-9]{0,40}?" + NUMBER, re.IGNORECASE)}
BEFORE = {"vcdr": re.compile(NUMBER + r"\s*(?:(?:in|on|along)\s+the\s+)?vertical(?:ly)?\b", re.IGNORECASE),
          "hcdr": re.compile(NUMBER + r"\s*(?:(?:in|on|along)\s+the\s+)?horizontal(?:ly)?\b", re.IGNORECASE)}
SUSPECT = re.compile(r"\b(?:suspect|suspicious)\b", re.IGNORECASE)
NEGATED_SUSPECT = re.compile(r"\b(?:not|no|non)\b[^.;,]{0,30}\b(?:suspect|suspicious)\b|\bsuspect\W{0,3}(?:no|not|none|false)\b",
                             re.IGNORECASE)
NORMAL = re.compile(r"\bnormal\b", re.IGNORECASE)
NEGATED_NORMAL = re.compile(r"\b(?:not|abnormal)\b[^.;,]{0,10}\bnormal\b|\babnormal\b", re.IGNORECASE)


def number(text, field):
    first = AXIS.search(text)
    numbers_first = bool(first and NUMBER_BEFORE_WORD.search(text[:first.start()]))
    for pattern in ((BEFORE[field], AFTER[field]) if numbers_first else (AFTER[field], BEFORE[field])):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def decision(text):
    suspect = bool(SUSPECT.search(text)) and not NEGATED_SUSPECT.search(text)
    normal = bool(NORMAL.search(text)) and not NEGATED_NORMAL.search(text)
    # "not a glaucoma suspect" (sospecha negada) o "normal" afirmado leen NORMAL; "not normal" o la
    # sospecha afirmada leen GLAUCOMA SUSPECT. Si leen las dos cosas, el texto es ambiguo.
    says_suspect = suspect or bool(NEGATED_NORMAL.search(text))
    says_normal = normal or bool(NEGATED_SUSPECT.search(text))
    if says_suspect == says_normal:
        return None
    return "GLAUCOMA SUSPECT" if says_suspect else "NORMAL"


def extract_prose(text):
    """{"vcdr", "hcdr", "decision"} leídos del texto; None en lo que no se pudo leer."""
    return {"vcdr": number(text, "vcdr"), "hcdr": number(text, "hcdr"),
            "decision": decision(text)}
