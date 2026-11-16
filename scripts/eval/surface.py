"""Métricas de superficie de la tarea de prosa (B) y su meta-evaluación.

    python -m scripts.eval.surface runs/chaksu_prose_v1_grid.txt      (necesita GPU)

Parte 1, rejilla: para cada corrida de la lista (y el modelo base) compara el texto con la frase de referencia
con BLEU-4, ROUGE-L, BERTScore-F1 (roberta-large, capa 17, y Bio_ClinicalBERT, capa 9) y coseno de embeddings
de frase (all-mpnet-base-v2 y MedCPT-Query-Encoder). Escribe `<root>/chaksu_<receta>/surface.csv` (media de las
semillas por celda) y calcula la correlación de cada métrica con el error del vCDR de `results.csv`.

Parte 2, meta-evaluación: la referencia es UNA de las 144 frases posibles con el mismo contenido, así que una
métrica de superficie compara también la forma. Con la gramática de `targets.py` se construyen, para cada
referencia de validación, textos en los que se sabe qué cambia: otra formulación con el contenido correcto,
la misma frase con el vCDR o la decisión equivocados, otra formulación con el vCDR equivocado y la frase de otra
imagen. Cada variante se lee con `fields.py` y se comprueba que dice lo que se quiso (si no, aborta). Se informa
la media de cada métrica por condición y la fracción de imágenes en las que la métrica puntúa MÁS al texto con
el contenido equivocado y la misma forma que al texto con el contenido correcto y otra forma.

Parte 3 (`--stage2`): las mismas métricas sobre el texto de B (`all-linear`) y de la etapa 2 de C (solo texto y con imagen) frente a la
frase de referencia, tres semillas, imagen real. Para cada brazo se informa la media de cada métrica y el Spearman, por imagen, entre la métrica
y el error del vCDR del texto: si la métrica midiera contenido, ese Spearman sería claramente negativo.

Elecciones de este instrumento (no de los modelos): el BERTScore de Bio_ClinicalBERT usa la capa 9, la que
`bert_score` recomienda para BERT-base, porque no trae una capa propia para ese modelo; el coseno de
all-mpnet-base-v2 usa promedio de tokens y el de MedCPT el token [CLS], como indican sus tarjetas; textos
truncados a 128 tokens (ninguno de estos textos llega).
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.eval.evaluate import load, training_medians
from scripts.eval.fields import extract_prose
from scripts.eval.results import GRIDS, parse_runs
from scripts.data.targets import CHAKSU_DECISION, chaksu_prose_v1

SHIFTS = (0.10, 0.20)
EMBEDDERS = {"SBERT-mpnet": ("sentence-transformers/all-mpnet-base-v2", "mean"),
             "MedCPT": ("ncbi/MedCPT-Query-Encoder", "cls")}
BERTS = {"BERTScore-roberta": dict(model_type="roberta-large"),
         "BERTScore-ClinicalBERT": dict(model_type="emilyalsentzer/Bio_ClinicalBERT", num_layers=9)}


class Similarities:
    """Parecido de cada texto con su referencia, por cada métrica; una fila de la salida por texto."""

    def __init__(self):
        import torch
        from bert_score import BERTScorer
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.berts = {name: BERTScorer(batch_size=64, device="cuda", **kwargs) for name, kwargs in BERTS.items()}
        self.embedders = {name: (AutoTokenizer.from_pretrained(model), AutoModel.from_pretrained(model).cuda().eval(), pool)
                          for name, (model, pool) in EMBEDDERS.items()}

    def embed(self, name, texts):
        tokenizer, model, pool = self.embedders[name]
        out = []
        with self.torch.no_grad():
            for first in range(0, len(texts), 64):
                batch = tokenizer(texts[first:first + 64], padding=True, truncation=True, max_length=128,
                                  return_tensors="pt").to("cuda")
                hidden = model(**batch).last_hidden_state
                if pool == "cls":
                    vector = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vector = (hidden * mask).sum(1) / mask.sum(1)
                out.append(self.torch.nn.functional.normalize(vector, dim=-1))
        return self.torch.cat(out)

    def __call__(self, predictions, references):
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        bleu = BLEU(effective_order=True)
        scores = {"BLEU-4": np.array([bleu.sentence_score(p, [r]).score for p, r in zip(predictions, references)]),
                  "ROUGE-L": np.array([rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)])}
        for name, scorer in self.berts.items():
            scores[name] = scorer.score(predictions, references)[2].numpy()
        for name in self.embedders:
            scores[name] = (self.embed(name, predictions) * self.embed(name, references)).sum(-1).cpu().numpy()
        return scores


def counterfactuals(rows):
    """Para cada fila de referencia, un dict condición -> texto. Cada texto se comprueba con `fields.py`."""
    others = [r["response"] for r in rows]
    result = {}
    for i, row in enumerate(rows):
        reference = {"vcdr": float(row["vcdr"]), "hcdr": float(row["hcdr"]), "decision": row["decision"]}
        text = row["response"]
        rng = random.Random(f"surface:{row['id']}")
        expect = {"vcdr": round(reference["vcdr"], 2), "hcdr": round(reference["hcdr"], 2), "decision": reference["decision"]}

        def other_form(fields):
            while True:
                candidate = chaksu_prose_v1(fields, rng)
                if candidate != text:
                    return candidate

        variants = {"misma frase": (text, expect),
                    "otra formulación, contenido correcto": (other_form(reference), expect)}
        for shift in SHIFTS:
            wrong = round(expect["vcdr"] + (shift if expect["vcdr"] <= 0.5 else -shift), 2)
            changed = {**expect, "vcdr": wrong}
            numbers = re.findall(r"\d\.\d+", text)
            assert float(numbers[0]) == expect["vcdr"], f"{row['id']}: el vCDR no es el primer decimal de {text!r}"
            variants[f"misma frase, vCDR ±{shift:.2f}"] = (text.replace(numbers[0], f"{wrong:.2f}", 1), changed)
            variants[f"otra formulación, vCDR ±{shift:.2f}"] = (other_form({**reference, "vcdr": wrong}), changed)
        flipped = "GLAUCOMA SUSPECT" if reference["decision"] == "NORMAL" else "NORMAL"
        phrase = max((p for p in CHAKSU_DECISION[reference["decision"]] if p in text.lower()), key=len)
        index = CHAKSU_DECISION[reference["decision"]].index(phrase)
        variants["misma frase, decisión contraria"] = (
            re.sub(re.escape(phrase), CHAKSU_DECISION[flipped][index], text, count=1, flags=re.IGNORECASE),
            {**expect, "decision": flipped})
        variants["frase de otra imagen"] = (others[(i + 1 + rng.randrange(len(rows) - 1)) % len(rows)], None)
        for condition, (candidate, fields) in variants.items():
            if fields is not None:
                assert extract_prose(candidate) == fields, f"{row['id']} {condition}: {candidate!r} no dice {fields}"
            result.setdefault(condition, []).append(candidate)
    return result


def meta_evaluation(similarity, rows, say):
    references = [r["response"] for r in rows]
    scores = {c: similarity(texts, references) for c, texts in counterfactuals(rows).items()}
    metrics = list(next(iter(scores.values())))
    say("META-EVALUACIÓN (171 referencias de validación; media por condición; todos los textos leídos con fields.py)")
    say(f"{'condición':<42}" + "".join(f"{m:>24}" for m in metrics))
    for condition, by_metric in scores.items():
        say(f"{condition:<42}" + "".join(f"{by_metric[m].mean():>24.3f}" for m in metrics))
    say("\nFracción de imágenes en que la métrica puntúa MÁS al texto con el contenido equivocado y la misma forma")
    say("que al texto con el contenido correcto y otra formulación (una métrica que mide contenido daría 0):")
    right = "otra formulación, contenido correcto"
    for condition in scores:
        if condition.startswith("misma frase,"):
            say(f"  {condition:<40}" + "".join(f"{(scores[condition][m] > scores[right][m]).mean():>24.2f}" for m in metrics))
    return scores


def grid(similarity, runs_path, root, say):
    dataset, recipe, cells = parse_runs(runs_path)
    folder = Path(root) / f"{dataset}_{recipe}"
    seeds = next(iter(cells.values()))
    table = []
    for cell in ["base", *cells]:
        folders = [folder / "base"] if cell == "base" else [folder / f"{cell}_s{s}" for s in seeds]
        per_run = []
        for run in folders:
            rows = load(run / "validation" / "pred_real.jsonl")
            scores = similarity([r["prediction"] for r in rows], [r["response"] for r in rows])
            per_run.append({m: v.mean() for m, v in scores.items()})
        table.append({"cell": cell, **{m: np.mean([r[m] for r in per_run]) for m in per_run[0]}})
    surface = pd.DataFrame(table)
    table_dir = GRIDS / f"{dataset}_{recipe}"
    surface.round(4).to_csv(table_dir / "surface.csv", index=False)
    results = pd.read_csv(table_dir / "results.csv").set_index("cell")
    error = results.loc[surface.cell, "vcdr_error_mean"].to_numpy()
    say(f"\nREJILLA {recipe}: {len(surface)} filas (base y {len(surface) - 1} celdas, media de {len(seeds)} semillas); "
        f"escrita en {table_dir / 'surface.csv'}")
    say("Spearman entre la métrica de cada celda y su error del vCDR (más negativo = la métrica ordena como el error):")
    for metric in surface.columns[1:]:
        say(f"  {metric:<24}{spearmanr(surface[metric], error).statistic:+.3f}   "
            f"sin el base {spearmanr(surface[metric][1:], error[1:]).statistic:+.3f}")
    return surface


# Brazo -> (archivo de predicciones con {s} = semilla, campo con el vCDR de referencia). `response` es la frase de referencia en todos.
STAGE2_ARMS = {
    "B, all-linear": ("experiments/chaksu_prose_v1/all-linear_s{s}/validation/pred_real.jsonl", "vcdr"),
    "C, etapa 2 base, solo texto": ("experiments/chaksu_C/base_all-linear_s{s}_real/pred_text.jsonl", "gold_vcdr"),
    "C, etapa 2 LoRA, solo texto": ("experiments/chaksu_C/lora_all-linear_s{s}_real/pred_text.jsonl", "gold_vcdr"),
    "C, etapa 2 base, con imagen": ("experiments/chaksu_C/img_base_all-linear_s{s}_real/pred_real.jsonl", "gold_vcdr"),
    "C, etapa 2 LoRA, con imagen": ("experiments/chaksu_C/img_lora_all-linear_s{s}_real/pred_real.jsonl", "gold_vcdr"),
}


def stage2(similarity, say):
    constant = training_medians("data/targets/chaksu__json_v1.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
    say("MÉTRICAS DE SUPERFICIE SOBRE B Y SOBRE LA ETAPA 2 DE C (171 imágenes de validación, media de tres semillas, imagen real)")
    say("Spearman por imagen entre la métrica (media de las semillas) y el error del vCDR del texto (media de las semillas); más negativo = la métrica baja cuando el vCDR falla\n")
    for arm, (pattern, key) in STAGE2_ARMS.items():
        scores, errors = [], []
        for seed in (42, 43, 44):
            rows = load(Path(pattern.format(s=seed)))
            scores.append(similarity([r["prediction"] for r in rows], [r["response"] for r in rows]))
            read = [extract_prose(r["prediction"])["vcdr"] for r in rows]
            errors.append(np.array([abs((v if v is not None else constant) - float(r[key])) for v, r in zip(read, rows)]))
        error = np.mean(errors, axis=0)
        say(f"{arm}: error del vCDR {error.mean():.4f}")
        for metric in scores[0]:
            per_image = np.mean([s[metric] for s in scores], axis=0)
            say(f"  {metric:<24}{per_image.mean():>8.3f}   Spearman con el error por imagen {spearmanr(per_image, error).statistic:+.3f}")
        say()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="?", help="lista de corridas de la rejilla de prosa (no se usa con --stage2)")
    parser.add_argument("--stage2", action="store_true", help="solo la parte 3, sobre B y la etapa 2 de C")
    parser.add_argument("--root", default="experiments", help="carpeta con las corridas; las tablas se escriben en results/grids/")
    args = parser.parse_args()
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    similarity = Similarities()
    if args.stage2:
        stage2(similarity, say)
        Path("results/audit/chaksu_superficie_C.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    assert args.runs, "falta la lista de corridas de la rejilla (o --stage2)"
    _, recipe, _ = parse_runs(args.runs)
    base_rows = load(Path(args.root) / f"chaksu_{recipe}" / "base" / "validation" / "pred_real.jsonl")
    meta_evaluation(similarity, base_rows, say)
    surface = grid(similarity, args.runs, args.root, say)
    say("\nCeldas de referencia:\n" + surface[surface.cell.isin(["base", "all-linear"])].round(3).to_string(index=False))
    Path("results/audit/chaksu_superficie.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
