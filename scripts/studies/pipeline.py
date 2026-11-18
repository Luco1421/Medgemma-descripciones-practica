"""Configuración C: etapa 1 = adapter de A (imagen → campos), etapa 2 = redacción solo texto.

    python -m scripts.studies.pipeline prepare [--cell all-linear] [--seeds 42 43 44]
    python -m scripts.studies.pipeline report  [--cell all-linear] [--seeds 42 43 44]

`prepare` lee las predicciones de validación de la etapa 1 (`experiments/chaksu_json_v1/<celda>_s<semilla>/validation/pred_<control>.jsonl`)
y, para cada semilla y control de imagen, escribe en `data/stage2/` una copia de los targets de `verbalize_v1` en la que los campos de
las imágenes de validación (`vcdr`, `hcdr`, `decision`: lo que entra en el prompt de la etapa 2) son los PREDICHOS por la etapa 1; los de
referencia quedan en `gold_vcdr`, `gold_hcdr`, `gold_decision`. Las demás particiones no se usan. Escribe también, en `runs/`, la lista de
generaciones de la etapa 2 (una por línea: TARGETS ADAPTER SALIDA) que cada trabajo de generación recorre. La prueba de estrés usa los campos
de referencia con la decisión invertida. Una predicción de la etapa 1 sin los tres campos válidos aborta: la etapa 2 no tendría entrada.

`prepare --recipe verbalize_img_v1` prepara la variante con la imagen real en la etapa 2 (solo imagen real, sin prueba de estrés) y `report_img` la compara con la
etapa 2 solo texto.

`report` lee lo que escribió la etapa 2 en `experiments/chaksu_C/` y aplica los seis criterios: error del vCDR del texto final frente a
la referencia (respuesta no legible = constante de entrenamiento) y su diferencia emparejada con A y con B, fidelidad a los campos de
entrada, hallazgos fuera de los campos, ganancias de los controles de imagen, copia de la decisión invertida y sensibilidad/especificidad.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.eval.evaluate import answer_fields, load, read_answer, training_medians
from scripts.eval.fields import extract_prose
from scripts.eval.results import MODES, block_interval, run_metrics
from scripts.data.splits import chaksu_groups, read_manifest
from scripts.eval.text_report import EXTRA_FINDINGS

STAGE1 = Path("experiments/chaksu_json_v1")
STAGE_B = Path("experiments/chaksu_prose_v1")
STAGE2 = Path("experiments/chaksu_C")
INPUTS = Path("data/stage2")
SPLITS = "data/splits/chaksu_seed42.csv"
# Etapa 2 solo texto o con la imagen real además de los campos predichos. Con imagen no hay controles ni prueba de estrés:
# la pregunta es si ver la imagen cambia el texto, no si la etapa 1 usa la imagen.
RECIPES = {
    "verbalize_v1": {"targets": "data/targets/chaksu__verbalize_v1.csv", "prefix": "", "modes": MODES,
                     "lora": "experiments/chaksu_verbalize_v1/block_language_s{seed}"},
    "verbalize_img_v1": {"targets": "data/targets/chaksu__verbalize_img_v1.csv", "prefix": "img_", "modes": ("real",),
                         "lora": "experiments/chaksu_verbalize_img_v1/block_language_s{seed}"},
}
TOLERANCE = 0.0051
RELEVANT = 0.004


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    cfg = RECIPES[args.recipe]
    targets = read_rows(cfg["targets"])
    validation = {r["image_id"] for r in read_rows(SPLITS) if r["split"] == "validation"}
    gold = [{**t, "gold_vcdr": t["vcdr"], "gold_hcdr": t["hcdr"], "gold_decision": t["decision"]} for t in targets]
    lists = {seed: [] for seed in args.seeds}

    def with_fields(fields_of):
        return [{**g, **fields_of[g["image_id"]]} if g["image_id"] in fields_of else g for g in gold]

    for seed in args.seeds:
        for mode in cfg["modes"]:
            predictions = load(STAGE1 / f"{args.cell}_s{seed}" / "validation" / f"pred_{mode}.jsonl")
            fields = answer_fields(predictions)
            read = {}
            for row in predictions:
                values = read_answer(row, fields)[1]
                assert all(v is not None for v in values.values()), f"{args.cell} s{seed} {mode} {row['id']}: la etapa 1 no dio los tres campos"
                read[row["id"]] = {k: str(v) for k, v in values.items()}
            assert set(read) == validation, f"{args.cell} s{seed} {mode}: la etapa 1 no cubre la validación"
            path = INPUTS / f"chaksu_C_{cfg['prefix']}{args.cell}_s{seed}_{mode}.csv"
            write_rows(path, with_fields(read))
            for variant, adapter in (("base", "base"), ("lora", cfg["lora"].format(seed=seed))):
                lists[seed].append(f"{path} {adapter} {STAGE2}/{cfg['prefix']}{variant}_{args.cell}_s{seed}_{mode}")
        if args.recipe == "verbalize_v1":
            flipped = {g["image_id"]: {"decision": "NORMAL" if g["decision"] == "GLAUCOMA SUSPECT" else "GLAUCOMA SUSPECT"}
                       for g in gold if g["image_id"] in validation}
            path = INPUTS / "chaksu_C_stress.csv"
            write_rows(path, with_fields(flipped))
            lists[seed].append(f"{path} {cfg['lora'].format(seed=seed)} {STAGE2}/lora_stress_s{seed}")
    if args.recipe == "verbalize_v1":
        lists[args.seeds[0]].append(f"{INPUTS / 'chaksu_C_stress.csv'} base {STAGE2}/base_stress")
    # Solo texto: una lista por semilla. Con imagen: una sola lista, que cabe en un job.
    files = {f"chaksu_C_etapa2_s{seed}.txt": lines for seed, lines in lists.items()} if args.recipe == "verbalize_v1" \
        else {"chaksu_C_img.txt": [line for lines in lists.values() for line in lines]}
    for name, lines in files.items():
        out = Path("runs") / name
        out.write_text("# Etapa 2 de C: TARGETS ADAPTER SALIDA, una generación por línea; `base` = sin adapter\n"
                       + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"{out}: {len(lines)} generaciones")


def analyze(rows, constant):
    """Por imagen: error del vCDR frente a la referencia, fidelidad a los campos de entrada, hallazgos nuevos y decisión leída."""
    read = [extract_prose(r["prediction"]) for r in rows]
    gold = np.array([float(r["gold_vcdr"]) for r in rows])
    error = np.array([abs((v["vcdr"] if v["vcdr"] is not None else constant) - g) for v, g in zip(read, gold)])
    faithful = np.array([all(v[f] is not None and abs(v[f] - round(float(r[f]), 2)) < TOLERANCE for f in ("vcdr", "hcdr"))
                         and v["decision"] == r["decision"] for v, r in zip(read, rows)])
    extra = np.array([any(t in r["prediction"].lower() for t in EXTRA_FINDINGS) for r in rows])
    decision = [v["decision"] for v in read]
    suspect = np.array([r["gold_decision"] == "GLAUCOMA SUSPECT" for r in rows])
    sensitivity = np.mean([d == "GLAUCOMA SUSPECT" for d, s in zip(decision, suspect) if s])
    specificity = np.mean([d == "NORMAL" for d, s in zip(decision, suspect) if not s])
    return {"error": error, "faithful": faithful, "extra": extra, "sensitivity": sensitivity, "specificity": specificity,
            "decision": decision, "unreadable": sum(v["vcdr"] is None for v in read)}


def report(args):
    constant = training_medians("data/targets/chaksu__json_v1.csv", SPLITS, ["vcdr"])["vcdr"]
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    def paired(a, b, blocks):
        mean, low, high = block_interval(a - b, blocks)
        return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"

    comparators = {}
    for name, folder, cell in (("A", STAGE1, args.cell), ("B", STAGE_B, args.cell)):
        runs = [run_metrics(folder / f"{cell}_s{s}", constant) for s in args.seeds]
        ids = runs[0][2]
        assert all(r[2] == ids for r in runs)
        comparators[name] = {"error": np.mean([r[1] for r in runs], axis=0), "metrics": [r[0] for r in runs], "ids": ids}
    ids = comparators["A"]["ids"]
    assert comparators["B"]["ids"] == ids
    blocks = np.array([groups[i] for i in ids])
    say(f"Configuración C, etapa 1 = {args.cell} de A, semillas {args.seeds}; 171 imágenes de validación; constante {constant:.4f}\n")
    say(f"Referencias, error del vCDR con la imagen correcta: A {comparators['A']['error'].mean():.4f}, B {comparators['B']['error'].mean():.4f} "
        f"(media de las semillas)\n")

    for variant in ("base", "lora"):
        say(f"=== Etapa 2: {variant}")
        results = {}
        for seed in args.seeds:
            for mode in MODES:
                rows = load(STAGE2 / f"{variant}_{args.cell}_s{seed}_{mode}" / "pred_text.jsonl")
                assert [r["id"] for r in rows] == ids, f"{variant} s{seed} {mode}: otras imágenes u otro orden"
                results[seed, mode] = analyze(rows, constant)
        mean_error = {m: np.mean([results[s, m]["error"] for s in args.seeds], axis=0) for m in MODES}
        say(f"  1 contenido  error del vCDR (real) por semilla {[round(results[s, 'real']['error'].mean(), 4) for s in args.seeds]}, "
            f"media {mean_error['real'].mean():.4f}")
        say(f"               C − A {paired(mean_error['real'], comparators['A']['error'], blocks)}   (criterio: IC dentro de ±{RELEVANT})")
        say(f"               C − B {paired(mean_error['real'], comparators['B']['error'], blocks)}")
        fidelity = {k: v["faithful"].mean() for k, v in results.items()}
        say(f"  2 fidelidad  mínima de las {len(fidelity)} combinaciones {min(fidelity.values()):.4f} (criterio ≥ 0.99); "
            f"textos sin vCDR legible: {sum(v['unreadable'] for v in results.values())}")
        say(f"  3 hallazgos fuera de los campos: {sum(int(v['extra'].sum()) for v in results.values())} de {len(results) * len(ids)} textos")
        gains = {m: mean_error[m].mean() - mean_error["real"].mean() for m in ("shuffled", "blank")}
        a_gains = {"shuffled": np.mean([m["image_gain"] for m in comparators["A"]["metrics"]]),
                   "blank": np.mean([m["blank_gain"] for m in comparators["A"]["metrics"]])}
        say(f"  4 controles  ganancia sobre la barajada {gains['shuffled']:+.4f} (A {a_gains['shuffled']:+.4f}), "
            f"sobre el blanco {gains['blank']:+.4f} (A {a_gains['blank']:+.4f})")
        say(f"    sensibilidad / especificidad (real, media de semillas): "
            f"{np.mean([results[s, 'real']['sensitivity'] for s in args.seeds]):.3f} / "
            f"{np.mean([results[s, 'real']['specificity'] for s in args.seeds]):.3f}  "
            f"(A {np.mean([m['sensitivity'] for m in comparators['A']['metrics']]):.3f} / "
            f"{np.mean([m['specificity'] for m in comparators['A']['metrics']]):.3f})")
        stress_paths = [STAGE2 / "base_stress"] if variant == "base" else [STAGE2 / f"lora_stress_s{s}" for s in args.seeds]
        copied, total, faithful = 0, 0, []
        for path in stress_paths:
            rows = load(path / "pred_text.jsonl")
            outcome = analyze(rows, constant)
            copied += sum(d == r["decision"] for d, r in zip(outcome["decision"], rows))
            total += len(rows)
            faithful.append(outcome["faithful"].mean())
        say(f"  6 estrés     decisión invertida copiada en {copied}/{total} textos ({copied / total:.3f}; criterio ≥ 0.99 en el LoRA); "
            f"fidelidad completa {np.mean(faithful):.3f}\n")
    Path("results/audit/chaksu_C.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_img(args):
    """La etapa 2 con la imagen real y los campos predichos frente a la etapa 2 solo texto (misma etapa 1, imagen real)."""
    constant = training_medians("data/targets/chaksu__json_v1.csv", SPLITS, ["vcdr"])["vcdr"]
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say(f"Etapa 2 con imagen, etapa 1 = {args.cell} de A, semillas {args.seeds}; imagen real en la etapa 1 y en la 2; constante {constant:.4f}\n")
    for variant in ("base", "lora"):
        with_image, text_only = [], []
        for seed in args.seeds:
            rows = load(STAGE2 / f"img_{variant}_{args.cell}_s{seed}_real" / "pred_real.jsonl")
            reference = load(STAGE2 / f"{variant}_{args.cell}_s{seed}_real" / "pred_text.jsonl")
            assert [r["id"] for r in rows] == [r["id"] for r in reference], f"{variant} s{seed}: otras imágenes u otro orden"
            with_image.append((rows, analyze(rows, constant)))
            text_only.append((reference, analyze(reference, constant)))
        ids = [r["id"] for r in with_image[0][0]]
        blocks = np.array([groups[i] for i in ids])
        error_img = np.mean([a["error"] for _, a in with_image], axis=0)
        error_text = np.mean([a["error"] for _, a in text_only], axis=0)
        mean, low, high = block_interval(error_img - error_text, blocks)
        say(f"=== Etapa 2 {variant}")
        say(f"  (i) fidelidad a los campos de entrada, por semilla: con imagen {[round(float(a['faithful'].mean()), 4) for _, a in with_image]}, "
            f"solo texto {[round(float(a['faithful'].mean()), 4) for _, a in text_only]} (criterio ≥ 0.99)")
        say(f"  (ii) error del vCDR: con imagen {error_img.mean():.4f}, solo texto {error_text.mean():.4f}; "
            f"con imagen − solo texto {mean:+.4f} [{low:+.4f}, {high:+.4f}] (criterio: IC dentro de ±{RELEVANT})")
        changed = agree = total = 0
        for rows, a in with_image:
            for r, d in zip(rows, a["decision"]):
                total += 1
                if d != r["decision"]:
                    changed += 1
                    agree += d == r["gold_decision"]
        say(f"  (iii) decisiones que la etapa 2 cambia respecto de la etapa 1: {changed} de {total}; de ellas coinciden con la referencia {agree}")
        say(f"        textos sin vCDR legible: {sum(a['unreadable'] for _, a in with_image)}; hallazgos fuera de los campos: "
            f"{sum(int(a['extra'].sum()) for _, a in with_image)} de {total}")
    Path("results/audit/chaksu_C_img.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "report", "report_img"])
    parser.add_argument("--recipe", default="verbalize_v1", choices=list(RECIPES), help="receta de la etapa 2 que `prepare` prepara")
    parser.add_argument("--cell", default="all-linear")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    {"prepare": prepare, "report": report, "report_img": report_img}[args.command](args)


if __name__ == "__main__":
    main()
