"""Prompting optimizado del modelo base como línea base (baseline 2).

    python -m scripts.studies.baseline_prompts write      # prompts, particiones, ejemplos de P5 y listas de generación
    python -m scripts.studies.baseline_prompts report     # elige el prompt y compara con `all-linear` de A

El conjunto de prompts es el fijado antes de ver resultados. Todos piden la receta `json_v1` (el modelo base, sin adapter) y
ninguno lleva información de un caso concreto: el mismo texto para todas las imágenes. Cada variante cambia una sola cosa de P0.

`write` crea, en `data/prompts/`, un archivo por prompt y los ejemplos de P5 (4 imágenes de entrenamiento con su respuesta); en
`data/splits/`, una partición derivada donde 171 imágenes de entrenamiento elegidas al azar (semilla 0) pasan a llamarse `selection`; y en
`runs/`, las listas de generación (`PROMPT SPLITS PARTICION EJEMPLOS CONTROLES SALIDA` por línea, para `lora.py generate`). Los ejemplos de P5 salen de las imágenes de entrenamiento que no
están en `selection`.

`report` elige el prompt de menor error del vCDR en `selection` (imagen correcta); si dos difieren menos de 0.004 gana el más corto. Luego
tabula todos los prompts en validación con los tres controles de imagen, su diferencia emparejada con `all-linear` de A (media de las tres
semillas, intervalo del 95 % por bloques de captura) y la dispersión entre P0, P0b y P0c, que es el ruido por redacción.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from scripts.eval.evaluate import answer_fields, errors, load, read_answer, training_medians
from scripts.eval.results import block_interval, run_metrics
from scripts.data.splits import chaksu_groups, read_manifest

TARGETS = "data/targets/chaksu__json_v1.csv"
SPLITS = "data/splits/chaksu_seed42.csv"
SELECTION_SPLITS = "data/splits/chaksu_seed42_seleccion.csv"
PROMPTS = Path("data/prompts")
EXPERIMENTS = Path("experiments/chaksu_prompts_A")
STAGE_A = Path("experiments/chaksu_json_v1")
SELECTION_SIZE = 171
SHOTS = 4
RELEVANT = 0.004

INTRO = "You are an ophthalmologist reading a color fundus photograph centered on the optic nerve head."
ANSWER = "Answer only with a JSON object with the keys vcdr, hcdr and decision."
TASK = ("Estimate the vertical cup-to-disc ratio (vcdr) and the horizontal cup-to-disc ratio (hcdr) as decimals, and classify the "
        f"optic disc as NORMAL or GLAUCOMA SUSPECT. {ANSWER}")
DEFINITIONS = ("The optic disc is the round or oval head of the optic nerve; the cup is the pale central depression within it. The "
               "vertical cup-to-disc ratio (vcdr) is the vertical diameter of the cup divided by the vertical diameter of the disc; "
               "the horizontal ratio (hcdr) is the same along the horizontal axis. Both are between 0 and 1.")
CRITERIA = ("A cup that is enlarged or elongated vertically, and thinning or notching of the neuroretinal rim, are signs associated "
            "with glaucoma. Classify the optic disc as GLAUCOMA SUSPECT when these signs suggest glaucomatous damage, and as NORMAL "
            "otherwise.")
CLOSING = f"Estimate vcdr and hcdr as decimals. {ANSWER}"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prior_sentence(train):
    def spread(field):
        values = np.array([float(r[field]) for r in train])
        return np.median(values), np.percentile(values, 5), np.percentile(values, 95)

    (vm, vl, vh), (hm, hl, hh) = spread("vcdr"), spread("hcdr")
    suspect = np.mean([r["decision"] == "GLAUCOMA SUSPECT" for r in train])
    return (f"For reference, in a collection of {len(train)} similar photographs graded by experts, the vertical cup-to-disc ratio "
            f"had a median of {vm:.2f} (90% of the values between {vl:.2f} and {vh:.2f}), the horizontal ratio a median of {hm:.2f} "
            f"(90% between {hl:.2f} and {hh:.2f}), and {100 * suspect:.1f}% of the optic discs were classified GLAUCOMA SUSPECT.")


def build_prompts(train):
    prior = prior_sentence(train)
    p0 = f"{INTRO} {TASK}"
    assert p0 == Path("data/targets/chaksu__json_v1.prompt.txt").read_text(encoding="utf-8").strip(), "P0 no es el prompt de A"
    return {
        "P0": p0,
        "P0b": ("You are an ophthalmologist examining a color fundus photograph that is centered on the optic nerve head. Report the "
                "vertical cup-to-disc ratio (vcdr) and the horizontal cup-to-disc ratio (hcdr) as decimal numbers, and label the optic "
                "disc as NORMAL or GLAUCOMA SUSPECT. Reply with a JSON object only, using the keys vcdr, hcdr and decision."),
        "P0c": ("Look at this color fundus photograph centered on the optic nerve head, as an ophthalmologist would. Give your estimate "
                "of the vertical cup-to-disc ratio (vcdr) and the horizontal cup-to-disc ratio (hcdr) as decimals, and classify the "
                "optic disc as NORMAL or GLAUCOMA SUSPECT. Respond with only a JSON object with the keys vcdr, hcdr and decision."),
        "P1": (f"{INTRO} {DEFINITIONS} Estimate vcdr and hcdr as decimals, and classify the optic disc as NORMAL or GLAUCOMA "
               f"SUSPECT. {ANSWER}"),
        "P2": f"{INTRO} {DEFINITIONS} {CRITERIA} {CLOSING}",
        "P3": f"{INTRO} {prior} {TASK}",
        "P4": f"{INTRO} {DEFINITIONS} {CRITERIA} {prior} {CLOSING}",
        "P5": p0,
    }


def write():
    targets = read_csv(TARGETS)
    split_of = {r["image_id"]: r["split"] for r in read_csv(SPLITS)}
    train = [t for t in targets if split_of[t["image_id"]] == "train"]
    prompts = build_prompts(train)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        (PROMPTS / f"chaksu_A_{name}.txt").write_text(text + "\n", encoding="utf-8")

    rng = random.Random(0)
    selection = {t["image_id"] for t in rng.sample(train, SELECTION_SIZE)}
    rows = read_csv(SPLITS)
    for r in rows:
        if r["image_id"] in selection:
            r["split"] = "selection"
    with Path(SELECTION_SPLITS).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Ejemplos de P5: dos de cada decisión, repartidos entre las cámaras, de las imágenes de entrenamiento que no están en `selection`.
    pool = [t for t in train if t["image_id"] not in selection]
    cameras = sorted({t["camera"] for t in pool})
    rng.shuffle(cameras)
    plan = [("GLAUCOMA SUSPECT", cameras[0]), ("NORMAL", cameras[1]), ("GLAUCOMA SUSPECT", cameras[2 % len(cameras)]),
            ("NORMAL", cameras[0])]
    assert len(plan) == SHOTS
    shots = []
    for decision, camera in plan:
        candidates = [t for t in pool if t["decision"] == decision and t["camera"] == camera
                      and t["image_id"] not in {s["id"] for s in shots}]
        assert candidates, f"sin imágenes {decision} de {camera}"
        t = rng.choice(candidates)
        shots.append({"id": t["image_id"], "image": t["image"], "response": t["text"], "decision": decision, "camera": camera})
    rng.shuffle(shots)
    shots_path = PROMPTS / "chaksu_A_P5_shots.json"
    shots_path.write_text(json.dumps(shots, indent=2), encoding="utf-8")

    def run(prompt, split, modes, shot_file, tag=""):
        return (f"{PROMPTS / f'chaksu_A_{prompt}.txt'} {SELECTION_SPLITS} {split} {shot_file} {modes} {EXPERIMENTS / (prompt + tag) / split}")

    batches = {1: ["P0b", "P0c", "P1"], 2: ["P2", "P3", "P4"], 3: ["P5", "P0"]}
    for job, names in batches.items():
        lines = []
        for name in names:
            shot_file = shots_path if name == "P5" else "-"
            lines += [run(name, "selection", "real", shot_file), run(name, "validation", "real,shuffled,blank", shot_file)]
        out = Path("runs") / f"chaksu_prompts_base_{job}.txt"
        out.write_text("# Prompting del modelo base: PROMPT SPLITS PARTICION EJEMPLOS|- CONTROLES SALIDA\n" + "\n".join(lines) + "\n",
                       encoding="utf-8")
        print(f"{out}: {len(lines)} generaciones")
    # P4 entra en modo de razonamiento («<unused94>thought») y 128 tokens no alcanzan para llegar al JSON: repetición declarada con 1024.
    out = Path("runs") / "chaksu_prompts_base_4.txt"
    out.write_text("# P4 con 1024 tokens (análisis de sensibilidad); lanzar con MAX_NEW_TOKENS=1024\n"
                   + run("P4", "validation", "real,shuffled,blank", "-", "_t1024") + "\n", encoding="utf-8")
    print(f"{out}: 1 generación")
    print(f"{SELECTION_SPLITS}: {len(selection)} imágenes de entrenamiento como `selection`; ejemplos de P5 en {shots_path}:")
    for s in shots:
        print(f"  {s['id']} {s['camera']} {s['decision']}")
    print(f"prior de P3: {prior_sentence(train)}")


def selection_error(prompt, constant):
    rows = load(EXPERIMENTS / prompt / "selection" / "pred_real.jsonl")
    fields = answer_fields(rows)
    reads = [read_answer(r, fields) for r in rows]
    return errors(rows, reads, "vcdr", constant).mean(), sum(not ok for ok, _ in reads)


def report():
    constant = training_medians(TARGETS, SPLITS, ["vcdr"])["vcdr"]
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    names = [p.stem.removeprefix("chaksu_A_") for p in sorted(PROMPTS.glob("chaksu_A_P*.txt"))]
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    prompts = {n: (PROMPTS / f"chaksu_A_{n}.txt").read_text(encoding="utf-8").strip() for n in names}
    selected = {n: selection_error(n, constant) for n in names}
    best = min(selected.values(), key=lambda v: v[0])[0]
    near = [n for n in names if selected[n][0] - best < RELEVANT]
    chosen = min(near, key=lambda n: len(prompts[n]))
    say(f"SELECCIÓN ({SELECTION_SIZE} imágenes de entrenamiento, imagen correcta; constante {constant:.4f})")
    for n in names:
        say(f"  {n:<4} error del vCDR {selected[n][0]:.4f}   respuestas ilegibles {selected[n][1]}   longitud {len(prompts[n])} caracteres")
    say(f"  mejor {best:.4f}; dentro de {RELEVANT}: {near}; elegido (el más corto): {chosen}\n")

    seeds = (42, 43, 44)
    runs = [run_metrics(STAGE_A / f"all-linear_s{s}", constant) for s in seeds]
    ids = runs[0][2]
    blocks = np.array([groups[i] for i in ids])
    reference = np.mean([r[1] for r in runs], axis=0)
    say(f"VALIDACIÓN (171 imágenes, tres controles). Referencia: `all-linear` de A, error {reference.mean():.4f} (media de {len(seeds)} semillas)")
    say(f"{'':<5}{'error':>8}{'barajada':>10}{'blanco':>8}{'formato':>9}{'sens':>7}{'espec':>7}   diferencia con all-linear [IC95]")
    validation = {}
    for n in names + [n for n in ("P4_t1024",) if (EXPERIMENTS / n / "validation" / "generate_config.json").exists()]:
        metrics, error, run_ids, _ = run_metrics(EXPERIMENTS / n, constant)
        assert run_ids == ids, f"{n}: otras imágenes"
        validation[n] = error
        mean, low, high = block_interval(error - reference, blocks)
        say(f"{n:<9}{metrics['vcdr_error']:>8.4f}{metrics['image_gain']:>+10.4f}{metrics['blank_gain']:>+8.4f}{metrics['format']:>9.3f}"
            f"{metrics['sensitivity']:>7.3f}{metrics['specificity']:>7.3f}   {mean:+.4f} [{low:+.4f}, {high:+.4f}]"
            f"{'   <- elegido' if n == chosen else ''}")
    wording = [validation[n].mean() for n in ("P0", "P0b", "P0c")]
    say(f"\nRuido por redacción: error de P0, P0b y P0c {[round(w, 4) for w in wording]}; rango {max(wording) - min(wording):.4f}")
    mean, low, high = block_interval(validation[chosen] - validation["P0"], blocks)
    say(f"Elegido ({chosen}) − P0: {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
    mean, low, high = block_interval(validation[chosen] - reference, blocks)
    holds = low > RELEVANT
    say(f"Elegido ({chosen}) − all-linear de A: {mean:+.4f} [{low:+.4f}, {high:+.4f}]  "
        f"=> «adaptar supera a mejorar el prompt, frente a este conjunto»: {'se sostiene' if holds else 'no se sostiene'} (criterio: límite inferior > +{RELEVANT})")

    base = load(STAGE_A / "base" / "validation" / "pred_real.jsonl")
    p0 = load(EXPERIMENTS / "P0" / "validation" / "pred_real.jsonl")
    same = sum(a["prediction"] == b["prediction"] for a, b in zip(base, p0))
    say(f"\nComprobación: P0 reproduce el modelo base de A en {same} de {len(base)} respuestas idénticas")
    Path("results/audit/chaksu_prompts_base.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["write", "report"])
    args = parser.parse_args()
    {"write": write, "report": report}[args.command]()


if __name__ == "__main__":
    main()
