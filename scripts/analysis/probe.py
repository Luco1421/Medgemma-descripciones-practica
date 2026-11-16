"""Sondas por profundidad: ¿en qué punto del cálculo de MedGemma se puede leer cada campo?

    python -m scripts.analysis.probe extract --targets T --splits S --prompt-file P --datasets-root R --output F.npz
    python -m scripts.analysis.probe fit --features F.npz --targets T --splits S --output PREFIJO
    python -m scripts.analysis.probe pooling --targets T --splits S --prompt-file P --datasets-root R --output PREFIJO

Una sonda es un modelo lineal pequeño (regresión ridge, o logística para la decisión) entrenado para
predecir un campo a partir de la representación que el modelo tiene en un punto de su cálculo, SIN
tocar el modelo. Si la sonda lee el vCDR a la salida del encoder pero no en el último token, la
información se perdió por el camino; si no lo lee ni en el encoder, nunca llegó. Es un diagnóstico de
dónde está la información, que no es lo mismo que dónde conviene adaptar con LoRA.

Puntos de lectura, todos con el modelo base y el mismo prompt que las corridas:
  vision        cada una de las 28 salidas de la torre de visión (embeddings + 27 bloques), promediada
                sobre los 4096 parches
  image_tokens  cada una de las 35 salidas de la torre de lenguaje (embeddings + 34 bloques), promediada
                sobre los 256 tokens de imagen
  last_token    esas mismas 35 salidas en el último token del prompt, justo donde el modelo empieza a
                escribir la respuesta

Dos cuidados de lectura. Que una sonda lea un campo no prueba que el modelo lo use: una sonda con
suficientes dimensiones puede memorizar. Por eso `fit` repite cada ajuste con las etiquetas de
entrenamiento PERMUTADAS (la tarea de control); lo que cuenta es la diferencia entre la sonda real y
la de control. Y solo se extraen entrenamiento y validación: el Test no se toca.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def cmd_extract(args):
    import torch
    from PIL import Image

    from scripts.model.lora import load_rows
    from scripts.model.medgemma import load_medgemma

    rows = []
    for split in ("train", "validation"):
        part, _ = load_rows(argparse.Namespace(**{**vars(args), "split": split}))
        rows += part

    processor, model = load_medgemma()
    model.eval()
    image_token = model.config.image_token_id
    vision, image_tokens, last_token = [], [], []
    for number, row in enumerate(rows, 1):
        image = Image.open(row["image"]).convert("RGB")
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": row["prompt"]}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            # La torre de visión se llama aparte porque el modelo completo no devuelve sus capas.
            vit = model.model.vision_tower(pixel_values=inputs["pixel_values"].to(model.dtype),
                                           output_hidden_states=True).hidden_states
            hidden = model(**inputs, output_hidden_states=True).hidden_states
        positions = (inputs["input_ids"][0] == image_token).nonzero().flatten()
        assert len(positions) == 256, f"se esperaban 256 tokens de imagen y hay {len(positions)}"
        assert len(vit) == 28 and len(hidden) == 35, (len(vit), len(hidden))
        vision.append(torch.stack([h[0].float().mean(0) for h in vit]).cpu().numpy())
        image_tokens.append(torch.stack([h[0, positions].float().mean(0) for h in hidden]).cpu().numpy())
        last_token.append(torch.stack([h[0, -1].float() for h in hidden]).cpu().numpy())
        if number % 50 == 0:
            print(f"[{number}/{len(rows)}]", flush=True)

    arrays = {"vision": np.stack(vision), "image_tokens": np.stack(image_tokens), "last_token": np.stack(last_token)}
    # float32 y no float16: las activaciones masivas de Gemma pueden pasar de 65 504.
    for name, array in arrays.items():
        assert np.isfinite(array).all(), f"valores no finitos en {name}"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=np.array([r["id"] for r in rows]), **arrays)
    print(f"{args.output}: {len(rows)} imágenes; " + ", ".join(f"{k} {v.shape[1:]}" for k, v in arrays.items()))


POOLINGS = ("avg", "max", "gradient", "gradient_peak")


def pooled_descriptors(hidden):
    """Resumen de una salida de la torre de visión (4096 parches × 1152) tras agrupar en ventanas de 4×4, como el conector.

    El conector promedia cada ventana de 4×4 parches (64×64 → 16×16 = 256 tokens). Aquí se comparan cuatro formas de resumir
    el mapa de parches en un vector de 1152 dimensiones, que alimenta una sonda lineal:
      avg            media de las medias por ventana (= la media de los 4096 parches; es lo que hace el conector)
      max            media de los máximos por ventana: conserva el rasgo más fuerte de cada ventana
      gradient       media de (máximo − mínimo) por ventana: el gradiente morfológico, grande donde hay un borde
      gradient_peak  máximo entre ventanas de ese gradiente: la ventana con el borde más marcado
    """
    grid = hidden.float().reshape(64, 64, -1)
    windows = grid.reshape(16, 4, 16, 4, -1).permute(0, 2, 1, 3, 4).reshape(16, 16, 16, -1)
    high, low = windows.max(dim=2).values, windows.min(dim=2).values
    edge = high - low
    return [windows.mean(dim=2).mean(dim=(0, 1)), high.mean(dim=(0, 1)), edge.mean(dim=(0, 1)), edge.amax(dim=(0, 1))]


def cmd_pooling(args):
    import torch
    from PIL import Image
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from scripts.model.lora import load_rows
    from scripts.model.medgemma import load_medgemma

    rows = []
    for split in ("train", "validation"):
        part, _ = load_rows(argparse.Namespace(**{**vars(args), "split": split}))
        rows += part
    processor, model = load_medgemma()
    model.eval()
    features = []
    for number, row in enumerate(rows, 1):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": Image.open(row["image"]).convert("RGB")},
                                          {"type": "text", "text": row["prompt"]}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            states = model.model.vision_tower(pixel_values=inputs["pixel_values"].to(model.dtype), output_hidden_states=True).hidden_states
        assert len(states) == 28 and states[0].shape[1] == 4096, (len(states), states[0].shape)
        features.append(torch.stack([torch.stack(pooled_descriptors(h[0])) for h in states]).cpu().numpy())
        if number % 100 == 0:
            print(f"[{number}/{len(rows)}]", flush=True)
    features = np.stack(features)  # imágenes × 28 capas × 4 resúmenes × 1152
    assert np.isfinite(features).all()
    ids = [r["id"] for r in rows]

    split_of = {r["image_id"]: r["split"] for r in read_csv(args.splits)}
    target_of = {r["image_id"]: r for r in read_csv(args.targets)}
    split = np.array([split_of[i] for i in ids])
    train, validation = split == "train", split == "validation"
    vcdr = np.array([float(target_of[i]["vcdr"]) for i in ids])
    suspect = np.array([target_of[i]["decision"] == "GLAUCOMA SUSPECT" for i in ids])
    permutation = np.random.default_rng(0).permutation(train.sum())
    alphas = np.logspace(-1, 5, 13)

    table, report = [], [f"Sondas sobre la torre de visión según el resumen de los parches: {train.sum()} imágenes de entrenamiento, "
                         f"{validation.sum()} de validación. Mismas sondas que `fit` (ridge para el vCDR, logística para la decisión).",
                         f"{'capa':>5} {'resumen':<14} | {'vcdr ρ':>7} {'control':>8} {'|error|':>8} | {'decisión AUC':>12}"]
    for layer in range(28):
        for k, name in enumerate(POOLINGS):
            x = StandardScaler().fit(features[train, layer, k]).transform(features[:, layer, k])
            predicted = RidgeCV(alphas=alphas).fit(x[train], vcdr[train]).predict(x[validation])
            control = RidgeCV(alphas=alphas).fit(x[train], vcdr[train][permutation]).predict(x[validation])
            score = LogisticRegressionCV(Cs=5, cv=3, scoring="roc_auc", class_weight="balanced", max_iter=300).fit(
                x[train], suspect[train]).decision_function(x[validation])
            row = {"layer": layer, "pooling": name, "vcdr_spearman": spearmanr(vcdr[validation], predicted).statistic,
                   "vcdr_control": spearmanr(vcdr[validation], control).statistic,
                   "vcdr_mae": np.abs(vcdr[validation] - predicted).mean(), "decision_auc": roc_auc_score(suspect[validation], score)}
            table.append(row)
            report.append(f"{layer:>5} {name:<14} | {row['vcdr_spearman']:>+7.3f} {row['vcdr_control']:>+8.3f} {row['vcdr_mae']:>8.3f} | {row['decision_auc']:>12.3f}")
    report.append("")
    for name in POOLINGS:
        best = max((t for t in table if t["pooling"] == name), key=lambda t: t["vcdr_spearman"])
        report.append(f"mejor capa de {name}: {best['layer']} (vcdr ρ {best['vcdr_spearman']:+.3f}, |error| {best['vcdr_mae']:.3f}, AUC {best['decision_auc']:.3f})")
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    prefix.with_suffix(".txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[-4:]))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cmd_fit(args):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    data = np.load(args.features)
    ids = list(data["ids"])
    split_of = {r["image_id"]: r["split"] for r in read_csv(args.splits)}
    target_of = {r["image_id"]: r for r in read_csv(args.targets)}
    split = np.array([split_of[i] for i in ids])
    train, validation = split == "train", split == "validation"
    assert train.any() and validation.any() and set(split) == {"train", "validation"}
    fields = ("vcdr", "hcdr")
    y = {f: np.array([float(target_of[i][f]) for i in ids]) for f in fields}
    suspect = np.array([target_of[i]["decision"] == "GLAUCOMA SUSPECT" for i in ids])
    camera = np.array([target_of[i]["camera"] for i in ids])
    rng = np.random.default_rng(0)
    permutation = rng.permutation(train.sum())

    report, table, predictions = [], [], {}

    def say(text=""):
        print(text)
        report.append(text)

    say(f"Sondas por profundidad: {train.sum()} imágenes de entrenamiento, {validation.sum()} de validación "
        f"({int(suspect[validation].sum())} sospechosos)")
    # Lo que se puede conseguir sin mirar la imagen: la media por cámara y un valor constante.
    for f in fields:
        constant = np.abs(y[f][validation] - np.median(y[f][train])).mean()
        per_camera = np.array([y[f][train & (camera == c)].mean() for c in camera[validation]])
        say(f"  sin imagen, {f}: constante |error| {constant:.3f}; media por cámara |error| "
            f"{np.abs(y[f][validation] - per_camera).mean():.3f}, Spearman {spearmanr(y[f][validation], per_camera).statistic:+.3f}")

    for family in ("vision", "image_tokens", "last_token"):
        features = data[family]
        say(f"\n--- {family} ({features.shape[1]} puntos, {features.shape[2]} dimensiones) ---")
        say(f"{'capa':>5} | {'vcdr ρ':>7} {'control':>8} {'|error|':>8} | {'hcdr ρ':>7} | {'decisión AUC':>12}")
        stored = {k: [] for k in ("vcdr", "hcdr", "control", "decision")}
        for layer in range(features.shape[1]):
            x = StandardScaler().fit(features[train, layer]).transform(features[:, layer])
            cell = {}
            for f in fields:
                real = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(x[train], y[f][train])
                predicted = real.predict(x[validation])
                cell[f] = (spearmanr(y[f][validation], predicted).statistic, np.abs(y[f][validation] - predicted).mean())
                stored[f].append(predicted)
                if f == "vcdr":
                    control = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(x[train], y[f][train][permutation])
                    control_predicted = control.predict(x[validation])
                    cell["control"] = spearmanr(y[f][validation], control_predicted).statistic
                    stored["control"].append(control_predicted)
            classifier = LogisticRegressionCV(Cs=5, cv=3, scoring="roc_auc", class_weight="balanced", max_iter=300)
            classifier.fit(x[train], suspect[train])
            score = classifier.decision_function(x[validation])
            auc = roc_auc_score(suspect[validation], score)
            stored["decision"].append(score)
            say(f"{layer:>5} | {cell['vcdr'][0]:>+7.3f} {cell['control']:>+8.3f} {cell['vcdr'][1]:>8.3f} | "
                f"{cell['hcdr'][0]:>+7.3f} | {auc:>12.3f}")
            table.append({"family": family, "layer": layer, "vcdr_spearman": cell["vcdr"][0],
                          "vcdr_control": cell["control"], "vcdr_mae": cell["vcdr"][1],
                          "hcdr_spearman": cell["hcdr"][0], "decision_auc": auc})
        best = max((t for t in table if t["family"] == family), key=lambda t: t["vcdr_spearman"])
        say(f"  mejor capa para vcdr: {best['layer']} (ρ {best['vcdr_spearman']:+.3f}; control {best['vcdr_control']:+.3f})")
        predictions[family] = {k: np.stack(v) for k, v in stored.items()}

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    prefix.with_suffix(".txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Predicciones de validación de cada sonda y capa: sirven para calcular intervalos por bloques de captura sin reajustar.
    np.savez_compressed(prefix.parent / f"{prefix.name}_predicciones.npz", ids=np.array(ids)[validation],
                        **{f"{family}__{k}": v for family, d in predictions.items() for k, v in d.items()})
    print(f"\n{prefix.with_suffix('.csv')}\n{prefix.with_suffix('.txt')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="activaciones por capa (necesita GPU)")
    extract.add_argument("--targets", required=True)
    extract.add_argument("--splits", required=True)
    extract.add_argument("--prompt-file", required=True)
    extract.add_argument("--datasets-root", required=True)
    extract.add_argument("--output", required=True)
    fit = sub.add_parser("fit", help="ajusta las sondas y la tarea de control")
    fit.add_argument("--features", required=True)
    fit.add_argument("--targets", required=True)
    fit.add_argument("--splits", required=True)
    fit.add_argument("--output", required=True, help="prefijo de los archivos de salida")
    pooling = sub.add_parser("pooling", help="sondas de la torre de visión con distintos resúmenes de los parches (necesita GPU)")
    for flag in ("--targets", "--splits", "--prompt-file", "--datasets-root", "--output"):
        pooling.add_argument(flag, required=True)
    args = parser.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "pooling": cmd_pooling}[args.command](args)


if __name__ == "__main__":
    main()
