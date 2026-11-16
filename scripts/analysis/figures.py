"""Figuras del trabajo a partir de las tablas de resultados (ningún número se escribe a mano).

    python -m scripts.analysis.figures [--split validation|test]

Escribe PNG y PDF en `results/figures/`:
  capas_<split>        error del vCDR de cada capa suelta (media y desviación entre semillas), por torre y receta
  eficiencia_<split>   error frente a parámetros entrenables (escala logarítmica); con --split test solo los finalistas
  predictores          profundidad, sonda y Auto-RGN de cada capa frente a su error de validación (solo con validation)
  sondas_perdida       señal de vCDR y de la decisión legible en cada punto del modelo y su pérdida respecto de la salida de la visión (validation)
  transferencia        error de validación frente al del Test de los 14 finalistas (solo con test)
  prompts              error del modelo base con cada prompt frente a `all-linear` y al predictor constante (solo con validation)
  configuraciones, particion_camaras, acuerdo_expertos, prediccion_vs_referencia, controles_imagen, sondas_profundidad,
  rgn_bloques, sorteos_control, validacion_externa, etapa2_C, superficie_vs_error   (validation; ver cada función)
  python -m scripts.analysis.figures pooling   dibuja pooling_sondas (no forma parte del conjunto por defecto)
Lee `results/grids/chaksu_{json_v1,prose_v1}/results[_test].csv`, `results/probes/`, `results/rgn/` y las predicciones de los prompts.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import re

from scripts.studies.baseline_prompts import EXPERIMENTS as PROMPT_RUNS, PROMPTS
from scripts.eval.evaluate import VCDR_TOLERANCE, answer_fields, errors, load, read_answer, training_medians
from scripts.analysis.finalists import A_PRIORI, EFFICIENT, folder_name
from scripts.eval.results import GRIDS, run_metrics
from scripts.data.splits import chaksu_groups, read_manifest

OUT = Path("results/figures")
RECIPES = {"A": "json_v1", "B": "prose_v1"}
COLORS = {"A": "#0072B2", "B": "#D55E00"}
COLORS.update({"C": "#7B3294", "base": "#8C8C8C", "constant": "#777777"})
# Familias de celdas de las rejillas: una torre, o `all-linear` (todas las capas lineales de las dos torres), o el control aleatorio
# (un presupuesto igual al de un bloque, repartido al azar entre las dos torres).
TOWER_COLORS = {"vision": "#009E73", "language": "#CC79A7", "all-linear": "#000000", "random": "#E69F00"}
TOWER_MARKERS = {"vision": "o", "language": "o", "all-linear": "s", "random": "D"}
ES = {"vision": "visión", "language": "lenguaje", "all-linear": "all-linear (todas las capas lineales)",
      "random": "control aleatorio (presupuesto de un bloque)", "validation": "validación", "test": "Test"}
CAMERAS = {"Bosch": ("#E69F00", "o"), "Forus": ("#56B4E9", "s"), "Remidio": ("#555555", "^")}
EXPERIMENTS = Path("experiments")
TARGETS, SPLITS = "data/targets/chaksu__json_v1.csv", "data/splits/chaksu_seed42.csv"
RELEVANT = 0.004
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})


def table(recipe, split):
    name = "results.csv" if split == "validation" else f"results_{split}.csv"
    return pd.read_csv(GRIDS / f"chaksu_{recipe}" / name).set_index("cell")


def family(t):
    """Etiqueta de familia de cada celda de una tabla de rejilla (ver TOWER_COLORS)."""
    return pd.Series(np.select([t.module == "random", t.module == "all-linear"], ["random", "all-linear"], t.tower), index=t.index)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(OUT / f"{name}.png")


def reference_lines(ax, t):
    ax.axhline(t.loc["constant", "vcdr_error_mean"], color="#777777", ls="--", lw=1)
    ax.axhline(t.loc["all-linear", "vcdr_error_mean"], color="black", lw=1)


def layers(split):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex="row", sharey="col", layout="constrained")
    for col, (label, recipe) in enumerate(RECIPES.items()):
        t = table(recipe, split)
        for row, (tower, n) in enumerate((("vision", 27), ("language", 34))):
            ax = axes[row, col]
            cells = [f"block_{tower}_layer{k:02d}" for k in range(n)]
            ax.errorbar(range(n), t.loc[cells, "vcdr_error_mean"], yerr=t.loc[cells, "vcdr_error_sd"], fmt="o", ms=3.5, lw=0.8,
                        color=TOWER_COLORS[tower], ecolor="#999999")
            reference_lines(ax, t)
            ax.set_title(f"{label}: torre de {ES[tower]}", loc="left")
            ax.set_xlabel("capa" if row == 1 else "")
            ax.set_ylabel("error absoluto medio del vCDR" if col == 0 else "")
    axes[0, 0].plot([], [], color="black", lw=1, label="all-linear")
    axes[0, 0].plot([], [], color="#777777", ls="--", lw=1, label="predictor constante")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Error del vCDR al adaptar una sola capa ({ES[split]}; media ± desviación entre 3 semillas)", x=0.01, ha="left")
    save(fig, f"capas_{split}")


def efficiency(split):
    finalists = {folder_name(c) for c in A_PRIORI + EFFICIENT}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (label, recipe) in zip(axes, RECIPES.items()):
        t = table(recipe, split).drop(index=["constant", "base"], errors="ignore")
        reference = t.loc["all-linear", "vcdr_error_mean"]
        if split != "validation":
            t = t[t.index.isin(finalists)]
        for tower, g in t.groupby(family(t)):
            ax.scatter(g.trainable_params / 1e6, g.vcdr_error_mean, s=18, color=TOWER_COLORS[tower], marker=TOWER_MARKERS[tower], label=ES[tower], alpha=0.9, zorder=3)
        for cell in t.index[t.index.isin(finalists)]:
            if split == "validation":
                ax.scatter(t.loc[cell, "trainable_params"] / 1e6, t.loc[cell, "vcdr_error_mean"], s=60, facecolors="none", edgecolors="black", lw=0.8)
        full = table(recipe, split)
        ax.axhspan(reference - RELEVANT, reference + RELEVANT, color="#DDDDDD", alpha=0.6, lw=0, zorder=0)
        ax.axhline(reference, color="black", lw=1)
        ax.axhline(full.loc["constant", "vcdr_error_mean"], color="#777777", ls="--", lw=1)
        ax.set_xscale("log")
        if label == "B" and split == "validation":
            top = 0.09
            outside = int((t.vcdr_error_mean > top).sum())
            ax.set_ylim(top=top)
            ax.text(0.98, 0.97, f"{outside} celdas por encima de {top} (fuera de escala)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("parámetros entrenables (millones, escala logarítmica)")
        ax.set_title(f"{label} ({recipe})", loc="left")
        ax.set_ylabel("error absoluto medio del vCDR" if label == "A" else "")
    axes[0].legend(frameon=False, title="qué se adapta", fontsize=7, title_fontsize=7)
    fig.suptitle(f"Error del vCDR frente a parámetros entrenables ({ES[split]}). Banda: ±{RELEVANT} alrededor de all-linear; línea discontinua: predictor constante;"
                 " círculos: finalistas", x=0.01, ha="left", fontsize=8)
    save(fig, f"eficiencia_{split}")


def transfer():
    finalists = [folder_name(c) for c in A_PRIORI + EFFICIENT]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, (label, recipe) in zip(axes, RECIPES.items()):
        val, test = table(recipe, "validation").loc[finalists], table(recipe, "test").loc[finalists]
        for tower, g in val.groupby(family(val)):
            ax.scatter(g.vcdr_error_mean, test.loc[g.index, "vcdr_error_mean"], s=24, color=TOWER_COLORS[tower], marker=TOWER_MARKERS[tower], label=ES[tower])
        lo, hi = min(val.vcdr_error_mean.min(), test.vcdr_error_mean.min()) - 0.001, min(max(val.vcdr_error_mean.max(), test.vcdr_error_mean.max()), 0.07) + 0.001
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8)
        for cell in ("all-linear", "mlp_vision", "block_vision", "random_block_vision", "random_block_language"):
            ax.annotate(cell, (val.loc[cell, "vcdr_error_mean"], test.loc[cell, "vcdr_error_mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        rho = spearmanr(val.vcdr_error_mean, test.vcdr_error_mean).statistic
        ax.set_title(f"{label} ({recipe}): ρ = {rho:+.2f}", loc="left")
        ax.set_xlabel("error del vCDR en validación (171 imágenes)")
        ax.set_ylabel("error del vCDR en el Test (336 imágenes)" if label == "A" else "")
    axes[0].legend(frameon=False, title="qué se adapta", fontsize=7, title_fontsize=7)
    fig.suptitle("Las 14 ubicaciones finalistas: validación frente a Test (línea gris: igual error; ρ = Spearman)", x=0.01, ha="left")
    save(fig, "transferencia")


def predictors():
    probes = pd.read_csv("results/probes/chaksu_json_v1_sondas.csv")
    rgn = {label: pd.read_csv(f"results/rgn/chaksu_{recipe}.csv") for label, recipe in RECIPES.items()}
    tables = {label: table(recipe, "validation") for label, recipe in RECIPES.items()}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey="row", layout="constrained")
    for row, (tower, n) in enumerate((("vision", 27), ("language", 34))):
        family = "vision" if tower == "vision" else "image_tokens"
        probe = probes[probes.family == family].set_index("layer").loc[range(1, n + 1), "vcdr_spearman"].to_numpy()
        for col, (name, kind) in enumerate((("profundidad (índice de la capa)", None), (f"sonda lineal ({family}): Spearman con el vCDR", "probe"),
                                            ("RGN del bloque (modelo base)", "rgn"))):
            ax = axes[row, col]
            notes = []
            for label in RECIPES:
                error = np.array([tables[label].loc[f"block_{tower}_layer{k:02d}", "vcdr_error_mean"] for k in range(n)])
                if kind is None:
                    x = np.arange(n)
                elif kind == "probe":
                    x = probe
                else:
                    x = rgn[label][rgn[label].tower == tower].set_index("layer").loc[range(n), "rgn_block_mean"].to_numpy()
                ax.scatter(x, error, s=14, color=COLORS[label], label=label, alpha=0.85)
                notes.append(f"{label}: ρ = {spearmanr(x, error).statistic:+.2f}")
            ax.set_title(f"torre de {ES[tower]}: " + ", ".join(notes), loc="left", fontsize=8)
            ax.set_xlabel(name)
            ax.set_ylabel("error del vCDR de la capa suelta" if col == 0 else "")
    axes[0, 0].legend(frameon=False, title="receta")
    fig.suptitle("Predictores de capa frente al error medido (61 capas por receta; ρ = Spearman, validación)", x=0.01, ha="left")
    save(fig, "predictores")


def probe_statistics(draws=1000):
    """Spearman del vCDR y AUC de la decisión de cada sonda, por capa: estimación puntual y distribución por remuestreo de bloques de captura."""
    pred = np.load("results/probes/chaksu_json_v1_sondas_predicciones.npz")
    ids = [str(i) for i in pred["ids"]]
    targets = pd.read_csv("data/targets/chaksu__json_v1.csv").set_index("image_id").loc[ids]
    groups, _ = chaksu_groups(read_manifest("chaksu"))
    block = np.array([groups[i] for i in ids])
    gt, suspect = targets.vcdr.to_numpy(), (targets.decision == "GLAUCOMA SUSPECT").to_numpy()
    members = [np.flatnonzero(block == b) for b in np.unique(block)]
    rng = np.random.default_rng(0)
    samples = [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(draws)]

    def rho(x, index):
        return np.corrcoef(rankdata(x[index]), rankdata(gt[index]))[0, 1]

    def auc(x, index):
        positive = suspect[index]
        n1, n0 = positive.sum(), (~positive).sum()
        return (rankdata(x[index])[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan

    everyone = np.arange(len(ids))
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for family in ("vision", "image_tokens", "last_token"):
            for name, key, statistic in (("rho", "vcdr", rho), ("auc", "decision", auc), ("control", "control", rho)):
                values = pred[f"{family}__{key}"]
                out[family, name] = (np.array([statistic(v, everyone) for v in values]),
                                     np.array([[statistic(v, s_) for s_ in samples] for v in values]))
    return out


def probes_loss():
    stats = probe_statistics()
    csv = pd.read_csv("results/probes/chaksu_json_v1_sondas.csv")
    for family in ("vision", "image_tokens", "last_token"):
        assert np.allclose(stats[family, "rho"][0], csv[csv.family == family].vcdr_spearman.to_numpy(), equal_nan=True, atol=1e-6), f"{family}: la sonda no reproduce el CSV"
    offset = {"vision": 0, "image_tokens": 28, "last_token": 28}
    label = {"vision": "torre de visión (promedio de 4 096 parches)", "image_tokens": "torre de lenguaje, promedio de los tokens de imagen",
             "last_token": "torre de lenguaje, último token del prompt"}
    color = {"vision": TOWER_COLORS["vision"], "image_tokens": TOWER_COLORS["language"], "last_token": "#E69F00"}
    reference = stats["vision", "rho"][1][27]  # salida de la torre de visión: lo que recibe el conector
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for family in color:
        n = len(stats[family, "rho"][0])
        x = offset[family] + np.arange(n)
        for ax, name in zip(axes[:2], ("rho", "auc")):
            point, dist = stats[family, name]
            low, high = np.nanpercentile(dist, [2.5, 97.5], axis=1)
            ax.fill_between(x, low, high, color=color[family], alpha=0.18, lw=0)
            ax.plot(x, point, color=color[family], lw=1.6, label=label[family] if name == "rho" else None)
        axes[0].scatter(x, stats[family, "control"][0], s=6, color="#888888", zorder=2, label="control: etiquetas de entrenamiento permutadas (una permutación)" if family == "vision" else None)
        delta = stats[family, "rho"][1] - reference
        axes[2].fill_between(x, *np.nanpercentile(delta, [2.5, 97.5], axis=1), color=color[family], alpha=0.18, lw=0)
        axes[2].plot(x, stats[family, "rho"][0] - stats["vision", "rho"][0][27], color=color[family], lw=1.6)
    axes[1].axhline(0.5, color="#888888", ls="--", lw=1)
    axes[2].axhline(0, color="black", lw=1)
    for ax in axes:
        ax.axvline(27.5, color="black", ls=":", lw=1)
    axes[0].text(27.7, 0.02, "conector", fontsize=8, va="bottom")
    axes[0].set_ylabel("Spearman del vCDR (validación)")
    axes[1].set_ylabel("AUC de la decisión (sospechoso)")
    axes[2].set_ylabel("Spearman − el de la salida de la visión")
    ticks = list(range(0, 28, 3)) + [28 + k for k in range(0, 35, 5)]
    axes[2].set_xticks(ticks, [str(t) for t in range(0, 28, 3)] + [str(k) for k in range(0, 35, 5)])
    axes[2].set_xlabel("capa (izquierda: bloques de la torre de visión; derecha: bloques de la torre de lenguaje; 0 = embeddings)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Señal de vCDR y de la decisión legible en cada punto del modelo base (sondas lineales; bandas: IC 95 % por bloques de captura)", x=0.01, ha="left", fontsize=10)
    save(fig, "sondas_perdida")


def prompts():
    constant = training_medians("data/targets/chaksu__json_v1.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
    names = [p.stem.removeprefix("chaksu_A_") for p in sorted(PROMPTS.glob("chaksu_A_P*.txt"))]
    names += [n for n in ("P4_t1024",) if (PROMPT_RUNS / n / "validation" / "generate_config.json").exists()]
    metrics = {n: run_metrics(PROMPT_RUNS / n, constant)[0] for n in names}
    t = table("json_v1", "validation")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(names))[::-1]
    legible = np.array([metrics[n]["format"] >= 0.5 for n in names])
    ax.scatter(np.array([metrics[n]["vcdr_error"] for n in names])[legible], y[legible], color="#0072B2", zorder=3)
    ax.scatter(np.array([metrics[n]["vcdr_error"] for n in names])[~legible], y[~legible], facecolors="none", edgecolors="#0072B2", zorder=3,
               label="respuestas ilegibles (cuenta como la constante)")
    ax.axvline(t.loc["all-linear", "vcdr_error_mean"], color="black", lw=1, label="all-linear (LoRA)")
    ax.axvline(t.loc["constant", "vcdr_error_mean"], color="#777777", ls="--", lw=1, label="predictor constante")
    ax.set_yticks(y, [f"{n}  (formato {metrics[n]['format']:.2f})" for n in names])
    ax.set_xlabel("error absoluto medio del vCDR en validación")
    ax.legend(frameon=False)
    ax.set_title("Modelo base con cada prompt (receta A)", loc="left")
    save(fig, "prompts")


def camera_legend(ax, **kwargs):
    for name, (color, marker) in CAMERAS.items():
        ax.scatter([], [], s=18, color=color, marker=marker, label=name)
    ax.legend(frameon=False, fontsize=7, title="cámara", title_fontsize=7, **kwargs)


def validation_predictions(folder, mode="real", experiment="chaksu_json_v1"):
    """Una fila por imagen de validación: vCDR de referencia, vCDR que dice el modelo (la constante de entrenamiento si no es legible),
    su error absoluto y la cámara. `folder` es la carpeta de la corrida bajo `experiments/<experiment>/`."""
    rows = load(EXPERIMENTS / experiment / folder / "validation" / f"pred_{mode}.jsonl")
    fields = answer_fields(rows)
    reads = [read_answer(r, fields) for r in rows]
    constant = training_medians(TARGETS, SPLITS, ["vcdr"])["vcdr"]
    said = np.array([v["vcdr"] if v["vcdr"] is not None else constant for _, v in reads], dtype=float)
    ref = np.array([float(r["vcdr"]) for r in rows])
    out = pd.DataFrame({"ref": ref, "pred": said, "error": errors(rows, reads, "vcdr", constant), "camera": [r["camera"] for r in rows]},
                       index=[r["id"] for r in rows])
    assert len(out) == 171 and out.index.is_unique
    return out


def box(ax, x, y, w, h, text, color, face=None, bold=False, size=7):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=face or "white", ec=color, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, x0, x1, y, color):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), zorder=1)


def configurations():
    """Las tres configuraciones del estudio. Esquema sin datos: qué entra y qué sale de cada modelo."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    image, lora, text = "#F0F0F0", "#E8F1F8", "#FFFFFF"
    h = 14
    # A
    y = 80
    ax.text(0, y + 21, "A  Imagen y prompt → JSON con los campos", fontsize=8.5, fontweight="bold", color=COLORS["A"])
    box(ax, 0, y, 20, h, "fotografía de\nfondo de ojo\n+ prompt", "#555555", image)
    box(ax, 30, y, 24, h, "MedGemma 1.5 4B\n+ LoRA all-linear", COLORS["A"], lora, True)
    box(ax, 64, y, 36, h, '{"vcdr": 0.52,\n "hcdr": 0.47,\n "decision": "NORMAL"}', COLORS["A"], text)
    arrow(ax, 20, 30, y + h / 2, COLORS["A"])
    arrow(ax, 54, 64, y + h / 2, COLORS["A"])
    # B
    y = 52
    ax.text(0, y + 21, "B  Imagen y prompt → prosa clínica", fontsize=8.5, fontweight="bold", color=COLORS["B"])
    box(ax, 0, y, 20, h, "fotografía de\nfondo de ojo\n+ prompt", "#555555", image)
    box(ax, 30, y, 24, h, "MedGemma 1.5 4B\n+ LoRA (misma capa\nque en A)", COLORS["B"], "#FBEBDD", True)
    box(ax, 64, y, 36, h, "«Optic nerve head: vertical\ncup-to-disc ratio 0.48, horizontal\n0.51, disc not suspicious…»", COLORS["B"], text, size=6.5)
    arrow(ax, 20, 30, y + h / 2, COLORS["B"])
    arrow(ax, 54, 64, y + h / 2, COLORS["B"])
    # C
    y = 8
    ax.text(0, y + 36, "C  Dos etapas: campos medidos en la imagen, redacción solo con los campos", fontsize=8.5, fontweight="bold", color=COLORS["C"])
    box(ax, 0, y + 14, 15, h, "fotografía\n+ prompt de A", "#555555", image, size=6.5)
    box(ax, 21, y + 14, 21, h, "Etapa 1\nadaptador all-linear\nde A", COLORS["A"], lora, True, 6.5)
    box(ax, 48, y + 14, 16, h, '{vcdr, hcdr,\n decision}', COLORS["A"], text, size=6.5)
    box(ax, 70, y + 14, 30, h, "Etapa 2 (solo texto):\nredacta los campos;\nmodelo base o LoRA", COLORS["C"], "#F1E8F5", True, 6.5)
    arrow(ax, 15, 21, y + 21, COLORS["C"])
    arrow(ax, 42, 48, y + 21, COLORS["C"])
    arrow(ax, 64, 70, y + 21, COLORS["C"])
    box(ax, 70, y - 8, 30, 13, "prosa con los campos\nde la etapa 1", COLORS["C"], text, size=6.5)
    ax.annotate("", xy=(85, y + 5), xytext=(85, y + 14), arrowprops=dict(arrowstyle="-|>", color=COLORS["C"], lw=1.4))
    ax.text(0, y - 4, "Variante de la etapa 2: recibe también la fotografía, además de los campos.", fontsize=6.5, color="#444444", va="center")
    save(fig, "configuraciones")


def split_cameras():
    """Imágenes de cada partición por cámara y prevalencia de «sospechoso»: lo que cada evaluación puede y no puede decir."""
    split = pd.read_csv(SPLITS).set_index("image_id")
    t = pd.read_csv(TARGETS).set_index("image_id").join(split)
    order = ["train", "validation", "excluded", "test"]
    names = {"train": "entrenamiento", "validation": "validación", "excluded": "excluidas", "test": "Test"}
    counts = pd.crosstab(t.split, t.camera).loc[order]
    assert counts.sum(axis=1).tolist() == [780, 171, 58, 336]
    prevalence = t.groupby("split").decision.apply(lambda d: (d == "GLAUCOMA SUSPECT").mean()).loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bottom = np.zeros(len(order))
    for camera, (color, _) in CAMERAS.items():
        ax.bar(range(len(order)), counts[camera], bottom=bottom, color=color, width=0.6, label=camera, edgecolor="white", lw=1)
        bottom += counts[camera].to_numpy()
    for i, name in enumerate(order):
        ax.text(i, bottom[i] + 12, f"n = {int(bottom[i])}\nsospechoso: {100 * prevalence[name]:.1f} %", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(order)), [names[o] for o in order])
    ax.set_ylabel("imágenes")
    ax.set_ylim(0, 960)
    ax.legend(frameon=False, title="cámara", loc="upper right", title_fontsize=7, fontsize=7.5)
    ax.set_title("Chákṣu: partición por bloques de captura, por cámara", loc="left")
    save(fig, "particion_camaras")


def expert_agreement():
    """Dónde discrepan los cinco expertos en el vCDR: la regla con la que se mide al modelo."""
    a = pd.read_csv("results/audit/chaksu.csv").set_index("image_id")
    camera = a.index.str.split("_").str[0].str.capitalize()
    e = a[[f"vcdr_e{i}" for i in range(1, 6)]].to_numpy()
    complete = ~np.isnan(e).any(axis=1)
    gap = np.concatenate([np.abs(e[complete, i] - np.median(np.delete(e[complete], i, axis=1), axis=1)) for i in range(5)])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), layout="constrained")
    ax = axes[0]
    ax.hist(gap, bins=np.arange(0, 0.45, 0.01), color="#8C8C8C", edgecolor="white", lw=0.5)
    ax.axvline(VCDR_TOLERANCE, color="black", ls="--", lw=1)
    ax.text(VCDR_TOLERANCE + 0.008, ax.get_ylim()[1] * 0.93, f"tolerancia {VCDR_TOLERANCE}\n({100 * (gap <= VCDR_TOLERANCE).mean():.0f} % de los casos\nla cumple)", fontsize=7, va="top")
    ax.set_xlabel("|vCDR de un experto − mediana de los otros cuatro|")
    ax.set_ylabel(f"pares experto-imagen (n = {len(gap)})")
    ax.set_title(f"(a) mediana {np.median(gap):.3f}", loc="left")
    ax = axes[1]
    for name, (color, marker) in CAMERAS.items():
        m = np.asarray(camera == name) & ~np.isnan(e[:, 0]) & ~np.isnan(e[:, 1])
        ax.scatter(e[m, 0], e[m, 1], s=7, color=color, marker=marker, alpha=0.6, lw=0)
    ok = ~np.isnan(e[:, 0]) & ~np.isnan(e[:, 1])
    ax.plot([0, 1], [0, 1], color="black", lw=0.8)
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.1, 0.95)
    ax.set_aspect("equal")
    ax.set_xlabel("vCDR del experto 1")
    ax.set_ylabel("vCDR del experto 2")
    ax.set_title(f"(b) ρ de Spearman = {spearmanr(e[ok, 0], e[ok, 1]).statistic:.2f}", loc="left")
    camera_legend(ax, loc="lower right", markerscale=1.6)
    fig.suptitle("Acuerdo entre los cinco expertos de Chákṣu sobre el vCDR (1 345 imágenes)", x=0.01, ha="left", fontsize=9)
    save(fig, "acuerdo_expertos")


def prediction_vs_reference():
    """vCDR que dice el modelo frente al de referencia en validación: base y all-linear (media de las tres semillas por imagen)."""
    base = validation_predictions("base")
    seeds = [validation_predictions(f"all-linear_s{s}") for s in (42, 43, 44)]
    tuned = seeds[0].copy()
    tuned["pred"] = np.mean([s.pred for s in seeds], axis=0)
    assert all((s.ref == tuned.ref).all() for s in seeds)
    mae_tuned = np.mean([s.error.mean() for s in seeds])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5), sharex=True, sharey=True, layout="constrained")
    for ax, (title, d, mae) in zip(axes, (("Modelo base", base, base.error.mean()), ("all-linear (LoRA)", tuned, mae_tuned))):
        lim = np.array([0.1, 0.95])
        ax.fill_between(lim, lim - VCDR_TOLERANCE, lim + VCDR_TOLERANCE, color="#DDDDDD", alpha=0.7, lw=0, label=f"±{VCDR_TOLERANCE} (tolerancia)")
        ax.plot(lim, lim, color="black", lw=0.8, label="identidad")
        for name, (color, marker) in CAMERAS.items():
            m = d.camera == name
            ax.scatter(d.ref[m], d.pred[m], s=9, color=color, marker=marker, alpha=0.75, lw=0, label=name)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("vCDR de referencia (mayoría de expertos)")
        ax.text(0.03, 0.97, f"error absoluto medio {mae:.4f}\nρ de Spearman {spearmanr(d.ref, d.pred).statistic:+.2f}", transform=ax.transAxes, va="top", fontsize=7.5)
    axes[0].set_ylabel("vCDR dicho por el modelo")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right", markerscale=1.5)
    fig.suptitle("Validación (171 imágenes): el vCDR dicho frente al de referencia; all-linear: media de 3 semillas por imagen, error medio de las 3", x=0.01, ha="left", fontsize=8)
    save(fig, "prediccion_vs_referencia")


def image_controls():
    """Error del vCDR con la imagen correcta, con la de otro paciente (permutación fija) y en blanco."""
    constant = table("json_v1", "validation").loc["constant", "vcdr_error_mean"]
    modes = (("real", "imagen correcta", "", 1.0), ("shuffled", "imagen de otro paciente", "///", 0.65), ("blank", "imagen en blanco", "...", 0.35))
    models = (("base", "modelo base\n(una corrida)", COLORS["base"], ["base"]), ("all-linear", "all-linear\n(3 semillas)", COLORS["A"], [f"all-linear_s{s}" for s in (42, 43, 44)]))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    width = 0.24
    for i, (_, label, color, folders) in enumerate(models):
        for j, (mode, _, hatch, alpha) in enumerate(modes):
            values = np.array([validation_predictions(f, mode).error.mean() for f in folders])
            err = values.std(ddof=1) if len(values) > 1 else 0
            ax.bar(i + (j - 1) * width, values.mean(), width * 0.92, yerr=err, color=color, alpha=alpha, hatch=hatch, edgecolor="white", error_kw=dict(lw=1, capsize=2))
            ax.text(i + (j - 1) * width, values.mean() + err + 0.0008, f"{values.mean():.3f}", ha="center", fontsize=7)
    ax.axhline(constant, color=COLORS["constant"], ls="--", lw=1, label="predictor constante")
    ax.set_xticks(range(2), [m[1] for m in models])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("error absoluto medio del vCDR")
    ax.set_ylim(0, 0.085)
    for _, label, hatch, alpha in modes:
        ax.bar(0, 0, color="#555555", alpha=alpha, hatch=hatch, edgecolor="white", label=label)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    ax.set_title("El error depende de la imagen en all-linear, no en el modelo base", loc="left", fontsize=9)
    save(fig, "controles_imagen")


def probe_depth():
    """Sondas lineales sobre el modelo base: cuánto vCDR y decisión se pueden leer en cada punto de la red."""
    p = pd.read_csv("results/probes/chaksu_json_v1_sondas.csv")
    families = {"vision": ("torre de visión (promedio de parches)", TOWER_COLORS["vision"]),
                "image_tokens": ("torre de lenguaje, tokens de imagen", TOWER_COLORS["language"]),
                "last_token": ("torre de lenguaje, último token", "#E69F00")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), layout="constrained")
    for family_, (label, color) in families.items():
        g = p[p.family == family_].sort_values("layer")
        depth = g.layer / g.layer.max()
        axes[0].plot(depth, g.vcdr_spearman, color=color, lw=1.6, label=label)
        axes[0].plot(depth, g.vcdr_control, color=color, lw=0.9, ls=":")
        axes[1].plot(depth, g.decision_auc, color=color, lw=1.6)
        best = g.vcdr_spearman.idxmax()
        axes[0].scatter(g.loc[best, "layer"] / g.layer.max(), g.loc[best, "vcdr_spearman"], color=color, s=22, zorder=3)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[1].axhline(0.5, color="black", ls="--", lw=0.8)
    axes[1].text(0.02, 0.51, "azar", fontsize=7)
    axes[0].plot([], [], color="#555555", ls=":", lw=0.9, label="control: etiquetas permutadas")
    axes[0].set_ylabel("ρ de Spearman del vCDR (validación)")
    axes[1].set_ylabel("AUC de la decisión (validación)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=2)
    for ax, t in zip(axes, ("(a) vCDR", "(b) decisión")):
        ax.set_xlabel("profundidad relativa")
        ax.set_title(t, loc="left")
    fig.suptitle("Sonda lineal ajustada en cada punto de lectura del modelo base (círculo: mejor capa; 0 = embeddings, 1 = última capa)", x=0.01, ha="left", fontsize=9)
    save(fig, "sondas_profundidad")


def rgn_blocks():
    """Norma de gradiente relativa de cada bloque del modelo base: dónde se movería más cada tensor con un paso de gradiente."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True, layout="constrained")
    for ax, (label, recipe) in zip(axes, RECIPES.items()):
        r = pd.read_csv(f"results/rgn/chaksu_{recipe}.csv")
        for tower in ("vision", "language"):
            g = r[r.tower == tower].sort_values("layer")
            assert len(g) == (27 if tower == "vision" else 34)
            x = g.layer / g.layer.max()
            ax.plot(x, g.rgn_block_mean, color=TOWER_COLORS[tower], lw=1.6, marker="o", ms=2.5, label=f"torre de {ES[tower]} ({len(g)} bloques)")
        ax.set_yscale("log")
        ax.set_xlabel("profundidad relativa del bloque")
        ax.set_title(f"{label} ({recipe})", loc="left", color=COLORS[label])
    axes[0].set_ylabel("RGN del bloque (escala log)")
    axes[0].legend(frameon=False, fontsize=7, loc="center left")
    fig.suptitle("Norma de gradiente relativa RGN = ‖gradiente‖ / ‖pesos‖ por bloque, modelo base (media entre pasos; 0 = primer bloque, 1 = último)", x=0.01, ha="left", fontsize=8)
    save(fig, "rgn_bloques")


def random_draws():
    """El bloque frente a cuatro sorteos aleatorios del mismo presupuesto: ¿el lugar importa más que el azar del sorteo?"""
    main_t = table("json_v1", "validation")
    extra = pd.read_csv(GRIDS / "chaksu_json_v1" / "results_sorteos.csv").set_index("cell")
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    reference = main_t.loc["all-linear", "vcdr_error_mean"]
    ax.axhspan(reference - RELEVANT, reference + RELEVANT, color="#DDDDDD", alpha=0.7, lw=0, label=f"all-linear ± {RELEVANT}")
    ax.axhline(reference, color="black", lw=1, label="all-linear")
    ax.axhline(main_t.loc["constant", "vcdr_error_mean"], color=COLORS["constant"], ls="--", lw=1, label="predictor constante")
    rng = np.random.default_rng(0)
    for i, tower in enumerate(("vision", "language")):
        draws = [main_t.loc[f"random_block_{tower}", "vcdr_error_mean"]] + [extra.loc[f"random_block_{tower}_{k}", "vcdr_error_mean"] for k in (1, 2, 3)]
        assert len(draws) == 4
        ax.scatter(i + rng.uniform(-0.08, 0.08, 4) + 0.18, draws, s=26, color=TOWER_COLORS["random"], marker="D", zorder=3, label="sorteo aleatorio del mismo presupuesto" if i == 0 else None)
        ax.scatter(i - 0.18, main_t.loc[f"block_{tower}", "vcdr_error_mean"], s=50, color=TOWER_COLORS[tower], zorder=3, label=f"bloque de {ES[tower]}" if i == 0 else "bloque de lenguaje")
        budget = main_t.loc[f"block_{tower}", "trainable_params"] / 1e6
        ax.text(i, 0.0345, f"≈ {budget:.1f} M parámetros\nentrenables", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1], ["presupuesto de un bloque\nde visión", "presupuesto de un bloque\nde lenguaje"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.033, 0.0605)
    ax.set_ylabel("error absoluto medio del vCDR (validación)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("A: bloque frente a cuatro sorteos (el primero, media de 3 semillas;\nlos otros tres, semilla 42)", loc="left", fontsize=8.5)
    save(fig, "sorteos_control")


def external_validation():
    """AUC en MM-ODIR-129 del vCDR predicho como puntuación de «patológico»: cuánto transfiere cada ubicación a otro dataset."""
    e = pd.read_csv("results/audit/mmodir129_externa.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.6), sharey=True, layout="constrained")
    for ax, label in zip(axes, RECIPES):
        d = e[e.modo == label].reset_index(drop=True)
        assert len(d) == 15 and d.celda.iloc[-1] == "base"
        y = np.arange(len(d))[::-1]
        isbase = (d.celda == "base").to_numpy()
        ax.axvline(0.5, color="#777777", ls="--", lw=1)
        ax.hlines(y, d.auc_etiqueta_low, d.auc_etiqueta_high, color=np.where(isbase, COLORS["base"], COLORS[label]), lw=1.3)
        ax.scatter(d.auc_etiqueta, y, s=20, color=np.where(isbase, COLORS["base"], COLORS[label]), zorder=3)
        ax.set_yticks(y, ["modelo base" if c == "base" else c for c in d.celda], fontsize=7)
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("AUC (IC 95 %, pacientes)")
        ax.set_title(f"{label} ({RECIPES[label]})", loc="left", color=COLORS[label])
        ax.text(0.5, y[-1] - 1.1, "azar", fontsize=7, ha="center", color="#555555")
    fig.suptitle("Validación externa en MM-ODIR-129 (129 imágenes; media de 3 semillas)", x=0.01, ha="left", fontsize=9)
    save(fig, "validacion_externa")


def parse_stage2():
    """Lee de los informes de texto el error del vCDR, la fidelidad (por semilla) y el BLEU-4 de B y de las cuatro variantes de la etapa 2."""
    img = Path("results/audit/chaksu_C_img.txt").read_text(encoding="utf-8")
    surface = Path("results/audit/chaksu_superficie_C.txt").read_text(encoding="utf-8")
    out = {}
    for model, body in re.findall(r"=== Etapa 2 (base|lora)\n(.*?)(?==== |\Z)", img, flags=re.S):
        fidelity = re.search(r"\(i\) fidelidad .*?con imagen \[([^\]]+)\], solo texto \[([^\]]+)\]", body)
        error = re.search(r"\(ii\) error del vCDR: con imagen ([\d.]+), solo texto ([\d.]+)", body)
        assert fidelity and error
        for key, k in (("con imagen", 1), ("solo texto", 2)):
            out[model, key] = {"fidelity": [float(v) for v in fidelity.group(k).split(",")], "error": float(error.group(k))}
    assert len(out) == 4
    headers = {"B, all-linear": "B", "C, etapa 2 base, solo texto": ("base", "solo texto"), "C, etapa 2 LoRA, solo texto": ("lora", "solo texto"),
               "C, etapa 2 base, con imagen": ("base", "con imagen"), "C, etapa 2 LoRA, con imagen": ("lora", "con imagen")}
    current, bleu, error_b = None, {}, None
    for line in surface.splitlines():
        h = re.match(r"(.+?): error del vCDR ([\d.]+)", line)
        if h:
            assert h.group(1) in headers, h.group(1)
            current = headers[h.group(1)]
            if current == "B":
                error_b = float(h.group(2))
            continue
        b = re.match(r"\s+BLEU-4\s+([\d.]+)", line)
        if b:
            bleu[current] = float(b.group(1))
    assert len(bleu) == 5 and error_b is not None
    for (model, key), v in out.items():
        v["bleu"] = bleu[model, key]
    return out, {"error": error_b, "bleu": bleu["B"]}


def stage2():
    """Configuración C: la etapa 2 (redacción) con modelo base o LoRA, con y sin imagen, frente a B."""
    c, b = parse_stage2()
    arms = [("B", None), ("base", "solo texto"), ("lora", "solo texto"), ("base", "con imagen"), ("lora", "con imagen")]
    labels = ["B", "base\ntexto", "LoRA\ntexto", "base\nimagen", "LoRA\nimagen"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.2), layout="constrained")
    for i, (model, key) in enumerate(arms):
        color = COLORS["B"] if model == "B" else COLORS["C"]
        alpha = 0.4 if model == "base" else 1.0
        hatch = "///" if key == "con imagen" else ""
        values = (b["error"], None, b["bleu"]) if model == "B" else (c[model, key]["error"], c[model, key]["fidelity"], c[model, key]["bleu"])
        for ax, v in zip(axes, values):
            if v is None:
                continue
            if isinstance(v, list):
                ax.bar(i, np.mean(v), color=color, alpha=alpha, hatch=hatch, edgecolor="white", width=0.7)
                ax.scatter([i] * len(v), v, s=8, color="black", zorder=3)
            else:
                ax.bar(i, v, color=color, alpha=alpha, hatch=hatch, edgecolor="white", width=0.7)
                if ax is not axes[0]:
                    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("error absoluto medio del vCDR")
    axes[0].set_ylim(0, 0.05)
    axes[0].text(0, b["error"] + 0.0006, f"{b['error']:.4f}", ha="center", fontsize=6.5)
    axes[0].text(2.5, 0.0415, f"{c['base', 'solo texto']['error']:.4f} en las\n4 variantes de C", ha="center", fontsize=6.5)
    axes[1].set_ylabel("fidelidad a los campos de la etapa 1\n(puntos: 3 semillas)")
    axes[1].set_ylim(0.85, 1.01)
    axes[1].text(0, 0.93, "sin\netapa 2", ha="center", fontsize=6.5, color="#555555")
    axes[2].set_ylabel("BLEU-4 contra la frase de referencia")
    axes[2].set_ylim(0, 30)
    for ax, t in zip(axes, ("(a) error del vCDR", "(b) fidelidad", "(c) BLEU-4")):
        ax.set_xticks(range(5), labels, fontsize=6.5)
        ax.set_title(t, loc="left", fontsize=8)
    fig.suptitle("Configuración C frente a B en validación (media de 3 semillas; B: prosa directa de la imagen; texto = solo los campos; imagen = campos y fotografía)", x=0.01, ha="left", fontsize=7)
    save(fig, "etapa2_C")


def surface_vs_error():
    """BLEU-4 y ROUGE-L por imagen de la prosa de B frente al error del vCDR del mismo texto: ¿la métrica de superficie detecta el error de contenido?"""
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu = BLEU(effective_order=True)
    runs = []
    for s in (42, 43, 44):
        rows = load(EXPERIMENTS / "chaksu_prose_v1" / f"all-linear_s{s}" / "validation" / "pred_real.jsonl")
        fields = answer_fields(rows)
        constant = training_medians(TARGETS, SPLITS, ["vcdr"])["vcdr"]
        reads = [read_answer(r, fields) for r in rows]
        runs.append(pd.DataFrame({"error": errors(rows, reads, "vcdr", constant),
                                  "bleu": [bleu.sentence_score(r["prediction"], [r["response"]]).score for r in rows],
                                  "rouge": [scorer.score(r["response"], r["prediction"])["rougeL"].fmeasure for r in rows]}, index=[r["id"] for r in rows]))
    assert all((runs[0].index == r.index).all() for r in runs)
    d = sum(runs) / 3
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), layout="constrained")
    for ax, (column, label) in zip(axes, (("bleu", "BLEU-4 por imagen (frase)"), ("rouge", "ROUGE-L por imagen"))):
        ax.scatter(d.error, d[column], s=10, color=COLORS["B"], alpha=0.6, lw=0)
        ax.set_xlabel("error absoluto del vCDR del texto")
        ax.set_ylabel(label)
        ax.text(0.97, 0.03, f"ρ de Spearman = {spearmanr(d.error, d[column]).statistic:+.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.suptitle("B, all-linear, validación: 171 imágenes (media de 3 semillas por imagen)", x=0.01, ha="left", fontsize=9)
    save(fig, "superficie_vs_error")


def pooling():
    """Sondas sobre la torre de visión con cuatro resúmenes de los parches: ¿el promedio esconde la señal que el máximo o el gradiente conservan?"""
    p = pd.read_csv("results/probes/chaksu_json_v1_pooling.csv")
    assert p.groupby("pooling").layer.nunique().eq(28).all()
    styles = {"avg": ("promedio (el del estudio)", "#000000"), "max": ("máximo", "#009E73"), "gradient": ("gradiente espacial", "#D55E00"),
              "gradient_peak": ("pico del gradiente", "#56B4E9")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), layout="constrained")
    for name, (label, color) in styles.items():
        g = p[p.pooling == name].sort_values("layer")
        axes[0].plot(g.layer, g.vcdr_spearman, color=color, lw=1.5, label=label)
        axes[0].plot(g.layer, g.vcdr_control, color=color, lw=0.8, ls=":")
        axes[1].plot(g.layer, g.decision_auc, color=color, lw=1.5)
    axes[0].plot([], [], color="#555555", ls=":", lw=0.8, label="control (etiquetas permutadas)")
    axes[1].axhline(0.5, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("ρ de Spearman del vCDR (validación)")
    axes[1].set_ylabel("AUC de la decisión (validación)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="outside lower center", ncol=3)
    for ax, t in zip(axes, ("(a) vCDR", "(b) decisión")):
        ax.set_xlabel("bloque de la torre de visión (0 = embeddings)")
        ax.set_title(t, loc="left")
    fig.suptitle("Sondas de la torre de visión según el resumen de los parches", x=0.01, ha="left", fontsize=9)
    save(fig, "pooling_sondas")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", nargs="?", default="all", choices=["all", "pooling"], help="`pooling` dibuja solo pooling_sondas (no está en el conjunto por defecto)")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    if args.figure == "pooling":
        pooling()
        return
    efficiency(args.split)
    if args.split == "test":
        transfer()
    if args.split == "validation":
        layers(args.split)
        predictors()
        probes_loss()
        prompts()
        configurations()
        split_cameras()
        expert_agreement()
        prediction_vs_reference()
        image_controls()
        probe_depth()
        rgn_blocks()
        random_draws()
        external_validation()
        stage2()
        surface_vs_error()


if __name__ == "__main__":
    main()
