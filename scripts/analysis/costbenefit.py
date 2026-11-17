"""Costo-beneficio de cada ubicación de LoRA frente a `all-linear`, con las rejillas A y B.

    python -m scripts.analysis.costbenefit

Para cada celda: parámetros entrenables (y % de all-linear), tiempo de entrenamiento medio (`train_runtime` de
`run_config.json`, media de semillas; y % de all-linear de su misma receta), tamaño del adaptador en disco, error del
vCDR en validación y, para los finalistas, en el Test.

Fracción de la mejora conservada = (error de la constante - error de la celda) / (error de la constante - error de all-linear):
1 es igualar a all-linear, 0 no mejorar al predictor constante, y es negativa si empeora. La constante de cada
partición sale de la fila `constant` de la tabla. Error adicional relativo = error de la celda / error de all-linear - 1.

Escribe `results/audit/costo_beneficio.{txt,csv}` `results/figures/costo_beneficio.{png,pdf}` (tres paneles), `costo_beneficio_articulo.{png,pdf}` (cuerpo del artículo) y `tiempo_parametros.{png,pdf}`.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.analysis.figures import ES, RECIPES, TOWER_COLORS, TOWER_MARKERS, family, save, table

EXPERIMENTS = "experiments"
AUDIT = "results/audit"
SEEDS = (42, 43, 44)
BUDGET = 0.10  # fracción de los parámetros de all-linear que define "ubicación barata"
MIB = 1024 ** 2


def cell_costs(recipe, cell):
    """Tiempo de entrenamiento medio y tamaño del adaptador de una celda, leídos de las carpetas de sus corridas."""
    times, sizes = [], []
    for seed in SEEDS:
        folder = f"{EXPERIMENTS}/chaksu_{recipe}/{cell}_s{seed}"
        times.append(json.load(open(f"{folder}/run_config.json"))["train_metrics"]["train_runtime"])
        sizes.append(os.path.getsize(f"{folder}/adapter_model.safetensors") / MIB)
    assert max(sizes) - min(sizes) < 1e-6, f"{cell}: el adaptador cambia de tamaño entre semillas"
    return np.mean(times), sizes[0]


def build(recipe_label, recipe):
    val, test = table(recipe, "validation"), table(recipe, "test")
    c_val, c_test = val.loc["constant", "vcdr_error_mean"], test.loc["constant", "vcdr_error_mean"]
    t = val.drop(index=["constant", "base"]).copy()
    assert len(t) == 88
    costs = pd.DataFrame({c: cell_costs(recipe, c) for c in t.index}, index=["train_seconds", "adapter_mib"]).T
    t = t.join(costs)
    ref = t.loc["all-linear"]
    out = pd.DataFrame({"recipe": recipe_label, "tower": t.tower, "family": family(t), "params": t.trainable_params})
    out["pct_params"] = 100 * t.trainable_params / ref.trainable_params
    out["train_seconds"], out["pct_time"] = t.train_seconds, 100 * t.train_seconds / ref.train_seconds
    out["adapter_mib"] = t.adapter_mib
    out["val_error"], out["val_diff"] = t.vcdr_error_mean, t.diff_vs_ref
    out["val_ci_low"], out["val_ci_high"] = t.diff_ci_low, t.diff_ci_high
    out["val_kept"] = (c_val - t.vcdr_error_mean) / (c_val - ref.vcdr_error_mean)
    out["val_extra_rel"] = t.vcdr_error_mean / ref.vcdr_error_mean - 1
    out["val_constant"], out["test_constant"] = c_val, c_test
    ref_test = test.loc["all-linear", "vcdr_error_mean"]
    f = test.drop(index=["constant", "base"])
    out["test_error"], out["test_diff"] = f.vcdr_error_mean, f.diff_vs_ref
    out["test_ci_low"], out["test_ci_high"] = f.diff_ci_low, f.diff_ci_high
    out["test_kept"] = (c_test - f.vcdr_error_mean) / (c_test - ref_test)
    out["test_extra_rel"] = f.vcdr_error_mean / ref_test - 1
    return out


def pareto(d, column):
    """Celdas con el menor error de todas las que usan igual o menos parámetros (frontera error-presupuesto)."""
    d = d.dropna(subset=[column]).sort_values(["params", column])
    return d[d[column] < d[column].cummin().shift(fill_value=np.inf)]


def fmt(d, test):
    cols = ["pct_params", "pct_time", "adapter_mib", "val_error", "val_diff", "val_kept", "val_extra_rel"]
    if test:
        cols += ["test_error", "test_diff", "test_kept", "test_extra_rel"]
    out = d[cols].copy()
    out["val_diff"] = [f"{r.val_diff:+.4f} [{r.val_ci_low:+.4f},{r.val_ci_high:+.4f}]" for r in d.itertuples()]
    if test:
        out["test_diff"] = [f"{r.test_diff:+.4f} [{r.test_ci_low:+.4f},{r.test_ci_high:+.4f}]" for r in d.itertuples()]
    d = out
    for c in ("val_kept", "val_extra_rel", "test_kept", "test_extra_rel"):
        if c in d:
            d[c] = (100 * d[c]).map("{:.0f}%".format)
    return d.round({"pct_params": 1, "pct_time": 1, "adapter_mib": 1, "val_error": 4, "test_error": 4}).to_string()


def report(all_cells):
    lines = []
    for label in RECIPES:
        d = all_cells[all_cells.recipe == label].drop(columns="recipe")
        ref = d.loc["all-linear"]
        lines.append(f"{'=' * 100}\nRECETA {label} ({RECIPES[label]}). all-linear: {ref.params:,.0f} parámetros, {ref.train_seconds:.0f} s, {ref.adapter_mib:.1f} MiB; "
                     f"error de validación {ref.val_error:.4f} (constante {ref.val_constant:.4f}), Test {ref.test_error:.4f} (constante {ref.test_constant:.4f})\n")
        cheap = d[d.pct_params <= 100 * BUDGET].sort_values("val_error")
        lines.append(f"(1) Ubicaciones con <= {100 * BUDGET:.0f}% de los parámetros de all-linear ({len(cheap)}), ordenadas por error de validación; "
                     "diff = ubicación - all-linear con IC por bloques de captura; kept = fracción de la mejora conservada\n")
        lines.append(fmt(cheap, test=True) + "\n")
        lines.append(f"(2a) Frontera de Pareto en validación (menor error para cada presupuesto de parámetros)\n")
        lines.append(fmt(pareto(d, "val_error"), test=True) + "\n")
        lines.append("(2b) Frontera de Pareto en el Test (solo los 14 finalistas)\n")
        lines.append(fmt(pareto(d, "test_error"), test=True) + "\n")
        fin = d.dropna(subset=["test_error"]).sort_values("test_kept", ascending=False)
        lines.append("Finalistas ordenados por fracción conservada en el Test\n" + fmt(fin, test=True) + "\n")
        sel = d[d.index != "all-linear"]
        r, rho = pearsonr(sel.pct_params, sel.pct_time)[0], spearmanr(sel.pct_params, sel.pct_time)[0]
        slope, intercept = np.polyfit(sel.pct_params, sel.pct_time, 1)
        small = d[d.pct_params <= 100 * BUDGET]
        lines.append(f"(3) Tiempo frente a parámetros ({len(sel)} celdas sin all-linear): Pearson {r:.3f}, Spearman {rho:.3f}; ajuste lineal "
                     f"%tiempo = {intercept:.1f} + {slope:.3f} x %parámetros. Celdas con <= {100 * BUDGET:.0f}% de parámetros: tiempo "
                     f"{small.pct_time.min():.1f}% a {small.pct_time.max():.1f}% (mediana {small.pct_time.median():.1f}%) del de all-linear. "
                     f"Rango global del tiempo: {d.pct_time.min():.1f}% a {d.pct_time.max():.1f}%.\n")
        mib = d.loc[["all-linear", "block_language_layer00"], "adapter_mib"]
        lines.append(f"Disco: all-linear {mib['all-linear']:.1f} MiB; una capa de lenguaje {mib['block_language_layer00']:.1f} MiB "
                     f"({100 * mib['block_language_layer00'] / mib['all-linear']:.1f}%). Los pesos se guardan en fp32: MiB = 4 bytes x parámetros / 2^20.\n")
    return "\n".join(lines)


def figure(all_cells):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    highlight = ["block_language_layer00", "block_language_layer01", "all-linear"]
    for ax, label in zip(axes[:2], RECIPES):
        d = all_cells[all_cells.recipe == label]
        ref, c = d.loc["all-linear", "val_error"], d.val_constant.iloc[0]
        for fam, g in d.groupby("family"):
            ax.scatter(g.params / 1e6, g.val_error, s=18, color=TOWER_COLORS[fam], marker=TOWER_MARKERS[fam], label=ES[fam], alpha=0.85, zorder=3)
        front = pareto(d, "val_error")
        front = pd.concat([front, d.loc[["all-linear"]]]).drop_duplicates()
        ax.step(front.params / 1e6, front.val_error, where="post", color="#555555", lw=1, zorder=2, label="frontera de Pareto")
        ax.axhline(ref, color="black", lw=1)
        ax.axhline(c, color="#777777", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(ref - 0.004, 0.065 if label == "A" else 0.09)
        for k, cell in enumerate(highlight[:2]):
            ax.annotate(f"{cell.replace('block_language_layer', 'lenguaje, capa ')}: {100 * d.loc[cell, 'val_kept']:.0f}% conservado", (d.loc[cell, "params"] / 1e6, d.loc[cell, "val_error"]),
                        xytext=(0.4, 0.8 - 0.08 * k), textcoords="axes fraction", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))
        ax.set_xlabel("parámetros entrenables (millones, escala logarítmica)")
        ax.set_title(f"{label} ({RECIPES[label]}): error en validación", loc="left")
        if label == "A":
            ax.set_ylabel("error absoluto medio del vCDR")
        sec = ax.secondary_yaxis("right", functions=(lambda e, c=c, r=ref: 100 * (c - e) / (c - r), lambda k, c=c, r=ref: c - k / 100 * (c - r)))
        sec.set_ylabel("mejora conservada frente a all-linear (%)" if label == "B" else "")
        ax = axes[2]
    for label, marker in zip(RECIPES, ("o", "^")):
        d = all_cells[all_cells.recipe == label]
        for fam, g in d.groupby("family"):
            ax.scatter(g.pct_params, g.pct_time, s=16, color=TOWER_COLORS[fam], marker=marker, alpha=0.8, label=f"{label}" if fam == "vision" else None)
    ax.plot([0.3, 100], [0.3, 100], color="#999999", ls=":", lw=1)
    ax.set_xlabel("parámetros entrenables (% de all-linear)")
    ax.set_ylabel("tiempo de entrenamiento (% de all-linear)")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.1))
    ax.set_title("Tiempo frente a parámetros (círculo: A, triángulo: B)", loc="left", fontsize=9)
    ax.text(0.03, 0.03, "línea punteada: tiempo proporcional a parámetros", transform=ax.transAxes, ha="left", fontsize=7, color="#666666")
    fig.suptitle("Costo-beneficio de cada ubicación frente a all-linear (línea negra); línea discontinua: predictor constante", x=0.01, ha="left", fontsize=9)
    save(fig, "costo_beneficio")


def figure_article(all_cells):
    """Versión para el cuerpo del artículo: error frente a parámetros (receta A, validación), una sola figura a ancho de columna,
    con la fracción conservada anotada solo en ubicaciones destacadas; y el tiempo frente a parámetros como figura aparte."""
    d = all_cells[all_cells.recipe == "A"]
    ref, c = d.loc["all-linear", "val_error"], d.val_constant.iloc[0]
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5}):
        fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
        for fam, g in d.groupby("family"):
            ax.scatter(g.params / 1e6, g.val_error, s=22, color=TOWER_COLORS[fam], marker=TOWER_MARKERS[fam], alpha=0.85, zorder=3,
                       label={"vision": "visión", "language": "lenguaje", "all-linear": "all-linear", "random": "control aleatorio"}[fam])
        ax.axhline(ref, color="black", lw=1)
        ax.axhline(c, color="#777777", ls="--", lw=1)
        ax.text(3.0, c + 0.0004, "predictor constante", fontsize=7.5, color="#555555")
        ax.text(0.31, ref + 0.0004, "all-linear", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_ylim(0.0345, 0.063)
        marks = {"block_language_layer01": ("capa 1 de lenguaje", (0.17, 0.45)), "mlp_vision": ("MLP de visión", (0.36, 0.05)),
                 "block_vision": ("bloque de visión", (0.60, 0.05)), "random_block_vision": ("control aleatorio", (0.58, 0.37)),
                 "block_language": ("bloque de lenguaje", (0.76, 0.50))}
        for cell, (name, pos) in marks.items():
            r = d.loc[cell]
            ax.annotate(f"{name}: {100 * r.val_kept:.0f}\u2009%", (r.params / 1e6, r.val_error), xytext=pos, textcoords="axes fraction", fontsize=7.5,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))
        ax.set_xlabel("parámetros entrenables (millones, escala logarítmica)")
        ax.set_ylabel("error del vCDR en validación (receta A)")
        ax.legend(frameon=False, loc="upper right", ncol=2)
        save(fig, "costo_beneficio_articulo")
        fig, ax = plt.subplots(figsize=(3.4, 2.8), layout="constrained")
        for label, marker in zip(RECIPES, ("o", "^")):
            g = all_cells[all_cells.recipe == label]
            for fam, h in g.groupby("family"):
                ax.scatter(h.pct_params, h.pct_time, s=14, color=TOWER_COLORS[fam], marker=marker, alpha=0.8)
        ax.plot([0.3, 100], [0.3, 100], color="#999999", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.set_xlabel("parámetros (% de all-linear)")
        ax.set_ylabel("tiempo de entrenamiento (% de all-linear)")
        save(fig, "tiempo_parametros")


def main():
    all_cells = pd.concat([build(label, recipe) for label, recipe in RECIPES.items()])
    all_cells.index.name = "cell"
    all_cells.round(6).to_csv(f"{AUDIT}/costo_beneficio.csv")
    text = report(all_cells)
    open(f"{AUDIT}/costo_beneficio.txt", "w").write(text)
    print(text)
    figure(all_cells)
    figure_article(all_cells)


if __name__ == "__main__":
    main()
