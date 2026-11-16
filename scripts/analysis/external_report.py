"""Informe de la validación externa en MM-ODIR-129 de todos los modos de evaluación, con los criterios de evaluación fijados de antemano.

    python -m scripts.analysis.external_report

Lee las predicciones externas (`<corrida>/external/pred_*.jsonl`, y `experiments/chaksu_C/ext_*`) y las de Chákṣu (Test para A y B,
validación para la cadena de razonamiento, los prompts y C, que no se corrieron en el Test) con `scripts.eval.normalize`, y escribe
`results/audit/mmodir129_externa.csv` y `.txt`.

Métricas externas (una respuesta no legible cuenta como la constante de entrenamiento, como en el resto del trabajo):
  auc_etiqueta   AUC del vCDR predicho como puntuación de «patológico» (129 imágenes)
  rho_escrito    Spearman del vCDR predicho con el cociente copa-disco que escribe la transcripción (las que lo traen)
  rho_mascara    Spearman del vCDR predicho con el vCDR de las máscaras (69 patológicas); mae_mascara su error absoluto medio
  sens, espec    de la decisión predicha frente a la etiqueta
Análogos en Chákṣu: auc_decision (vCDR predicho frente a la decisión por mayoría de los expertos) y rho_gt (Spearman con el vCDR de referencia).
Cada métrica es la media de las tres semillas; los intervalos del 95 % y las diferencias emparejadas remuestrean pacientes (76) y promedian
la métrica de las semillas en cada remuestreo.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scripts.analysis.finalists import A_PRIORI, EFFICIENT, folder_name
from scripts.eval.evaluate import training_medians
from scripts.eval.normalize import rows_of
from scripts.eval.results import GRIDS
from scripts.eval.text_report import EXTRA_FINDINGS

CONSTANT = training_medians("data/targets/chaksu__json_v1.csv", "data/splits/chaksu_seed42.csv", ["vcdr"])["vcdr"]
DRAWS = 1000
AUC_RELEVANT, RHO_RELEVANT = 0.05, 0.10
WRITTEN = re.compile(r"(?:C/?D|cup[- ]to[- ]disc)[^0-9.]{0,40}?(\d?\.\d+)|(\d?\.\d+)\s*(?:cup[- ]to[- ]disc|C/?D)", re.IGNORECASE)
SEEDS = ("42", "43", "44")
CELLS = [folder_name(c) for c in A_PRIORI + EFFICIENT]


def load(paths):
    frames = []
    for path in paths:
        frames.append(pd.DataFrame(list(rows_of(path))))
    return pd.concat(frames, ignore_index=True)


def fast_auc(score, positive):
    n1, n0 = positive.sum(), (~positive).sum()
    return (rankdata(score)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def fast_rho(x, y):
    """Spearman; indefinido (NaN) si una de las dos series es constante, como en el modelo base."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(rankdata(x), rankdata(y))[0, 1] if len(x) > 2 else np.nan


class External:
    """Las 129 imágenes con sus referencias, y las métricas de una predicción (vector de vCDR en el orden de `ids`)."""

    def __init__(self):
        targets = pd.read_csv("data/targets/mmodir129__external.csv", dtype=str, keep_default_na=False)
        self.ids = list(targets.image_id)
        self.positive = (targets.label == "Pathological").to_numpy()
        found = targets.text.map(lambda t: (lambda m: float(m.group(1) or m.group(2)) if m else np.nan)(WRITTEN.search(t))).to_numpy()
        self.written = found
        self.has_written = ~np.isnan(found)
        self.mask = targets.vcdr_mask_bbox.map(lambda v: float(v) if v else np.nan).to_numpy()
        self.has_mask = ~np.isnan(self.mask)
        patients = targets.patient_id.to_numpy()
        members = [np.flatnonzero(patients == p) for p in np.unique(patients)]
        rng = np.random.default_rng(0)
        self.samples = [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(DRAWS)]
        self.everyone = np.arange(len(self.ids))

    def metrics(self, v, index):
        w, m = self.has_written[index], self.has_mask[index]
        return {"auc_etiqueta": fast_auc(v[index], self.positive[index]),
                "rho_escrito": fast_rho(v[index][w], self.written[index][w]),
                "rho_mascara": fast_rho(v[index][m], self.mask[index][m]),
                "mae_mascara": np.abs(v[index][m] - self.mask[index][m]).mean()}


def vcdr_vector(df, ids):
    """vCDR predicho por imagen, en el orden de `ids`; lo no legible vale la constante."""
    series = df.set_index("id").pred_vcdr.reindex(ids)
    return pd.to_numeric(series, errors="coerce").fillna(CONSTANT).to_numpy()


def decision_rates(df, positive_ids):
    said = df.set_index("id").pred_decision
    pos = np.array([said.get(i, "") == "GLAUCOMA SUSPECT" for i in positive_ids["pos"]])
    neg = np.array([said.get(i, "") == "NORMAL" for i in positive_ids["neg"]])
    return pos.mean(), neg.mean()


def main():
    ext = External()
    ids = ext.ids
    pos_ids = {"pos": [i for i, p in zip(ids, ext.positive) if p], "neg": [i for i, p in zip(ids, ext.positive) if not p]}
    experiments = Path("experiments")
    sources = {
        "A": ("chaksu_json_v1", "test"), "B": ("chaksu_prose_v1", "test"), "CoT": ("chaksu_cot_v1", "validation"),
        "Prompts": ("chaksu_prompts_A", "validation"),
    }
    lines, table, by_mode = [], [], {}

    def say(text=""):
        print(text)
        lines.append(text)

    def runs_of(experiment, cells):
        """{celda: [carpetas de corrida]}; con semilla `cell_s42…`, sin semilla `base` o `P0`."""
        out = {}
        for cell in cells:
            seeded = sorted(experiments.glob(f"{experiment}/{cell}_s[0-9]*"))
            out[cell] = seeded or [experiments / experiment / cell]
        return out

    say("VALIDACIÓN EXTERNA EN MM-ODIR-129. Métricas con la imagen correcta, media de las semillas; entre corchetes, IC 95 % remuestreando pacientes.\n")
    for mode, (experiment, analog) in sources.items():
        cells = {"A": CELLS + ["base"], "B": CELLS + ["base"], "CoT": ["all-linear", "base"],
                 "Prompts": [p.stem.removeprefix("chaksu_A_") for p in sorted(Path("data/prompts").glob("chaksu_A_P*.txt")) if "shots" not in p.stem] + ["P4_t1024"]}[mode]
        say(f"=== {mode} ({experiment}); análogo en Chákṣu: {analog}")
        say(f"{'celda':<24}{'Chákṣu AUC dec.':>16}{'ρ gt':>8} | {'AUC etiqueta [IC]':>24}{'ρ escrito':>11}{'ρ máscara':>11}{'MAE másc.':>10}{'sens':>7}{'espec':>7} | {'gan. AUC barajada':>18}{'blanco':>8}")
        results = {}
        for cell, folders in runs_of(experiment, cells).items():
            per_seed, chaksu, gains, rates = [], [], [], []
            for folder in folders:
                if not (folder / "external" / "generate_config.json").exists():
                    continue
                e = load(sorted((folder / "external").glob("pred_*.jsonl")))
                v = {c: vcdr_vector(e[e.control == c], ids) for c in ("real", "shuffled", "blank")}
                per_seed.append(v)
                rates.append(decision_rates(e[e.control == "real"], pos_ids))
                c = load(sorted((folder / analog).glob("pred_*.jsonl"))) if (folder / analog / "generate_config.json").exists() else None
                if c is not None:
                    r = c[c.control == "real"]
                    gt = (r.gt_decision == "GLAUCOMA SUSPECT").to_numpy()
                    pv = pd.to_numeric(r.pred_vcdr, errors="coerce").fillna(CONSTANT).to_numpy()
                    chaksu.append((fast_auc(pv, gt), fast_rho(pv, pd.to_numeric(r.gt_vcdr).to_numpy())))
            if not per_seed:
                say(f"{cell:<24}  (sin corrida externa todavía)")
                continue
            point = {k: np.nanmean([ext.metrics(v["real"], ext.everyone)[k] for v in per_seed]) for k in ("auc_etiqueta", "rho_escrito", "rho_mascara", "mae_mascara")}
            boot = np.array([np.nanmean([ext.metrics(v["real"], s)["auc_etiqueta"] for v in per_seed]) for s in ext.samples])
            low, high = np.nanpercentile(boot, [2.5, 97.5])
            g = {c: np.nanmean([ext.metrics(v["real"], ext.everyone)["auc_etiqueta"] - ext.metrics(v[c], ext.everyone)["auc_etiqueta"] for v in per_seed]) for c in ("shuffled", "blank")}
            sens, spec = np.mean(rates, axis=0)
            results[cell] = {"point": point, "boot": boot, "chaksu": np.mean(chaksu, axis=0) if chaksu else (np.nan, np.nan), "gain": g}
            ca, cr = results[cell]["chaksu"]
            say(f"{cell:<24}{ca:>16.3f}{cr:>+8.3f} | {point['auc_etiqueta']:>8.3f} [{low:.3f}, {high:.3f}]{point['rho_escrito']:>+11.3f}{point['rho_mascara']:>+11.3f}"
                f"{point['mae_mascara']:>10.3f}{sens:>7.3f}{spec:>7.3f} | {g['shuffled']:>+18.3f}{g['blank']:>+8.3f}")
            table.append({"modo": mode, "celda": cell, "chaksu_auc_decision": ca, "chaksu_rho_gt": cr, **point, "auc_etiqueta_low": low, "auc_etiqueta_high": high,
                          "sens": sens, "espec": spec, "ganancia_auc_barajada": g["shuffled"], "ganancia_auc_blanco": g["blank"]})
        if mode in ("A", "B") and "all-linear" in results:
            ref = results["all-linear"]
            say("  diferencia de AUC con `all-linear` (IC 95 %, pacientes):")
            for cell, r in results.items():
                if cell in ("all-linear", "base"):
                    continue
                d = r["boot"] - ref["boot"]
                mean = r["point"]["auc_etiqueta"] - ref["point"]["auc_etiqueta"]
                low, high = np.nanpercentile(d, [2.5, 97.5])
                say(f"    {cell:<24}{mean:>+8.3f} [{low:+.3f}, {high:+.3f}]{'   más que lo relevante' if abs(mean) > AUC_RELEVANT else ''}")
            fin = [c for c in CELLS if c in results]
            test_error = pd.read_csv(GRIDS / experiment / "results_test.csv").set_index("cell").loc[fin, "vcdr_error_mean"]
            rho = spearmanr(test_error, [results[c]["point"]["auc_etiqueta"] for c in fin]).statistic
            best = max(fin, key=lambda c: results[c]["point"]["auc_etiqueta"])
            say(f"  H3: Spearman entre el error en el Test de Chákṣu y el AUC externo de las {len(fin)} ubicaciones {rho:+.3f} (negativo = el orden transfiere); mejor AUC externo: {best} "
                f"({results[best]['point']['auc_etiqueta']:.3f}); `all-linear` {ref['point']['auc_etiqueta']:.3f}")
        by_mode[mode] = results
        say()

    # ---- Criterio 5: B, cadena de razonamiento y prompts frente a `all-linear` de A, con pacientes remuestreados ----
    reference = by_mode["A"]["all-linear"]
    say("=== Diferencia de AUC con `all-linear` de A (IC 95 %, pacientes; relevante 0.05)")
    for mode in ("B", "CoT", "Prompts"):
        for cell, r in by_mode[mode].items():
            if mode in ("B", "CoT") and cell != "all-linear":
                continue
            d = r["boot"] - reference["boot"]
            mean = r["point"]["auc_etiqueta"] - reference["point"]["auc_etiqueta"]
            low, high = np.nanpercentile(d, [2.5, 97.5])
            say(f"  {mode:<8}{cell:<24}{mean:>+8.3f} [{low:+.3f}, {high:+.3f}]{'   más que lo relevante' if abs(mean) > AUC_RELEVANT else ''}")
    say()

    # ---- C ------------------------------------------------------------------------------------------------
    say("=== C (etapa 1 = `all-linear` de A; etapa 2 solo texto), MM-ODIR-129")
    say(f"{'etapa 2':<8}{'control':>10}{'fidelidad mín.':>16}{'hallazgos nuevos':>18}{'AUC etiqueta':>14}{'AUC de A (etapa 1)':>20}")
    a_all = [load(sorted((experiments / f"chaksu_json_v1/all-linear_s{s}/external").glob("pred_*.jsonl"))) for s in SEEDS]
    for variant in ("base", "lora"):
        for control in ("real", "shuffled", "blank"):
            fidelity, extra, aucs, a_aucs = [], 0, [], []
            for seed, a in zip(SEEDS, a_all):
                folder = experiments / f"chaksu_C/ext_{variant}_all-linear_s{seed}_{control}"
                if not (folder / "generate_config.json").exists():
                    continue
                c = load([folder / "pred_text.jsonl"])
                ok = (c.pred_vcdr.astype(float) - c.input_vcdr.astype(float)).abs().lt(0.0051) & (c.pred_hcdr.astype(float) - c.input_hcdr.astype(float)).abs().lt(0.0051) \
                    & (c.pred_decision == c.input_decision)
                fidelity.append(ok.mean())
                extra += int(c.prediction.str.lower().map(lambda t: any(k in t for k in EXTRA_FINDINGS)).sum())
                aucs.append(ext.metrics(vcdr_vector(c, ids), ext.everyone)["auc_etiqueta"])
                a_aucs.append(ext.metrics(vcdr_vector(a[a.control == control], ids), ext.everyone)["auc_etiqueta"])
            if fidelity:
                say(f"{variant:<8}{control:>10}{min(fidelity):>16.3f}{extra:>18}{np.mean(aucs):>14.3f}{np.mean(a_aucs):>20.3f}")
    say()
    Path("results/audit/mmodir129_externa.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(table).round(4).to_csv("results/audit/mmodir129_externa.csv", index=False)


if __name__ == "__main__":
    main()
