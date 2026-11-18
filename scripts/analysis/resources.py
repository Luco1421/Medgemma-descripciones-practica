"""Mide los recursos que consumió el proyecto: cómputo, almacenamiento y memoria por adaptador.

Cada cifra sale de un archivo o comando real (run_config.json, sacct, du); nada se estima salvo la
proyección de la sección 3, que multiplica cifras medidas. Escribe results/audit/recursos.txt.
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess

SACCT_SINCE = "2026-08-01"
HF_MODEL = "/data/jperez/models/huggingface/hub/models--google--medgemma-1.5-4b-it"
DATASETS = {"Chákṣu": "../datasets/chaksu/raw", "MM-ODIR-129": "../datasets/mmodir129/raw"}
GIB = 1024 ** 3


def run(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def du(path):
    """Bytes aparentes del árbol (du -sb), la misma medida que ocupa un archivo al copiarlo."""
    return int(run(f"du -sb {path}").split()[0])


def gib(n):
    return f"{n / GIB:.2f} GiB"


def hms(seconds):
    return f"{seconds / 3600:.2f} h"


def parse_elapsed(s):
    days, _, rest = s.partition("-") if "-" in s else ("0", "", s)
    h, m, sec = (int(x) for x in rest.split(":"))
    return int(days) * 86400 + h * 3600 + m * 60 + sec


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = []

    def p(line=""):
        out.append(line)

    # ---- 1. Cómputo -------------------------------------------------------------------------
    configs = sorted(glob.glob("experiments/*/*/run_config.json"))
    runs = []
    for c in configs:
        d = os.path.dirname(c)
        assert os.path.exists(f"{d}/adapter_model.safetensors"), d
        cfg = json.load(open(c))
        runs.append((d.split("/")[1], cfg["train_metrics"]["train_runtime"], cfg["trainable_params"]))
    rt = [r[1] for r in runs]
    p("1. CÓMPUTO")
    p(f"Entrenamientos (carpetas con adapter_model.safetensors y run_config.json en experiments/*/*): {len(runs)}")
    p(f"  Método: glob experiments/*/*/run_config.json; tiempo = train_metrics.train_runtime de cada run_config.json.")
    for rec in sorted({r[0] for r in runs}):
        t = [r[1] for r in runs if r[0] == rec]
        p(f"  {rec}: {len(t)} entrenamientos, {hms(sum(t))} en total, {hms(statistics.mean(t))} de media")
    p(f"Tiempo de entrenamiento total: {hms(sum(rt))}; media {hms(statistics.mean(rt))} por entrenamiento "
      f"(mín {hms(min(rt))}, máx {hms(max(rt))})")

    # Los jobs del proyecto se identifican por los logs de experiments/logs/<nombre>_<jobid>.out|.err;
    # sacct lista también los jobs de las líneas hermanas del mismo usuario, que así quedan fuera.
    ids = sorted({int(m.group(2)) for f in os.listdir("experiments/logs")
                  if (m := re.fullmatch(r"(.+)_(\d+)\.(out|err)", f))})
    rows = [l.split("|") for l in run(
        f"sacct -u $USER -S {SACCT_SINCE} -X -n -P --format=JobID,JobName,Partition,Elapsed,State,NodeList").splitlines()]
    mine = [r for r in rows if r[0].isdigit() and int(r[0]) in ids]
    assert all(r[2] == "nukwa-long" for r in mine)
    elapsed = [parse_elapsed(r[3]) for r in mine]
    nodes = sorted({r[5] for r in mine})
    p(f"Jobs SLURM del proyecto con log en experiments/logs: {len(ids)} ids; {len(mine)} presentes en sacct "
      f"desde {SACCT_SINCE}, todos en nukwa-long")
    p(f"  Método: ids de los nombres de experiments/logs/*_<id>.out|err cruzados con sacct -X; "
      f"los jobs sin log (humo, verificaciones, borrados) no se pueden separar de las líneas hermanas y no cuentan.")
    p(f"  Estados: { {s: sum(r[4] == s for r in mine) for s in sorted({r[4] for r in mine})} }")
    p(f"  Tiempo reservado de GPU (suma de Elapsed): {hms(sum(elapsed))}; media {hms(statistics.mean(elapsed))} por job")
    for name in sorted({r[1] for r in mine}):
        e = [parse_elapsed(r[3]) for r in mine if r[1] == name]
        p(f"    {name}: {len(e)} jobs, {hms(sum(e))}")
    p(f"  Nodos usados: {', '.join(nodes)} (L40S de 46 GB en nukwa-05..10 según la configuración del clúster; el modelo de GPU no lo da sacct)")
    inference = sum(parse_elapsed(r[3]) for r in mine if r[1] not in ("lora", "batch", "rgn", "probe", "pooling"))
    p(f"  Elapsed de jobs que no son entrenamiento ni diagnóstico (inferencia, evaluación, métricas): {hms(inference)}")
    p("  Los jobs 'batch' entrenan y evalúan, por lo que su Elapsed incluye también inferencia; "
      "no se puede separar sin medir cada fase.")

    # ---- 2. Almacenamiento ------------------------------------------------------------------
    p()
    p("2. ALMACENAMIENTO (du -sb, bytes aparentes)")
    p(f"experiments/ total: {gib(du('experiments'))}")
    for rec in sorted(os.listdir("experiments")):
        p(f"  experiments/{rec}: {gib(du(f'experiments/{rec}'))}")
    adapters = glob.glob("experiments/*/*/adapter_model.safetensors")
    sizes = [os.stat(a).st_size for a in adapters]
    p(f"Adaptadores guardados (adapter_model.safetensors): {len(adapters)}, total {gib(sum(sizes))}, "
      f"media {statistics.mean(sizes) / 1024 ** 2:.1f} MiB (mín {min(sizes) / 1024 ** 2:.1f}, máx {max(sizes) / 1024 ** 2:.1f})")
    preds = run("find experiments -name 'pred_*.jsonl' -printf '%s\\n'").split()
    p(f"Predicciones crudas (pred_*.jsonl): {len(preds)} archivos, {gib(sum(int(x) for x in preds))}")
    p(f"data/: {gib(du('data'))}")
    p(f"results/: {gib(du('results'))}")
    probes = glob.glob("results/probes/*.npz")
    p(f"Activaciones de sondas (results/probes/*.npz): {len(probes)} archivos, {gib(sum(os.stat(f).st_size for f in probes))}")
    p(f"Código: scripts/ {du('scripts') / 1024 ** 2:.2f} MiB, "
      f"runs/ {du('runs') / 1024 ** 2:.2f} MiB")
    tracked = int(run("git ls-files -z | xargs -0 du -cb | tail -1").split()[0])
    p(f"Repositorio versionado: archivos rastreados (git ls-files, du -cb) {tracked / 1024 ** 2:.2f} MiB; "
      f"git count-objects -vH:")
    p("  " + run("git count-objects -vH").replace("\n", "\n  ").rstrip())
    for name, path in DATASETS.items():
        p(f"Dataset {name} ({path}): {gib(du(path))}")
    p(f"Pesos de MedGemma en HF_HUB_CACHE ({HF_MODEL}): {gib(du(HF_MODEL))}")

    # ---- 3. Memoria por adaptador y proyección ---------------------------------------------
    p()
    p("3. MEMORIA POR ADAPTADOR Y PROYECCIÓN")
    params = [r[2] for r in runs]
    p(f"Parámetros entrenables por adaptador: media {statistics.mean(params) / 1e6:.2f} M "
      f"(mín {min(params) / 1e6:.2f} M, máx {max(params) / 1e6:.2f} M; run_config.trainable_params), "
      f"frente a 4338.58 M totales")
    mean_size, mean_rt = statistics.mean(sizes), statistics.mean(rt)
    p(f"Bytes en disco por adaptador (media medida): {mean_size / 1024 ** 2:.1f} MiB; "
      f"entrenamiento medio {hms(mean_rt)} (run_config)")
    p("Proyección = N adaptadores x media medida. Suposiciones: 88 ubicaciones, 3 semillas, "
      "mismo tamaño y tiempo medio por adaptador (sobreestima los de pocas capas, subestima all-linear).")
    p(f"  {'configuración':<46}{'adaptadores':>12}{'disco':>12}{'GPU-horas':>11}{'días (4 jobs)':>15}")
    for label, n in [("528 (88 ubicaciones x 3 semillas x 2 recetas)", 528),
                     ("1 tasa x 88 x 3 semillas", 264),
                     ("3 tasas x 88 x 3 semillas", 792),
                     ("5 tasas x 88 x 3 semillas", 1320),
                     ("5 tasas x 88 x 3 semillas x 2 recetas", 2640)]:
        h = n * mean_rt / 3600
        p(f"  {label:<46}{n:>12}{gib(n * mean_size):>12}{h:>11.0f}{h / 4 / 24:>15.1f}")
    p("  'días (4 jobs)': GPU-horas / 4 jobs simultáneos (límite por usuario del clúster) / 24, sin inferencia ni colas.")

    text = "\n".join(out) + "\n"
    open("results/audit/recursos.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
