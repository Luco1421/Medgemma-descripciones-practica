"""Mide la latencia de inferencia por imagen de las configuraciones A, B, C (redactor LoRA y redactor base) y del modelo base.

    python -m scripts.analysis.latency [--passes 2] [--images 40] [--warmup 3]

Un solo proceso, una sola GPU y un solo modelo cargado: los adaptadores se montan con nombre y `set_adapter` elige cuál actúa,
como en el prototipo. Cada configuración procesa las mismas imágenes de validación de Chákṣu, en el mismo orden, con la
decodificación voraz y el tope de tokens de los experimentos. Las primeras `--warmup` imágenes (las que siguen a las medidas) se
descartan: absorben la compilación de kernels y la asignación inicial de memoria. Cada tramo se cronometra con
`torch.cuda.synchronize()` antes y después, porque las llamadas a la GPU son asíncronas y sin ello el reloj mediría el lanzamiento.
Tramos: carga de la imagen del disco, preprocesado (plantilla de chat y procesador de imagen, en CPU), generación, y en C el
cambio de adaptador y la lectura de los campos entre etapas. Escribe results/audit/latencia.txt y latencia.csv (una fila por imagen).
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image

from scripts.eval.evaluate import DECISIONS, NUMERIC, is_number, parse_answer
from scripts.model.lora import fill_prompt, load_rows
from scripts.model.medgemma import load_medgemma
from scripts.studies.pipeline import RECIPES, STAGE1, STAGE_B

SPLITS = "data/splits/chaksu_seed42.csv"
MAX_NEW_TOKENS = 128
SEED = 42
CELL = "all-linear"
NAMES = {"A": "A", "B": "B", "S2": "verbalize"}
CONFIGS = ["A", "B", "C_lora", "C_base", "base"]


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def stage(model, processor, prompt, image_path, adapter):
    """Una generación voraz con el adaptador pedido (None = modelo base). Devuelve texto, tokens y los tiempos (ms) de cada tramo."""
    t0 = clock()
    image = Image.open(image_path).convert("RGB") if image_path else None
    t1 = clock()
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    t2 = clock()
    if adapter is None:
        with model.disable_adapter(), torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    else:
        assert model.active_adapters == [adapter], f"adaptador activo {model.active_adapters}, se esperaba {adapter}"
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    t3 = clock()
    new = output[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(new, skip_special_tokens=True).strip()
    return text, len(new), {"load": (t1 - t0) * 1e3, "prep": (t2 - t1) * 1e3, "gen": (t3 - t2) * 1e3}


def switch(model, adapter):
    t0 = clock()
    model.set_adapter(adapter)
    elapsed = (clock() - t0) * 1e3
    assert model.active_adapters == [adapter], f"adaptador activo {model.active_adapters}, se esperaba {adapter}"
    return elapsed


def fields_of(raw):
    """Los tres campos de la etapa 1 (el lector de la evaluación); una respuesta ilegible aborta: la etapa 2 no tendría entrada."""
    answer = parse_answer(raw)
    assert answer is not None, f"la etapa 1 no dio un JSON legible: {raw!r}"
    fields = {k: answer.get(k) for k in ("vcdr", "hcdr", "decision")}
    assert all(is_number(fields[k]) for k in NUMERIC) and fields["decision"] in DECISIONS, f"campos inválidos: {raw!r}"
    return fields


def one_image(model, processor, config, row_a, row_b, template2):
    """Una imagen en una configuración. Devuelve tokens totales y el desglose de tiempos en ms."""
    if config == "A":
        switch(model, NAMES["A"])
        _, tokens, t = stage(model, processor, row_a["prompt"], row_a["image"], NAMES["A"])
        return tokens, {**t, "switch": 0.0, "stage1": t["load"] + t["prep"] + t["gen"], "stage2": 0.0}
    if config == "B":
        switch(model, NAMES["B"])
        _, tokens, t = stage(model, processor, row_b["prompt"], row_b["image"], NAMES["B"])
        return tokens, {**t, "switch": 0.0, "stage1": t["load"] + t["prep"] + t["gen"], "stage2": 0.0}
    if config == "base":
        _, tokens, t = stage(model, processor, row_a["prompt"], row_a["image"], None)
        return tokens, {**t, "switch": 0.0, "stage1": t["load"] + t["prep"] + t["gen"], "stage2": 0.0}
    # C: etapa 1 con el adaptador de A, lectura de campos, cambio de adaptador y etapa 2 solo texto.
    switch(model, NAMES["A"])
    raw, tokens1, t1 = stage(model, processor, row_a["prompt"], row_a["image"], NAMES["A"])
    t0 = clock()
    f = fields_of(raw)
    prompt2 = fill_prompt(template2, f["vcdr"], f["hcdr"], f["decision"])
    parse_ms = (clock() - t0) * 1e3
    if config == "C_lora":
        switch_ms = switch(model, NAMES["S2"])
        adapter2 = NAMES["S2"]
    else:
        switch_ms = 0.0
        adapter2 = None
    _, tokens2, t2 = stage(model, processor, prompt2, None, adapter2)
    s1 = t1["load"] + t1["prep"] + t1["gen"]
    s2 = t2["prep"] + t2["gen"]
    return tokens1 + tokens2, {"load": t1["load"], "prep": t1["prep"] + t2["prep"], "gen": t1["gen"] + t2["gen"],
                               "switch": switch_ms + parse_ms, "stage1": s1, "stage2": s2}


def summarize(values):
    v = np.array(values)
    return v.mean(), v.std(ddof=1), np.median(v), np.percentile(v, 95)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="results/audit/latencia")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "hace falta GPU"

    def rows_of(recipe):
        ns = argparse.Namespace(targets=f"data/targets/chaksu__{recipe}.csv", splits=SPLITS, split="validation",
                                prompt_file=f"data/targets/chaksu__{recipe}.prompt.txt", datasets_root="../datasets")
        return load_rows(ns)[0]

    n = args.images + args.warmup
    rows_a, rows_b = rows_of("json_v1")[:n], rows_of("prose_v1")[:n]
    assert [r["id"] for r in rows_a] == [r["id"] for r in rows_b]
    template2 = Path("data/targets/chaksu__verbalize_v1.prompt.txt").read_text(encoding="utf-8").strip()
    measured = list(range(args.images))
    warm = list(range(args.images, n))

    processor, model = load_medgemma()
    processor.tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(model, STAGE1 / f"{CELL}_s{SEED}", adapter_name=NAMES["A"])
    model.load_adapter(STAGE_B / f"{CELL}_s{SEED}", adapter_name=NAMES["B"])
    model.load_adapter(RECIPES["verbalize_v1"]["lora"].format(seed=SEED), adapter_name=NAMES["S2"])
    model.eval()
    assert set(model.peft_config) == set(NAMES.values()), sorted(model.peft_config)
    print(f"GPU: {torch.cuda.get_device_name(0)}; adaptadores {sorted(model.peft_config)}; {args.images} imágenes + {args.warmup} de calentamiento", flush=True)

    records = []
    peak = {}
    for p in range(args.passes):
        for config in CONFIGS:
            for i in warm:
                one_image(model, processor, config, rows_a[i], rows_b[i], template2)
            torch.cuda.reset_peak_memory_stats()
            for i in measured:
                tokens, t = one_image(model, processor, config, rows_a[i], rows_b[i], template2)
                records.append({"pass": p + 1, "config": config, "image_id": rows_a[i]["id"], "tokens": tokens,
                                **{k: round(v, 2) for k, v in t.items()}, "total": round(t["stage1"] + t["stage2"] + t["switch"], 2)})
            peak[(p, config)] = torch.cuda.max_memory_allocated() / 2**30
            print(f"pasada {p + 1} {config}: {np.mean([r['total'] for r in records if r['config'] == config and r['pass'] == p + 1]):.0f} ms", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    labels = {"A": "A all-linear (imagen -> JSON)", "B": "B all-linear (imagen -> prosa)", "C_lora": "C, redactor LoRA",
              "C_base": "C, redactor base", "base": "base sin adaptar, prompt de A"}
    lines = [f"Latencia de inferencia por imagen. GPU {torch.cuda.get_device_name(0)}, bf16, decodificación voraz, max_new_tokens {MAX_NEW_TOKENS}.",
             f"{args.images} imágenes de validación de Chákṣu (primeras de {SPLITS}), {args.warmup} de calentamiento descartadas, "
             f"{args.passes} pasada(s). Adaptadores all-linear semilla {SEED} (A, B) y {RECIPES['verbalize_v1']['lora'].format(seed=SEED)}.",
             "Total = carga + preprocesado + generación + cambio de adaptador y lectura de campos (C). ms por imagen.", ""]
    means = {}
    for p in range(args.passes):
        lines += [f"=== Pasada {p + 1}", f"{'configuración':32s} {'media±desv':>16s} {'mediana':>8s} {'p95':>8s} {'tokens':>7s} {'pico GiB':>9s}"]
        for c in CONFIGS:
            rs = [r for r in records if r["config"] == c and r["pass"] == p + 1]
            m, s, med, p95 = summarize([r["total"] for r in rs])
            means[(p, c)] = m
            lines.append(f"{labels[c]:32s} {m:8.0f}±{s:6.0f} {med:8.0f} {p95:8.0f} {np.mean([r['tokens'] for r in rs]):7.1f} {peak[(p, c)]:9.2f}")
        lines += ["", "Desglose medio (ms): carga imagen | preprocesado | generación | cambio+lectura | etapa 1 | etapa 2"]
        for c in CONFIGS:
            rs = [r for r in records if r["config"] == c and r["pass"] == p + 1]
            lines.append(f"{labels[c]:32s} " + " | ".join(f"{np.mean([r[k] for r in rs]):8.1f}" for k in ("load", "prep", "gen", "switch", "stage1", "stage2")))
        lines.append("")
        for num, den in (("C_lora", "B"), ("C_lora", "A"), ("B", "A"), ("C_base", "B"), ("C_base", "A"), ("A", "base")):
            lines.append(f"razón {num}/{den}: {means[(p, num)] / means[(p, den)]:.2f}")
        lines.append("")
    text = "\n".join(lines)
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
