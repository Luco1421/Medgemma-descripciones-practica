"""Norma de gradiente relativa por bloque (RGN de Auto-RGN, Lee et al., ICLR 2023), sobre el modelo base.

    python -m scripts.analysis.rgn --targets T --splits S --split train --prompt-file P --datasets-root R --output F.csv

Para cada tensor de pesos, RGN = ||g||₂ / ||θ||₂: cuánto cambiaría el tensor, en proporción a su tamaño, si
se diera un paso en la dirección del gradiente de la tarea. Auto-RGN propone ajustar donde es mayor. Aquí se
usa solo como PREDICTOR de qué bloque conviene ajustar, para compararlo con lo que midió la rejilla (la
misma comparación que la sonda de H7) y con la profundidad.

Qué se mide exactamente (decisiones de esta implementación, no del paper):
  - Pesos de las proyecciones que LoRA adapta en `block_language` y `block_vision`; los sesgos no entran.
  - Un tensor se agrega a su bloque como RGN del bloque = sqrt(Σ||g||²) / sqrt(Σ||θ||²) sobre las
    proyecciones de atención (`attention`), las del MLP (`mlp`) o todas (`block`). El paper trabaja por tensor.
  - El gradiente es el de un paso de entrenamiento del modelo sin adaptar: `grad_accum` ejemplos (como en
    `lora.py train`) con la pérdida de la receta; se recorre la partición una vez, sin actualizar pesos, y se
    informan la media y la desviación del RGN entre pasos. El paper lo recalcula en cada época.
No se usa Auto-RGN para escalar tasas de aprendizaje: eso sería otro método de entrenamiento. Se escribe en
un CSV: una fila por (torre, capa).
"""

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np


def main():
    import torch

    from scripts.model.lora import LAYER_GROUPS, add_data_arguments, build_collator, load_rows
    from scripts.model.medgemma import load_medgemma

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_arguments(parser)
    parser.add_argument("--output", required=True, help="CSV de salida")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="corta el número de ejemplos; para pruebas de humo")
    args = parser.parse_args()

    rows, _ = load_rows(args)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit] if args.limit else rows
    processor, model = load_medgemma()
    collate = build_collator(processor)

    # Qué tensores se miden: la clave es (torre, capa, tipo).
    towers = {"language": LAYER_GROUPS["block_language"], "vision": LAYER_GROUPS["block_vision"]}
    tracked = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        module = name.removesuffix(".weight")
        for tower, regex in towers.items():
            if name.endswith(".weight") and re.fullmatch(regex, module):
                key = (tower, int(re.search(r"layers\.(\d+)\.", name)[1]), "attention" if "self_attn" in name else "mlp")
                tracked[name] = key
                parameter.requires_grad = True
    counts = {t: sum(k[0] == t for k in tracked.values()) for t in towers}
    assert counts == {"language": 34 * 7, "vision": 27 * 6}, f"tensores medidos inesperados: {counts}"
    # Sin un parámetro entrenable antes del primer bloque, la recomputación del bloque 0 no propaga gradiente.
    model.enable_input_require_grads()
    patch = [p for n, p in model.named_parameters() if "embeddings.patch_embedding" in n]
    assert len(patch) == 2, "no se encontró el embedding de parches de la torre de visión"
    for p in patch:
        p.requires_grad = True
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    parameters = dict(model.named_parameters())
    groups = sorted({k for k in tracked.values()})
    theta_sq = {g: sum(parameters[n].detach().float().pow(2).sum().item() for n, k in tracked.items() if k == g)
                for g in groups}
    steps = {g: [] for g in groups}
    for first in range(0, len(rows) - args.grad_accum + 1, args.grad_accum):
        for row in rows[first:first + args.grad_accum]:
            batch = {k: v.to(model.device) for k, v in collate([row]).items()}
            batch["pixel_values"] = batch["pixel_values"].to(model.dtype)
            loss = model(**batch).loss / args.grad_accum
            assert torch.isfinite(loss), f"pérdida no finita en {row['id']}"
            loss.backward()
        for g in groups:
            grad_sq = sum(parameters[n].grad.float().pow(2).sum().item() for n, k in tracked.items() if k == g)
            assert grad_sq > 0, f"gradiente nulo en {g}: la recomputación cortó el backward"
            steps[g].append((grad_sq ** 0.5) / (theta_sq[g] ** 0.5))
        for p in parameters.values():
            p.grad = None
        done = first // args.grad_accum + 1
        if done % 10 == 0:
            print(f"[{done * args.grad_accum}/{len(rows)}]", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for tower in towers:
        for layer in sorted({g[1] for g in groups if g[0] == tower}):
            row = {"tower": tower, "layer": layer}
            per_step = {}
            for kind in ("attention", "mlp"):
                per_step[kind] = np.array(steps[(tower, layer, kind)])
                row[f"rgn_{kind}_mean"], row[f"rgn_{kind}_sd"] = per_step[kind].mean(), per_step[kind].std(ddof=1)
            both = [theta_sq[(tower, layer, k)] for k in ("attention", "mlp")]
            grad_both = [(per_step[k] ** 2) * theta_sq[(tower, layer, k)] for k in ("attention", "mlp")]
            block = np.sqrt(sum(grad_both) / sum(both))
            row["rgn_block_mean"], row["rgn_block_sd"] = block.mean(), block.std(ddof=1)
            table.append(row)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in table)
    print(f"{out}: {len(steps[groups[0]])} pasos de {args.grad_accum} ejemplos, {len(table)} capas")


if __name__ == "__main__":
    main()
