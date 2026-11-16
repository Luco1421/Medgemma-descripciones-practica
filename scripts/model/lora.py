"""LoRA sobre MedGemma: anatomía del modelo, grupos de capas y finetuning.

    python -m scripts.model.lora inspect [--full] [--pattern REGEX]
    python -m scripts.model.lora preview --layers GRUPO_O_REGEX
    python -m scripts.model.lora train --layers GRUPO --output-dir DIR DATOS
    python -m scripts.model.lora train --layers random:GRUPO --output-dir DIR DATOS   # control aleatorio
    python -m scripts.model.lora generate --output-dir DIR [--adapter DIR] DATOS

DATOS son cinco argumentos que definen qué ve el modelo: `--targets` (qué respuesta tiene cada
imagen, de `scripts.data.targets`), `--splits` y `--split` (qué imágenes entran, de `scripts.data.splits`),
`--prompt-file` (la pregunta, igual para todas) y `--datasets-root` (de dónde cuelgan las rutas de
imagen). Se corre desde la raíz del repo, en un nodo con GPU (vía sbatch).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.model.medgemma import MODEL_ID, load_medgemma

# ---------------------------------------------------------------------------
# Grupos de capas
#
# MedGemma tiene dos torres transformer y un cabezal (verlo con `inspect`):
#
#   model.vision_tower.vision_model...  (SigLIP) "lee" la imagen. Cada bloque
#       tiene atención (q/k/v/out_proj) y un MLP clásico de dos matrices
#       (fc1, fc2). Ojo con el `.vision_model.` intermedio: el vision_tower es
#       un SiglipVisionModel que envuelve al transformer propiamente dicho.
#   model.language_model... (Gemma) genera el texto. Cada bloque tiene
#       atención (q/k/v/o_proj) y un MLP "gated" de tres matrices:
#       gate_proj y up_proj se multiplican entre sí, down_proj comprime.
#   lm_head                 proyecta del espacio latente al vocabulario.
#
# El conector (model.multi_modal_projector) queda fuera de estos grupos: no
# tiene ninguna nn.Linear, es un AvgPool2d + una norma + un nn.Parameter suelto,
# y LoRA solo envuelve capas. Adaptarlo exigiría otro mecanismo.
#
# En la atención, cada token se proyecta con tres matrices aprendidas:
# Query (lo que busco), Key (lo que ofrezco), Value (lo que entrego);
# o_proj/out_proj mezcla las cabezas al final. Como las torres nombran
# distinto sus matrices (o_proj vs out_proj, gate/up/down vs fc1/fc2), un
# regex puede apuntar a una torre, a la otra, o a ambas. PEFT compara cada
# regex contra los nombres de módulos con re.fullmatch.
# ---------------------------------------------------------------------------

LAYER_GROUPS = {
    # Atención, por torre y combinada
    "attention_all": r".*\.(q_proj|k_proj|v_proj|o_proj|out_proj)$",
    "attention_language": r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    "attention_vision": r"^model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$",
    # Proyecciones individuales de atención
    "q_proj_all": r".*\.q_proj$",
    "q_proj_language": r"^model\.language_model\.layers\.\d+\.self_attn\.q_proj$",
    "q_proj_vision": r"^model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.q_proj$",
    "k_proj_all": r".*\.k_proj$",
    "k_proj_language": r"^model\.language_model\.layers\.\d+\.self_attn\.k_proj$",
    "k_proj_vision": r"^model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.k_proj$",
    "v_proj_all": r".*\.v_proj$",
    "v_proj_language": r"^model\.language_model\.layers\.\d+\.self_attn\.v_proj$",
    "v_proj_vision": r"^model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.self_attn\.v_proj$",
    "o_proj_all": r".*\.(o_proj|out_proj)$",
    "o_proj_language": r".*\.o_proj$",
    "o_proj_vision": r".*\.out_proj$",
    # MLP
    "mlp_all": r".*\.(gate_proj|up_proj|down_proj|fc1|fc2)$",
    "mlp_language": r".*\.(gate_proj|up_proj|down_proj)$",
    "mlp_vision": r".*\.(fc1|fc2)$",
    # Bloques completos (atención + MLP) por torre
    "block_language": r"^model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$",
    "block_vision": r"^model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))$",
    # Cabezal de salida
    "lm_head": r"lm_head",
}

# Bloques completos por tercio de profundidad de cada torre (índices de bloque, ambos extremos incluidos):
# lenguaje 34 bloques = 0-10 / 11-22 / 23-33; visión 27 bloques = 0-8 / 9-17 / 18-26.
DEPTH_THIRDS = {"language": {"early": (0, 10), "mid": (11, 22), "late": (23, 33)},
                "vision": {"early": (0, 8), "mid": (9, 17), "late": (18, 26)}}
for _tower, _thirds in DEPTH_THIRDS.items():
    for _third, (_first, _last) in _thirds.items():
        _block = LAYER_GROUPS[f"block_{_tower}"]
        LAYER_GROUPS[f"block_{_tower}_{_third}"] = _block.replace(r"\d+", "(" + "|".join(map(str, range(_first, _last + 1))) + ")")

# Completan el cruce tercio × módulo × torre (atención y MLP por separado en cada tercio) y el barrido de
# un bloque a la vez (`block_<torre>_layer<NN>`, una sola capa con atención y MLP). Cada grupo se
# construye sobre el regex de su familia, así que la ruta de módulos es la misma que la de los grupos de arriba.
_MLP_BLOCK = {"language": r"mlp\.(gate_proj|up_proj|down_proj)", "vision": r"mlp\.(fc1|fc2)"}
for _tower, _thirds in DEPTH_THIRDS.items():
    _attention = LAYER_GROUPS[f"attention_{_tower}"]
    _prefix = LAYER_GROUPS[f"block_{_tower}"].split(r"\d+")[0]
    for _third, (_first, _last) in _thirds.items():
        _layers = "(" + "|".join(map(str, range(_first, _last + 1))) + ")"
        LAYER_GROUPS[f"attention_{_tower}_{_third}"] = _attention.replace(r"\d+", _layers)
        LAYER_GROUPS[f"mlp_{_tower}_{_third}"] = _prefix + _layers + r"\." + _MLP_BLOCK[_tower] + "$"
    for _layer in range(_thirds["late"][1] + 1):
        LAYER_GROUPS[f"block_{_tower}_layer{_layer:02d}"] = LAYER_GROUPS[f"block_{_tower}"].replace(r"\d+", str(_layer))


def resolve_target_modules(spec):
    # 'all-linear' es palabra reservada de PEFT (todas las capas lineales);
    # un nombre de grupo se traduce a su regex; cualquier otra cosa se asume
    # regex crudo. Un nombre mal escrito termina como regex que no matchea
    # nada y PEFT falla con "target modules not found" — error visible.
    if spec == "all-linear":
        return spec
    return LAYER_GROUPS.get(spec, spec)


def lora_cost(module, rank):
    # LoRA añade A (rank x in) y B (out x rank): rank * (in + out) parámetros por módulo.
    return rank * (module.in_features + module.out_features)


def resolve_target_modules_for(spec, model, rank):
    """Como `resolve_target_modules`, más `random:GRUPO`: el control aleatorio de un grupo.

    Sortea módulos de TODOS los bloques de las dos torres hasta igualar el presupuesto de
    parámetros entrenables del grupo de referencia (a menos de 1 %). Si una ubicación elegida
    rinde más que ese sorteo, la ganancia no se debe solo a cuántos parámetros se entrenan sino
    a dónde. El sorteo depende solo del nombre del grupo y su sufijo `@K`: las semillas de entrenamiento repiten
    el mismo conjunto de módulos, así que su dispersión no incluye la variación entre sorteos.
    """
    if not spec.startswith("random:"):
        return resolve_target_modules(spec)
    group = spec.removeprefix("random:").split("@")[0]   # `@K` elige el sorteo K (sin sufijo: el sorteo 0)
    reference = LAYER_GROUPS[group]
    pool = [(n, m) for n, m in model.named_modules()
            if re.fullmatch(LAYER_GROUPS["block_language"], n) or re.fullmatch(LAYER_GROUPS["block_vision"], n)]
    budget = sum(lora_cost(m, rank) for n, m in pool if re.fullmatch(reference, n))
    assert budget > 0, f"el grupo de referencia {spec} no alcanza ningún módulo"
    random.Random(spec).shuffle(pool)
    chosen, total = [], 0
    for name, module in pool:               # rellena en orden aleatorio sin pasarse del presupuesto
        if total + lora_cost(module, rank) <= budget:
            chosen.append(name)
            total += lora_cost(module, rank)
    assert abs(total - budget) / budget < 0.01, f"presupuesto {budget} y sorteo {total} difieren más de 1 %"
    print(f"Control aleatorio de {spec.removeprefix('random:')}: {len(chosen)} módulos, {total:,} parámetros "
          f"LoRA (referencia {budget:,}); {sum('vision_tower' in n for n in chosen)} en visión")
    return sorted(chosen)


# ---------------------------------------------------------------------------
# inspect: desnudar el modelo
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    _, model = load_medgemma()

    # named_parameters() recorre cada matriz de pesos con su nombre jerárquico;
    # esos nombres son exactamente lo que los regex de LAYER_GROUPS matchean.
    if args.full or args.pattern:
        regex = re.compile(args.pattern) if args.pattern else None
        for name, param in model.named_parameters():
            if regex is None or regex.search(name):
                print(f"{name:<105} shape={tuple(param.shape)} dtype={param.dtype}")
        return

    total = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_ID}: {total / 1e9:.2f}B parámetros\n")

    for tower, prefix in (
        ("visión (SigLIP)", "model.vision_tower"),
        ("lenguaje (Gemma)", "model.language_model"),
    ):
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        blocks = {m.group(1) for n, _ in params if (m := re.search(r"\.layers\.(\d+)\.", n))}
        subtotal = sum(p.numel() for _, p in params)
        print(f"Torre de {tower}: {len(blocks)} bloques, {subtotal / 1e6:.0f}M parámetros")
        # La anatomía de un bloque (el 0): todos son iguales, solo cambia la
        # profundidad. Las matrices 2D son los candidatos a LoRA; los vectores
        # 1D son normalizaciones (no se les aplica LoRA).
        for name, p in params:
            if ".layers.0." in name:
                print(f"  {name.split('.layers.0.')[1]:<45} {tuple(p.shape)}")
        print()

    if not any("lm_head" in n for n, _ in model.named_parameters()):
        print("lm_head: comparte pesos con la tabla de embeddings (weight tying)\n")

    # Cada grupo se contrasta contra los nombres reales con la misma regla que
    # aplica PEFT (re.fullmatch). Un grupo con 0 módulos es el primer fallo
    # silencioso del pipeline: el regex no matchea nada y no se entrena nada.
    module_names = [name for name, _ in model.named_modules()]
    print("Grupos para --layers (además de 'all-linear' o un regex crudo):")
    for group, pattern in LAYER_GROUPS.items():
        count = sum(1 for name in module_names if re.fullmatch(pattern, name))
        print(f"  {group:<20} {count:>4} módulos" + ("   <-- NO MATCHEA NADA" if count == 0 else ""))


# ---------------------------------------------------------------------------
# preview: qué tocaría LoRA ahí, sin entrenar
# ---------------------------------------------------------------------------

def cmd_preview(args):
    from peft import LoraConfig, get_peft_model

    _, model = load_medgemma()
    target_modules = resolve_target_modules_for(args.layers, model, args.rank)

    print(f"Target: {args.layers} -> {target_modules}")
    if target_modules != "all-linear":
        matched = [n for n, _ in model.named_modules()
                   if (n in target_modules if isinstance(target_modules, list) else re.fullmatch(target_modules, n))]
        print(f"Módulos alcanzados ({len(matched)}):")
        for name in matched:
            print(f"  {name}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    get_peft_model(model, lora_config).print_trainable_parameters()


# ---------------------------------------------------------------------------
# train: el finetuning quirúrgico
# ---------------------------------------------------------------------------

def build_collator(processor):
    """Tokeniza en el momento y enmascara imagen, prompt y padding: la loss
    solo cubre la respuesta del asistente."""
    from PIL import Image

    # El checkpoint trae padding a la izquierda, que es lo correcto para
    # generar. Al entrenar no: las posiciones de RoPE se numeran desde 0 sin
    # mirar la máscara, así que el relleno a la izquierda corre la posición de
    # todos los tokens reales. El padding va a la derecha.
    processor.tokenizer.padding_side = "right"

    def collate(rows):
        images, full_texts, prompt_texts = [], [], []
        for row in rows:
            # Una receta sin imagen (etapa de redacción) entrena solo con texto.
            image = Image.open(row["image"]).convert("RGB") if row["image"] else None
            messages = [
                {"role": "user", "content": ([{"type": "image", "image": image}] if image is not None else [])
                 + [{"type": "text", "text": row["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]},
            ]
            # Una lista por ejemplo: el procesador interpreta una lista plana de
            # N imágenes como UN ejemplo con N imágenes, no como N ejemplos.
            images.append([image] if image is not None else None)
            full_texts.append(
                processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            )
            # Sin .strip(): el prompt tiene que ser prefijo exacto, token a
            # token, del ejemplo completo. El encabezado del turno es
            # "<start_of_turn>model\n" y recortar ese salto de línea desplaza
            # el corte un token.
            prompt_texts.append(
                processor.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
            )

        # Se tokeniza dos veces: el ejemplo completo (la entrada del modelo) y
        # solo el prompt, únicamente para medir cuántos tokens ocupa por fila.
        # Sin padding en el segundo: interesa el largo real, no el rellenado.
        images = images if any(images) else None
        batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        prompt_lengths = [
            len(ids) for ids in processor(text=prompt_texts, images=images)["input_ids"]
        ]

        # -100 es el valor que la cross-entropy de HF ignora: todo lo marcado
        # así no genera gradiente. Con padding a la derecha, los primeros
        # prompt_length tokens de cada fila son exactamente el prompt (imagen
        # incluida) y el resto del relleno lo delata la máscara de atención.
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :prompt_length] = -100

        # El enmascarado es el segundo fallo silencioso del pipeline, así que se
        # comprueba en vez de suponerse: ningún token de imagen puede quedar
        # bajo la loss, y ninguna fila puede quedarse sin nada que aprender.
        assert not (labels == processor.tokenizer.image_token_id).any(), "tokens de imagen bajo la loss"
        assert (labels != -100).any(dim=1).all(), "hay ejemplos sin un solo token supervisado"

        batch["labels"] = labels
        return batch

    return collate


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_data_arguments(parser):
    parser.add_argument("--targets", required=True,
                        help="CSV de scripts.data.targets: image_id, image y text, más los campos de referencia")
    parser.add_argument("--splits", required=True, help="CSV de scripts.data.splits: image_id, patient_id, split")
    parser.add_argument("--split", required=True, help="partición a usar (train, validation o test)")
    parser.add_argument("--prompt-file", required=True, help="texto del prompt, idéntico para todas las imágenes")
    parser.add_argument("--datasets-root", required=True,
                        help="carpeta de la que cuelgan las rutas `image` de los targets")


def fill_prompt(template, vcdr, hcdr, decision):
    """Prompt de la etapa de redacción: los dos cocientes con dos decimales y la decisión. Lo usan `load_rows` y el prototipo."""
    return template.format(vcdr=f"{float(vcdr):.2f}", hcdr=f"{float(hcdr):.2f}", decision=decision)


def load_rows(args):
    """Los ejemplos de una partición: imagen, prompt y respuesta, más los campos de referencia.

    La respuesta es el texto de `targets` y el reparto es el de `splits`. El prompt es el mismo para
    todas las filas, salvo si es una plantilla con {vcdr}, {hcdr} y {decision} (etapa de redacción de
    la configuración C): entonces se rellena con los campos de cada fila. Una receta sin imagen
    (columna `image` vacía) tiene que ser una plantilla, porque es la única vía por la que le llega
    la información. Se devuelve también el registro de qué archivos definieron los datos,
    con su suma sha256: el run_config dice así exactamente con qué se entrenó o se evaluó, y un
    cambio de una sola línea en los targets se nota.
    """
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    split_of = {r["image_id"]: r["split"] for r in read_csv(args.splits)}
    targets = read_csv(args.targets)
    assert {r["image_id"] for r in targets} == set(split_of), \
        "los targets y las particiones no cubren las mismas imágenes"
    text_only = not targets[0]["image"]
    assert all(bool(r["image"]) != text_only for r in targets), "los targets mezclan filas con y sin imagen"
    templated = "{vcdr}" in prompt
    assert templated or not text_only, "una receta sin imagen necesita un prompt con los campos"
    assert not templated or all(f"{{{k}}}" in prompt for k in ("hcdr", "decision")), "plantilla incompleta"
    rows = [
        {"id": r["image_id"], "image": None if text_only else str(Path(args.datasets_root) / r["image"]),
         "prompt": fill_prompt(prompt, r["vcdr"], r["hcdr"], r["decision"]) if templated else prompt,
         "response": r["text"],
         **{k: v for k, v in r.items() if k not in ("image_id", "image", "text")}}
        for r in targets if split_of[r["image_id"]] == args.split
    ]
    assert rows, f"la partición {args.split!r} está vacía"
    record = {
        "targets": args.targets, "targets_sha256": sha256_of(args.targets),
        "splits": args.splits, "splits_sha256": sha256_of(args.splits), "split": args.split,
        "prompt": prompt, "prompt_sha256": sha256_of(args.prompt_file),
        "datasets_root": args.datasets_root,
    }
    return rows, record


def git_record():
    """Qué código corrió: el commit y, si el árbol de `scripts/` tenía cambios sin commitear, cuáles.
    Un commit solo identifica la corrida si no hay nada pendiente; si la lista no está vacía, ese
    commit no reproduce lo que se ejecutó."""
    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_scripts": git("status", "--porcelain", "--", "scripts").splitlines()}


def cmd_train(args):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows, data_record = load_rows(args)

    processor, model = load_medgemma()
    target_modules = resolve_target_modules_for(args.layers, model, args.rank)

    # LoRA congela W y aprende un desvío de rango bajo: h = Wx + (alpha/rank)·B·A·x
    #   rank:    capacidad del adaptador (ancho de B y A)
    #   alpha:   escala del desvío; heurística común alpha = 2*rank
    #   dropout: apaga entradas del adaptador al azar durante el entrenamiento
    #            (regulariza; con datos escasos evita memorizar el train set)
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,  # batch efectivo = 1 * grad_accum
        learning_rate=args.learning_rate,
        bf16=True,
        # Recalcula activaciones en el backward en vez de guardarlas: menos
        # VRAM. Tiene que declararse acá y no llamando a
        # model.gradient_checkpointing_enable() a mano, porque con LoRA la
        # entrada del bloque recomputado no requiere gradiente (los pesos base
        # están congelados) y el backward se corta; TRL solo llama a
        # enable_input_require_grads() si ve esta bandera.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",                 # solo el adapter final, sin checkpoints intermedios
        remove_unused_columns=False,        # TRL no debe tocar nuestras columnas...
        dataset_kwargs={"skip_prepare_dataset": True},  # ...ni tokenizar: el collator hace todo
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,                # fija también el orden de barajado de los datos
    )
    collate = build_collator(processor)
    # Antes de entrenar, ver el enmascarado: esto es literalmente lo que va a
    # recibir gradiente. Si acá no aparece la respuesta del asistente, y solo
    # ella, no tiene sentido seguir.
    probe = collate(rows[:1])
    supervised = probe["input_ids"][0][probe["labels"][0] != -100]
    print(f"Tokens bajo la loss: {repr(processor.tokenizer.decode(supervised))}\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=rows,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate,
    )
    trainer.model.print_trainable_parameters()

    train_output = trainer.train()
    trainer.save_model(args.output_dir)     # guarda solo el adapter (MB), no el modelo base
    processor.save_pretrained(args.output_dir)

    trainable, total = trainer.model.get_nb_trainable_parameters()
    run_config = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_ID,
        **git_record(),
        "layers": args.layers,
        "target_modules": target_modules,
        "adapted_modules": sum(
            1 for n, _ in trainer.model.named_modules() if n.endswith("lora_A.default")
        ),
        "trainable_params": trainable,
        "total_params": total,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        **data_record,
        "examples": len(rows),
        "train_metrics": dict(train_output.metrics),
    }
    config_path = Path(args.output_dir) / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Adapter y {config_path} guardados")


# ---------------------------------------------------------------------------
# generate: inferencia, con o sin adapter
# ---------------------------------------------------------------------------

def sattolo_order(n):
    """Permutación cíclica de 0..n-1 con semilla fija (algoritmo de Sattolo): ninguna posición conserva su
    lugar y es la misma para todas las celdas, así que el control de imagen barajada es reproducible y
    emparejado entre ellas."""
    order, rng = list(range(n)), random.Random(0)
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    assert n > 1 and all(order[i] != i for i in range(n)), "la permutación deja una imagen en su sitio"
    return order


def cmd_generate(args):
    import torch
    from PIL import Image

    rows, data_record = load_rows(args)
    if args.limit:
        rows = rows[: args.limit]

    # Ejemplos con imagen (prompting con pocos ejemplos): cada uno es un turno de usuario (imagen y el mismo
    # prompt) seguido de la respuesta del asistente, antes de la consulta. Los controles de imagen cambian solo
    # la imagen de la consulta; los ejemplos son los mismos para todas las filas.
    shot_turns = []
    if args.shots:
        assert rows[0]["image"] is not None, "los ejemplos con imagen no se usan en una receta sin imagen"
        assert len({r["prompt"] for r in rows}) == 1, "con ejemplos el prompt tiene que ser el mismo para todas las filas"
        for shot in json.loads(Path(args.shots).read_text(encoding="utf-8")):
            shot_turns += [
                {"role": "user", "content": [
                    {"type": "image", "image": Image.open(Path(args.datasets_root) / shot["image"]).convert("RGB")},
                    {"type": "text", "text": rows[0]["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": shot["response"]}]},
            ]

    processor, model = load_medgemma()
    if args.adapter:
        from peft import PeftModel
        # Carga los pesos del adapter sobre el modelo base congelado. Sin esto,
        # "evaluar un adapter" evaluaría en realidad el modelo sin adaptar y el
        # resultado parecería válido: no hay error, solo números equivocados.
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # El checkpoint ya trae padding a la izquierda, que es lo correcto para
    # generar: con relleno a la derecha el modelo continuaría a partir del
    # relleno en vez del último token real.
    processor.tokenizer.padding_side = "left"

    # Los tres controles de imagen. Sirven para medir la contribución de la
    # imagen: si el rendimiento no cae al romper el vínculo
    # imagen-referencia, el modelo no está mirando la imagen y cualquier
    # comparación entre configuraciones de capas es ruido.
    #   real      la imagen que corresponde
    #   shuffled  la imagen de otra fila según una permutación fija (`sattolo_order`)
    #   blank     un rectángulo negro: mide cuánto sale del prior del lenguaje
    text_only = rows[0]["image"] is None
    assert not (text_only and args.image_modes), "una receta sin imagen no tiene controles de imagen"
    modes = ["text"] if text_only else args.image_modes or ["real", "shuffled", "blank"]
    order = sattolo_order(len(rows))
    sources = {
        "text": [None] * len(rows),
    } if text_only else {
        "real": [row["image"] for row in rows],
        "shuffled": [rows[order[i]]["image"] for i in range(len(rows))],
        "blank": [None] * len(rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        predictions = []
        for row, source in zip(rows, sources[mode]):
            if mode == "text":
                image = None
            else:
                image = (
                    Image.new("RGB", Image.open(row["image"]).size, "black")
                    if source is None
                    else Image.open(source).convert("RGB")
                )
            inputs = processor.apply_chat_template(
                shot_turns + [{"role": "user", "content": ([{"type": "image", "image": image}] if image is not None else [])
                              + [{"type": "text", "text": row["prompt"]}]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            if args.duplicate_bos:
                # Reproduce la entrada del entrenamiento: la plantilla ya escribe <bos> y el procesador del collator
                # añade otro. Se antepone un segundo <bos> a todos los tensores de la entrada para que la longitud
                # siga alineada (la máscara de atención y el tipo de token valen 1 y 0 en esa posición).
                bos = torch.tensor([[processor.tokenizer.bos_token_id]], device=model.device)
                assert inputs["input_ids"][0, 0] == bos[0, 0] and inputs["input_ids"][0, 1] != bos[0, 0]
                inputs["input_ids"] = torch.cat([bos, inputs["input_ids"]], dim=1)
                inputs["attention_mask"] = torch.cat([torch.ones_like(bos), inputs["attention_mask"]], dim=1)
                inputs["token_type_ids"] = torch.cat([torch.zeros_like(bos), inputs["token_type_ids"]], dim=1)
                assert inputs["input_ids"][0, :2].tolist() == [bos.item()] * 2
            with torch.inference_mode():
                # Decodificación voraz: sin muestreo la salida es determinista, así
                # que dos corridas de la misma configuración son comparables y la
                # varianza que se mida viene de las semillas de entrenamiento, no
                # del decodificador.
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            prediction = processor.decode(
                output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            predictions.append({**row, "image_used": source, "prediction": prediction})
            print(f"[{mode} {len(predictions):>3}/{len(rows)}] {row['id']}: {prediction[:110]}", flush=True)
        # Se guardan las predicciones crudas, no solo las métricas: el protocolo de
        # evaluación va a cambiar y regenerar cuesta GPU.
        (output_dir / f"pred_{mode}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8"
        )

    # Se escribe al final: su presencia indica que los tres controles terminaron.
    run_config = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_ID,
        **git_record(),
        "adapter": args.adapter,
        **data_record,
        "shots": args.shots, "shots_sha256": sha256_of(args.shots) if args.shots else None,
        "image_modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "duplicate_bos": args.duplicate_bos,
        "examples": len(rows),
    }
    config_path = output_dir / "generate_config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"\nPredicciones y {config_path} guardados en {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="anatomía del modelo: torres, bloques, proyecciones")
    inspect.add_argument("--full", action="store_true", help="lista todos los parámetros")
    inspect.add_argument("--pattern", help="filtro regex sobre los nombres")

    preview = sub.add_parser("preview", help="qué módulos tocaría LoRA y cuántos parámetros entrenaría")
    train = sub.add_parser("train", help="finetuning quirúrgico con LoRA")

    # preview y train comparten los hiperparámetros de LoRA: lo que previsualizas
    # es lo que entrenas
    for p in (preview, train):
        p.add_argument("--layers", required=True, help="'all-linear', grupo canónico, regex o random:GRUPO (control aleatorio)")
        p.add_argument("--rank", type=int, default=16)
        p.add_argument("--alpha", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
    add_data_arguments(train)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=float, default=5)
    train.add_argument("--grad-accum", type=int, default=4)

    generate = sub.add_parser("generate", help="inferencia sobre una partición, con o sin adapter")
    add_data_arguments(generate)
    generate.add_argument("--output-dir", required=True, help="carpeta donde se escriben pred_<modo>.jsonl")
    generate.add_argument("--adapter", help="directorio del adapter; sin esto, el modelo base")
    generate.add_argument(
        "--image-modes", nargs="+", choices=("real", "shuffled", "blank"),
        help="controles de contribución de la imagen; por defecto los tres",
    )
    generate.add_argument("--shots", help="JSON con ejemplos [{image, response}] que preceden a cada consulta (pocos ejemplos)")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--duplicate-bos", action="store_true", help="antepone un segundo <bos>, como ve la entrada el entrenamiento")
    generate.add_argument("--limit", type=int, help="corta el dataset; para pruebas de humo")

    args = parser.parse_args()
    {
        "inspect": cmd_inspect,
        "preview": cmd_preview,
        "train": cmd_train,
        "generate": cmd_generate,
    }[args.command](args)


if __name__ == "__main__":
    main()
