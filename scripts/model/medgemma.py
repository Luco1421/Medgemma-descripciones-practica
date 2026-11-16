import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/medgemma-1.5-4b-it"


def load_medgemma(device_map="auto"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device_map
    )
    return processor, model


def generate(processor, model, prompt, image=None, max_new_tokens=256):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


if __name__ == "__main__":
    # Prototipo de inferencia: entra una fotografía de fondo de ojo y un prompt, sale la descripción del modelo y los campos que
    # se leen de ella. Sin --adapter usa el modelo sin adaptar; con él, el adaptador LoRA indicado.
    parser = argparse.ArgumentParser(description="Genera la descripción de una fotografía de fondo de ojo con MedGemma, con o sin adaptador LoRA")
    parser.add_argument("--image", required=True, help="fotografía de fondo de ojo centrada en el disco óptico")
    parser.add_argument("--adapter", help="carpeta del adaptador LoRA (p. ej. experiments/chaksu_json_v1/all-linear_s42)")
    parser.add_argument("--recipe", default="json_v1", choices=("json_v1", "prose_v1"),
                        help="prompt con el que se entrenó el adaptador: campos en JSON o una frase")
    parser.add_argument("--prompt", help="prompt propio; sustituye al de la receta")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from scripts.eval.evaluate import parse_answer
    from scripts.eval.fields import extract_prose

    prompt = args.prompt or Path(f"data/targets/chaksu__{args.recipe}.prompt.txt").read_text(encoding="utf-8").strip()
    processor, model = load_medgemma()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    answer = generate(processor, model, prompt, Image.open(args.image).convert("RGB"), args.max_new_tokens)
    print(f"Prompt: {prompt}\nAdaptador: {args.adapter or 'ninguno (modelo sin adaptar)'}\n\nRespuesta del modelo:\n{answer}\n")
    fields = parse_answer(answer) if args.recipe == "json_v1" else extract_prose(answer)
    print(f"Campos leídos de la respuesta: {fields}")
