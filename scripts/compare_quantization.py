#!/usr/bin/env python
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(mode: str, model_id: str):
    kwargs = {"device_map": "auto"}
    if mode == "8bit":
        kwargs["load_in_8bit"] = True
    elif mode == "4bit":
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def format_mem():
    if not torch.cuda.is_available():
        return "N/A"
    return f"{torch.cuda.max_memory_allocated()/1024**2:.1f} MB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--modes", nargs="+", default=["fp16", "8bit", "4bit"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    results = []
    for mode in args.modes:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0_load = time.time()
        model, tokenizer = load_model(mode, args.model_id)
        load_time = time.time() - t0_load

        # warmup
        for _ in range(args.warmup):
            _ = generate(model, tokenizer, args.prompt, max_new_tokens=8)

        t0 = time.time()
        text = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
        gen_time = time.time() - t0

        results.append({
            "mode": mode,
            "load_s": load_time,
            "gen_s": gen_time,
            "peak_mem": format_mem(),
            "output_preview": text[:120].replace("\n", " ")
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Mode | Load(s) | Gen(s) | PeakMem | OutputPreview")
    for r in results:
        print(f"{r['mode']:>4} | {r['load_s']:.2f} | {r['gen_s']:.2f} | {r['peak_mem']:>8} | {r['output_preview']}")

if __name__ == "__main__":
    main()