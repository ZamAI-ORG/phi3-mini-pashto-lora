#!/usr/bin/env python
"""Evaluation script for ZamAI Phi-3 Pashto model."""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.metrics import compute_all_metrics


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from model."""
    # Format prompt for chat
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy for evaluation consistency
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_perplexity(model, tokenizer, texts: List[str], batch_size: int = 8) -> float:
    """Compute perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Count non-padding tokens
            attention_mask = inputs["attention_mask"]
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def evaluate_generation(model, tokenizer, data: List[Dict], args) -> Dict[str, Any]:
    """Evaluate generation quality using BLEU/chrF metrics."""
    predictions = []
    references = []

    print(f"Generating responses for {len(data)} samples...")

    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(data)}")

        # Get source text (instruction/input)
        if args.source_field in item:
            source = item[args.source_field]
        else:
            # Fallback: try common field names
            source = item.get("instruction", item.get("input", item.get("prompt", "")))

        # Get reference text (expected output)
        if args.reference_field in item:
            reference = item[args.reference_field]
        else:
            # Fallback: try common field names
            reference = item.get("output", item.get("response", item.get("target", "")))

        if not source or not reference:
            print(f"Warning: Missing source or reference for item {i}")
            continue

        # Generate prediction
        try:
            prediction = generate_text(model, tokenizer, source, args.max_new_tokens)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(f"Error generating for item {i}: {e}")
            continue

    if not predictions:
        raise ValueError("No valid predictions generated")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(predictions, references)

    return {
        "metrics": metrics,
        "num_predictions": len(predictions),
        "predictions": predictions[:5],  # Sample predictions for inspection
        "references": references[:5],  # Sample references for inspection
    }


def evaluate_perplexity(model, tokenizer, data: List[Dict], args) -> Dict[str, float]:
    """Evaluate perplexity on the dataset."""
    print(f"Computing perplexity for {len(data)} samples...")

    # Extract texts for perplexity computation
    texts = []
    for item in data:
        # Use the full text (instruction + output) for perplexity
        if args.source_field in item and args.reference_field in item:
            full_text = f"{item[args.source_field]}\n{item[args.reference_field]}"
        else:
            # Fallback to any available text
            text = item.get("text", "")
            if not text:
                # Combine available fields
                parts = []
                for key in ["instruction", "input", "output", "response"]:
                    if key in item and item[key]:
                        parts.append(str(item[key]))
                text = "\n".join(parts)
            full_text = text

        if full_text.strip():
            texts.append(full_text)

    if not texts:
        raise ValueError("No valid texts found for perplexity computation")

    perplexity = compute_perplexity(model, tokenizer, texts, args.batch_size)

    return {"perplexity": perplexity, "num_samples": len(texts)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate ZamAI Phi-3 Pashto model")
    parser.add_argument("--model_id", required=True, help="Model ID or path")
    parser.add_argument("--data_file", required=True, help="Evaluation data file (JSONL)")
    parser.add_argument(
        "--eval_type", choices=["generation", "perplexity", "both"], default="both", help="Type of evaluation to run"
    )

    # Field mapping
    parser.add_argument("--source_field", default="instruction", help="Field name for input/source text")
    parser.add_argument("--reference_field", default="output", help="Field name for reference/target text")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens for generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for perplexity computation")

    # Limits
    parser.add_argument("--max_samples", type=int, help="Limit number of samples to evaluate")

    # Output
    parser.add_argument("--output_file", default="evaluation_results.json", help="Output file for results")

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    print(f"Loading data from: {args.data_file}")
    data = load_jsonl(args.data_file)

    if args.max_samples:
        data = data[: args.max_samples]

    print(f"Evaluating on {len(data)} samples")

    # Load model
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")

    # Run evaluation
    results = {"model_id": args.model_id, "data_file": args.data_file, "num_samples": len(data), "args": vars(args)}

    if args.eval_type in ["generation", "both"]:
        print("\n" + "=" * 50)
        print("GENERATION EVALUATION")
        print("=" * 50)

        gen_results = evaluate_generation(model, tokenizer, data, args)
        results["generation"] = gen_results

        print("\n=== GENERATION RESULTS ===")
        for metric, value in gen_results["metrics"].items():
            if isinstance(value, float):
                print(f"{metric.upper()}: {value:.2f}")
            else:
                print(f"{metric.upper()}: {value}")

    if args.eval_type in ["perplexity", "both"]:
        print("\n" + "=" * 50)
        print("PERPLEXITY EVALUATION")
        print("=" * 50)

        ppl_results = evaluate_perplexity(model, tokenizer, data, args)
        results["perplexity"] = ppl_results

        print("\n=== PERPLEXITY RESULTS ===")
        print(f"PERPLEXITY: {ppl_results['perplexity']:.2f}")
        print(f"SAMPLES: {ppl_results['num_samples']}")

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation complete! Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
