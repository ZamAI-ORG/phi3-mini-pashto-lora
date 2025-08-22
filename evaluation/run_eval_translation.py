#!/usr/bin/env python
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluation.metrics import compute_all_metrics
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Model ID")
    parser.add_argument("--file", help="JSONL file path")
    parser.add_argument("--dataset", help="HF dataset name")
    parser.add_argument("--source_field", default="instruction", help="Input field name")
    parser.add_argument("--reference_field", default="output", help="Reference field name")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, help="Limit number of samples")
    args = parser.parse_args()

    # Load data
    if args.file:
        data = load_jsonl(args.file)
    elif args.dataset:
        dataset = load_dataset(args.dataset, split="test")
        data = [dict(item) for item in dataset]
    else:
        raise ValueError("Either --file or --dataset must be provided")

    if args.max_samples:
        data = data[:args.max_samples]

    # Load model
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate predictions
    predictions = []
    references = []
    
    print(f"Evaluating {len(data)} samples...")
    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(data)}")
        
        source = item[args.source_field]
        reference = item[args.reference_field]
        
        prediction = generate_text(model, tokenizer, source, args.max_new_tokens)
        
        predictions.append(prediction)
        references.append(reference)

    # Compute metrics
    metrics = compute_all_metrics(predictions, references)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"BLEU: {metrics['bleu']:.2f}")
    print(f"chrF: {metrics['chrf']:.2f}")
    print(f"Samples: {metrics['num_predictions']}")
    
    # Save detailed results
    results = {
        "metrics": metrics,
        "predictions": predictions[:5],  # Sample predictions
        "references": references[:5],   # Sample references
        "model_id": args.model_id,
        "args": vars(args)
    }
    
    with open("translation_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: translation_eval_results.json")

if __name__ == "__main__":
    main()