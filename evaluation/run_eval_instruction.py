#!/usr/bin/env python
"""
Instruction evaluation placeholder script.

This script demonstrates how to evaluate a model on instruction-following tasks.
Currently outputs generations for manual review.

Future enhancements:
- Add automatic instruction-following metrics (e.g., reward models)
- Integration with MT-Bench style evaluation
- Custom Pashto instruction evaluation criteria
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

def load_prompts(file_path: str) -> List[str]:
    """Load prompts from file (one per line or JSON)."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # Try JSON format first
                    data = json.loads(line)
                    if isinstance(data, dict) and 'prompt' in data:
                        prompts.append(data['prompt'])
                    else:
                        prompts.append(str(data))
                except json.JSONDecodeError:
                    # Plain text format
                    prompts.append(line)
    return prompts

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Model ID")
    parser.add_argument("--prompts_file", required=True, help="File with prompts")
    parser.add_argument("--output_file", default="instruction_eval_outputs.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompts", type=int, help="Limit number of prompts")
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

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

    # Generate responses
    results = []
    print(f"Generating responses for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        if i % 5 == 0:
            print(f"Progress: {i}/{len(prompts)}")
        
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        
        result = {
            "prompt": prompt,
            "response": response,
            "model_id": args.model_id,
            "index": i
        }
        results.append(result)

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n=== INSTRUCTION EVALUATION COMPLETE ===")
    print(f"Generated {len(results)} responses")
    print(f"Results saved to: {args.output_file}")
    print("\nSample outputs:")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Response: {result['response'][:200]}...")

if __name__ == "__main__":
    main()