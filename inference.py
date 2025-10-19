#!/usr/bin/env python
"""Inference script for ZamAI Phi-3 Pashto model."""

import argparse
import json
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: Optional[str] = None):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto" if device is None else device, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_chat_format: bool = True,
) -> str:
    """Generate response from the model."""

    # Format prompt for chat if requested
    if use_chat_format:
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        formatted_prompt = prompt

    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Extract only the generated part
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


def interactive_mode(model, tokenizer, args):
    """Run interactive chat mode."""
    print("=" * 60)
    print("🧠 ZamAI Phi-3 Pashto Interactive Mode")
    print("=" * 60)
    print("Enter your messages in Pashto or English. Type 'quit' to exit.")
    print("د پښتو یا انګریزي کې خپل پیغامونه ولیکئ. د وتلو لپاره 'quit' وټایپ کړئ.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye! / تور مل وي!")
                break

            if not user_input:
                continue

            print("🤖 Assistant:", end=" ", flush=True)

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                use_chat_format=args.use_chat_format,
            )

            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye! / تور مل وي!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def batch_mode(model, tokenizer, args):
    """Run batch inference mode."""
    if args.input_file:
        # Read prompts from file
        with open(args.input_file, "r", encoding="utf-8") as f:
            if args.input_file.endswith(".jsonl"):
                prompts = []
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, dict):
                        prompts.append(data.get("prompt", str(data)))
                    else:
                        prompts.append(str(data))
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin
        prompts = [line.strip() for line in sys.stdin if line.strip()]

    results = []

    print(f"Processing {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(prompts)}")

        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                use_chat_format=args.use_chat_format,
            )

            result = {"prompt": prompt, "response": response, "index": i}
            results.append(result)

            if args.output_file:
                # Save incrementally to file
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                # Print to stdout
                print(f"\n--- Result {i + 1} ---")
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print("-" * 40)

        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            continue

    print(f"\n✅ Processed {len(results)} prompts successfully")

    if args.output_file:
        print(f"Results saved to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="ZamAI Phi-3 Pashto Inference")
    parser.add_argument("--model_id", required=True, help="Model ID or path")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive", help="Inference mode")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Use sampling (vs greedy decoding)")
    parser.add_argument("--no_sampling", action="store_true", help="Disable sampling (use greedy decoding)")
    parser.add_argument("--use_chat_format", action="store_true", default=True, help="Use chat format for prompts")
    parser.add_argument("--no_chat_format", action="store_true", help="Disable chat format")

    # Batch mode parameters
    parser.add_argument("--input_file", help="Input file for batch mode")
    parser.add_argument("--output_file", help="Output file for batch mode")

    # Single prompt for quick testing
    parser.add_argument("--prompt", help="Single prompt for quick testing")

    args = parser.parse_args()

    # Handle flag conflicts
    if args.no_sampling:
        args.do_sample = False
    if args.no_chat_format:
        args.use_chat_format = False

    # Load model
    model, tokenizer = load_model(args.model_id)

    # Single prompt mode
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print("-" * 40)

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            use_chat_format=args.use_chat_format,
        )

        print(f"Response: {response}")
        return

    # Run inference mode
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args)
    elif args.mode == "batch":
        batch_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
