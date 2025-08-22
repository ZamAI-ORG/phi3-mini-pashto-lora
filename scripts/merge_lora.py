#!/usr/bin/env python
import argparse

from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model ID")
    parser.add_argument("--lora_model", required=True, help="LoRA adapter path/ID")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", help="Hub model ID for upload")
    parser.add_argument("--hf_token", help="Hugging Face token")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.lora_model}")
    model = PeftModel.from_pretrained(model, args.lora_model)

    print("Merging LoRA adapter...")
    model = model.merge_and_unload()

    print(f"Saving to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id required when --push_to_hub is set")
        print(f"Pushing to hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    print("✅ Merge complete!")


if __name__ == "__main__":
    main()
